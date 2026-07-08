"""Email notification hook for training runs.

Sends notifications on three events:
  * `OK`   — clean training completion (after_run)
  * `EVAL` — each validation milestone with mIoU (after_val_epoch)
  * `ERR`  — uncaught exception during training (via send_crash_email, called
             from a try/except wrapper in tools/train.py)

Also adds a NaN/Inf loss guard that converts the most common silent failure
mode (non-finite loss) into a reportable RuntimeError.

Configuration is split:
  * Non-secret knobs live in the MMCV config (subject_prefix, milestone_every,
    log_tail_lines).
  * Secrets are read from environment variables at hook instantiation
    (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_USE_SSL,
    NOTIFY_RECIPIENT). Missing creds disable the hook with a WARNING; training
    proceeds normally.

The hook never raises during a send — failures are logged and (best-effort)
persisted to `{work_dir}/.notification_failed_{event}.json` so undelivered
notifications can be recovered.
"""
import json
import logging
import os
import os.path as osp
import smtplib
import socket
import ssl
import time
import traceback
from email.message import EmailMessage

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info

_REQUIRED_ENV = ('SMTP_HOST', 'SMTP_USER', 'SMTP_PASSWORD', 'NOTIFY_RECIPIENT')
_RETRY_DELAYS = (1, 4, 16)
_SMTP_TIMEOUT = 30
_DOTENV_LOADED = False


def _find_dotenv():
    """Walk up from CWD looking for a `.env` file. Returns the path or None."""
    here = osp.abspath(os.getcwd())
    for _ in range(6):
        candidate = osp.join(here, '.env')
        if osp.isfile(candidate):
            return candidate
        parent = osp.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _autoload_dotenv():
    """Parse a `.env` file in the project root and populate os.environ.

    Existing environment variables always win (so a value set in the shell
    overrides the file). Runs at most once per process.

    Supported syntax: `KEY=VALUE` lines, `#` comments, blank lines, optional
    surrounding single/double quotes on VALUE. No interpolation, no `export`
    keyword required (but tolerated).
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    path = _find_dotenv()
    if not path:
        return
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[len('export '):].lstrip()
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                if (len(value) >= 2
                        and value[0] == value[-1]
                        and value[0] in ('"', "'")):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def _load_smtp_config():
    """Read SMTP settings from env. Returns (config, missing_keys).

    SMTP_PASSWORD is normalized: spaces and dashes are stripped so the user
    can store a Gmail app password in any of these equivalent forms:
        yaklbmnjiybmvxdy
        yakl bmnj iybm vxdy
        yakl-bmnj-iybm-vxdy
    """
    _autoload_dotenv()
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        return None, missing
    raw_pw = os.environ['SMTP_PASSWORD']
    password = raw_pw.replace(' ', '').replace('-', '')
    cfg = dict(
        host=os.environ['SMTP_HOST'],
        port=int(os.environ.get('SMTP_PORT', '465')),
        user=os.environ['SMTP_USER'],
        password=password,
        recipient=os.environ['NOTIFY_RECIPIENT'],
        use_ssl=os.environ.get('SMTP_USE_SSL', '1') not in ('0', 'false', 'False', ''),
    )
    return cfg, []


def _tail(path, n):
    """Return the last `n` lines of `path` as a single string. Empty string if
    the file does not exist or cannot be read."""
    if not path or not osp.isfile(path):
        return ''
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b''
            while size > 0 and data.count(b'\n') <= n:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
            lines = data.splitlines()[-n:]
            return b'\n'.join(lines).decode('utf-8', errors='replace')
    except Exception:
        return ''


def _persist_failure(work_dir, event, subject, body, exc_text):
    """Write a breadcrumb file so a failed send is recoverable."""
    if not work_dir:
        return
    try:
        os.makedirs(work_dir, exist_ok=True)
        path = osp.join(work_dir, f'.notification_failed_{event}.json')
        with open(path, 'w') as f:
            json.dump({
                'event': event,
                'subject': subject,
                'body': body,
                'exception': exc_text,
                'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            }, f, indent=2)
    except Exception:
        pass


def _send(subject, body, smtp_cfg, work_dir=None, event='generic',
          logger=None):
    """Send a plain-text email. Never raises; returns True on success.

    Retries with exponential backoff on transient SMTP errors. On terminal
    failure, writes a breadcrumb file under `work_dir`.
    """
    log = logger or logging.getLogger('mmseg')

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = smtp_cfg['user']
    msg['To'] = smtp_cfg['recipient']
    host = socket.gethostname()
    msg['Message-ID'] = f'<{time.time_ns()}.{event}@{host}>'
    msg.set_content(body)

    last_exc = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            if smtp_cfg['use_ssl']:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(smtp_cfg['host'], smtp_cfg['port'],
                                      timeout=_SMTP_TIMEOUT, context=ctx) as s:
                    s.login(smtp_cfg['user'], smtp_cfg['password'])
                    s.send_message(msg)
            else:
                with smtplib.SMTP(smtp_cfg['host'], smtp_cfg['port'],
                                  timeout=_SMTP_TIMEOUT) as s:
                    s.starttls(context=ssl.create_default_context())
                    s.login(smtp_cfg['user'], smtp_cfg['password'])
                    s.send_message(msg)
            log.info(f'EmailNotificationHook: sent {subject!r} to '
                     f'{smtp_cfg["recipient"]}')
            return True
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < len(_RETRY_DELAYS) - 1:
                log.warning(f'EmailNotificationHook: send failed '
                            f'({type(e).__name__}: {e}); retrying in {delay}s')
                time.sleep(delay)

    log.error(f'EmailNotificationHook: send failed after '
              f'{len(_RETRY_DELAYS)} attempts: {last_exc!r}')
    _persist_failure(work_dir, event, subject, body, repr(last_exc))
    return False


def _format_duration(seconds):
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f'{h}h{m:02d}m{s:02d}s'
    if m:
        return f'{m}m{s:02d}s'
    return f'{s}s'


def send_test_email():
    """Standalone helper: validate SMTP credentials without launching a run.

    Usage:
        set -a; . .env; set +a
        python -c "from mmseg.core.hook.notification_hook \
            import send_test_email; send_test_email()"
    """
    logging.basicConfig(level=logging.INFO)
    smtp_cfg, missing = _load_smtp_config()
    if missing:
        print(f'Missing env vars: {missing}')
        return False
    host = socket.gethostname()
    ok = _send(
        subject=f'[daformer][TEST] smoke test @ {host}',
        body=(f'EmailNotificationHook smoke test from {host}.\n'
              f'If you see this, SMTP credentials are working.\n'),
        smtp_cfg=smtp_cfg, event='test')
    print('OK' if ok else 'FAILED')
    return ok


def send_crash_email(cfg, exc, work_dir, timestamp):
    """Send a crash notification from outside the runner.

    Called from the try/except wrapper in tools/train.py. Best-effort: any
    exception inside this function is swallowed so the original training
    error is never masked.
    """
    try:
        rank, _ = get_dist_info()
        if rank != 0:
            return
        smtp_cfg, missing = _load_smtp_config()
        if missing:
            return
        exp_name = osp.splitext(osp.basename(
            getattr(cfg, 'filename', '') or ''))[0] or 'unknown'
        host = socket.gethostname()
        tb = ''.join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        log_file = osp.join(work_dir, f'{timestamp}.log') if work_dir else ''
        tail = _tail(log_file, 50)
        subject = f'[daformer][ERR] {exp_name} @ {host}'
        body = (
            f'Training crashed.\n\n'
            f'exp_name : {exp_name}\n'
            f'host     : {host}\n'
            f'work_dir : {work_dir}\n'
            f'log_file : {log_file}\n'
            f'exception: {type(exc).__name__}: {exc}\n\n'
            f'--- traceback ---\n{tb}\n'
            f'--- last 50 log lines ---\n{tail}\n')
        _send(subject, body, smtp_cfg, work_dir=work_dir, event='err')
    except Exception:
        pass


@HOOKS.register_module()
class EmailNotificationHook(Hook):
    """MMCV hook that emails on successful completion, eval milestones, and
    raises a clean RuntimeError on non-finite loss (so crashes route to the
    train.py wrapper which sends the error email).
    """

    def __init__(self,
                 subject_prefix='[daformer]',
                 milestone_every=1,
                 log_tail_lines=50,
                 nan_guard=True,
                 smoke_test=None):
        self.subject_prefix = subject_prefix
        self.milestone_every = max(1, int(milestone_every))
        self.log_tail_lines = int(log_tail_lines)
        self.nan_guard = bool(nan_guard)
        if smoke_test is None:
            smoke_test = os.environ.get('NOTIFY_SMOKE_TEST', '0') == '1'
        self.smoke_test = bool(smoke_test)

        self._smtp_cfg, missing = _load_smtp_config()
        self.enabled = not missing
        self._missing = missing
        self._milestone_count = 0
        self._start_monotonic = None
        self._log_file = None
        self._exp_name = None
        self._host = socket.gethostname()
        self._work_dir = None

    def _is_rank0(self):
        rank, _ = get_dist_info()
        return rank == 0

    def _logger(self, runner):
        return getattr(runner, 'logger', None) or logging.getLogger('mmseg')

    def _warn_disabled_once(self, runner):
        if self.enabled:
            return False
        self._logger(runner).warning(
            f'EmailNotificationHook disabled: missing env vars '
            f'{self._missing}. Training will proceed without notifications.')
        return True

    def before_run(self, runner):
        if not self._is_rank0():
            return
        self._start_monotonic = time.monotonic()
        self._work_dir = runner.work_dir
        self._log_file = osp.join(
            runner.work_dir, f'{runner.timestamp}.log')
        self._exp_name = (runner.meta or {}).get('exp_name', 'unknown')

        if self._warn_disabled_once(runner):
            return

        if self.smoke_test:
            subject = (f'{self.subject_prefix}[START] {self._exp_name} '
                       f'@ {self._host}')
            body = (f'Training starting.\n\n'
                    f'exp_name : {self._exp_name}\n'
                    f'host     : {self._host}\n'
                    f'work_dir : {self._work_dir}\n'
                    f'log_file : {self._log_file}\n'
                    f'max_iters: {getattr(runner, "max_iters", "?")}\n')
            _send(subject, body, self._smtp_cfg, work_dir=self._work_dir,
                  event='start', logger=self._logger(runner))

    def after_train_iter(self, runner):
        if not self.nan_guard:
            return
        outputs = getattr(runner, 'outputs', None)
        if not outputs:
            return
        loss = outputs.get('loss')
        if loss is None or not torch.is_tensor(loss):
            return
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f'Non-finite loss {loss.item()} at iter {runner.iter}')

    def after_val_epoch(self, runner):
        if not self._is_rank0() or not self.enabled:
            return
        self._milestone_count += 1
        if (self._milestone_count - 1) % self.milestone_every != 0:
            return
        metrics = self._extract_metrics(runner)
        miou = metrics.get('mIoU')
        subject = (f'{self.subject_prefix}[EVAL] {self._exp_name} '
                   f'mIoU={miou:.4f}' if isinstance(miou, float)
                   else f'{self.subject_prefix}[EVAL] {self._exp_name}')
        subject = f'{subject} @ {self._host}'
        body = self._compose_body(runner, status='EVAL', metrics=metrics)
        _send(subject, body, self._smtp_cfg, work_dir=self._work_dir,
              event='eval', logger=self._logger(runner))

    def after_run(self, runner):
        if not self._is_rank0() or not self.enabled:
            return
        metrics = self._extract_metrics(runner)
        miou = metrics.get('mIoU')
        suffix = (f'mIoU={miou:.4f}' if isinstance(miou, float) else 'done')
        subject = (f'{self.subject_prefix}[OK] {self._exp_name} {suffix} '
                   f'@ {self._host}')
        body = self._compose_body(runner, status='OK', metrics=metrics)
        _send(subject, body, self._smtp_cfg, work_dir=self._work_dir,
              event='ok', logger=self._logger(runner))

    def _extract_metrics(self, runner):
        """Pull mIoU/mAcc/aAcc from runner.log_buffer.output if present."""
        out = {}
        buf = getattr(runner, 'log_buffer', None)
        if buf is None:
            return out
        for k in ('mIoU', 'mAcc', 'aAcc'):
            v = buf.output.get(k)
            if v is None:
                continue
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        return out

    def _compose_body(self, runner, status, metrics):
        elapsed = (time.monotonic() - self._start_monotonic
                   if self._start_monotonic else 0)
        metrics_str = '\n'.join(
            f'  {k:5s}: {v:.4f}' if isinstance(v, float) else f'  {k:5s}: {v}'
            for k, v in metrics.items()) or '  (none reported yet)'
        tail = _tail(self._log_file, self.log_tail_lines)
        return (
            f'status   : {status}\n'
            f'exp_name : {self._exp_name}\n'
            f'host     : {self._host}\n'
            f'work_dir : {self._work_dir}\n'
            f'log_file : {self._log_file}\n'
            f'iter     : {getattr(runner, "iter", "?")} / '
            f'{getattr(runner, "max_iters", "?")}\n'
            f'elapsed  : {_format_duration(elapsed)}\n'
            f'metrics  :\n{metrics_str}\n\n'
            f'--- last {self.log_tail_lines} log lines ---\n{tail}\n')
