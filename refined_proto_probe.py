"""Decisive probe: do the EM-REFINED DynamicAnchor prototypes organize into
interior (class-pure) vs interface (class-mixed) structure on real OEM data?

Tests the actual claim of DAPCN_Report section 4.7.3 / Figure 7, using the
refined prototypes computed on real features (not the static seed the viz tool
plots). Read-only: builds student + DAM from the real config + checkpoint.
"""
import glob, os.path as osp, numpy as np, torch, torch.nn.functional as F
import mmcv
from mmcv.utils import Config
from mmseg.models.builder import build_segmentor, MODELS

CFG = "work_dirs/t9_dapcn_unetformer_rx101_5pct/ssl_oem_dapcn_unetformer_resnext101.py"
CKPT = "work_dirs/t9_dapcn_unetformer_rx101_5pct/iter_40000.pth"
N_IMG, SIZE = 16, 512
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config.fromfile(CFG)
sd = torch.load(CKPT, map_location="cpu")["state_dict"]

# ---- student backbone ----
student = build_segmentor(cfg.model)
msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
miss, unexp = student.load_state_dict(msd, strict=False)
print(f"[student] loaded; missing={len(miss)} unexpected={len(unexp)}")
student.eval().to(dev)

# ---- DynamicAnchorModule ----
da_cfg = dict(cfg.uda.dynamic_anchor); da_cfg["feature_dim"] = 2048
dam = MODELS.build(da_cfg)
dsd = {k[len("dynamic_anchor."):]: v for k, v in sd.items() if k.startswith("dynamic_anchor.")}
dam.load_state_dict(dsd, strict=False)
dam.eval().to(dev)

# ---- data ----
root = cfg.data_root
anns = sorted(glob.glob(osp.join(root, "annotations/val", "*.tif")))[:N_IMG]
imgs = [a.replace("annotations/val", "images/val") for a in anns]
assert imgs, f"no annotations under {osp.join(root,'annotations/val')}"
mean = np.array(cfg.img_norm_cfg["mean"]); std = np.array(cfg.img_norm_cfg["std"])
to_rgb = cfg.img_norm_cfg.get("to_rgb", True)

def load(pth):
    img = mmcv.imread(pth)                      # BGR HWC
    ann_p = pth.replace("images/val", "annotations/val")
    base = osp.splitext(ann_p)[0]
    cand = [ann_p] + [base + e for e in (".png", ".tif", ".tiff")]
    ann_p = next((c for c in cand if osp.exists(c)), None)
    gt = mmcv.imread(ann_p, flag="unchanged") if ann_p else None
    img = mmcv.imresize(img, (SIZE, SIZE))
    if gt is not None:
        gt = mmcv.imresize(gt, (SIZE, SIZE), interpolation="nearest")
        if gt.ndim == 3: gt = gt[..., 0]
    im = mmcv.imnormalize(img, mean, std, to_rgb)
    return torch.from_numpy(im.transpose(2, 0, 1)).float(), gt

batch, gts = [], []
for p in imgs:
    t, gt = load(p); batch.append(t); gts.append(gt)
batch = torch.stack(batch).to(dev)

with torch.no_grad():
    feats = student.backbone(batch)
    f = feats[-1]                                # (B,2048,h,w)
    B, C, h, w = f.shape
    assign, proto, quality = dam(f)              # (N,K'), (K',C), (K',)

print(f"[feat] {tuple(f.shape)}  -> N={B*h*w} pixels | refined proto {tuple(proto.shape)} | quality mean {quality.mean():.3f}")

# ---- refined inter-prototype structure ----
pn = F.normalize(proto, dim=1)
S = (pn @ pn.t()).cpu().numpy()
K = S.shape[0]; off = S[~np.eye(K, dtype=bool)]
# count near-duplicate prototypes (collapse indicator)
dup = int(((S - np.eye(K)) > 0.95).sum() // 2)
print(f"[refined protos] K'={K} | inter-sim mean {off.mean():.3f} mean|.| {np.abs(off).mean():.3f} | near-dupe pairs(>0.95): {dup}")

# ---- class correlation of refined prototypes ----
gt_small = []
for gt in gts:
    g = torch.from_numpy(mmcv.imresize(gt, (w, h), interpolation="nearest").astype("int64"))
    gt_small.append(g.reshape(-1))
labels = torch.cat(gt_small).numpy()             # (N,)
hard = assign.argmax(1).cpu().numpy()            # (N,) proto id per pixel
uniq = np.unique(labels[labels != 255])
print(f"[gt] classes present: {uniq.tolist()}")

purity, active = [], 0
interior = interface = 0
for k in range(K):
    m = (hard == k) & (labels != 255)
    n = int(m.sum())
    if n < 20:  # prototype claims too few pixels
        continue
    active += 1
    hist = np.bincount(labels[m], minlength=int(labels.max()) + 1).astype(float)
    hist /= hist.sum()
    top = np.sort(hist)[::-1]
    pur = top[0]                                   # dominant-class fraction
    purity.append(pur)
    if pur > 0.5: interior += 1
    elif top[0] + top[1] > 0.6 and top[1] > 0.2: interface += 1
purity = np.array(purity)
print(f"[assignment] active protos(>=20px)={active} | how many distinct protos used: {len(np.unique(hard))}")
print(f"[purity] mean {purity.mean():.3f} median {np.median(purity):.3f} | interior(>0.5): {interior} | interface(mixed 2-class): {interface}")
frac_top = np.bincount(hard, minlength=K); frac_top = np.sort(frac_top)[::-1] / frac_top.sum()
print(f"[load] top-5 protos capture {frac_top[:5].sum()*100:.1f}% of pixels (collapse if ~100%)")
