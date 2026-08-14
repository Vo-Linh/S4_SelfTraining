"""Generate labeled/unlabeled SSL splits at the paper's 1/5/10% ratios.

Partitions the OEM train pool into a labeled subset and a DISJOINT unlabeled
subset ("the remaining training images form the unlabeled pool", STIC_semi
§V-A) -- unlike the legacy train_500_fixed / train_3500_fixed pair, where the
labeled images reappear in the unlabeled list.

Deterministic: same --seed always yields the same partition, so every method
compared in Table IX sees the identical split.

Usage (on the box, from the S4 repo root):
    python tools/make_ssl_splits.py \
        --pool /home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/train_3500_fixed.txt \
        --out-dir /home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat \
        --ratios 1 5 10 --seed 0
"""
import argparse
import os.path as osp
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool', required=True,
                   help='split file listing the full train pool')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--ratios', type=float, nargs='+', default=[1, 5, 10],
                   help='labeled percentages')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    with open(args.pool) as f:
        items = [ln.strip() for ln in f if ln.strip()]
    # A pool with duplicates would silently leak labeled images into the
    # unlabeled half, so collapse them before partitioning.
    items = sorted(set(items))
    n = len(items)

    for ratio in args.ratios:
        rng = random.Random(args.seed)
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_lab = max(1, round(n * ratio / 100.0))
        labeled, unlabeled = shuffled[:n_lab], shuffled[n_lab:]

        tag = f'{ratio:g}pct'
        for name, subset in (('labeled', labeled), ('unlabeled', unlabeled)):
            path = osp.join(args.out_dir, f'train_{tag}_{name}.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(sorted(subset)) + '\n')
            print(f'{path}  ({len(subset)})')
        assert not (set(labeled) & set(unlabeled))
    print(f'pool = {n} unique images')


if __name__ == '__main__':
    main()
