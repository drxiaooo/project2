# src/seg/qc_filter_masks.py
# QC + filter pseudo masks (npz) based on simple heuristics.
#
# Expected outputs from seg scripts (SAM2/MedSAM2/etc):
#   - mask stats csv: mask_stats_*_image.csv (auto-detect) OR user-specified via --stats
#   - npz dir: pseudo_masks_*_npz (auto-detect) OR "pseudo_masks_npz" if you standardize
#
# Produces:
#   - qc_keep.csv / qc_drop.csv under out_root

import argparse
from pathlib import Path
import pandas as pd


def pick_stats_csv(out_root: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"--stats not found: {p}")
        return p

    # 1) preferred pattern
    cand = sorted(out_root.glob("mask_stats_*_image.csv"))
    if cand:
        return cand[0]

    # 2) fallback names
    cand = sorted(out_root.glob("mask_stats*.csv"))
    if cand:
        return cand[0]

    raise FileNotFoundError(
        f"stats_csv not found in {out_root}. Expected mask_stats_*_image.csv (or pass --stats)."
    )


def pick_npz_dir(out_root: Path, override: str | None) -> Path | None:
    if override:
        p = Path(override)
        if not p.exists():
            return None
        return p

    # prefer pseudo_masks_*_npz
    cand = sorted([p for p in out_root.glob("pseudo_masks_*_npz") if p.is_dir()])
    if cand:
        return cand[0]

    # fallback
    p2 = out_root / "pseudo_masks_npz"
    if p2.exists() and p2.is_dir():
        return p2

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--stats", type=str, default=None, help="path to stats csv (optional). If not set, auto-detect in out_root.")
    ap.add_argument("--npz_dir", type=str, default=None, help="npz dir (optional). If not set, auto-detect in out_root.")
    ap.add_argument("--require_npz", action="store_true")

    # QC thresholds
    ap.add_argument("--min_area_ratio", type=float, default=1e-4)
    ap.add_argument("--max_area_ratio", type=float, default=3e-2)
    ap.add_argument("--max_expand_mean", type=float, default=8.0)
    ap.add_argument("--min_masks", type=int, default=1)

    args = ap.parse_args()
    out_root = Path(args.out_root)

    stats_csv = pick_stats_csv(out_root, args.stats)
    npz_dir = pick_npz_dir(out_root, args.npz_dir)

    df = pd.read_csv(stats_csv)

    # normalize expected columns (be tolerant)
    # We expect:
    #   stem, image_path, n_masks, union_area_ratio, expand_mean
    # If your stats use different names, adapt here.
    colmap = {}
    if "union_area_ratio" not in df.columns:
        # some scripts might name it "area_ratio"
        if "area_ratio" in df.columns:
            colmap["area_ratio"] = "union_area_ratio"
    if "expand_mean" not in df.columns:
        # sometimes "expand" metrics differ; if absent, create zeros so QC won't drop by expand
        df["expand_mean"] = 0.0
    if "n_masks" not in df.columns:
        # if absent, assume 1 when union_area_ratio>0 else 0
        if "union_area_ratio" in df.columns:
            df["n_masks"] = (df["union_area_ratio"] > 0).astype(int)
        else:
            df["n_masks"] = 1

    if colmap:
        df = df.rename(columns=colmap)

    # require columns exist now
    for c in ["stem", "image_path", "n_masks", "union_area_ratio", "expand_mean"]:
        if c not in df.columns:
            raise ValueError(f"stats csv missing required column: {c}. columns={list(df.columns)}")

    # apply QC rules
    reasons = []
    keep_flags = []

    for _, r in df.iterrows():
        rs = []
        n_masks = int(r["n_masks"])
        area_ratio = float(r["union_area_ratio"])
        expand_mean = float(r["expand_mean"])

        if n_masks < args.min_masks:
            rs.append(f"n_masks<{args.min_masks}")
        if area_ratio < args.min_area_ratio:
            rs.append(f"area_ratio<{args.min_area_ratio}")
        if area_ratio > args.max_area_ratio:
            rs.append(f"area_ratio>{args.max_area_ratio}")
        if expand_mean > args.max_expand_mean:
            rs.append(f"expand_mean>{args.max_expand_mean}")

        # optional: require npz exists
        if args.require_npz:
            if npz_dir is None:
                rs.append("npz_dir_missing")
            else:
                stem = str(r["stem"])
                npz_path = npz_dir / f"{stem}.npz"
                if not npz_path.exists():
                    rs.append("npz_missing")

        reasons.append("|".join(rs) if rs else "")
        keep_flags.append(0 if rs else 1)

    out = df.copy()
    out["qc_keep"] = keep_flags
    out["qc_drop_reason"] = reasons

    keep_df = out[out["qc_keep"] == 1].copy()
    drop_df = out[out["qc_keep"] == 0].copy()

    print("=== QC SUMMARY ===")
    print(f"stats: {stats_csv}")
    if args.require_npz:
        print(f"npz_dir: {npz_dir}")
    print(f"total: {len(out)}")
    print(f"keep : {len(keep_df)} ({(len(keep_df)/max(1,len(out))*100):.2f}%)")
    print(f"drop : {len(drop_df)} ({(len(drop_df)/max(1,len(out))*100):.2f}%)")
    print(
        "thresholds: "
        f"min_area_ratio={args.min_area_ratio}, max_area_ratio={args.max_area_ratio}, "
        f"max_expand_mean={args.max_expand_mean}, min_masks={args.min_masks}"
    )

    if len(drop_df) > 0:
        print("\nTop drop reasons:")
        print(drop_df["qc_drop_reason"].value_counts().head(10))

    keep_csv = out_root / "qc_keep.csv"
    drop_csv = out_root / "qc_drop.csv"
    keep_df.to_csv(keep_csv, index=False)
    drop_df.to_csv(drop_csv, index=False)
    print(f"\nSaved: {keep_csv}")
    print(f"Saved: {drop_csv}")


if __name__ == "__main__":
    main()
