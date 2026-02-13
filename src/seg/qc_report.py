import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def summarize(df: pd.DataFrame):
    # pass rate defined as: images with n_valid_masks>=1 / total images
    total = len(df)
    pass_img = int((df["n_valid_masks"] >= 1).sum())
    pass_rate = pass_img / total if total else 0.0

    # union area stats
    ar = df["union_area_ratio"].to_numpy()
    return {
        "n_images": total,
        "img_pass_rate(n_valid>=1)": pass_rate,
        "union_area_ratio_median": float(np.median(ar)) if total else 0.0,
        "union_area_ratio_p95": float(np.percentile(ar, 95)) if total else 0.0,
        "n_valid_masks_mean": float(df["n_valid_masks"].mean()) if total else 0.0,
        "n_valid_masks_p95": float(np.percentile(df["n_valid_masks"], 95)) if total else 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--models", type=str, default="medsam2,medsam,sam2,sam_med2d")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []
    for m in models:
        p = out_root / f"mask_stats_{m}_image.csv"
        if not p.exists():
            print("[WARN] missing:", p)
            continue
        df = pd.read_csv(p)
        s = summarize(df)
        s["seg_model"] = m
        rows.append(s)

    rep = pd.DataFrame(rows).sort_values("img_pass_rate(n_valid>=1)", ascending=False)
    out_csv = out_root / "qc_compare_report.csv"
    rep.to_csv(out_csv, index=False)

    # also print a compact markdown table
    print("\n=== QC Compare Report ===")
    print(rep.to_markdown(index=False))
    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()
