import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def unletterbox_xyxy(x1, y1, x2, y2, scale, pad_left, pad_top, orig_w, orig_h):
    """
    Convert bbox from letterbox canvas coords -> original image coords.
    letterbox: x_lb = x_orig*scale + pad_left
              y_lb = y_orig*scale + pad_top
    => x_orig = (x_lb - pad_left)/scale
    """
    x1o = (float(x1) - float(pad_left)) / float(scale)
    y1o = (float(y1) - float(pad_top)) / float(scale)
    x2o = (float(x2) - float(pad_left)) / float(scale)
    y2o = (float(y2) - float(pad_top)) / float(scale)

    # clamp to original image bounds
    x1o = clamp(x1o, 0, orig_w - 1)
    y1o = clamp(y1o, 0, orig_h - 1)
    x2o = clamp(x2o, 0, orig_w - 1)
    y2o = clamp(y2o, 0, orig_h - 1)

    # ensure proper ordering
    x1o, x2o = (x1o, x2o) if x1o <= x2o else (x2o, x1o)
    y1o, y2o = (y1o, y2o) if y1o <= y2o else (y2o, y1o)

    return int(round(x1o)), int(round(y1o)), int(round(x2o)), int(round(y2o))


def draw_boxes_on_orig(img_path: str, boxes_df: pd.DataFrame, save_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # try load a default font; if fails, PIL will use basic
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # draw each box
    for _, r in boxes_df.iterrows():
        x1, y1, x2, y2 = int(r["x1o"]), int(r["y1o"]), int(r["x2o"]), int(r["y2o"])
        bid = int(r.get("box_id", 0))
        hs = float(r.get("heat_score", 0.0))
        prob = float(r.get("y_prob", -1.0))

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        txt = f"{bid} hs={hs:.2f}"
        # label background
        tx, ty = x1, max(0, y1 - 14)
        draw.text((tx, ty), txt, fill=(255, 255, 255), font=font)

    # top-left prob
    prob = float(boxes_df["y_prob"].iloc[0]) if "y_prob" in boxes_df.columns else -1.0
    draw.text((10, 10), f"prob={prob:.3f}", fill=(255, 255, 255), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="outputs/weakloc_letterbox",
                    help="where pseudo_boxes_test.csv and letterbox_meta_test.csv are located")
    ap.add_argument("--boxes_csv", default=None,
                    help="override boxes csv; default {out_root}/pseudo_boxes_test.csv")
    ap.add_argument("--meta_csv", default=None,
                    help="override meta csv; default {out_root}/letterbox_meta_test.csv")
    ap.add_argument("--save_csv", default=None,
                    help="output csv path; default {out_root}/pseudo_boxes_test_orig.csv")

    ap.add_argument("--vis_dir", default=None,
                    help="output visualization dir; default {out_root}/test_boxes_vis_orig")
    ap.add_argument("--max_imgs", type=int, default=200,
                    help="max unique images to draw (sorted by y_prob desc)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    boxes_csv = Path(args.boxes_csv) if args.boxes_csv else (out_root / "pseudo_boxes_test.csv")
    meta_csv = Path(args.meta_csv) if args.meta_csv else (out_root / "letterbox_meta_test.csv")
    save_csv = Path(args.save_csv) if args.save_csv else (out_root / "pseudo_boxes_test_orig.csv")
    vis_dir = Path(args.vis_dir) if args.vis_dir else (out_root / "test_boxes_vis_orig")

    if not boxes_csv.exists():
        raise FileNotFoundError(f"boxes_csv not found: {boxes_csv}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta_csv not found: {meta_csv} (run make_cam_test.py after code update)")

    boxes = pd.read_csv(boxes_csv)
    meta = pd.read_csv(meta_csv)

    # meta key by image_path
    meta = meta.drop_duplicates(subset=["image_path"]).copy()
    merged = boxes.merge(meta, on="image_path", how="left", suffixes=("", "_m"))

    missing_meta = merged["scale"].isna().sum()
    if missing_meta > 0:
        print(f"[WARN] missing meta for {missing_meta} box rows. They will be dropped.")
        merged = merged[~merged["scale"].isna()].copy()

    # compute original coords
    x1o_list, y1o_list, x2o_list, y2o_list = [], [], [], []
    for _, r in merged.iterrows():
        x1o, y1o, x2o, y2o = unletterbox_xyxy(
            r["x1"], r["y1"], r["x2"], r["y2"],
            r["scale"], r["pad_left"], r["pad_top"],
            int(r["orig_w"]), int(r["orig_h"])
        )
        x1o_list.append(x1o); y1o_list.append(y1o); x2o_list.append(x2o); y2o_list.append(y2o)

    merged["x1o"] = x1o_list
    merged["y1o"] = y1o_list
    merged["x2o"] = x2o_list
    merged["y2o"] = y2o_list

    # save csv
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(save_csv, index=False)
    print(f"[OK] saved original-coordinate boxes to: {save_csv}")

    # draw visualizations: sort by y_prob desc, then heat_score desc
    if "heat_score" in merged.columns:
        merged_sorted = merged.sort_values(["y_prob", "heat_score"], ascending=[False, False]).copy()
    else:
        merged_sorted = merged.sort_values(["y_prob"], ascending=[False]).copy()

    # unique images
    uniq_paths = []
    seen = set()
    for p in merged_sorted["image_path"].tolist():
        if p not in seen:
            seen.add(p)
            uniq_paths.append(p)
        if len(uniq_paths) >= args.max_imgs:
            break

    for img_path in uniq_paths:
        sub = merged_sorted[merged_sorted["image_path"] == img_path].copy()
        # keep same order (by heat_score)
        stem = Path(img_path).stem
        prob = float(sub["y_prob"].iloc[0]) if "y_prob" in sub.columns else -1.0
        out_path = vis_dir / f"{stem}_prob{prob:.3f}.png"
        draw_boxes_on_orig(img_path, sub, out_path)

    print(f"[DONE] saved {len(uniq_paths)} original-image visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()
