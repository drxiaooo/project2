import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.letterbox import letterbox_pil


def main():
    ap = argparse.ArgumentParser()
    # NEW: for consistent CLI (optional)
    ap.add_argument("--exp_dir", default=None,
                    help="(optional) cls exp dir; only used for record/consistency, not required here")
    ap.add_argument("--out_root", default="outputs/weakloc_letterbox",
                    help="weakloc output root (default: outputs/weakloc_letterbox)")

    # Old args (still supported)
    ap.add_argument("--boxes_csv", default=None,
                    help="path to pseudo boxes csv; default: {out_root}/pseudo_boxes_test.csv")
    ap.add_argument("--out_dir", default=None,
                    help="dir to save visualizations; default: {out_root}/test_boxes_vis")

    ap.add_argument("--max_imgs", type=int, default=100)
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1024)
    args = ap.parse_args()

    out_root = Path(args.out_root)

    boxes_csv = Path(args.boxes_csv) if args.boxes_csv else (out_root / "pseudo_boxes_test.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (out_root / "test_boxes_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not boxes_csv.exists():
        raise FileNotFoundError(f"boxes_csv not found: {boxes_csv}")

    df = pd.read_csv(boxes_csv)
    if df.empty:
        raise ValueError(f"boxes_csv is empty: {boxes_csv}. Try lowering min_area or increasing top_frac.")

    # Sort by prob then heat score
    if "heat_score" in df.columns:
        df = df.sort_values(["y_prob", "heat_score"], ascending=[False, False])
    else:
        df = df.sort_values(["y_prob"], ascending=[False])

    seen = set()
    saved = 0

    for _, r in df.iterrows():
        img_path = r["image_path"]
        if img_path in seen:
            continue
        seen.add(img_path)

        img_pil = Image.open(img_path).convert("RGB")
        img_pad, _, _, _ = letterbox_pil(img_pil, out_w=args.img_w, out_h=args.img_h)
        arr = np.asarray(img_pad).copy()

        sub = df[df["image_path"] == img_path]
        if "heat_score" in sub.columns:
            sub = sub.sort_values("heat_score", ascending=False)

        for _, b in sub.iterrows():
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 0, 0), 2)

            hs = float(b["heat_score"]) if "heat_score" in b else 0.0
            bid = int(b["box_id"]) if "box_id" in b else 0
            cv2.putText(
                arr,
                f"{bid} hs={hs:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        prob = float(r["y_prob"]) if "y_prob" in r else -1.0
        cv2.putText(arr, f"prob={prob:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out_path = out_dir / f"{Path(img_path).stem}_prob{prob:.3f}.png"
        Image.fromarray(arr).save(out_path)

        saved += 1
        if saved >= args.max_imgs:
            break

    print(f"[INFO] exp_dir={args.exp_dir}" if args.exp_dir else "[INFO] exp_dir=(not set)")
    print(f"[DONE] saved {saved} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
