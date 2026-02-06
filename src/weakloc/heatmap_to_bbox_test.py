import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def peak_boxes_from_heatmap(
    heat: np.ndarray,
    topk: int = 3,
    box_w: int = 96,
    box_h: int = 96,
    nms_iou: float = 0.30,
    candidates_multiplier: int = 50,
    score_min: float = 0.0,
):
    """
    Peak-based pseudo boxes:
    1) pick top-K peaks (from a larger candidate pool)
    2) create fixed-size boxes centered at peaks
    3) NMS to remove overlaps
    Returns: list of (score, area, x1, y1, x2, y2)
    """
    h = np.clip(heat.astype(np.float32), 0.0, 1.0)
    H, W = h.shape

    flat = h.reshape(-1)
    if flat.size == 0:
        return []

    # pick a candidate pool bigger than topk to be robust
    cand_n = min(max(topk * candidates_multiplier, topk), flat.size)
    # argpartition for speed
    idxs = np.argpartition(-flat, cand_n - 1)[:cand_n]
    # sort candidates by score desc
    idxs = idxs[np.argsort(-flat[idxs])]

    props = []
    for idx in idxs:
        score = float(flat[idx])
        if score <= score_min:
            break

        y = int(idx // W)
        x = int(idx % W)

        x1 = max(0, x - box_w // 2)
        y1 = max(0, y - box_h // 2)
        x2 = min(W, x1 + box_w)
        y2 = min(H, y1 + box_h)

        area = int((x2 - x1) * (y2 - y1))
        props.append((score, area, x1, y1, x2, y2))

        # early stop if we already have enough proposals before NMS
        if len(props) >= topk * 10:
            pass

    # NMS
    picked = []
    for score, area, x1, y1, x2, y2 in props:
        box = (x1, y1, x2, y2)
        keep = True
        for ps, pa, px1, py1, px2, py2 in picked:
            if iou_xyxy(box, (px1, py1, px2, py2)) > nms_iou:
                keep = False
                break
        if keep:
            picked.append((score, area, x1, y1, x2, y2))
        if len(picked) >= topk:
            break

    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="outputs/cls/resnetrs50_POST_letterbox",
                    help="exp dir containing preds_test.csv; used to locate pred file")
    ap.add_argument("--pred_csv", default=None, help="override pred csv; default exp_dir/preds_test.csv")

    ap.add_argument("--out_root", default="outputs/weakloc_letterbox",
                    help="weakloc output root (contains test_cam_npy)")
    ap.add_argument("--heat_dir", default=None,
                    help="override heatmap dir; default {out_root}/test_cam_npy")

    # Peak-box params
    ap.add_argument("--max_boxes", type=int, default=3, help="top-k boxes per image")
    ap.add_argument("--box_w", type=int, default=96, help="fixed box width in pixels (on model input size)")
    ap.add_argument("--box_h", type=int, default=96, help="fixed box height in pixels (on model input size)")
    ap.add_argument("--nms_iou", type=float, default=0.30, help="NMS IoU threshold")
    ap.add_argument("--score_min", type=float, default=0.0, help="min peak score to accept")
    ap.add_argument("--candidates_multiplier", type=int, default=50,
                    help="candidate pool size = max_boxes * multiplier (capped by image pixels)")

    # optional filter
    ap.add_argument("--min_prob", type=float, default=None,
                    help="if set, only generate boxes for rows with y_prob >= min_prob")

    ap.add_argument("--save_csv", default=None,
                    help="override output csv; default {out_root}/pseudo_boxes_test.csv")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    pred_csv = Path(args.pred_csv) if args.pred_csv else (exp_dir / "preds_test.csv")
    if not pred_csv.exists():
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")

    out_root = Path(args.out_root)
    heat_dir = Path(args.heat_dir) if args.heat_dir else (out_root / "test_cam_npy")
    if not heat_dir.exists():
        raise FileNotFoundError(f"heat_dir not found: {heat_dir} (run make_cam_test.py first)")

    out_csv = Path(args.save_csv) if args.save_csv else (out_root / "pseudo_boxes_test.csv")

    # index heatmaps by stem
    heat_files = list(heat_dir.glob("*.npy"))
    if not heat_files:
        raise FileNotFoundError(f"No .npy heatmaps found in {heat_dir}. Run make_cam_test.py first.")

    index = {}
    for p in heat_files:
        stem = p.name.split("_prob")[0]
        index.setdefault(stem, []).append(p)

    df = pd.read_csv(pred_csv)
    if args.min_prob is not None:
        df = df[df["y_prob"] >= float(args.min_prob)].copy()

    rows = []
    miss = 0
    used = 0

    for _, r in df.iterrows():
        img_path = r["image_path"]
        stem = Path(img_path).stem
        prob = float(r["y_prob"])

        candidates = index.get(stem, [])
        if not candidates:
            miss += 1
            continue

        # choose the latest one (if multiple)
        heat_path = sorted(candidates)[-1]
        heat = np.load(heat_path)

        boxes = peak_boxes_from_heatmap(
            heat,
            topk=int(args.max_boxes),
            box_w=int(args.box_w),
            box_h=int(args.box_h),
            nms_iou=float(args.nms_iou),
            candidates_multiplier=int(args.candidates_multiplier),
            score_min=float(args.score_min),
        )

        if len(boxes) == 0:
            continue

        used += 1
        for bid, (score, area, x1, y1, x2, y2) in enumerate(boxes):
            rows.append({
                "image_path": img_path,
                "y_prob": prob,
                "heat_path": str(heat_path),
                "box_id": bid,
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "area": int(area),
                "heat_score": float(score),
                "method": "peak_fixed_box",
                "box_w": int(args.box_w),
                "box_h": int(args.box_h),
                "nms_iou": float(args.nms_iou),
                "score_min": float(args.score_min),
            })

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[INFO] pred_csv={pred_csv}")
    print(f"[INFO] heat_dir={heat_dir}")
    print(f"[DONE] saved {len(rows)} boxes to {out_csv} (images used={used}, missing heatmaps for {miss} images)")


if __name__ == "__main__":
    main()
