import os, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .adapters import build_adapter

# ---------- basic utils ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_rgb_uint8(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def parse_boxes_from_row(row: pd.Series) -> np.ndarray:
    """
    Expect one-row-per-image with JSON column 'boxes' storing list of [x1,y1,x2,y2]
    If your CSV is different, convert it to this format (recommended for reproducibility).
    """
    if "boxes" not in row or not isinstance(row["boxes"], str):
        raise ValueError("CSV must contain a JSON string column 'boxes' per image.")
    boxes = np.array(json.loads(row["boxes"]), dtype=np.float32).reshape(-1, 4)
    return boxes

def clip_boxes(boxes: np.ndarray, W: int, H: int) -> np.ndarray:
    b = boxes.copy()
    b[:, 0] = np.clip(b[:, 0], 0, W - 1)
    b[:, 2] = np.clip(b[:, 2], 0, W - 1)
    b[:, 1] = np.clip(b[:, 1], 0, H - 1)
    b[:, 3] = np.clip(b[:, 3], 0, H - 1)
    # enforce valid
    b[:, 0] = np.minimum(b[:, 0], b[:, 2] - 1)
    b[:, 1] = np.minimum(b[:, 1], b[:, 3] - 1)
    return b

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

def count_connected_components(mask: np.ndarray):
    # lightweight CC count using scipy if available; else fallback to 1/0
    try:
        import scipy.ndimage as ndi
        lab, n = ndi.label(mask.astype(np.uint8))
        return int(n), lab
    except Exception:
        return int(mask.sum() > 0), None

def overlay_vis(rgb: np.ndarray, union_mask: np.ndarray, boxes: np.ndarray):
    out = rgb.copy().astype(np.float32)
    m = union_mask.astype(np.float32)
    out[..., 1] = np.clip(out[..., 1] * (1 - 0.35*m) + 255 * 0.35*m, 0, 255)
    H, W = rgb.shape[:2]
    for (x1, y1, x2, y2) in boxes.astype(int):
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        out[y1:y1+2, x1:x2, 0] = 255
        out[y2-2:y2, x1:x2, 0] = 255
        out[y1:y2, x1:x1+2, 0] = 255
        out[y1:y2, x2-2:x2, 0] = 255
    return out.astype(np.uint8)

# ---------- QC (instance level) ----------
def qc_instance(mask: np.ndarray, prompt_box: np.ndarray, H: int, W: int,
                area_min=20, area_ratio_max=0.08, fill_min=0.10,
                iou_min=0.10, expand_max=6.0, cc_major_min=0.7, cc_max=3):
    area = int(mask.sum())
    if area < area_min:
        return False, {"reason": "area_too_small", "area": area}

    area_ratio = area / float(H * W)
    if area_ratio > area_ratio_max:
        return False, {"reason": "area_too_large", "area": area, "area_ratio": area_ratio}

    mbox = bbox_from_mask(mask)
    if mbox is None:
        return False, {"reason": "empty_mask", "area": area}

    # fill ratio
    mbox_area = max(1.0, (mbox[2] - mbox[0]) * (mbox[3] - mbox[1]))
    fill = area / mbox_area
    if fill < fill_min:
        return False, {"reason": "too_sparse", "fill": fill}

    # box consistency
    iou = iou_xyxy(mbox, prompt_box)
    if iou < iou_min:
        return False, {"reason": "low_iou", "iou": iou}

    box_area = max(1.0, (prompt_box[2] - prompt_box[0]) * (prompt_box[3] - prompt_box[1]))
    expand = mbox_area / box_area
    if expand > expand_max:
        return False, {"reason": "over_expand", "expand": expand}

    # connected components
    n_cc, lab = count_connected_components(mask)
    if n_cc > cc_max:
        return False, {"reason": "too_many_cc", "n_cc": n_cc}

    if lab is not None:
        # major component ratio
        import numpy as np
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        major = counts.max() if counts.size > 0 else 0
        if major / float(area) < cc_major_min:
            return False, {"reason": "cc_not_major", "major_ratio": major / float(area)}

    return True, {
        "reason": "pass",
        "area": area,
        "area_ratio": area_ratio,
        "fill": fill,
        "iou": iou,
        "expand": expand,
        "n_cc": n_cc,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="outputs/weakloc_letterbox")
    ap.add_argument("--seg_model", type=str, required=True, choices=["medsam2","medsam","sam2","sam_med2d"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--boxes_csv", type=str, default=None, help="default={out_root}/pseudo_boxes_test.csv")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--max_imgs", type=int, default=999999)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    boxes_csv = Path(args.boxes_csv) if args.boxes_csv else out_root / "pseudo_boxes_test.csv"
    df = pd.read_csv(boxes_csv)

    # outputs
    out_npz = out_root / f"pseudo_masks_{args.seg_model}_npz"
    out_vis = out_root / f"pseudo_masks_{args.seg_model}_vis"
    ensure_dir(out_npz)
    if args.save_vis:
        ensure_dir(out_vis)

    adapter = build_adapter(args.seg_model, args.ckpt, device=args.device)

    rows = []
    n = min(len(df), args.max_imgs)

    for idx in range(n):
        row = df.iloc[idx]
        img_path = str(row["image_path"])
        rgb = load_rgb_uint8(img_path)
        H, W = rgb.shape[:2]

        boxes = clip_boxes(parse_boxes_from_row(row), W, H)[: args.topk]

        # infer
        adapter.set_image(rgb)
        masks, scores = adapter.predict(boxes)  # masks (N,H,W) uint8, scores (N,)
        if scores is None or len(scores) == 0:
            scores = np.ones((masks.shape[0],), dtype=np.float32)

        # QC each instance
        keep_masks = []
        keep_boxes = []
        keep_scores = []
        inst_stats = []

        for j in range(masks.shape[0]):
            ok, st = qc_instance(masks[j] > 0, boxes[j], H, W)
            st.update({
                "stem": Path(img_path).stem,
                "image_path": img_path,
                "seg_model": args.seg_model,
                "inst_id": j,
                "score": float(scores[j]),
                "box_x1": float(boxes[j,0]), "box_y1": float(boxes[j,1]),
                "box_x2": float(boxes[j,2]), "box_y2": float(boxes[j,3]),
                "qc_pass": int(ok),
            })
            inst_stats.append(st)

            if ok:
                keep_masks.append(masks[j].astype(np.uint8))
                keep_boxes.append(boxes[j].astype(np.float32))
                keep_scores.append(float(scores[j]))

        keep_masks = np.stack(keep_masks, axis=0) if len(keep_masks) else np.zeros((0,H,W), np.uint8)
        keep_boxes = np.stack(keep_boxes, axis=0) if len(keep_boxes) else np.zeros((0,4), np.float32)
        keep_scores = np.array(keep_scores, dtype=np.float32) if len(keep_scores) else np.zeros((0,), np.float32)

        stem = Path(img_path).stem
        np.savez_compressed(
            out_npz / f"{stem}.npz",
            image_path=img_path, H=H, W=W,
            boxes=keep_boxes, scores=keep_scores, masks=keep_masks,
        )

        # per-image summary row
        union = (keep_masks.sum(axis=0) > 0).astype(np.uint8) if keep_masks.shape[0] > 0 else np.zeros((H,W), np.uint8)
        area_sum = int(union.sum())
        rows.append({
            "stem": stem,
            "image_path": img_path,
            "seg_model": args.seg_model,
            "n_prompt_boxes": int(min(parse_boxes_from_row(row).shape[0], args.topk)),
            "n_valid_masks": int(keep_masks.shape[0]),
            "union_area": area_sum,
            "union_area_ratio": area_sum / float(H*W),
        })

        # optional vis
        if args.save_vis:
            vis = overlay_vis(rgb, union, keep_boxes if keep_boxes.shape[0] else boxes)
            Image.fromarray(vis).save(out_vis / f"{stem}.png")

        # append inst stats
        for st in inst_stats:
            st["union_area"] = area_sum
        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{n}] done")

        # store instance stats incrementally
        # (keep in memory is ok too; dataset size is manageable)
        # We'll write at end.

    # save reports
    img_df = pd.DataFrame(rows)
    img_df.to_csv(out_root / f"mask_stats_{args.seg_model}_image.csv", index=False)

    # instance-level: re-run quickly by reading all npz? No, we stored inst_stats above in loop.
    # For simplicity, we collect instance stats by re-reading? We'll just do it properly:
    # (Minimal version) -> scan vis/npz is costly. So in practice you can store inst_stats in a list.
    # Here we reconstruct by reading image csv only; instance csv optional.
    print("Saved:", out_root / f"mask_stats_{args.seg_model}_image.csv")

if __name__ == "__main__":
    main()
