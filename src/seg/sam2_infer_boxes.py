# src/seg/sam2_infer_boxes.py
# FINAL+ROI VERSION (recommended for BS-80K wholeBodyPOST)
#
# Fixes:
# - Apply LETTERBOX (1024x512) before segmentation so boxes align (your boxes are in canvas coords).
# - Use ROI crop around each box to prevent SAM2 from segmenting the whole background/person.
# - Optional tighten within ROI (crop to box±margin + keep best-overlap CC).
# - Save: NPZ masks (canvas coords), canvas vis, optional orig vis, stats, meta.
#
# Run (avoid SAM2 parent-dir shadowing):
#   cd E:\project\project2\sam2
#   $env:PYTHONPATH="E:\project\project2"
#
# Recommended test:
#   python -m src.seg.sam2_infer_boxes --out_root E:\project\project2\outputs\weakloc_letterbox `
#     --use_letterbox --out_w 1024 --out_h 512 `
#     --use_crop --crop_size 320 `
#     --topk 1 --max_imgs 20 --save_vis_canvas
#
# If still too big:
#   ... --tighten --margin 8

import json
import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.utils.letterbox import letterbox_pil


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_boxes_str(s: str) -> np.ndarray:
    s = s.strip()
    try:
        boxes = json.loads(s)
    except Exception:
        boxes = ast.literal_eval(s)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def normalize_boxes_csv(df: pd.DataFrame, topk: int = 3) -> pd.DataFrame:
    """
    Return per-image rows with:
      - image_path
      - y_prob (if exists)
      - boxes (JSON string list of [x1,y1,x2,y2]) sorted by heat_score/score if available

    Supports your current per-box CSV:
      image_path,y_prob,heat_path,box_id,x1,y1,x2,y2,area,heat_score,...
    """
    if "image_path" in df.columns and "boxes" in df.columns:
        keep = ["image_path", "boxes"]
        if "y_prob" in df.columns:
            keep.append("y_prob")
        return df[keep].copy()

    required = {"image_path", "x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        raise ValueError("Unsupported boxes CSV format.")

    score_col = None
    for cand in ["heat_score", "score", "peak_score", "conf", "prob"]:
        if cand in df.columns:
            score_col = cand
            break

    rows = []
    for img_path, g in df.groupby("image_path", sort=False):
        gg = g
        if score_col:
            gg = gg.sort_values(score_col, ascending=False)
        gg = gg.head(topk)
        boxes = gg[["x1", "y1", "x2", "y2"]].astype(float).values.tolist()
        row = {"image_path": img_path, "boxes": json.dumps(boxes)}
        if "y_prob" in gg.columns:
            row["y_prob"] = float(gg["y_prob"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def clip_boxes_xyxy(boxes: np.ndarray, W: int, H: int) -> np.ndarray:
    b = boxes.copy().astype(np.float32)
    b[:, 0] = np.clip(b[:, 0], 0, W - 1)
    b[:, 2] = np.clip(b[:, 2], 0, W - 1)
    b[:, 1] = np.clip(b[:, 1], 0, H - 1)
    b[:, 3] = np.clip(b[:, 3], 0, H - 1)
    b[:, 0] = np.minimum(b[:, 0], b[:, 2] - 1)
    b[:, 1] = np.minimum(b[:, 1], b[:, 3] - 1)
    return b


def overlay_vis(rgb: np.ndarray, union_mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    m = union_mask.astype(np.float32)
    out[..., 1] = np.clip(out[..., 1] * (1 - 0.35 * m) + 255 * 0.35 * m, 0, 255)

    H, W = rgb.shape[:2]
    for (x1, y1, x2, y2) in boxes.astype(int):
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        out[y1:y1 + 2, x1:x2, 0] = 255
        out[y2 - 2:y2, x1:x2, 0] = 255
        out[y1:y2, x1:x1 + 2, 0] = 255
        out[y1:y2, x2 - 2:x2, 0] = 255

    return out.astype(np.uint8)


def squeeze_masks(masks) -> np.ndarray:
    m = masks
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()

    if m.ndim == 4 and m.shape[1] == 1:
        m = m[:, 0, :, :]
    if m.ndim == 4 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {getattr(masks,'shape',None)} -> {m.shape}")

    return (m > 0).astype(np.uint8)


def bbox_from_mask(mask2d_u8: np.ndarray):
    ys, xs = np.where(mask2d_u8 > 0)
    if xs.size == 0:
        return None
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], dtype=np.float32)


def box_area_xyxy(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def transform_boxes_for_sam2(predictor, boxes_xyxy_t: torch.Tensor, orig_hw: tuple[int, int]) -> torch.Tensor:
    tfm = getattr(predictor, "_transforms", None)
    if tfm is not None and hasattr(tfm, "transform_boxes"):
        return tfm.transform_boxes(boxes_xyxy_t, normalize=False, orig_hw=orig_hw)

    tfm2 = getattr(predictor, "transform", None)
    if tfm2 is not None and hasattr(tfm2, "apply_boxes_torch"):
        return tfm2.apply_boxes_torch(boxes_xyxy_t, orig_hw)

    if tfm is not None and hasattr(tfm, "apply_boxes"):
        return tfm.apply_boxes(boxes_xyxy_t, orig_hw)

    return boxes_xyxy_t


def predict_with_box(predictor, boxes_t: torch.Tensor):
    try:
        return predictor.predict(
            point_coords=None,
            point_labels=None,
            boxes=boxes_t,
            multimask_output=False,
        )
    except TypeError:
        return predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_t,
            multimask_output=False,
        )


def tighten_mask_by_box(mask2d_u8: np.ndarray, box_xyxy: np.ndarray, margin: int = 8) -> np.ndarray:
    """
    1) hard crop to box±margin
    2) keep CC with max overlap inside crop (robust for peak boxes)
    """
    m = (mask2d_u8 > 0).astype(np.uint8)
    H, W = m.shape
    x1, y1, x2, y2 = box_xyxy.astype(int)

    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(W - 1, x2 + margin)
    y2m = min(H - 1, y2 + margin)

    tight = np.zeros_like(m, dtype=np.uint8)
    tight[y1m:y2m, x1m:x2m] = m[y1m:y2m, x1m:x2m]

    try:
        import scipy.ndimage as ndi
        lab, n = ndi.label(tight > 0)
        if n == 0:
            return tight

        crop = (slice(y1m, y2m), slice(x1m, x2m))
        best_id, best_overlap = 0, -1
        for k in range(1, n + 1):
            overlap = int((lab[crop] == k).sum())
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = k
        return (lab == best_id).astype(np.uint8)
    except Exception:
        return tight


def crop_around_box(rgb: np.ndarray, box: np.ndarray, crop_size: int) -> tuple[np.ndarray, int, int]:
    """
    Square crop around box center.
    Returns: rgb_crop, x0, y0 (top-left in original image coords)
    """
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    half = crop_size // 2
    x0 = cx - half
    y0 = cy - half
    x1c = x0 + crop_size
    y1c = y0 + crop_size

    # shift to keep inside
    if x0 < 0:
        x1c -= x0
        x0 = 0
    if y0 < 0:
        y1c -= y0
        y0 = 0
    if x1c > W:
        dx = x1c - W
        x0 = max(0, x0 - dx)
        x1c = W
    if y1c > H:
        dy = y1c - H
        y0 = max(0, y0 - dy)
        y1c = H

    rgb_crop = rgb[y0:y1c, x0:x1c].copy()
    return rgb_crop, int(x0), int(y0)


def paste_crop_mask(mask_crop: np.ndarray, H: int, W: int, x0: int, y0: int) -> np.ndarray:
    out = np.zeros((H, W), np.uint8)
    h, w = mask_crop.shape
    out[y0:y0 + h, x0:x0 + w] = (mask_crop > 0).astype(np.uint8)
    return out


def mask_canvas_to_orig(mask_canvas_u8: np.ndarray, orig_w: int, orig_h: int, scale: float, pad_left: int, pad_top: int) -> np.ndarray:
    Hc, Wc = mask_canvas_u8.shape
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    x1 = int(pad_left)
    y1 = int(pad_top)
    x2 = int(pad_left + new_w)
    y2 = int(pad_top + new_h)

    x1 = max(0, min(Wc, x1))
    x2 = max(0, min(Wc, x2))
    y1 = max(0, min(Hc, y1))
    y2 = max(0, min(Hc, y2))

    crop = mask_canvas_u8[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((orig_h, orig_w), np.uint8)

    pil = Image.fromarray((crop > 0).astype(np.uint8) * 255)
    pil = pil.resize((orig_w, orig_h), resample=Image.NEAREST)
    return (np.array(pil) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--boxes_csv", type=str, default=None)

    ap.add_argument("--model_id", type=str, default="facebook/sam2-hiera-large")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_imgs", type=int, default=50)

    # letterbox (CRITICAL for your pipeline)
    ap.add_argument("--use_letterbox", action="store_true")
    ap.add_argument("--out_w", type=int, default=1024)
    ap.add_argument("--out_h", type=int, default=512)

    # ROI crop (STRONGLY recommended)
    ap.add_argument("--use_crop", action="store_true")
    ap.add_argument("--crop_size", type=int, default=320)

    # outputs
    ap.add_argument("--save_vis_canvas", action="store_true")
    ap.add_argument("--save_vis_orig", action="store_true")

    # tighten
    ap.add_argument("--tighten", action="store_true")
    ap.add_argument("--margin", type=int, default=8)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    boxes_csv = Path(args.boxes_csv) if args.boxes_csv else out_root / "pseudo_boxes_test.csv"

    out_npz = out_root / "pseudo_masks_sam2_npz"
    out_vis_canvas = out_root / "pseudo_masks_sam2_vis_canvas"
    out_vis_orig = out_root / "pseudo_masks_sam2_vis_orig"
    ensure_dir(out_npz)
    if args.save_vis_canvas:
        ensure_dir(out_vis_canvas)
    if args.save_vis_orig:
        ensure_dir(out_vis_orig)

    df = pd.read_csv(boxes_csv)
    df = normalize_boxes_csv(df, topk=args.topk)

    predictor = SAM2ImagePredictor.from_pretrained(args.model_id, device=args.device)

    stats_rows = []
    meta_rows = []

    n = min(len(df), args.max_imgs)
    for i in range(n):
        row = df.iloc[i]
        img_path = str(row["image_path"])

        img0 = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img0.size

        if args.use_letterbox:
            canvas, scale, pad_left, pad_top = letterbox_pil(img0, out_w=args.out_w, out_h=args.out_h, fill=0)
            rgb = np.array(canvas)
            H, W = rgb.shape[:2]  # out_h, out_w
        else:
            rgb = np.array(img0)
            H, W = rgb.shape[:2]
            scale, pad_left, pad_top = 1.0, 0, 0

        boxes = clip_boxes_xyxy(parse_boxes_str(row["boxes"]), W=W, H=H)[: args.topk]

        all_masks = []
        all_scores = []

        if args.use_crop:
            # per-box ROI segmentation
            for j in range(len(boxes)):
                rgb_crop, x0, y0 = crop_around_box(rgb, boxes[j], crop_size=args.crop_size)
                h, w = rgb_crop.shape[:2]

                predictor.set_image(rgb_crop)

                bx = boxes[j].copy()
                bx[0] -= x0; bx[2] -= x0
                bx[1] -= y0; bx[3] -= y0
                bx = clip_boxes_xyxy(bx.reshape(1, 4), W=w, H=h)[0]

                bxt = torch.from_numpy(bx.reshape(1, 4)).to(predictor.device)
                bxt = transform_boxes_for_sam2(predictor, bxt, orig_hw=(h, w))

                masks, scores, _ = predict_with_box(predictor, bxt)
                mu8 = squeeze_masks(masks)  # (1,h,w)
                m2d = mu8[0]

                if args.tighten:
                    m2d = tighten_mask_by_box(m2d, bx, margin=args.margin)

                m_full = paste_crop_mask(m2d, H=H, W=W, x0=x0, y0=y0)
                all_masks.append(m_full)

                if isinstance(scores, torch.Tensor):
                    all_scores.append(float(scores.detach().cpu().numpy()[0]))
                else:
                    all_scores.append(float(np.array(scores)[0]))

            masks_u8 = np.stack(all_masks, axis=0) if all_masks else np.zeros((0, H, W), np.uint8)
            scores_f = np.array(all_scores, dtype=np.float32)

        else:
            # full-canvas segmentation (not recommended for your data)
            predictor.set_image(rgb)

            boxes_t = torch.from_numpy(boxes).to(predictor.device)
            boxes_t = transform_boxes_for_sam2(predictor, boxes_t, orig_hw=(H, W))

            masks, scores, _ = predict_with_box(predictor, boxes_t)

            masks_u8 = squeeze_masks(masks)
            scores_f = scores.detach().cpu().numpy().astype(np.float32) if isinstance(scores, torch.Tensor) else np.array(scores, np.float32)

            if args.tighten and masks_u8.shape[0] > 0:
                tight_list = []
                for j in range(masks_u8.shape[0]):
                    tight_list.append(tighten_mask_by_box(masks_u8[j], boxes[j], margin=args.margin))
                masks_u8 = np.stack(tight_list, axis=0) if tight_list else masks_u8

        union_canvas = (masks_u8.sum(axis=0) > 0).astype(np.uint8) if masks_u8.shape[0] else np.zeros((H, W), np.uint8)

        # stats
        union_area = int(union_canvas.sum())
        union_area_ratio = union_area / float(H * W)
        expands = []
        for j in range(masks_u8.shape[0]):
            mbox = bbox_from_mask(masks_u8[j])
            if mbox is None:
                continue
            expands.append(box_area_xyxy(mbox) / max(1.0, box_area_xyxy(boxes[j])))

        stats_rows.append({
            "stem": Path(img_path).stem,
            "image_path": img_path,
            "orig_w": orig_w, "orig_h": orig_h,
            "canvas_w": W, "canvas_h": H,
            "use_letterbox": int(bool(args.use_letterbox)),
            "use_crop": int(bool(args.use_crop)),
            "crop_size": int(args.crop_size) if args.use_crop else 0,
            "topk": int(args.topk),
            "n_masks": int(masks_u8.shape[0]),
            "union_area": union_area,
            "union_area_ratio": union_area_ratio,
            "expand_mean": float(np.mean(expands)) if expands else 0.0,
            "expand_p95": float(np.percentile(expands, 95)) if expands else 0.0,
            "tighten": int(bool(args.tighten)),
            "margin": int(args.margin),
        })

        meta_rows.append({
            "stem": Path(img_path).stem,
            "image_path": img_path,
            "orig_w": orig_w, "orig_h": orig_h,
            "out_w": int(args.out_w if args.use_letterbox else W),
            "out_h": int(args.out_h if args.use_letterbox else H),
            "scale": float(scale),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
        })

        # save NPZ (canvas coords)
        stem = Path(img_path).stem
        np.savez_compressed(
            out_npz / f"{stem}.npz",
            image_path=img_path,
            orig_w=orig_w, orig_h=orig_h,
            canvas_h=H, canvas_w=W,
            use_letterbox=bool(args.use_letterbox),
            use_crop=bool(args.use_crop),
            crop_size=int(args.crop_size),
            scale=float(scale), pad_left=int(pad_left), pad_top=int(pad_top),
            boxes=boxes.astype(np.float32),
            scores=scores_f,
            masks=masks_u8.astype(np.uint8),
        )

        # vis on canvas
        if args.save_vis_canvas:
            vis = overlay_vis(rgb, union_canvas, boxes)
            Image.fromarray(vis).save(out_vis_canvas / f"{stem}.png")

        # vis back on original (mask only; boxes omitted to avoid coord confusion)
        if args.save_vis_orig:
            if not args.use_letterbox:
                # cannot invert without letterbox meta
                vis0 = np.array(img0).copy().astype(np.float32)
                Image.fromarray(vis0.astype(np.uint8)).save(out_vis_orig / f"{stem}.png")
            else:
                union_orig = mask_canvas_to_orig(
                    union_canvas, orig_w=orig_w, orig_h=orig_h, scale=scale, pad_left=pad_left, pad_top=pad_top
                )
                vis0 = np.array(img0).copy().astype(np.float32)
                m = union_orig.astype(np.float32)
                vis0[..., 1] = np.clip(vis0[..., 1] * (1 - 0.35 * m) + 255 * 0.35 * m, 0, 255)
                Image.fromarray(vis0.astype(np.uint8)).save(out_vis_orig / f"{stem}.png")

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{n}] done (letterbox={args.use_letterbox}, crop={args.use_crop}, tighten={args.tighten})")

    stats_path = out_root / "mask_stats_sam2_image.csv"
    meta_path = out_root / "letterbox_meta_used.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    print("Saved npz to:", out_npz)
    if args.save_vis_canvas:
        print("Saved canvas vis to:", out_vis_canvas)
    if args.save_vis_orig:
        print("Saved orig vis to:", out_vis_orig)
    print("Saved stats to:", stats_path)
    print("Saved meta  to:", meta_path)


if __name__ == "__main__":
    main()
