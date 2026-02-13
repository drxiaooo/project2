# src/seg/medsam2_infer_boxes.py
# MedSAM2 (SAM2.1-hiera-tiny) box-prompt inference on BS-80K LETTERBOX canvas (1024x512).
#
# Reads:  out_root/pseudo_boxes_test.csv   (boxes in letterbox canvas coords)
# Uses:   original image_path (thin) -> letterbox_pil(...) -> 1024x512 canvas
# Outputs:
#   - out_root/pseudo_masks_medsam2_npz/*.npz     (uint8 mask, HxW in canvas coords)
#   - out_root/mask_stats_medsam2_image.csv
#   - optional: out_root/pseudo_masks_medsam2_vis/*.png  (canvas vis)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch

from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# IMPORTANT: build 1024x512 canvas that matches your weakloc pipeline
from src.utils.letterbox import letterbox_pil


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _clip_box_xyxy(box, W, H):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _crop_with_context(img_rgb: np.ndarray, box_xyxy, crop_size: int):
    """Center crop around the box center. Return cropped image and (x0,y0) offset."""
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    half = crop_size // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = max(0, min(x0, W - crop_size))
    y0 = max(0, min(y0, H - crop_size))

    crop = img_rgb[y0:y0 + crop_size, x0:x0 + crop_size]
    return crop, x0, y0


def _tighten_mask_by_box(mask_u8: np.ndarray, box_xyxy, margin: int):
    """Keep only mask inside (box expanded by margin). mask_u8: HxW or 1xHxW."""
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[0]
    H, W = mask_u8.shape
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W - 1, x2 + margin)
    y2 = min(H - 1, y2 + margin)

    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[y1:y2 + 1, x1:x2 + 1] = mask_u8[y1:y2 + 1, x1:x2 + 1]
    return out


def _mask_stats(stem, image_path, union_mask_u8):
    H, W = union_mask_u8.shape
    area = int((union_mask_u8 > 0).sum())
    area_ratio = float(area) / float(H * W)

    ys, xs = np.where(union_mask_u8 > 0)
    if len(xs) == 0:
        return dict(
            stem=stem, image_path=image_path,
            n_masks=0, union_area=0, union_area_ratio=0.0,
            union_bbox_area=0, expand_ratio=0.0, expand_mean=0.0
        )

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox_area = int((x2 - x1 + 1) * (y2 - y1 + 1))
    expand = bbox_area - area
    expand_ratio = float(expand) / float(max(1, bbox_area))
    expand_mean = float(bbox_area) / float(max(1, area)) - 1.0

    return dict(
        stem=stem, image_path=image_path,
        n_masks=1, union_area=area, union_area_ratio=area_ratio,
        union_bbox_area=bbox_area, expand_ratio=expand_ratio, expand_mean=expand_mean
    )


def _save_vis(canvas_rgb: np.ndarray, box_xyxy, mask_u8: np.ndarray, out_png: Path):
    im = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(im)

    # red box
    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    # green overlay
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[0]
    m = (mask_u8 > 0)
    if m.any():
        arr = np.array(im).astype(np.uint8)
        arr[m, 1] = np.clip(arr[m, 1].astype(np.int32) + 120, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)


def build_medsam2_predictor(device: str):
    # MedSAM2_latest.pt matches SAM2.1-hiera-tiny
    cfg = hf_hub_download("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_t.yaml")
    ckpt = hf_hub_download("wanglab/MedSAM2", "MedSAM2_latest.pt")
    model = build_sam2(cfg, ckpt, device=device)
    return SAM2ImagePredictor(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True,
                    help="folder that contains pseudo_boxes_test.csv and where outputs will be written")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--max_imgs", type=int, default=999999)

    ap.add_argument("--use_letterbox", action="store_true")
    ap.add_argument("--out_w", type=int, default=1024)
    ap.add_argument("--out_h", type=int, default=512)

    ap.add_argument("--use_crop", action="store_true")
    ap.add_argument("--crop_size", type=int, default=320)

    ap.add_argument("--tighten", action="store_true")
    ap.add_argument("--margin", type=int, default=8)

    ap.add_argument("--save_vis_canvas", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    boxes_csv = out_root / "pseudo_boxes_test.csv"
    if not boxes_csv.exists():
        raise FileNotFoundError(f"pseudo_boxes_test.csv not found: {boxes_csv}")

    npz_dir = out_root / "pseudo_masks_medsam2_npz"
    vis_dir = out_root / "pseudo_masks_medsam2_vis"
    _ensure_dir(npz_dir)
    if args.save_vis_canvas:
        _ensure_dir(vis_dir)

    predictor = build_medsam2_predictor(args.device)

    df = pd.read_csv(boxes_csv)

    sort_col = "heat_score" if "heat_score" in df.columns else None

    def agg(g):
        if sort_col is not None:
            g2 = g.sort_values(sort_col, ascending=False)
        else:
            g2 = g
        return g2.head(args.topk)

    tmp = df.groupby("image_path", sort=False, group_keys=False).apply(agg).reset_index(drop=True)
    images = tmp["image_path"].drop_duplicates().tolist()
    images = images[: min(len(images), args.max_imgs)]

    stats_rows = []
    n_total = len(images)

    for i, img_path in enumerate(images, 1):
        stem = Path(img_path).stem

        # ---- IMPORTANT FIX: build letterbox canvas (1024x512) to match box coords ----
        im_pil = Image.open(img_path).convert("RGB")
        if args.use_letterbox:
            canvas_pil, scale, pad_left, pad_top = letterbox_pil(im_pil, out_w=args.out_w, out_h=args.out_h)
            canvas = np.array(canvas_pil)
        else:
            canvas = np.array(im_pil)
        # ---------------------------------------------------------------------------

        H, W = canvas.shape[:2]

        g = tmp[tmp["image_path"] == img_path]
        boxes = g[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        boxes_clipped = [_clip_box_xyxy(b, W, H) for b in boxes]

        crop_x0 = crop_y0 = 0
        if args.use_crop:
            crop, crop_x0, crop_y0 = _crop_with_context(canvas, boxes_clipped[0], args.crop_size)
            work_img = crop
            work_H, work_W = work_img.shape[:2]
            work_boxes = []
            for b in boxes_clipped:
                x1, y1, x2, y2 = b
                work_boxes.append([x1 - crop_x0, y1 - crop_y0, x2 - crop_x0, y2 - crop_y0])
            work_boxes = np.array(work_boxes, dtype=np.float32)
        else:
            work_img = canvas
            work_H, work_W = H, W
            work_boxes = np.array(boxes_clipped, dtype=np.float32)

        predictor.set_image(work_img)

        masks, _, _ = predictor.predict(
            box=work_boxes,
            multimask_output=False
        )
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        masks_u8 = (masks > 0.5).astype(np.uint8) * 255  # (N,H,W)

        union = np.zeros((work_H, work_W), dtype=np.uint8)
        for j in range(masks_u8.shape[0]):
            m = masks_u8[j]
            if args.tighten:
                m = _tighten_mask_by_box(m, _clip_box_xyxy(work_boxes[j], work_W, work_H), margin=args.margin)
            union = np.maximum(union, m)

        if args.use_crop:
            full = np.zeros((H, W), dtype=np.uint8)
            full[crop_y0:crop_y0 + work_H, crop_x0:crop_x0 + work_W] = union
            union_full = full
        else:
            union_full = union

        np.savez_compressed(npz_dir / f"{stem}.npz", mask=union_full)

        stats_rows.append(_mask_stats(stem, img_path, union_full))

        if args.save_vis_canvas:
            out_png = vis_dir / f"{stem}.png"
            _save_vis(canvas, boxes_clipped[0], union_full, out_png)

        if (i % 10 == 0) or (i == n_total):
            print(f"[{i}/{n_total}] done (MedSAM2 tiny, letterbox={args.use_letterbox}, crop={args.use_crop}, tighten={args.tighten})")

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = out_root / "mask_stats_medsam2_image.csv"
    stats_df.to_csv(stats_csv, index=False)
    print("Saved npz  to:", str(npz_dir))
    print("Saved stats:", str(stats_csv))
    if args.save_vis_canvas:
        print("Saved vis  to:", str(vis_dir))


if __name__ == "__main__":
    main()
