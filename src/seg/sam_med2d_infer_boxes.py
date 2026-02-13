# src/seg/sam_med2d_infer_boxes.py
import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch


def letterbox_pil(img: Image.Image, out_w: int, out_h: int, color=(0, 0, 0)):
    w, h = img.size
    scale = min(out_w / w, out_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), color)
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, pad_left, pad_top


def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def crop_around_box(img_np: np.ndarray, box, crop_size: int):
    H, W = img_np.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = crop_size // 2
    x0 = int(cx - half)
    y0 = int(cy - half)
    x0 = max(0, min(W - crop_size, x0))
    y0 = max(0, min(H - crop_size, y0))
    crop = img_np[y0:y0 + crop_size, x0:x0 + crop_size].copy()
    new_box = [x1 - x0, y1 - y0, x2 - x0, y2 - y0]
    new_box = clamp_box_xyxy(new_box, crop_size, crop_size)
    return crop, new_box, (x0, y0)


def paste_mask_back(mask_crop_u8: np.ndarray, H: int, W: int, offset):
    x0, y0 = offset
    out = np.zeros((H, W), dtype=np.uint8)
    h, w = mask_crop_u8.shape[:2]
    out[y0:y0 + h, x0:x0 + w] = (mask_crop_u8[:h, :w] > 0).astype(np.uint8)
    return out


def tighten_mask_by_box(mask_u8: np.ndarray, box, margin: int):
    # mask_u8: (H,W) uint8 0/255 OR 0/1 -> 都行
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    H, W = mask_u8.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W - 1, x2 + margin)
    y2 = min(H - 1, y2 + margin)
    out = np.zeros((H, W), dtype=np.uint8)
    out[y1:y2 + 1, x1:x2 + 1] = mask_u8[y1:y2 + 1, x1:x2 + 1]
    return out


def overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha=0.45):
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    out = img_bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    m = (mask_u8 > 0)[:, :, None]
    out = np.where(m, (out * (1 - alpha) + green * alpha).astype(np.uint8), out)
    return out


def load_ckpt_to_sam(model, ckpt_path: str):
    """
    PyTorch 2.6+ 默认 weights_only=True 可能无法加载包含optimizer等对象的ckpt。
    若你信任ckpt来源，这里显式 weights_only=False。
    同时兼容多种 checkpoint 格式：
    - raw state_dict
    - dict with keys: state_dict / model / sam
    """
    # ✅ 关键修复：weights_only=False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "sam" in ckpt and isinstance(ckpt["sam"], dict):
            sd = ckpt["sam"]
        else:
            # 有些ckpt会把权重放在更深层；这里兜底：优先找“像权重”的键
            # 如果 dict 里大部分 value 都是 Tensor，就当它是 state_dict
            tensor_like = sum([1 for v in ckpt.values() if torch.is_tensor(v)])
            if tensor_like > 0 and tensor_like / max(1, len(ckpt)) > 0.5:
                sd = ckpt
            else:
                # 最后兜底：常见字段
                for k in ["net", "network", "sam_model", "model_state_dict"]:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        sd = ckpt[k]
                        break
                else:
                    raise RuntimeError(
                        f"Cannot find state_dict in checkpoint keys: {list(ckpt.keys())[:30]}"
                    )
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # clean module. prefix
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_sd[nk] = v
    sd = new_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def build_sam_med2d_vit_b(sam_model_registry, device: str):
    """
    SAM-Med2D 的 sam_model_registry["vit_b"] 可能是两种形式：
    1) fn() -> model
    2) fn(args) -> model   (你现在遇到的就是这种)
    """
    build_fn = sam_model_registry["vit_b"]
    sig = inspect.signature(build_fn)

    if len(sig.parameters) == 0:
        return build_fn()

    # SAM-Med2D README 示例 image_size=256；并经常需要 encoder_adapter=True
    # 这里把常见字段都补齐，避免内部访问 AttributeError
    med_args = SimpleNamespace(
        image_size=256,
        encoder_adapter=True,
        checkpoint=None,
        sam_checkpoint=None,
        model_type="vit_b",
        device=device,
    )
    return build_fn(med_args)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--boxes_csv", type=str, required=True)

    ap.add_argument("--ckpt", type=str, required=True)
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
    return ap.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    boxes_csv = Path(args.boxes_csv)
    if not boxes_csv.exists():
        raise FileNotFoundError(f"boxes_csv not found: {boxes_csv}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # IMPORTANT: 来自 third_party/SAM-Med2D 的 segment_anything
    from segment_anything import sam_model_registry, SamPredictor

    # output dirs
    npz_dir = out_root / "pseudo_masks_sam_med2d_npz"
    vis_dir = out_root / "pseudo_masks_sam_med2d_vis"
    npz_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis_canvas:
        vis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(boxes_csv)
    sort_col = "heat_score" if "heat_score" in df.columns else None

    def agg(g):
        if sort_col:
            g = g.sort_values(sort_col, ascending=False)
        else:
            g = g.sort_values("box_id", ascending=True) if "box_id" in g.columns else g
        return g.head(args.topk)

    tmp = (
        df.groupby("image_path", sort=False, group_keys=False)[df.columns]
        .apply(agg)
        .reset_index(drop=True)
    )

    image_list = tmp["image_path"].drop_duplicates().tolist()
    if args.max_imgs < len(image_list):
        image_list = image_list[: args.max_imgs]

    img2boxes = {}
    for p in image_list:
        gi = tmp[tmp["image_path"] == p]
        boxes = gi[["x1", "y1", "x2", "y2"]].values.astype(np.int32).tolist()
        img2boxes[str(p)] = boxes

    # ---- build model (兼容 SAM-Med2D 的 registry 签名) ----
    model = build_sam_med2d_vit_b(sam_model_registry, device=args.device)
    missing, unexpected = load_ckpt_to_sam(model, str(ckpt_path))
    model.to(args.device)
    model.eval()

    print(f"[INFO] loaded ckpt: {ckpt_path}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")

    predictor = SamPredictor(model)

    rows = []
    pbar = tqdm(image_list, total=len(image_list))
    for i, img_path in enumerate(pbar, start=1):
        img_path = str(img_path)
        stem = Path(img_path).stem
        pil = Image.open(img_path).convert("RGB")

        if args.use_letterbox:
            canvas, scale, pad_left, pad_top = letterbox_pil(pil, args.out_w, args.out_h)
            img_rgb = np.array(canvas)
            meta = dict(
                orig_w=pil.size[0], orig_h=pil.size[1],
                out_w=args.out_w, out_h=args.out_h,
                scale=scale, pad_left=pad_left, pad_top=pad_top
            )
        else:
            img_rgb = np.array(pil)
            meta = dict(
                orig_w=pil.size[0], orig_h=pil.size[1],
                out_w=img_rgb.shape[1], out_h=img_rgb.shape[0],
                scale=1.0, pad_left=0, pad_top=0
            )

        H, W = img_rgb.shape[:2]
        boxes = [clamp_box_xyxy(b, W, H) for b in img2boxes[img_path]]

        predictor.set_image(img_rgb)

        masks_full = []
        expands = []

        for b in boxes:
            if args.use_crop:
                crop_rgb, crop_box, offset = crop_around_box(img_rgb, b, args.crop_size)
                predictor.set_image(crop_rgb)

                crop_box_np = np.array(crop_box, dtype=np.float32)
                masks, scores, _ = predictor.predict(box=crop_box_np, multimask_output=False)

                mask_u8 = (masks[0].astype(np.uint8) * 255)
                if args.tighten:
                    mask_u8 = tighten_mask_by_box(mask_u8, crop_box, margin=args.margin)

                mask_full = paste_mask_back(mask_u8, H, W, offset)

                predictor.set_image(img_rgb)
            else:
                box_np = np.array(b, dtype=np.float32)
                masks, scores, _ = predictor.predict(box=box_np, multimask_output=False)
                mask_u8 = (masks[0].astype(np.uint8) * 255)
                if args.tighten:
                    mask_u8 = tighten_mask_by_box(mask_u8, b, margin=args.margin)
                mask_full = (mask_u8 > 0).astype(np.uint8)

            masks_full.append(mask_full.astype(np.uint8))
            area = float(mask_full.sum())
            box_area = float(max(1, (b[2] - b[0] + 1) * (b[3] - b[1] + 1)))
            expands.append(area / box_area)

        union = np.zeros((H, W), dtype=np.uint8) if len(masks_full) == 0 else \
            (np.clip(np.sum(np.stack(masks_full, axis=0), axis=0), 0, 1)).astype(np.uint8)

        union_area = float(union.sum())
        area_ratio = union_area / float(H * W)
        expand_mean = float(np.mean(expands)) if len(expands) else 0.0

        np.savez_compressed(
            npz_dir / f"{stem}.npz",
            mask=union.astype(np.uint8),
            masks=np.stack(masks_full, axis=0).astype(np.uint8) if len(masks_full) else np.zeros((0, H, W), dtype=np.uint8),
            boxes=np.array(boxes, dtype=np.int32),
            meta=meta,
        )

        if args.save_vis_canvas:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            vis = overlay_mask(img_bgr, (union * 255).astype(np.uint8))
            for b in boxes:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
            cv2.imwrite(str(vis_dir / f"{stem}.png"), vis)

        rows.append({
            "image_path": img_path,
            "stem": stem,
            "H": H, "W": W,
            "num_boxes": len(boxes),
            "num_masks": int(len(masks_full)),
            "union_area": union_area,
            "area_ratio": area_ratio,
            "expand_mean": expand_mean,
            "use_letterbox": bool(args.use_letterbox),
            "use_crop": bool(args.use_crop),
            "crop_size": int(args.crop_size) if args.use_crop else 0,
            "tighten": bool(args.tighten),
            "margin": int(args.margin) if args.tighten else 0,
            "ckpt": str(ckpt_path),
        })

        pbar.set_description(f"[{i}/{len(image_list)}] SAM-Med2D(vit_b)")

    stats_csv = out_root / "mask_stats_sam_med2d_image.csv"
    pd.DataFrame(rows).to_csv(stats_csv, index=False)
    print(f"Saved npz  to: {npz_dir}")
    print(f"Saved stats: {stats_csv}")
    if args.save_vis_canvas:
        print(f"Saved vis  to: {vis_dir}")


if __name__ == "__main__":
    main()
