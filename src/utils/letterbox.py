from typing import Tuple
from PIL import Image
import numpy as np

def letterbox_pil(
    img: Image.Image,
    out_w: int,
    out_h: int,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, float, int, int]:
    """
    Keep aspect ratio, resize + pad to (out_w,out_h).
    Returns: padded_img, scale, pad_left, pad_top
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = min(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_r = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (out_w, out_h), fill)
    pad_left = (out_w - nw) // 2
    pad_top = (out_h - nh) // 2
    canvas.paste(img_r, (pad_left, pad_top))
    return canvas, scale, pad_left, pad_top

def imagenet_normalize(np_rgb: np.ndarray) -> np.ndarray:
    """np_rgb: HWC uint8 or float in [0,255] -> float normalized"""
    x = np_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (x - mean) / std
