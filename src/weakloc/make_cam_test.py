import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import timm

from src.utils.letterbox import letterbox_pil


def load_thr(exp_dir: Path) -> float:
    p = exp_dir / "selected_threshold.json"
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj["thr"])


def preprocess_letterbox(img_pil: Image.Image, img_w: int, img_h: int) -> torch.Tensor:
    """
    Letterbox to (img_w,img_h) then ImageNet normalize -> tensor [1,3,H,W]
    NOTE: returns only tensor; meta is saved separately in main loop.
    """
    img_pad, _, _, _ = letterbox_pil(img_pil.convert("RGB"), out_w=img_w, out_h=img_h)
    arr = np.asarray(img_pad).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    x = torch.from_numpy(arr).unsqueeze(0).float()
    return x


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha=0.45) -> np.ndarray:
    heat_u8 = np.uint8(np.clip(heat * 255, 0, 255))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    out = (rgb * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="outputs/cls/resnetrs50_POST_letterbox",
                    help="exp dir containing best.pt / preds_test.csv / selected_threshold.json")
    ap.add_argument("--pred_csv", default=None, help="override pred csv; default exp_dir/preds_test.csv")
    ap.add_argument("--out_root", default="outputs/weakloc_letterbox",
                    help="output root folder")
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--max_imgs", type=int, default=999999)
    ap.add_argument("--min_prob", type=float, default=None,
                    help="override threshold; default uses selected_threshold.json")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)

    pred_csv = Path(args.pred_csv) if args.pred_csv else (exp_dir / "preds_test.csv")
    if not pred_csv.exists():
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")

    thr = args.min_prob if args.min_prob is not None else load_thr(exp_dir)
    print(f"[INFO] exp_dir={exp_dir}")
    print(f"[INFO] pred_csv={pred_csv}")
    print(f"[INFO] using thr={thr:.3f}")

    # ---- load model checkpoint (ResNet-RS50) ----
    ckpt = torch.load(exp_dir / "best.pt", map_location="cpu")
    model = timm.create_model("resnetrs50", pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # ---- target layer for Grad-CAM ----
    target_layer = model.layer4[-1]
    feats, grads = {}, {}

    def fwd_hook(_, __, output):
        feats["value"] = output

    def bwd_hook(_, grad_in, grad_out):
        grads["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    df = pd.read_csv(pred_csv)
    df = df[df["y_prob"] >= thr].copy()
    df = df.sort_values("y_prob", ascending=False).head(args.max_imgs).reset_index(drop=True)
    print(f"[INFO] selected {len(df)} images (y_prob>=thr)")

    out_root = Path(args.out_root)
    out_dir = out_root / "test_cam"
    out_npy = out_root / "test_cam_npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy.mkdir(parents=True, exist_ok=True)

    # ---- save letterbox meta for future unletterbox (orig coord export) ----
    meta_rows = []

    for i, row in df.iterrows():
        img_path = row["image_path"]
        prob = float(row["y_prob"])

        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        # IMPORTANT: use letterbox for overlay image too (same coords as model input)
        img_pad, scale, pad_left, pad_top = letterbox_pil(img_pil, out_w=args.img_w, out_h=args.img_h)
        rgb = np.asarray(img_pad).astype(np.uint8)

        # record meta
        meta_rows.append({
            "image_path": img_path,
            "orig_w": int(orig_w),
            "orig_h": int(orig_h),
            "out_w": int(args.img_w),
            "out_h": int(args.img_h),
            "scale": float(scale),
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "y_prob": float(prob),
        })

        x = preprocess_letterbox(img_pil, img_w=args.img_w, img_h=args.img_h).to(device)

        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            logit = model(x).squeeze(1)
            score = logit[0]

        score.backward()

        f = feats["value"]          # [1,C,Hf,Wf]
        g = grads["value"]          # [1,C,Hf,Wf]
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = (w * f).sum(dim=1, keepdim=True)
        cam = F.relu(cam)[0, 0].detach().float().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)

        stem = Path(img_path).stem
        npy_path = out_npy / f"{stem}_prob{prob:.3f}.npy"
        np.save(npy_path, cam)

        overlay = overlay_heatmap(rgb, cam, alpha=0.45)
        cv2.putText(overlay, f"prob={prob:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out_path = out_dir / f"{stem}_prob{prob:.3f}.png"
        Image.fromarray(overlay).save(out_path)

        if (i + 1) % 50 == 0:
            print(f"[OK] CAM {i+1}/{len(df)}")

    # save meta
    meta_csv = out_root / "letterbox_meta_test.csv"
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    print(f"[OK] saved letterbox meta to {meta_csv}")

    h1.remove()
    h2.remove()
    print(f"[DONE] saved overlays to {out_dir} and npy to {out_npy}")


if __name__ == "__main__":
    main()
