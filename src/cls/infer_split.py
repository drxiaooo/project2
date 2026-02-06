import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

from src.utils.letterbox import letterbox_pil


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, view_filter: str, img_h: int, img_w: int):
        df = pd.read_csv(csv_path)
        df = df[df["folder"].isin(["wholeBodyANT", "wholeBodyPOST"])].copy()

        view_filter = view_filter.upper()
        if view_filter in ("ANT", "POST"):
            df = df[df["folder"] == f"wholeBody{view_filter}"].copy()
        elif view_filter == "BOTH":
            pass
        else:
            raise ValueError("view must be ANT/POST/BOTH")

        df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
        missing = int((~df["exists"]).sum())
        if missing > 0:
            print(f"[WARN] {missing} files missing in {csv_path}, dropping them.")
        df = df[df["exists"]].copy()
        self.df = df.reset_index(drop=True)

        self.img_h = img_h
        self.img_w = img_w

        # NOTE: no Resize here; we letterbox in __getitem__
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        # STRATEGY B: letterbox keep ratio, pad to model input size
        img_pad, _, _, _ = letterbox_pil(img, out_w=self.img_w, out_h=self.img_h)

        x = self.tf(img_pad)
        y = float(row["label"])
        return x, y, str(row["image_path"])


def eval_metrics(y_true, y_prob, thr=0.5):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true.astype(int), y_pred, average="binary", zero_division=0)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    return {
        "auc": float(auc),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "sens": float(sens),
        "spec": float(spec),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n": int(tp + tn + fp + fn),
    }


def load_thr_from_json(path: str) -> float:
    p = Path(path)
    if p.is_dir():
        p = p / "selected_threshold.json"
    if not p.exists():
        raise FileNotFoundError(f"thr json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    thr = obj.get("thr", None)
    if thr is None:
        raise ValueError(f"'thr' not found in {p}")
    return float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--exp_dir", required=True, help="e.g. outputs/cls/resnetrs50_POST_letterbox")
    ap.add_argument("--model_name", required=True, help="e.g. resnetrs50 (same as training)")
    ap.add_argument("--view", default="POST", choices=["ANT", "POST", "BOTH"])
    ap.add_argument("--img_h", type=int, default=512)
    ap.add_argument("--img_w", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--thr", type=float, default=None, help="manual threshold; ignored if --thr_from is provided")
    ap.add_argument("--thr_from", type=str, default=None,
                    help="path to selected_threshold.json OR its directory; if set, overrides --thr")
    ap.add_argument("--amp", action="store_true", help="use AMP for inference to save memory")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)

    # threshold
    if args.thr_from is not None:
        thr = load_thr_from_json(args.thr_from)
    else:
        thr = float(args.thr) if args.thr is not None else 0.5

    # load checkpoint
    ckpt = torch.load(exp_dir / "best.pt", map_location="cpu")
    model = timm.create_model(args.model_name, pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    ds = CSVDataset(args.split_csv, args.view, args.img_h, args.img_w)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    y_true_list, y_prob_list, paths = [], [], []

    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logit = model(x).squeeze(1)
                prob = torch.sigmoid(logit)

            y_prob_list.append(prob.detach().cpu().numpy())
            y_true_list.append(np.asarray(y, dtype=np.float32))
            paths.extend(list(p))

    y_true = np.concatenate(y_true_list).astype(np.int32)
    y_prob = np.concatenate(y_prob_list).astype(np.float32)

    out_csv = exp_dir / f"preds_{Path(args.split_csv).stem}.csv"
    pd.DataFrame({"image_path": paths, "y_true": y_true, "y_prob": y_prob}).to_csv(out_csv, index=False)

    met = eval_metrics(y_true, y_prob, thr=thr)
    met["thr_used"] = float(thr)

    print("[METRICS]", json.dumps(met, indent=2, ensure_ascii=False))
    print("[OK] saved:", out_csv)


if __name__ == "__main__":
    main()
