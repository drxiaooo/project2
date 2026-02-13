# -*- coding: utf-8 -*-
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


def find_one(root: Path, patterns):
    """Return first matching path under root for any glob pattern."""
    for pat in patterns:
        hits = list(root.glob(pat))
        if hits:
            # prefer directory if multiple
            hits = sorted(hits, key=lambda p: (not p.is_dir(), str(p)))
            return hits[0]
    return None


def safe_stem_from_path(x: str) -> str:
    try:
        return Path(str(x)).stem
    except Exception:
        return str(x)


def infer_canvas_hw(model_dir: Path, default_hw=(512, 1024)):
    """
    Try to infer (H, W) for area_ratio from letterbox_meta_used.csv.
    Fallback to default_hw = (512,1024).
    """
    meta = model_dir / "letterbox_meta_used.csv"
    if meta.exists():
        try:
            df = pd.read_csv(meta)
            # allow both out_h/out_w or H/W naming
            if "out_h" in df.columns and "out_w" in df.columns:
                H = int(df["out_h"].iloc[0])
                W = int(df["out_w"].iloc[0])
                return (H, W)
            if "H" in df.columns and "W" in df.columns:
                H = int(df["H"].iloc[0])
                W = int(df["W"].iloc[0])
                return (H, W)
        except Exception:
            pass
    return default_hw


def ensure_area_ratio(df_stats: pd.DataFrame, model_dir: Path, default_hw=(512, 1024)):
    """
    Ensure df_stats has 'area_ratio' column.
    If missing, compute from union_area / (H*W) where H,W inferred per model.
    """
    if "area_ratio" in df_stats.columns and df_stats["area_ratio"].notna().any():
        return df_stats

    # Determine union area column name
    union_col = None
    for c in ["union_area", "mask_area", "area_union", "area"]:
        if c in df_stats.columns:
            union_col = c
            break

    if union_col is None:
        # cannot compute; keep as NaN
        df_stats["area_ratio"] = np.nan
        return df_stats

    H, W = infer_canvas_hw(model_dir, default_hw=default_hw)
    denom = float(H * W)
    df_stats["area_ratio"] = df_stats[union_col].astype(float) / denom
    return df_stats


def quantiles(series: pd.Series):
    if series is None or len(series) == 0:
        return (np.nan, np.nan, np.nan)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(s.quantile(0.25)), float(s.quantile(0.50)), float(s.quantile(0.75)))


def load_model_bundle(model_dir: Path, default_hw=(512, 1024)):
    """
    Return dict with:
      stats_df (with area_ratio ensured, stem ensured),
      keep_df, drop_df,
      paths
    """
    stats_csv = find_one(model_dir, ["mask_stats_*_image.csv", "mask_stats*.csv"])
    keep_csv = model_dir / "qc_keep.csv"
    drop_csv = model_dir / "qc_drop.csv"

    if stats_csv is None or (not stats_csv.exists()):
        return None

    df = pd.read_csv(stats_csv)

    # Ensure stem exists
    if "stem" not in df.columns:
        if "image_path" in df.columns:
            df["stem"] = df["image_path"].map(safe_stem_from_path)
        elif "img" in df.columns:
            df["stem"] = df["img"].map(safe_stem_from_path)
        else:
            df["stem"] = np.arange(len(df)).astype(str)

    # Ensure union_area exists (if different naming, harmonize to union_area)
    if "union_area" not in df.columns:
        for c in ["mask_area", "area_union", "area"]:
            if c in df.columns:
                df["union_area"] = df[c]
                break

    # Ensure expand_mean exists (if different naming, harmonize)
    if "expand_mean" not in df.columns:
        for c in ["expand_mean_px", "expand_px_mean", "expand"]:
            if c in df.columns:
                df["expand_mean"] = df[c]
                break

    # Ensure area_ratio exists (this is what you want to fix)
    df = ensure_area_ratio(df, model_dir, default_hw=default_hw)

    keep = pd.read_csv(keep_csv) if keep_csv.exists() else pd.DataFrame()
    drop = pd.read_csv(drop_csv) if drop_csv.exists() else pd.DataFrame()

    return {
        "model_dir": model_dir,
        "stats_csv": stats_csv,
        "keep_csv": keep_csv if keep_csv.exists() else None,
        "drop_csv": drop_csv if drop_csv.exists() else None,
        "stats_df": df,
        "keep_df": keep,
        "drop_df": drop,
    }


def summarize_one(model: str, bundle):
    df = bundle["stats_df"]
    keep_df = bundle["keep_df"]
    total = int(df["stem"].nunique()) if "stem" in df.columns else int(len(df))

    keep = total
    keep_rate = 1.0
    if len(keep_df) > 0 and "stem" in keep_df.columns:
        keep = int(keep_df["stem"].nunique())
        keep_rate = float(keep / max(1, total))

    ar_q1, ar_med, ar_q3 = quantiles(df.get("area_ratio"))
    ex_q1, ex_med, ex_q3 = quantiles(df.get("expand_mean"))
    ua_q1, ua_med, ua_q3 = quantiles(df.get("union_area"))

    return {
        "model": model,
        "total": total,
        "keep": keep,
        "keep_rate": keep_rate,
        "area_ratio_q1": ar_q1,
        "area_ratio_med": ar_med,
        "area_ratio_q3": ar_q3,
        "expand_mean_q1": ex_q1,
        "expand_mean_med": ex_med,
        "expand_mean_q3": ex_q3,
        "union_area_q1": ua_q1,
        "union_area_med": ua_med,
        "union_area_q3": ua_q3,
        "stats_csv": str(bundle["stats_csv"]),
        "keep_csv": str(bundle["keep_csv"]) if bundle["keep_csv"] else "",
        "drop_csv": str(bundle["drop_csv"]) if bundle["drop_csv"] else "",
        "model_dir": str(bundle["model_dir"]),
    }


def _draw_label(im: Image.Image, text: str):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    pad = 10
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(255, 255, 255))
    draw.text((pad, pad), text, fill=(0, 0, 0), font=font)
    return im


def make_collage(df_map, outdir: Path, n: int, seed: int):
    """
    拼图（自动写模型名）：
    - 4个模型：2x2
    - 3个模型：2x2（最后一格灰）
    - 2个模型：1x2
    - <2：跳过
    """
    random.seed(seed)

    vis_patterns = [
        "pseudo_masks_*_vis",
        "*_vis",
        "*vis*",
        "*canvas*",
        "*overlay*",
        "*mask*vis*",
    ]

    model_stems = {}
    vis_dirs = {}

    for name, df in df_map.items():
        model_dir = Path(df["model_dir"].iloc[0])
        vis_dir = find_one(model_dir, vis_patterns)
        if vis_dir is None or (not vis_dir.is_dir()):
            print(f"[WARN] {name}: cannot find vis dir under {model_dir}")
            continue
        vis_dirs[name] = vis_dir

        stems = set(df["stem"].astype(str).tolist()) if "stem" in df.columns else set()
        ok = set()
        for s in stems:
            if (vis_dir / f"{s}.png").exists():
                ok.add(s)
        model_stems[name] = ok

    names = sorted(list(vis_dirs.keys()))
    if len(names) < 2:
        print("[WARN] collage skipped: need at least 2 models with vis dirs.")
        return

    common = None
    for name in names:
        common = model_stems[name] if common is None else (common & model_stems[name])
    common = sorted(list(common))
    if len(common) == 0:
        print("[WARN] collage skipped: no common stems across models.")
        return

    picks = random.sample(common, k=min(n, len(common)))

    collage_dir = outdir / "collage_2x2"
    collage_dir.mkdir(parents=True, exist_ok=True)

    for s in picks:
        imgs = []
        for name in names:
            p = vis_dirs[name] / f"{s}.png"
            if p.exists():
                im = Image.open(p).convert("RGB")
                im = ImageOps.expand(im, border=6, fill=(255, 255, 255))
                im = _draw_label(im, name)  # <-- label here
                imgs.append((name, im))
            else:
                im2 = Image.new("RGB", (800, 400), (240, 240, 240))
                im2 = _draw_label(im2, name)
                imgs.append((name, im2))

        valid = [im for _, im in imgs if im is not None]
        if not valid:
            continue

        w = min(im.size[0] for im in valid)
        h = min(im.size[1] for im in valid)

        tiles = []
        for name, im in imgs:
            im2 = im.resize((w, h))
            tiles.append((name, im2))

        k = len(tiles)
        if k >= 4:
            canvas = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
            canvas.paste(tiles[0][1], (0, 0))
            canvas.paste(tiles[1][1], (w, 0))
            canvas.paste(tiles[2][1], (0, h))
            canvas.paste(tiles[3][1], (w, h))
        elif k == 3:
            canvas = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
            canvas.paste(tiles[0][1], (0, 0))
            canvas.paste(tiles[1][1], (w, 0))
            canvas.paste(tiles[2][1], (0, h))
            canvas.paste(Image.new("RGB", (w, h), (240, 240, 240)), (w, h))
        elif k == 2:
            canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
            canvas.paste(tiles[0][1], (0, 0))
            canvas.paste(tiles[1][1], (w, 0))
        else:
            continue

        canvas.save(collage_dir / f"{s}_collage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/seg_compare", help="seg_compare root dir")
    ap.add_argument("--make_collage", action="store_true")
    ap.add_argument("--collage_n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--default_out_w", type=int, default=1024)
    ap.add_argument("--default_out_h", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = root / "_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    default_hw = (args.default_out_h, args.default_out_w)

    # Scan model dirs (exclude _summary)
    model_dirs = [p for p in root.iterdir() if p.is_dir() and p.name != "_summary"]
    bundles = {}
    for md in model_dirs:
        b = load_model_bundle(md, default_hw=default_hw)
        if b is None:
            continue
        bundles[md.name] = b

    if not bundles:
        raise RuntimeError(f"No valid model dirs with mask_stats found under: {root}")

    # Build merged stats for plots / collage
    merged = []
    df_map = {}
    summary_rows = []

    for model, bundle in bundles.items():
        df = bundle["stats_df"].copy()
        df["model"] = model
        df["model_dir"] = str(bundle["model_dir"])
        merged.append(df)

        # keep only useful columns for collage mapping
        df_map[model] = df

        summary_rows.append(summarize_one(model, bundle))

    merged_df = pd.concat(merged, axis=0, ignore_index=True)
    merged_csv = outdir / "merged_stats.csv"
    merged_df.to_csv(merged_csv, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"Saved summary: {summary_csv}")
    print(f"Saved merged:  {merged_csv}")

    # Hist plots (optional: if you already have, can keep as-is)
    # Keep simple: just save CSVs; figures can be added later if needed.

    if args.make_collage:
        make_collage(df_map, outdir, n=args.collage_n, seed=args.seed)
        print(f"Saved collages to: {outdir / 'collage_2x2'}")


if __name__ == "__main__":
    main()
