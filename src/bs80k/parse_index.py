import re
import csv
from pathlib import Path

# 文件夹名结构: part + (L/R可选) + (ANT/POST)
FOLDER_RE = re.compile(r"^(?P<part>[A-Za-z]+)(?P<side>[LR])?(?P<view>ANT|POST)$")

def parse_folder_name(name: str):
    m = FOLDER_RE.match(name)
    if not m:
        return "unknown", "", ""
    return m.group("part"), (m.group("side") or ""), m.group("view")

def find_xml_path(data_root: Path, folder_name: str, img_stem: str) -> str:
    """
    BS-80K wholeBody:
    - wholeBodyANT\ant\{stem}.xml
    - wholeBodyPOST\post\{stem}.xml
    """
    if folder_name == "wholeBodyANT":
        xml_path = data_root / folder_name / "ant" / f"{img_stem}.xml"
        return str(xml_path.resolve()) if xml_path.exists() else ""
    if folder_name == "wholeBodyPOST":
        xml_path = data_root / folder_name / "post" / f"{img_stem}.xml"
        return str(xml_path.resolve()) if xml_path.exists() else ""
    return ""

def main():
    # BS-80K 根目录（包含 wholeBodyANT、kneeLANT 等文件夹的那一层）
    DATA_ROOT = Path(r"E:\project\project2\data")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bs80k_index.csv"

    rows = []
    folders = [p for p in DATA_ROOT.iterdir() if p.is_dir()]

    for folder in folders:
        folder_name = folder.name
        txt_path = folder / f"{folder_name}.txt"
        if not txt_path.exists():
            continue

        part, side, view = parse_folder_name(folder_name)

        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = line.split()  # 兼容多空格 / tab
                if len(toks) < 2:
                    continue
                filename, label = toks[0], toks[1]

                img_path = folder / filename
                img_path_str = str(img_path.resolve()) if img_path.exists() else str(img_path)

                img_stem = Path(filename).stem
                xml_path = find_xml_path(DATA_ROOT, folder_name, img_stem)

                rows.append({
                    "folder": folder_name,
                    "part": part,
                    "side": side,
                    "view": view,
                    "image_path": img_path_str,
                    "label": int(label),
                    "xml_path": xml_path,
                })

    fieldnames = ["folder", "part", "side", "view", "image_path", "label", "xml_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote: {out_csv}  rows={len(rows)}")

if __name__ == "__main__":
    main()
