
本项目面向**全身骨显像（bone scintigraphy）**中的疑似病灶定位与分割，采用“**弱监督定位（CAM/heatmap → bbox）→ 基于 bbox 的分割模型推理（SAM 系列/医学适配模型）→ 质量控制（QC）→ 汇总对比**”的工程化流程，最终将 **MedSAM (ViT-B)** 作为后续实验的主分割模型。

> 课题目标（文档主线）：围绕“骨显像病灶精确分割 →（后续）良恶性判别 →（后续）全身负荷/分布评估”，构建可复现、可审查的 baseline 工程与结果证据包；并按周输出周报、持续 commit。

---

## 1. 项目结构（建议理解路径）

```

project2/
src/
weakloc/                    # 弱监督定位：CAM/heatmap 生成与 bbox 提取
seg/                        # 分割：SAM2 / MedSAM2 / MedSAM / SAM-Med2D 推理 + QC + 汇总
sam2/                         # SAM2 相关代码与依赖（推理脚本入口在 src/seg）
third_party/
SAM-Med2D/                  # 第三方：SAM-Med2D 源码（不一定是 pip 包形式）
outputs/
cls/                        # 分类模型实验输出（弱监督定位所依赖的 preds_test.csv）
weakloc_fulltest/           # ✅ 全量 CAM/heatmap + bbox 输出（本周新增）
seg_compare/
sam2_full/                # ✅ SAM2 full 推理输出
medsam2_full/             # ✅ MedSAM2 full 推理输出（tiny）
medsam_full/              # ✅ MedSAM full 推理输出（ViT-B，最终选用）
sam_med2d_full/           # ✅ SAM-Med2D full 推理输出（ViT-B）
_summary_full/            # ✅ 汇总目录（summary.csv / merged_stats.csv / 直方图 / 拼图等）
_archive/
2026-02-12_subset120_box48_crop320_topk1_tighten8/   # ✅ 子集实验归档（防止污染）

````

---

## 2. 数据与预处理约定

- 输入图像：全身骨显像 JPG（示例路径：`data/wholeBodyPOST/*.jpg`）
- 统一推理几何策略（本周全量一致）：
  - `--use_letterbox --out_w 1024 --out_h 512`：统一到 1024×512 的画布（保持纵横比+padding）
  - `--use_crop --crop_size 320`：围绕 bbox 做局部裁剪推理
  - `--topk 1`：每张图只取 top1 bbox（简化管线，便于稳定对比）
  - `--tighten --margin 8`：对 bbox 收紧/扩边，避免过松引入噪声或过紧截断目标

---

## 3. 环境（Conda）

本项目采用两个环境（按你这周的实践）：

- `boneai`：弱监督定位、QC、汇总对比（pandas/绘图/统计等）
- `seg_sam2`：分割推理（SAM2/MedSAM2/SAM-Med2D/MedSAM 推理依赖）

建议你在仓库后续补充：
- `environment_boneai.yml`
- `environment_seg_sam2.yml`
以满足“可复现/可复查”的要求。

---

## 4. 本周已跑通的全量流程（341 → 275 的说明见 4.3）

> 下面命令均在 `E:\project\project2` 路径示例；Linux/Mac 同理改路径。

### 4.1 弱监督定位：生成 CAM/heatmap（全量）
```powershell
conda activate boneai
cd E:\project\project2

python -m src.weakloc.make_cam_test `
  --exp_dir outputs/cls/resnetrs50_POST_letterbox `
  --pred_csv outputs/cls/resnetrs50_POST_letterbox/preds_test.csv `
  --out_root outputs/weakloc_fulltest `
  --min_prob 0 `
  --amp --max_imgs 999999
````

输出：

* `outputs/weakloc_fulltest/test_cam/`（可视化叠加图）
* `outputs/weakloc_fulltest/test_cam_npy/`（heatmap npy）
* `outputs/weakloc_fulltest/letterbox_meta_test.csv`

### 4.2 heatmap → bbox（全量）

```powershell
python -m src.weakloc.heatmap_to_bbox_test `
  --exp_dir outputs/cls/resnetrs50_POST_letterbox `
  --out_root outputs/weakloc_fulltest `
  --max_boxes 5 --box_w 48 --box_h 48 --nms_iou 0.15
```

输出：

* `outputs/weakloc_fulltest/pseudo_boxes_test.csv`


## 5. 分割推理（四模型对比）与最终选择

### 5.1 SAM2（full）

```powershell
conda activate seg_sam2
cd E:\project\project2\sam2
$env:PYTHONPATH="E:\project\project2;E:\project\project2\sam2"

python -m src.seg.sam2_infer_boxes `
  --out_root E:\project\project2\outputs\seg_compare\sam2_full `
  --boxes_csv E:\project\project2\outputs\weakloc_fulltest\pseudo_boxes_test.csv `
  --use_letterbox --out_w 1024 --out_h 512 `
  --use_crop --crop_size 320 `
  --topk 1 --max_imgs 999999 `
  --tighten --margin 8 `
  --save_vis_canvas
```

### 5.2 MedSAM2（full，当前使用 tiny）

```powershell
conda activate seg_sam2
cd E:\project\project2\sam2
$env:PYTHONPATH="E:\project\project2;E:\project\project2\sam2"

python -m src.seg.medsam2_infer_boxes `
  --out_root E:\project\project2\outputs\seg_compare\medsam2_full `
  --use_letterbox --out_w 1024 --out_h 512 `
  --use_crop --crop_size 320 `
  --topk 1 --max_imgs 999999 `
  --tighten --margin 8 `
  --save_vis_canvas
```

### 5.3 MedSAM（full，ViT-B，✅后续主力）

```powershell
conda activate seg_sam2
cd E:\project\project2\sam2
$env:PYTHONPATH="E:\project\project2;E:\project\project2\sam2"

python -m src.seg.medsam_infer_boxes `
  --out_root E:\project\project2\outputs\seg_compare\medsam_full `
  --boxes_csv E:\project\project2\outputs\weakloc_fulltest\pseudo_boxes_test.csv `
  --ckpt E:\project\project2\checkpoints\medsam\medsam_vit_b.pth `
  --use_letterbox --out_w 1024 --out_h 512 `
  --use_crop --crop_size 320 `
  --topk 1 --max_imgs 999999 `
  --tighten --margin 8 `
  --save_vis_canvas
```

### 5.4 SAM-Med2D（full，ViT-B）

```powershell
conda activate seg_sam2
cd E:\project\project2
$env:PYTHONPATH="E:\project\project2;E:\project\project2\third_party\SAM-Med2D"

python -m src.seg.sam_med2d_infer_boxes `
  --out_root outputs\seg_compare\sam_med2d_full `
  --boxes_csv outputs\weakloc_fulltest\pseudo_boxes_test.csv `
  --ckpt E:\project\project2\checkpoints\sam_med2d\sam-med2d_b.pth `
  --use_letterbox --out_w 1024 --out_h 512 `
  --use_crop --crop_size 320 `
  --topk 1 --max_imgs 999999 `
  --tighten --margin 8 `
  --save_vis_canvas
```

---

## 6. QC（质量控制）与“阈值”的作用

### 6.1 运行 QC

```powershell
conda activate boneai
cd E:\project\project2

python -m src.seg.qc_filter_masks --out_root outputs/seg_compare/sam2_full --require_npz
python -m src.seg.qc_filter_masks --out_root outputs/seg_compare/medsam2_full --require_npz
python -m src.seg.qc_filter_masks --out_root outputs/seg_compare/medsam_full --require_npz
python -m src.seg.qc_filter_masks --out_root outputs/seg_compare/sam_med2d_full --require_npz
```



## 7. 汇总对比（full）

> 说明：当前 `compare_models.py` 的 CLI 以 `--root` 为主（一次跑一个或脚本内部扫描）。如果你需要 `--roots/--outdir` 这种一次性入口，可以后续补一个 wrapper/参数解析。

典型输出文件：

* `summary.csv`：每个模型的 keep_rate、area_ratio/expand_mean/union_area 分位数等
* `merged_stats.csv`：按 image 维度合并四模型统计，便于做进一步分析/画图/抽样

---

## 8. 本周 full 对比结果（摘要）

基于 `summary.csv`（275 张有 bbox 的样本）：

* **MedSAM (ViT-B)**：keep_rate = 1.00（QC 全保留），mask 视觉效果整体最好 → ✅后续选用
* **MedSAM2 (tiny)**：keep_rate = 1.00（QC 全保留）
* **SAM-Med2D (ViT-B)**：keep_rate ≈ 0.993（少量极小 mask 被过滤）
* **SAM2**：keep_rate ≈ 0.709（大量 mask 因 area_ratio 过小被过滤，说明在本任务设置下更容易产出“极小/无效”mask）

> 后续实验将以 MedSAM (ViT-B) 为主线推进。

---

