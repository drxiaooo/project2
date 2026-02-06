
本仓库用于在 **BS-80K 数据集**上完成 2D 全身骨显像（wholeBody ANT/POST）的两阶段流程：

1) **弱分类（normal/abnormal）**：输出异常概率 `y_prob` + 自动阈值选择  
2) **弱定位（Grad-CAM）**：输出 heatmap  
3) **伪框（pseudo boxes）**：从 heatmap 峰值生成固定大小小框（用于后续分割/大模型 prompt）  
4) （进行中）**伪 mask**：计划用 MedSAM2 等分割大模型把伪框转为伪 mask，再训练专用分割网络

> 当前主线视角：**POST**  
> 关键预处理：**letterbox 保比例到 1024×512**（避免 resize 拉伸导致定位失真）

---

## 1. 环境

- Python: **3.10.19**
- PyTorch: **2.10.0+cu128**
- GPU: **NVIDIA RTX 5070 12GB**（推理/弱定位时 batch 需要控制）
- 运行方式建议统一使用：`python -m src.xxx`（避免 `No module named 'src'`）

缓存位置（可选参考）：
- HuggingFace: `C:\Users\<USER>\.cache\huggingface\hub`
- Torch Hub: `C:\Users\<USER>\.cache\torch\hub`

---

## 2. 数据说明（BS-80K）

数据集文件夹约定（来自 BS-80K 说明）：
- 文件夹名结构：`part + L/R(optional) + view`，例如 `kneeLANT`
- 每个文件夹包含同名 `.txt`：每行记录 `image_filename label`，`0=normal`，`1=abnormal`
- whole body 文件夹：`wholeBodyANT`、`wholeBodyPOST`
  - 其下额外包含 `ant/`、`post/` 子文件夹，内部 `.xml` 提供检测框信息（若存在）

你本地示例：
- wholeBody 图片路径：`E:\project\project2\data\wholeBodyANT` / `...wholeBodyPOST`
- xml 示例：`E:\project\project2\data\wholeBodyANT\ant\1.xml`

---

## 3. 项目结构

```

project2/
src/
**init**.py
bs80k/
parse_index.py
split_wholebody.py
cls/
train_wholebody_cls.py
select_threshold.py
infer_split.py
make_leaderboard.py
weakloc/
make_cam_test.py
heatmap_to_bbox_test.py
visualize_boxes_test.py
export_boxes_to_orig_and_draw.py
utils/
letterbox.py
outputs/
bs80k_index.csv
splits/
train.csv val.csv test.csv
cls/
<exp_dir>/
best.pt
preds_test.csv
selected_threshold.json
...
weakloc_letterbox/
test_cam/
test_cam_npy/
pseudo_boxes_test.csv
letterbox_meta_test.csv
test_boxes_vis/
pseudo_boxes_test_orig.csv
test_boxes_vis_orig/

````

---

## 4. 从零开始复现（推荐命令）

### Step 1：生成索引与划分（wholeBody）
```powershell
python -m src.bs80k.parse_index
python -m src.bs80k.split_wholebody
````

输出：

* `outputs/bs80k_index.csv`（全量索引）
* `outputs/splits/train.csv` / `val.csv` / `test.csv`（wholeBody 划分）

> 参考运行结果：index 约 82545 行；wholeBody 划分：train=5194 / val=648 / test=652

---

### Step 2：训练弱分类（ResNet-RS50 + POST + letterbox 1024×512）

```powershell
python -m src.cls.train_wholebody_cls `
  --model "hf_hub:timm/resnetrs50.tf_in1k" `
  --view POST `
  --img_h 512 --img_w 1024 `
  --bs 16 --epochs 20 --lr 1e-4 `
  --amp `
  --outdir resnetrs50_POST_letterbox
```

输出目录：

* `outputs/cls/resnetrs50_POST_letterbox/`

---

### Step 3：自动选择阈值（val 上 Youden + sens_min）

```powershell
python -m src.cls.select_threshold `
  --exp_dir outputs/cls/resnetrs50_POST_letterbox `
  --mode youden `
  --save
```

输出：

* `outputs/cls/resnetrs50_POST_letterbox/selected_threshold.json`

示例（一次实验结果）：

* best_epoch=9
* thr=0.361
* val：sens≈0.874 spec≈0.827 f1≈0.772

---

### Step 4：在 test 集推理评估（使用 thr_from）

```powershell
python -m src.cls.infer_split `
  --split_csv outputs/splits/test.csv `
  --exp_dir outputs/cls/resnetrs50_POST_letterbox `
  --model_name resnetrs50 `
  --view POST `
  --img_h 512 --img_w 1024 `
  --bs 4 `
  --amp `
  --thr_from outputs/cls/resnetrs50_POST_letterbox
```

输出：

* `outputs/cls/resnetrs50_POST_letterbox/preds_test.csv`

示例（一次实验结果）：

* AUC=0.8827
* Acc=0.7830
* Sens=0.8208
* Spec=0.7660
* TP=87 TN=180 FP=55 FN=19 (n=341)

---

## 5. 弱定位与伪框（CAM → boxes）

### Step 5：生成 CAM（仅对 y_prob >= thr 的样本）

```powershell
python -m src.weakloc.make_cam_test `
  --exp_dir outputs/cls/resnetrs50_POST_letterbox `
  --out_root outputs/weakloc_letterbox `
  --amp `
  --max_imgs 999999
```

输出：

* `outputs/weakloc_letterbox/test_cam/`（叠加图）
* `outputs/weakloc_letterbox/test_cam_npy/`（heatmap npy）
* `outputs/weakloc_letterbox/letterbox_meta_test.csv`（反变换元信息：scale/pad 等）

> 说明：heatmap 坐标系为 **letterbox 画布 1024×512**。

---

### Step 6：从 heatmap 生成伪框（peak-fixed-box + NMS）

推荐参数（当前使用 48×48 小框）：

```powershell
python -m src.weakloc.heatmap_to_bbox_test `
  --exp_dir outputs/cls/resnetrs50_POST_letterbox `
  --out_root outputs/weakloc_letterbox `
  --max_boxes 5 `
  --box_w 48 --box_h 48 `
  --nms_iou 0.15
```

输出：

* `outputs/weakloc_letterbox/pseudo_boxes_test.csv`

---

### Step 7：伪框可视化（画在 1024×512 画布上）

```powershell
python -m src.weakloc.visualize_boxes_test `
  --out_root outputs/weakloc_letterbox `
  --max_imgs 80
```

输出：

* `outputs/weakloc_letterbox/test_boxes_vis/`

---

### Step 8：导出原图坐标框 + 原图叠框可视化（可选）

```powershell
python -m src.weakloc.export_boxes_to_orig_and_draw `
  --out_root outputs/weakloc_letterbox `
  --max_imgs 120
```

输出：

* `outputs/weakloc_letterbox/pseudo_boxes_test_orig.csv`
* `outputs/weakloc_letterbox/test_boxes_vis_orig/`

> 该步骤使用 `letterbox_meta_test.csv` 做坐标反变换，保证框在原图位置正确。

---

## 6. 关键设计选择（Why 1024×512 letterbox）

全身骨显像图像通常细长，直接 resize 会造成几何变形，进而影响：

* CAM heatmap 的空间解释
* 伪框/后续分割 prompt 的坐标一致性

因此统一采用 **letterbox 保比例到 1024×512**：

* 不变形（scale+padding）
* 全流程统一坐标系（训练 / CAM / boxes / 可视化）
* 同时保存 `scale/pad_left/pad_top`，支持将结果反变换回原图坐标

---

## 7. 下一步计划（TODO）

* [ ] 使用 MedSAM2（box prompt）将 `pseudo_boxes_test.csv` 转换为 `pseudo masks`
* [ ] 伪 mask 质量控制：面积阈值、连通域过滤、与 y_prob/heat_score 一致性检查
* [ ] （可选）用伪 mask 训练专用 2D 分割模型（U-Net/SegFormer 等），形成可部署方案
* [ ] 输出“一键管线”：train → select_thr → infer → CAM → boxes → masks → export orig

---

## 8. 常见问题（FAQ）

### Q1：为什么建议用 `python -m`？

避免 `ModuleNotFoundError: No module named 'src'`，并统一包导入路径。

### Q2：推理/弱定位 OOM 怎么办？

降低 batch size（例如 `--bs 4`），开启 `--amp`，必要时减少 `max_imgs`。

### Q3：伪框太大怎么办？

优先使用 peak-fixed-box（当前 48×48），可进一步：

* 减小 `box_w/box_h`
* 增大 `max_boxes`
* 调整 `nms_iou`

---
