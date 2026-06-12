# 第五章實驗框架與結果觀察工具

把 `experiments/` 整個資料夾放到 `court_corner/` 的**同層目錄**（與
`detect_corners.py` 同層），即可使用。所有腳本以 module 方式執行
（`python -m experiments.xxx`），結果統一存到 `results/*.json`，
最後由 `make_report.py` 彙整、由 `results_viewer.py` 互動觀察。

```
court_corner_tool/
  court_corner/          ← 既有管線
  experiments/           ← 本框架
  data/
    test_imgs/           ← 測試影像
    gt/                  ← GT 標註（亦可與影像放同目錄）
  results/               ← 各實驗輸出（自動建立）
```

## GT 標註格式

每張影像一個 `<影像檔名去副檔名>.gt.json`：

```json
{
  "image": "court001.jpg",
  "scene": "single",
  "corners": [
    {"cid": 77, "x": 619.50, "y": 271.86, "visibility": "visible"},
    {"cid": 78, "x": 640.12, "y": 270.40, "visibility": "occluded"}
  ]
}
```

- `cid` 為 8-bit 全域角點編碼（與管線輸出一致，可用你的 GT 標註工具
  H-mode 蓋章產生）。
- `visibility` 可省略（預設 visible）；遮蔽點用 `occluded`，
  主實驗會分層報告（驗證「遮蔽交點仍可推估」的主張）。
- `scene` 為 `single` / `multi`（單／多球場），供門檻掃描分列；
  也可改用 `--multi_list` 提供多球場影像清單。

## 實驗腳本 × 論文表格對應

| 腳本 | 對應論文 | 產出 |
|------|---------|------|
| `run_main_eval.py` | 表 5.5（本方法列）、表 5.8、表 5.11、5.3 之 H 摘要 | `results/main_eval.json` |
| `run_baselines.py` | 表 5.5（Harris / Förstner 列） | `results/baselines.json` |
| `run_conf_sweep.py` | 表 5.3b | `results/conf_sweep.json` |
| `run_ablation.py` | 表 5.9 | `results/ablation.json` |
| `run_quality_discrim.py` | 表 5.6、表 5.7 | `results/quality_discrim.json` |
| `make_report.py` | 全部 → Markdown 表＋論文用圖 | `results/report/report.md`、`figures/*.png` |
| `results_viewer.py` | 互動觀察（含逐圖檢視） | — |

## 建議執行順序

```bash
# 1. 主實驗（corner_conf=0 輸出全部候選，分析端再切信心）
python -m experiments.run_main_eval --img_dir datasets/gt --gt_dir datasets/gt --weights ./weight/best.pt --yolo_conf 0.4 --save_viz --out results

# 2. 局部基準（GT 為中心之固定視窗，偏向有利基準）
python -m experiments.run_baselines --img_dir datasets/gt --gt_dir datasets/gt --win 21 --out results

# 3. 信心門檻掃描（YOLO 每張只推論一次，再依門檻過濾重跑後段）
python -m experiments.run_conf_sweep --img_dir datasets/gt --gt_dir datasets/gt --weights ./weight/best.pt --thresholds 0.8 0.6 0.4 0.25 --out results

# 4. 管線消融
python -m experiments.run_ablation --img_dir datasets/gt --gt_dir datasets/gt --weights ./weight/best.pt --out results

# 5. 證據判別性 + 組成消融（H 直接取自步驟 1 的結果）
python -m experiments.run_quality_discrim --img_dir datasets/gt --gt_dir datasets/gt --main_result results/main_eval.json --with_component_ablation --out results

# 6. 彙整報表（表格可直接貼回論文）
python -m experiments.make_report --results results --out results/report

# 7. 互動觀察
python -m experiments.results_viewer --results results --img_dir datasets/gt

# YOLOPoint 
python experiments/run_yolopoint_eval.py --csv datasets/YOLOPoint/pretrain_weight/YOLOPoint_full.csv --img_dir datasets/YOLOPoint/gt --gt_dir datasets/YOLOPoint/gt --win 21 --out results

# PnP
python experiments/run_extrinsics.py --data_dir datasets/PnP/corner_gt --weights ./weight/best.pt --conf_hi 0.8 --out results

# 純信心排序
python experiments/run_extrinsics.py --data_dir datasets/PnP/corner_gt --weights ./weight/best.pt --topk 6 10 --out results

# 分散選取
python experiments/run_extrinsics.py --data_dir datasets/PnP/corner_gt --weights ./weight/best.pt --topk 6 10 --spread --out results_spread

# 
python experiments/h_selfassessment_eval.py --main_eval results/main_eval.json --extrinsics datasets/PnP/lab_camera_pose/results/topX_spread/extrinsics_spread.json

# 內參去形變資料集標註檢查
python experiments/audit_gt.py datasets/real/gt --gt_dir datasets/real/gt --out results/audit
# 無去形變資料集標註檢查
python experiments/audit_gt.py datasets/roboflow/outputs/gt2 --gt_dir datasets/datasets/roboflow/outputs/gt2 --out results/audit_roboflow

python experiments/fix_mirror_gt.py datasets/roboflow/outputs/gt2 --images img_0308 img_0310 img_0314 img_0316
```

## 各腳本設計重點

### run_main_eval
- `--corner_conf 0` 輸出全部候選 → 同一次結果同時餵表 5.5（事後可依
  門檻過濾）與表 5.8（信心分組）。
- 每張影像保存 `H`、homography 摘要、`stage_times`、漏報 cid 清單，
  供 5.3 / 5.8 失敗案例分析與 viewer 逐圖檢視。
- `--save_viz` 另存疊圖（GT 橘十字、預測依信心上色）。

### run_baselines
- Harris：`cv2.cornerHarris` 取視窗最大響應 + `cornerSubPix` 次像素。
- Förstner：視窗內梯度收斂點閉式解 `(Σ∇I∇Iᵀ)⁻¹ Σ(∇I∇Iᵀ x)`。
- YOLOPoint 等外部方法：推論後存成 `<stem>.pred.json`（與 GT 同格式），
  以 `--extern_pred 目錄` 一併計分。

### run_conf_sweep
- 每張影像 YOLO 僅以 conf=0.05 推論一次，之後純粹過濾偵測結果重跑
  Stage 2–4，速度快很多且各門檻間完全可比。
- 「模板對應正確」以「配對角點誤差中位數 ≤ 5px」近似（錯誤對應必然
  造成大幅誤差）；門檻可用 `--correct_thresh` 調整。

### run_ablation
五個變體（`--variants` 可指定子集）：
- `full` 完整方法
- `no_steger`：`CornerGenerator.h_refine_enabled=False`（角點 = H 投影）
- `no_jacobian`：monkeypatch `compute_line_width_px` → 場地中心估一次的
  全圖固定線寬（不隨透視變化）；做法是先跑一次取得 H，再以固定線寬重跑
- `no_quality`：`corner_conf=0`，照單全收
- `no_topology`：只保留距 YOLO 偵測中心 ≤ 25px 之交點（喪失遮蔽 /
  未偵測交點的推估能力，輸出率下降）

YOLO 偵測結果跨變體共用快取，每張影像只推論一次。

### run_quality_discrim
- 負樣本三類依論文 5.5.1：背景隨機點、線上硬負樣本（沿 H 投影之
  模板線段取樣）、偏移正樣本（預設 3px）。
- `gradgeo`（本研究）直接呼叫 `vertex_quality.run_harris_steger_analysis`；
  `legacy`（原 Harris–Steger 差分）以正規化 Harris-R − 正規化脊強度
  重建證據圖，峰過濾改用脊鄰近性（≤6px），ANMS 與評分公式相同。
- `--with_component_ablation` 跑表 5.7 的四個拆項變體。
- 兩種證據均量測單點耗時（表 5.6 的耗時欄）。

### results_viewer（PyQt6）
- **摘要**：每個結果 JSON 的彙整表格，「複製為 Markdown」一鍵帶走。
- **圖表**：誤差 CDF / 直方圖、信心—誤差散佈＋分箱中位數曲線、
  tier 信心分佈、判別性 ROC、各階段耗時、消融比較。
- **逐圖檢視**：選影像目錄後疊圖顯示 GT（橘十字）vs 輸出（信心上色
  圓圈）與紅色誤差連線、未配對輸出（灰圈）；右側表格依誤差降冪排序，
  點選列會紅圈定位該角點；滾輪縮放、拖曳平移；左側清單直接顯示每張
  的誤差中位數，方便快速找失敗案例（5.8 節素材）。

## 尚未涵蓋（需另行準備）

- **表 5.3 求解方法比較**：基準（最近鄰+RANSAC）與「僅 RMSE 排序」變體
  需要進 solver 內部加開關；main_eval 已輸出三路徑 `solver_method` 統計
  可先填觸發比例。
- **表 5.10 外參估計**：需要多視角已知內參資料與相機位置真值，
  屬獨立 PnP 實驗；main_eval 輸出的角點 + H 可直接作為其輸入。
- **表 5.3 YOLO P/R/mAP**：建議直接用 ultralytics `model.val()` 對
  真實測試集 YOLO 格式標註計算，近／遠場分層可依 bbox 尺寸切。

---

## GT 標註工具 `gt_annotator.py`

```bash
python experiments/gt_annotator.py [影像資料夾]
```

**標註流程（H 蓋章）**
1. 開啟資料夾後選影像（左側清單，`✓` 表示已有 GT）。
2. 切到「對應點模式」(C)：在右側球場小地圖點選一個 junction，再到影像點對應位置；重複 ≥4 組（建議取分散的外圍交點，≥5 組會自動啟用 RANSAC）。
3. 按「解 H 並蓋章」：自動投影全部 80 個物理角點（cid = corner_code，與管線輸出一致；junction ± 0.02 m 線寬偏移、依 NODE_TABLE 有效角遮罩），框外點自動略過；同時疊上格線供目視確認，狀態列顯示對應點殘差。
4. 切回「編輯模式」(E) 微調：滾輪縮放、中鍵/Ctrl+左鍵平移、拖曳點、方向鍵 0.25px 微調（Shift=1px）、`V` 切換 visible/occluded、`Delete` 刪除。手動調過的點（白圈標記）重蓋章時不會被覆蓋。
5. 場景類型 single/multi 用右側單選鈕設定；`Ctrl+S` 存檔，或開啟「切換影像時自動存檔」。

**輸出** `<stem>.gt.json` 直接放影像旁，`run_main_eval.py` 等腳本即可讀取；檔內另存 `annot`（對應點與 H）供下次續標。`node`/`lcid`/`manual` 為輔助欄位，`load_gt` 會自動忽略。
