# 羽球場角點定位工具（四階段管線）

由單張影像偵測羽球場交點、估計單應矩陣、推導白線外緣角點，並以幾何與影像
證據評估後輸出最終角點集合 `(cid, x, y, conf)`。以物件導向實作，可作為指令
工具或函式庫使用。

## 安裝需求

- Python 3.8+
- `numpy`、`opencv-python`、`scipy`
- `ultralytics`（第一階段 YOLO 偵測需要；採延遲載入）
- `PyQt6`（僅 GUI 需要）

```bash
pip install -r requirements.txt
```

線為主求 H 的演算法（原 `court_homography_tool.py` / `folder_yolo_tool.py` 的非 GUI
部分）已**直接移植內嵌**於 `court_corner/homography/`，本工具自足、不需另外提供
那兩支檔案。

## 使用方式

### 函式庫用法
可參考專案目錄 example 
```python
from court_corner.pipeline import CourtCornerPipeline

pipe = CourtCornerPipeline("best.pt", yolo_conf=0.25, corner_conf=0.6)
result = pipe.run("court.jpg")
for cid, x, y, conf in result.corners_as_tuples():
    print(cid, x, y, conf)
```
### 指令用法
```bash
python detect_corners.py --img_path court.jpg
python detect_corners.py --img_path court.jpg --yolo.pt weights/best.pt \
                         --yolo_conf 0.3 --corner_conf 0.6 --viz out.png
```

#### 指令參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--yolo.pt` | YOLO 權重路徑 | 與本程式同目錄下的 `best.pt` |
| `--img_path` | 單張影像路徑（必填） | — |
| `--yolo_conf` | YOLO 偵測信心門檻 | `0.25` |
| `--corner_conf` | 角點輸出信心門檻 | `0.6` |
| `--min_line_support` | H 白線支持度門檻（投影格線須落在影像白線上）；低於此標為不可靠 | `0.45` |
| `--out` | 輸出 JSON 路徑 | `<影像名>_corners.json` |
| `--viz` | 視覺化疊圖輸出路徑（可選） | 不輸出 |
| `--dark_lines` | 球場線為暗色時加此旗標（少見） | 關閉 |
| `--quiet` | 關閉逐階段訊息 | 關閉 |

#### 輸出

主控台與 JSON 皆輸出最終角點 `(cid, x, y, conf)`，其中 `cid` 為 8-bit 全域
角點編碼（corner_id）。JSON 另含單應矩陣 `H`、拓樸求解摘要與各角點的
`junction_idx`、`corner_type`、`source`、`reproj_err_m` 等診斷欄位。

```json
{
  "status": "ok",
  "message": "完成：輸出 34 個角點 （strong 32 + weak 2，hidden 0；候選 34，門檻 conf≥0.6）",
  "elapsed_s": 1.795,
  "stage_times": { // 每階段耗時
    "detect": 1.424,
    "solve_H": 0.341,
    "corners": 0.008,
    "quality": 0.021
  },
  "H": [ //找到的 H 矩陣
    [
      14.344371639189117,
      -41.47582440101188,
      585.0004416401545
    ],
    [
      14.441666257340517,
      14.263656508534114,
      -71.55428817553278
    ],
    [
      -0.03982989436837715,
      -0.03268681975314165,
      1.0
    ]
  ],
  "n_detections": 12, // 偵測到的交點數
  "homography": { // H 矩陣拓樸評估
    "method": "line",
    "confidence": "high",
    "line_consistency": 1.0,
    "type_consistency": 1.0,
    "line_support": 0.829,
    "line_support_ok": true,
    "solver_method": "link-enum+steger-refine+hungarian-refit",
    "n_steger_refined": 10,
    "n_courts": 1,
    "n_junctions": 12
  },
  "report": { // 針對該圖的所有統計結果
    "n_candidates": 34,
    "n_strong": 32,
    "n_weak": 2,
    "n_hidden": 0,
    "n_passed": 34,
    "corner_conf": 0.6,
    "geom_quality": "high",
    "mean_conf_passed": 0.792,
    "reproj_err_m": {
      "mean": 0.0118,
      "max": 0.0375
    }
  },
  "corners": [ // 角點資訊
    {
      "cid": 77,
      "x": 619.505,
      "y": 271.861,
      "conf": 0.9501,
      "tier": "strong",
      "junction_idx": 13,
      "corner_type": "-+",
      "source": "fused",
      "reproj_err_m": 0.0027
    },...]}
```


## 圖形介面（GUI）

提供 PyQt6 圖形介面 `court_corner_gui.py`，可載入權重與影像（或整個資料夾）、
執行管線並把角點畫在影像上。

```bash
python court_corner_gui.py
```

功能：

- **載入權重 (.pt)**、**載入影像** 或 **載入資料夾**（資料夾會列出所有影像，可在
  左側清單切換瀏覽）。
- 可選「暗線球場」；可調 `yolo_conf` 與 `corner_conf`；勾選「選取後自動執行」則
  切換影像時自動跑。
- 影像上繪出角點（依信心值上色：綠≥0.8、黃≥0.65、橙其餘）、cid 標籤、可選的
  H 格線與偵測交點；顯示選項即時重繪，不需重跑。
- 右側角點表格 `(cid, x, y, conf, type, source)`；點選表格列會在影像上以紅圈標出
  該角點。滑鼠滾輪縮放、拖曳平移、「符合視窗」一鍵還原。
- **批次處理資料夾…**：對整個資料夾逐張執行，將標註圖（`*_annotated.png`）與
  `*_corners.json` 存到指定輸出資料夾。
- **儲存標註圖** / **儲存 JSON**：輸出目前影像的結果。

模型只在第一次執行時載入一次，之後切換影像或調整參數都沿用同一模型；管線在
背景執行緒執行，介面不會卡住。

## 四階段架構

```
影像 ─► Stage 1 交點偵測 ─► Stage 2 拓樸求解 ─► Stage 3 角點生成 ─► Stage 4 品質評估 ─► (cid,x,y,conf)
        JunctionDetector     TopologySolver       CornerGenerator      QualityEvaluator
```

- **第一階段　交點偵測（`stages/detection.py`）**
  以 YOLO 偵測交點，取 bbox 中心為位置、由類別判定型別（L／T／X）。自動
  判讀模型 class 名稱關鍵字（`x`/`cross`→X、`t`→T、`l`/`corner`→L），純數字
  名稱則退回索引對應表。

- **第二階段　單應求解（線為主，`stages/topology_line.py` + `homography/`）**
  以**內嵌移植**的線為主求解器求 H——全域 Steger 抽線 → cross-ratio 線標號 →
  PROSAC → Steger 次像素精修，並以 line/type 一致性挑解。演算法移植自原
  `court_homography_tool.py`（求解）與 `folder_yolo_tool.py`（抽線），去除其 GUI 後
  放在 `court_corner/homography/solver.py` 與 `court_lines.py`。`topology_line.py`
  為橋接：把 YOLO 偵測整理成 `Annotation` 清單、呼叫 `solver.solve_image()`，再把
  投影點（`row*5+col` 即 junction_idx）整理成 Stage 3 需要的交點清單。其場地範本與
  `shared.court_model` 完全相同，故輸出的 H 與 Stage 3 / 4 完全相容。

- **第三階段　角點生成（`stages/corners.py`）**
  以單應矩陣將每個交點依型別（X→4 角、T→2、L→2）投影出白線外緣角點作為
  完整幾何候選，再以 Steger 中線偏移法於局部 ROI 萃取白線、做次像素精修，並
  透過 `corner_code(cid)` 配對融合。Steger 確認者用融合位置（精度較高），未確認
  者保留 H 投影位置，是否輸出留待第四階段裁決。

- **第四階段　品質評估與輸出（`stages/quality.py`）**
  同時以幾何與影像證據計算每個角點信心：

  ```
  conf = g × image_support
  g            = topo_quality_weight × exp(−reproj_err_m / τ_g)        # 幾何證據
  image_support = max(Harris/Steger composite, w · 白線亮度支持)        # 影像證據
  ```

  其中幾何證據反映 H 信心與該角點重投影一致性；影像證據取「Harris/Steger
  角點分數」與「白線亮度支持」之較大者——後者作為遮蔽偵測，即使角點壓在
  乾淨白線上（無明顯角點紋理）亦可確認其影像存在性。最終以 `corner_conf`
  門檻過濾。


## 套件結構

```
court_corner_tool/
  README.md
  requirements.txt             相依套件
  detect_corners.py            指令入口（CLI）
  court_corner_gui.py          圖形介面（PyQt6）
  court_corner/
    __init__.py
    config.py                  各階段參數（逆向推得之預設，可調）
    pipeline.py                四階段編排（CourtCornerPipeline，線為主求 H）
    stages/                    四階段管線
      __init__.py
      detection.py             第一階段：JunctionDetector
      topology_line.py         第二階段：橋接 homography 求解器（線為主求 H）
      line_support.py          H 白線支持度驗證（投影格線是否落在影像白線上）
      corners.py               第三階段：CornerGenerator（H 投影 + Steger 精修）
      quality.py               第四階段：QualityEvaluator（幾何 + 影像證據）
    homography/                線為主求解器（移植自原工具，去除 GUI）
      __init__.py
      court_lines.py           全域 Steger 抽線 / 交點掛線（移植自 folder_yolo_tool）
      solver.py                cross-ratio / PROSAC / Steger 精修 + solve_image
                               （移植自 court_homography_tool）
    vertex/                    角點精修與品質支援模組
      __init__.py
      steger_vertex_finder.py  Steger 中線偏移角點萃取
      h_refine.py              H 投影角點與 cid 融合精修
      vertex_finder.py         邊緣式角點輔助（拓樸約束）
      vertex_quality.py        Harris/Steger 角點品質評分
      reprojection.py          角點重投影誤差
    shared/                    場地模型與幾何基元
      __init__.py
      court_model.py           場地模板點位、型別、角點編碼
      homography.py            單應幾何工具
      steger.py                Steger 脊線基元（重新實作補回）
```


