# 羽球場角點定位工具（四階段管線）

由單張影像偵測羽球場交點、估計單應矩陣、推導白線外緣角點，並以幾何與影像
證據評估後輸出最終角點集合 `(cid, x, y, conf)`。以物件導向實作，可作為指令
工具或函式庫使用。

## 安裝需求

- Python 3.8+
- `numpy`、`opencv-python`、`scipy`
- `ultralytics`（僅第一階段 YOLO 偵測需要；採延遲載入，未用到偵測時不需安裝）

```bash
pip install numpy opencv-python scipy ultralytics
```

## 使用方式

```bash
python detect_corners.py --img_path court.jpg
python detect_corners.py --img_path court.jpg --yolo.pt weights/best.pt \
                         --yolo_conf 0.3 --corner_conf 0.6 --viz out.png
```

### 指令參數

| 參數 | 說明 | 預設 |
|------|------|------|
| `--yolo.pt` | YOLO 權重路徑 | 與本程式同目錄下的 `best.pt` |
| `--img_path` | 單張影像路徑（必填） | — |
| `--yolo_conf` | YOLO 偵測信心門檻 | `0.25` |
| `--corner_conf` | 角點輸出信心門檻 | `0.6` |
| `--out` | 輸出 JSON 路徑 | `<影像名>_corners.json` |
| `--viz` | 視覺化疊圖輸出路徑（可選） | 不輸出 |
| `--dark_lines` | 球場線為暗色時加此旗標（少見） | 關閉 |
| `--quiet` | 關閉逐階段訊息 | 關閉 |

### 輸出

主控台與 JSON 皆輸出最終角點 `(cid, x, y, conf)`，其中 `cid` 為 8-bit 全域
角點編碼（corner_id）。JSON 另含單應矩陣 `H`、拓樸求解摘要與各角點的
`junction_idx`、`corner_type`、`source`、`reproj_err_m` 等診斷欄位。

## 圖形介面（GUI）

提供 PyQt6 圖形介面 `court_corner_gui.py`，可載入權重與影像（或整個資料夾）、
執行管線並把角點畫在影像上。

```bash
python court_corner_gui.py
```

功能：

- **載入權重 (.pt)**、**載入影像** 或 **載入資料夾**（資料夾會列出所有影像，可在
  左側清單切換瀏覽）。
- 可調 `yolo_conf` 與 `corner_conf`；勾選「選取後自動執行」則切換影像時自動跑。
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

- **第二階段　拓樸求解（`stages/topology.py`）**
  以「點對應」估計單應矩陣（template → image）。先取四個極值點對應模板四角，
  嘗試二面體群（D2）的各種排列、以 DLT 求解並驗證；失敗則退回型別約束的
  取樣 RANSAC。內含格網翻轉檢查、型別一致性、型別加權最近鄰指派、引導重擬合
  與 Steger 次像素 H 精修。輸出 `H`、偵測↔模板對應與信心等級。

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

## 設計說明

- **點對應式第二階段**：附件原始碼的拓樸求解依賴未隨附的全域白線抽取模組
  （`folder_yolo_tool` / `s1_detection.steger_center` 等）。本工具改以偵測交點與
  模板的「點對應」直接估計 H，不需事先抽取整場白線，較為自足且穩定。

- **重新實作 Steger 脊線基元（`shared/steger.py`）**：補回缺漏的
  `_steger_ridge_points_simple` 等函式。修正了 Hessian 特徵向量在軸對齊脊線
  退化的問題（同時計算兩種特徵向量公式並取範數較大者），亮線極性以合成資料
  驗證正確。

- **信心融合**：原始 `VertexQualityScorer` 的 composite 對「位於脊線上的角點」
  結構性偏低（差異圖 Harris−Steger 在線上相消），故第四階段改以幾何證據與
  影像支持相乘，兼顧幾何一致性與遮蔽偵測，輸出更合理之 `[0,1]` 信心值。

- **單張影像的固有方向歧義**：羽球場版面在二面體群 D2（左右翻轉、上下翻轉、
  180° 旋轉）下型別不變，故型別一致的有效 H 有四個。本工具沿用原始碼的方向
  慣例（模板 +x 對應影像向右、+y 對應影像向下）的 `orient` 分數來穩定挑選，
  與原始實作一致；此歧義為單張影像本質使然。

- **參數來源**：因原始碼缺少設定檔，`config.py` 內各參數為依演算法逆向推得之
  合理預設，皆已逐項註記，可視實際資料調整。

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
    pipeline.py                四階段編排（CourtCornerPipeline）
    stages/                    四階段管線
      __init__.py
      detection.py             第一階段：JunctionDetector
      topology.py              第二階段：TopologySolver（點對應 + H + 拓樸對應）
      corners.py               第三階段：CornerGenerator（H 投影 + Steger 精修）
      quality.py               第四階段：QualityEvaluator（幾何 + 影像證據）
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

## 函式庫用法

```python
from court_corner.pipeline import CourtCornerPipeline

pipe = CourtCornerPipeline("best.pt", yolo_conf=0.25, corner_conf=0.6)
result = pipe.run("court.jpg")
for cid, x, y, conf in result.corners_as_tuples():
    print(cid, x, y, conf)
```
