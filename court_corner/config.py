"""
config.py — 全管線參數預設值
================================================================
原始程式碼將參數放在外部 `config` 模組（未隨附），此處依各模組的
使用方式（clip 範圍、threshold_pct 換算、Steger/Harris 慣用值）回推出
一組合理預設，並集中管理供四個階段共用。使用者可於建立各 Stage 物件時
以參數覆寫，或直接修改本檔。

命名沿用原始碼：
  S2_*  拓樸求解（Stage 2，本專案新增的點式單應求解器）
  S4_STEGER_*  Steger 角點搜尋（Stage 3）
  VF_*  Vertex finder / H-refine（Stage 3）
  VQ_*  Vertex quality（Stage 4）
"""

import cv2

# ──────────────────────────────────────────────────────────────
# 共用
# ──────────────────────────────────────────────────────────────
LINE_WIDTH_M = 0.04            # 羽球場白線寬 40mm（與 court_model 一致）
DIST_L2 = cv2.DIST_L2

# ──────────────────────────────────────────────────────────────
# Stage 1 — 交點偵測（YOLO）
# ──────────────────────────────────────────────────────────────
YOLO_DEFAULT_CONF = 0.25       # 預設 yolo_conf
YOLO_IOU = 0.45                # NMS IoU
YOLO_MAX_DET = 300
# class id → junction type；沿用 court_model 的 TYPE 編碼 0=L,1=T,2=X。
# 若模型 class 名稱本身是 'L'/'T'/'X' 或含關鍵字，偵測器會自動辨識，
# 此表僅作為「class 名稱為純數字索引」時的後援。
YOLO_CLASSID_TO_TYPE = {0: "L", 1: "T", 2: "X"}

# ──────────────────────────────────────────────────────────────
# Stage 2 — 拓樸求解（點式單應；不依賴球場線抽取）
# ──────────────────────────────────────────────────────────────
S2_GRID_TWIST_MAX_RATIO = 120.0   # 投影格面積比上限（防折疊/扭轉）
S2_NN_INLIER_RATIO = 0.02         # NN 指派 inlier 門檻 = max(6, ratio*span)
S2_NN_INLIER_MIN_PX = 6.0
S2_GUIDED_REFIT_ITERS = 5         # guided refit 迭代次數
S2_RANSAC_FALLBACK_ITERS = 4000   # 後援取樣 RANSAC 次數
S2_MIN_INLIERS = 6                # 接受 H 的最少 inlier 數
S2_STEGER_REFINE_H = True         # 是否對 H 做 Jacobian 導引 Steger 次像素精修
S2_STEGER_REFINE_ITERS = 2

# Stage 2 — 線為主求解的多階段重試（line_consistency / 白線支持過低時逐步升級）
#   Attempt 1 strict  ：目前預設抽線
#   Attempt 2 relaxed ：放寬 Steger 閾值 / 降門檻 / 加 RANSAC 迭代 / 放寬合併
#   Attempt 3 masked  ：YOLO 交點凸包外擴遮罩內抽線（去除柱子/人/觀眾等干擾結構）
#   Attempt 4 rerank  ：跨所有嘗試保留 white-line support 前 K 個候選，重排挑最佳
#   Attempt 5 fail    ：只有當最佳候選 線/型/白線支持 全部低於底線才算徹底失敗
S2_RETRY_ENABLED = True
S2_RETRY_LC_OK = 0.5               # line_consistency 達此值且白線支持足 → 視為穩，提前結束重試
S2_RELAXED_LINE_PARAMS = {
    "threshold_pct": 0.08, "min_inliers": 20, "min_span": 22.0,
    "merge_ang_deg": 6.0, "merge_rho": 8.0, "max_iter": 120, "line_thr": 2.5,
}
S2_MASK_DILATE_RATIO = 1.0         # 遮罩外擴半徑 = ratio × 中位交點框邊長
S2_TOPK_CANDIDATES = 8             # 重排保留的候選數
S2_FAIL_SUPPORT_FLOOR = 0.20       # 徹底失敗底線：白線支持
S2_FAIL_LC_FLOOR = 0.30            # 徹底失敗底線：line_consistency
S2_FAIL_TC_FLOOR = 0.50            # 徹底失敗底線：type_consistency

# ──────────────────────────────────────────────────────────────
# Stage 3 — 角點生成：Steger 中線偏移搜尋
# ──────────────────────────────────────────────────────────────
S4_STEGER_SIGMA = 1.2              # 與原始 court_homography_tool 用法一致
S4_STEGER_THRESHOLD_PCT = 15.0     # /100 = 0.15
S4_STEGER_BRIGHT_LINES = True      # 球場白線為亮線
S4_STEGER_MIN_RIDGE_POINTS = 10
S4_STEGER_RANSAC_ITERATIONS = 100
S4_STEGER_RANSAC_THRESHOLD = 2.0
S4_STEGER_RANSAC_MIN_INLIERS = 5
S4_STEGER_DIRECTION_TOL_DEG = 22.0
S4_STEGER_CENTER_DIST_RATIO = 2.0
S4_STEGER_ROI_HALF_WIDTH_RATIO = 4.0   # R = ratio * max(line_width_px)

# ──────────────────────────────────────────────────────────────
# Stage 3 — 角點生成：H 投影精修（h_refine）
# ──────────────────────────────────────────────────────────────
VF_H_REFINE_ENABLED = True
VF_H_REFINE_OUTLIER_PX = 6.0       # Steger 與 H 投影距離 > 此值 → 視為 outlier，改用 H
VF_H_REFINE_BLEND_WEIGHT = 0.5     # 融合權重：final = w*v_h + (1-w)*v_s

# Stage 3 — EdgeBasedVertexFinder 用（topology 約束所需）
VF_RANSAC_ITERATIONS = 100
VF_RANSAC_THRESHOLD = 2.0

# ──────────────────────────────────────────────────────────────
# Stage 4 — 品質評估（Harris × Steger diff map）
# ──────────────────────────────────────────────────────────────
VQ_HARRIS_K = 0.04
VQ_HARRIS_SIGMA = 1.5
VQ_HARRIS_THRESHOLD_PCT = 1
VQ_STEGER_SIGMA = 1.5
VQ_STEGER_THRESHOLD_PCT = 15
VQ_STEGER_DILATION_RADIUS = 2
VQ_STEGER_DARK_RIDGES = False
VQ_TOP_K = 8                        # 每個 ROI 取的 peak 數
VQ_PEAK_RADIUS_PX = 5.0             # dist_score 的 tau
VQ_INSET = 0
VQ_ANMS_C = 0.9
VQ_ANMS_CANDIDATE_POOL = 200
VQ_ANMS_LOOSE_NMS_RADIUS = 3
VQ_PROX_ENABLED = True
VQ_PROX_MIN_AREA = 10
VQ_PROX_MAX_DIST = 6.0
VQ_PROX_CLOSING_RADIUS = 3
VQ_ROI_HALF_WIDTH_RATIO = 4.0      # quality ROI 半徑 = ratio * line_width_px
VQ_ROI_MIN_HALF = 12

# Stage 4 — composite 分數權重（幾何證據 vs 影像證據）
VQ_DIST_WEIGHT = 0.5               # exp(-d/tau)：靠近 Harris/Steger peak
VQ_HEATMAP_WEIGHT = 0.5            # diff map 在 vertex 處的值

# 最終輸出
CORNER_CONF_DEFAULT = 0.6          # 預設 corner_conf（最終信心門檻）

# 第四階段信心融合：conf = 幾何證據 × 影像支持
#   幾何證據 g = topo_quality_weight × exp(-reproj_err_m / VQ_GEOM_TAU_M)
#   影像支持   = max(VertexQualityScorer composite, VQ_IMG_LINE_WEIGHT × 白線亮度支持)
# 白線亮度支持作為遮蔽偵測：角點鄰域確實有亮線→高；被遮蔽/出界→低。
# 在乾淨合成白線與真實影像皆有效（composite 需紋理，亮度支持不需要）。
VQ_GEOM_TAU_M = 0.12               # 3× 線寬；重投影誤差的幾何容忍尺度
VQ_IMG_LINE_WEIGHT = 0.9           # （保留；新版 final_conf 直接取 max(vertex, line_support)）
VQ_TOPO_QUALITY_WEIGHT = {"high": 1.0, "medium": 0.85, "low": 0.7}
VQ_LINE_SUPPORT_RADIUS_RATIO = 1.5 # 亮度取樣半徑 = ratio × 線寬（下限 4px）
VQ_LINE_SUPPORT_MIN_RADIUS = 4

# 第四階段三層輸出（strong / weak / hidden）保底規則
#   strong : final_conf >= corner_conf（幾何 + 影像都夠強，正式輸出）
#   weak   : 未達 strong，但 H 信心 >= medium 且白線亮度支持 >= 此門檻
#            → 即使 Harris/Steger 角點響應弱也保留（標 low confidence）
#   hidden : 其餘（遮蔽 / 白線支持太低 / H 信心 low）→ 不列入 corners，只進 corner_candidates
VQ_WEAK_LINE_SUPPORT_MIN = 0.45    # weak 保底所需的最低白線亮度支持
VQ_WEAK_GEOM_QUALITIES = ("high", "medium")   # weak 保底所需的最低 H 信心
