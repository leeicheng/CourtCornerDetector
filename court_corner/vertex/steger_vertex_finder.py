"""
Steger Vertex Finder
======================
方向 B 實作：以 Steger sub-pixel ridge 抽取「中心線」，再沿法向偏移
±half_line_width 形成兩條虛擬邊界線。兩個方向各做一次 → 4 條邊界線
→ 4 種組合的交點即為 4 個外緣角點，與舊 EdgeBasedVertexFinder /
MaskLineFinder 的輸出語意一致。

優點 vs 舊兩條 path：
  - mask path 受地板紋理 / 反光大幅干擾；Steger 對大型亮塊穩定
  - edge path 雙邊掃描需要 line_width 估得很準才不會錯位；
    Steger 抽中線只需要把線寬乘到輸出階段
  - 中線對應到「球場理論線」，跟 reproj GT 的物理意義對齊度高，
    最終 H_refine 比較不會把 Steger 結果當 outlier 丟掉

公開 API：
  StegerVertexFinder.find_vertices_for_junction(...) → 與 EdgeBasedVertexFinder
  完全相同的回傳 dict 形狀，drop-in 取代用。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..shared.court_model import (
    TEMPLATE_POINTS,
    TEMPLATE_TYPES,
    LINE_WIDTH_M,
    compute_junction_principal_directions,
    corner_type_to_corner_code,
)
from ..shared.homography import HomographyUtils
from ..shared.steger import (
    _steger_ridge_points_simple,
    _build_line_candidates_by_angle_offset,
    _score_line_pair,
    _select_line_pair_by_voting,
    _line_intersection_simple,
    _axis_angle_deg,
)

from ..config import (
    S4_STEGER_SIGMA,
    S4_STEGER_THRESHOLD_PCT,
    S4_STEGER_BRIGHT_LINES,
    S4_STEGER_MIN_RIDGE_POINTS,
    S4_STEGER_RANSAC_ITERATIONS,
    S4_STEGER_RANSAC_THRESHOLD,
    S4_STEGER_RANSAC_MIN_INLIERS,
    S4_STEGER_DIRECTION_TOL_DEG,
    S4_STEGER_CENTER_DIST_RATIO,
    S4_STEGER_ROI_HALF_WIDTH_RATIO,
)


# ---------------------------------------------------------------------------
# Direction-clustering line fit (preferred over RANSAC when ridge tangents
# are available — avoids随机性 and 不需要 inlier/outlier 假設)
# ---------------------------------------------------------------------------

def _fit_lines_by_direction_clustering(
    pts: np.ndarray,
    tangents: np.ndarray,
    strengths: np.ndarray,
    target_dir_A: np.ndarray,
    target_dir_B: np.ndarray,
    direction_tol_deg: float = 20.0,
    expected_through_pt: Optional[np.ndarray] = None,
    max_dist_to_pt: Optional[float] = None,
    min_pts_per_line: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    用每個 ridge point 自帶的切線方向 (來自 Steger Hessian eigenvector)，
    把點分成 A 線群、B 線群，再對各群做 TLS fit。

    這比 RANSAC 好的地方：
      * 沒有採樣隨機性 — 每個點明確歸屬，結果可重現
      * 不需要「inlier vs outlier」假設 — 你的場地圖點都是真 ridge points,
        只是「分屬不同條線」，這是分群問題不是 outlier rejection 問題
      * 一個 pass 就好，O(N) 而非 O(N · iters)

    分群邏輯：
      對每個 point i：
        cosA = |tangent_i · target_A|
        cosB = |tangent_i · target_B|
        如果 cosA > cos_tol 且 cosA > cosB: → 屬於 A 群
        如果 cosB > cos_tol 且 cosB > cosA: → 屬於 B 群
        否則: → 雜訊（兩個方向都不像，可能在 junction 中心或外角附近）

    後續對每群做：
      1. 距離過濾：點到「expected center」距離 < max_dist_to_pt（沿切線法向）
      2. 用 strength 加權 TLS fit
      3. 最終驗證：fit 出來的線方向跟 target 夾角 < tol

    Args:
        pts                : (N, 2) sub-pixel ridge points
        tangents           : (N, 2) unit tangent vectors per point
        strengths          : (N,)  ridge strength per point (用作 TLS 權重)
        target_dir_A       : (2,)  H-projected A 方向 unit vector
        target_dir_B       : (2,)  H-projected B 方向 unit vector
        direction_tol_deg  : 切線方向跟 target 的最大夾角（度）
        expected_through_pt: 線「應該」通過的點（junction center）；None 不約束
        max_dist_to_pt     : 線到 expected center 的距離上限（px）
        min_pts_per_line   : 每群至少需這麼多點才 fit 線

    Returns:
        (lineA, lineB, idxA, idxB)
        其中 lineX = (vx, vy, x0, y0) format（與 cv2.fitLine 同），失敗為 None
        idxA / idxB = 該群點的 index list（給 debug visualization 用）
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    tans = np.asarray(tangents, dtype=np.float64).reshape(-1, 2)
    strs = np.asarray(strengths, dtype=np.float64).reshape(-1)
    n = len(pts)
    if n == 0 or len(tans) != n or len(strs) != n:
        return None, None, None, None

    # Normalize target directions
    tA = np.asarray(target_dir_A, dtype=np.float64).reshape(2)
    tB = np.asarray(target_dir_B, dtype=np.float64).reshape(2)
    nA = float(np.linalg.norm(tA))
    nB = float(np.linalg.norm(tB))
    if nA < 1e-9 or nB < 1e-9:
        return None, None, None, None
    tA = tA / nA
    tB = tB / nB

    cos_tol = math.cos(math.radians(float(direction_tol_deg)))

    # Per-point: |tangent · tA| 跟 |tangent · tB|（切線方向跟 target 的夾角絕對值）
    # 取 abs 是因為 ±tangent 都是同一條線
    cosA = np.abs(tans @ tA)   # (N,)
    cosB = np.abs(tans @ tB)   # (N,)

    # 分群
    is_A = (cosA > cos_tol) & (cosA >= cosB)
    is_B = (cosB > cos_tol) & (cosB > cosA)

    # 中心通過約束（如果給了）— 點到 center 沿 normal_target 的距離 < max_dist
    if expected_through_pt is not None and max_dist_to_pt is not None:
        center = np.asarray(expected_through_pt, dtype=np.float64).reshape(2)
        # 對 A 群：normal of A = perp(tA) → A 線應通過 center 表示「點離 center 沿 nA 方向距離小」
        # 但更簡單：「點本身離 center 沿 nA 方向距離小」即可
        nA_perp = np.array([-tA[1], tA[0]])
        nB_perp = np.array([-tB[1], tB[0]])
        max_d = float(max_dist_to_pt)
        # 距離到 (center, A-line direction)：點到 center 的位移在 nA_perp 上的投影
        offsetsA = np.abs((pts - center) @ nA_perp)
        offsetsB = np.abs((pts - center) @ nB_perp)
        # A 群點必須沿 nA_perp 距離 center < max_d
        is_A = is_A & (offsetsA < max_d)
        is_B = is_B & (offsetsB < max_d)

    idxA = np.where(is_A)[0]
    idxB = np.where(is_B)[0]

    lineA = _tls_line_weighted(pts[idxA], strs[idxA]) if len(idxA) >= min_pts_per_line else None
    lineB = _tls_line_weighted(pts[idxB], strs[idxB]) if len(idxB) >= min_pts_per_line else None

    # 最終再驗一次方向（TLS 結果可能略偏 target）
    if lineA is not None:
        v = np.array([lineA[0], lineA[1]])
        if abs(float(v @ tA)) < cos_tol:
            lineA = None
    if lineB is not None:
        v = np.array([lineB[0], lineB[1]])
        if abs(float(v @ tB)) < cos_tol:
            lineB = None

    return lineA, lineB, idxA, idxB


def _tls_line_weighted(pts: np.ndarray, weights: Optional[np.ndarray] = None
                      ) -> Optional[np.ndarray]:
    """
    Total Least Squares line fit via weighted SVD.
    回傳 (vx, vy, x0, y0) format 跟 cv2.fitLine 相容。
    失敗回 None。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return None
    if weights is None:
        weights = np.ones(len(pts), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    w_total = weights.sum()
    if w_total < 1e-9:
        return None
    # weighted centroid
    centroid = (pts * weights[:, None]).sum(axis=0) / w_total
    centered = pts - centroid
    # weight 應用到 row（np.sqrt 因為 covariance 是 X^T X）
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    weighted = centered * sqrt_w
    try:
        _U, _S, Vt = np.linalg.svd(weighted, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = Vt[0]   # 主方向（最大 singular value 對應）
    return np.array([direction[0], direction[1], centroid[0], centroid[1]],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Deterministic line-pair fit (採用 S1 同款 doubled-angle 2-means + ρ-clustering
# + Huber-IRLS TLS + multi-criteria scoring)
# ---------------------------------------------------------------------------

def _fit_line_pair_deterministic(
    pts: np.ndarray,
    tangents: np.ndarray,
    strengths: np.ndarray,
    junction_type: str,
    center_prior: Optional[np.ndarray] = None,
    target_dir_A: Optional[np.ndarray] = None,
    target_dir_B: Optional[np.ndarray] = None,
    direction_tol_deg: float = 25.0,
    min_pts_per_line: int = 5,
):
    """
    跟 S1 _extract_two_axes_simple 同款的 deterministic line-pair fit，但接
    S4 的概念：
      - junction_type 從 template (X / T / L) 拿（S4 永遠知道，因為已過 S2 matching）
      - center_prior 從 H projection 拿（S4 沒有 YOLO bbox 中心，但有 H 的 projection）
      - target_dir_A/B 是 H-projected 主方向（optional 用來篩候選線；
        若 H 估計很差可不傳，純資料驅動）

    Returns:
        (lineA, lineB, idxA, idxB)
        lineX = (vx, vy, x0, y0)。失敗任一條為 None。
        idxX = 該線 inlier indices（ridge_pts 的 index list）。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < max(2, min_pts_per_line):
        return None, None, None, None

    # Step 1: 用 S1 同款的候選線生成器
    candidates = _build_line_candidates_by_angle_offset(
        pts,
        tangents=np.asarray(tangents, dtype=np.float64).reshape(-1, 2),
        strengths=np.asarray(strengths, dtype=np.float64).reshape(-1),
        min_points=min_pts_per_line,
    )
    if len(candidates) < 2:
        return None, None, None, None

    # Step 2: 若給了 H prior，用方向 tolerance 過濾候選線（保留靠近 tA 或 tB 的）
    if target_dir_A is not None and target_dir_B is not None:
        tA = np.asarray(target_dir_A, dtype=np.float64).reshape(2)
        tB = np.asarray(target_dir_B, dtype=np.float64).reshape(2)
        nA = float(np.linalg.norm(tA)); nB = float(np.linalg.norm(tB))
        if nA > 1e-9 and nB > 1e-9:
            tA = tA / nA; tB = tB / nB
            cos_tol = math.cos(math.radians(float(direction_tol_deg)))
            kept = []
            for c in candidates:
                v = np.asarray(c['line'][:2], dtype=np.float64)
                v = v / max(1e-9, float(np.linalg.norm(v)))
                # 候選線方向必須跟 tA 或 tB 至少一個夠近
                if abs(float(v @ tA)) >= cos_tol or abs(float(v @ tB)) >= cos_tol:
                    kept.append(c)
            candidates = kept
    if len(candidates) < 2:
        return None, None, None, None

    # Step 3: O(N²) enumerate pair，但不再用單一 best score 決策。
    # 改成 S1 helper 的 weighted voting：每個有效 line-pair 都投票給兩條線，
    # 最後選「pair 本身可靠 + 兩條線都有共識支持」的組合。
    cp = (None if center_prior is None
          else (float(center_prior[0]), float(center_prior[1])))
    line1, line2, idx1, idx2, _vote_meta = _select_line_pair_by_voting(
        candidates,
        pts,
        bbox_center=cp,
        junction_type=junction_type,
        min_angle_deg=35.0,
    )

    if line1 is None or line2 is None:
        return None, None, None, None

    # Step 4: 決定哪條是 A、哪條是 B（用 H prior 對應；無 prior 則隨意）
    if target_dir_A is not None:
        tA = np.asarray(target_dir_A, dtype=np.float64).reshape(2)
        nA = float(np.linalg.norm(tA))
        if nA > 1e-9:
            tA = tA / nA
            v1 = np.asarray(line1[:2], dtype=np.float64)
            v2 = np.asarray(line2[:2], dtype=np.float64)
            v1 = v1 / max(1e-9, float(np.linalg.norm(v1)))
            v2 = v2 / max(1e-9, float(np.linalg.norm(v2)))
            # |v · tA| 大者為 A 線
            if abs(float(v2 @ tA)) > abs(float(v1 @ tA)):
                line1, line2 = line2, line1
                idx1, idx2 = idx2, idx1

    return line1, line2, idx1, idx2


# ---------------------------------------------------------------------------
# Direction-prior RANSAC line fit (legacy, kept for fallback)
# ---------------------------------------------------------------------------

def _fit_line_with_direction_prior(
    pts: np.ndarray,
    target_dir: np.ndarray,
    iterations: int = S4_STEGER_RANSAC_ITERATIONS,
    threshold: float = S4_STEGER_RANSAC_THRESHOLD,
    min_inliers: int = S4_STEGER_RANSAC_MIN_INLIERS,
    direction_tol_deg: float = S4_STEGER_DIRECTION_TOL_DEG,
    seed: int = 17,
    expected_through_pt: Optional[np.ndarray] = None,
    max_dist_to_pt: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    在 ridge 點雲上 RANSAC fit 一條直線，要求其方向跟 target_dir 夾角 < tol，
    且（可選）線必須通過 expected_through_pt 附近 max_dist_to_pt 範圍內。

    Args:
        pts                : (N, 2) ridge points (sub-pixel x, y, image coords)
        target_dir         : (2,)   H-projected 主方向 unit vector
        iterations         : RANSAC 採樣次數
        threshold          : ridge 點到候選線的最大垂直距離（px）
        min_inliers        : 接受 fit 的最少 inlier 數
        direction_tol_deg  : 候選線方向與 target_dir 夾角上限（度）
        seed               : RNG seed
        expected_through_pt: (2,) image-coord，候選線「應該」通過的點（通常是
                             junction center）。設定後啟用「中心通過」約束。
        max_dist_to_pt     : 候選線到 expected_through_pt 的距離上限（px）。
                             典型值 = max(wA_px, wB_px)。

    Returns:
        (line, inlier_idx) where line is (vx, vy, x0, y0) parametric form
        from TLS fit on the inlier set; or None if fail.

    跟舊版差異：
      1. 新增「中心通過約束」 — RANSAC 採樣後若候選線不通過 expected center
         max_dist_to_pt 範圍內就 reject。這對 ridge 點少且分布偏一側時非常重要
         （例如 T-junction 的 stem 邊只有一段 ridge points）。
      2. 最終 line fit 從 cv2.fitLine(L2) 改為 TLS (SVD-based orthogonal regression),
         對 leverage 點更穩定。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < max(2, min_inliers):
        return None

    target = np.asarray(target_dir, dtype=np.float64).reshape(2)
    n_target = float(np.linalg.norm(target))
    if n_target < 1e-9:
        return None
    target = target / n_target
    cos_tol = math.cos(math.radians(float(direction_tol_deg)))

    # 中心通過約束（optional）
    has_center_constraint = (expected_through_pt is not None
                             and max_dist_to_pt is not None)
    if has_center_constraint:
        center_pt = np.asarray(expected_through_pt, dtype=np.float64).reshape(2)
        max_d = float(max_dist_to_pt)
    else:
        center_pt = None
        max_d = None

    rng = np.random.default_rng(seed)
    best_inliers: List[int] = []

    for _ in range(int(iterations)):
        idx = rng.choice(len(pts), 2, replace=False)
        p0, p1 = pts[idx[0]], pts[idx[1]]
        d = p1 - p0
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-6:
            continue
        v = (d / nrm).astype(np.float64)

        # Direction prior：候選線方向與 target_dir 夾角必須小於 tol
        cosv = abs(float(np.dot(v, target)))
        if cosv < cos_tol:
            continue

        # 法向量（給點對線距離計算 + 中心通過約束）
        n = np.array([-v[1], v[0]], dtype=np.float64)

        # 中心通過約束：若這條候選線離 expected center 太遠就 reject
        if has_center_constraint:
            center_dist = abs(float((center_pt - p0.astype(np.float64)) @ n))
            if center_dist > max_d:
                continue

        # 點到候選線的距離（用法向量 dot）
        dist = np.abs((pts.astype(np.float64) - p0) @ n)
        inliers = np.where(dist < threshold)[0].tolist()
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < min_inliers:
        return None

    # 最終 line fit：用 TLS (orthogonal regression via SVD)
    # 這比 cv2.fitLine(DIST_L2) 對 leverage 點更穩，也是真正最小化「點到線」距離。
    inlier_pts = pts[best_inliers].astype(np.float64)
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    # SVD：主方向 = 最大 singular value 對應的右奇異向量
    try:
        _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
        fit_v = Vt[0]  # (2,) 主方向 unit vector
    except np.linalg.LinAlgError:
        return None
    # cv2.fitLine 形式 = (vx, vy, x0, y0)
    line = np.array([fit_v[0], fit_v[1], centroid[0], centroid[1]],
                    dtype=np.float64)

    # 最終再驗一次方向（TLS 的方向可能跟兩點採樣略有差異）
    fit_n = float(np.linalg.norm(fit_v))
    if fit_n > 1e-9:
        fit_v = fit_v / fit_n
        if abs(float(np.dot(fit_v, target))) < cos_tol:
            return None

    # （可選）最終驗中心通過約束
    if has_center_constraint:
        n_final = np.array([-fit_v[1], fit_v[0]], dtype=np.float64)
        center_dist = abs(float((center_pt - centroid) @ n_final))
        if center_dist > max_d:
            return None

    return line, np.asarray(best_inliers, dtype=int)


def _shift_line_along_normal(line: np.ndarray, offset: float) -> np.ndarray:
    """
    把 (vx, vy, x0, y0) 的線沿其單位法向偏移 offset (px)。

    line.direction = (vx, vy) → normal = (-vy, vx)
    new x0 = x0 + offset * normal_x
    new y0 = y0 + offset * normal_y
    direction 不變。
    """
    vx, vy, x0, y0 = [float(v) for v in line[:4]]
    nx, ny = -vy, vx
    return np.array([vx, vy, x0 + offset * nx, y0 + offset * ny], dtype=np.float64)


def _line_intersection_param(line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
    """
    Intersection of two lines in parametric (vx, vy, x0, y0) form.

    Why a private helper? HomographyUtils.line_intersection takes the standard
    form (a, b, c) for a*x + b*y + c = 0; our pipeline (Steger) produces lines
    in cv2.fitLine's parametric form. Convert internally rather than mixing
    representations across files.

    Returns (x, y) ndarray or None if the lines are near-parallel.
    """
    vx1, vy1, x1, y1 = [float(v) for v in line1[:4]]
    vx2, vy2, x2, y2 = [float(v) for v in line2[:4]]
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    if abs(float(np.linalg.det(A))) < 1e-9:
        return None
    try:
        t, _ = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float64)


# ---------------------------------------------------------------------------
# StegerVertexFinder
# ---------------------------------------------------------------------------

class StegerVertexFinder:
    """
    Steger 中線 + 線寬偏移當虛擬邊界線的角點搜尋器。

    與 EdgeBasedVertexFinder 對外介面一致：
      finder.find_vertices_for_junction(img_gray, H, junction_idx, center_px,
                                          do_multi_shift=False, Ix=None, Iy=None,
                                          roi_bounds=None) -> dict

    回傳 dict 形狀：
      {
        'vertices': [{'pos_px', 'pos_init', 'width_px',
                       'confidence', 'junction_idx',
                       'corner_type', 'score'}, ...],
        'confidence': float,
        'line_widths': {'A': wA_px, 'B': wB_px},
        'directions':  {'tA': tA_px, 'tB': tB_px},
        'method': 'steger_offset',
        'debug': {ridge_pts, lineA_mid, lineB_mid, ...}  # for GUI inspection
      }
    """

    def __init__(self, line_width_m: float = LINE_WIDTH_M):
        self.line_width_m = line_width_m
        # 借用舊 EdgeBasedVertexFinder 的 _apply_topology_constraints 來篩 T/L
        # 之所以不直接複製：那段邏輯依賴 self.topo_constraints / self.line_width_m，
        # 直接 instantiate 一個物件最快也最一致。
        from .vertex_finder import EdgeBasedVertexFinder
        self._topo_filter = EdgeBasedVertexFinder(line_width_m=line_width_m)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def find_vertices_for_junction(
        self,
        img_gray: np.ndarray,
        H: np.ndarray,
        junction_idx: int,
        center_px: np.ndarray,
        do_multi_shift: bool = False,         # 與舊介面相容；此 finder 內部不需要
        Ix: np.ndarray = None,                # 與舊介面相容；此 finder 不使用 Sobel
        Iy: np.ndarray = None,                # 與舊介面相容
        roi_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> dict:
        # Step 1: 兩條主方向 + Jacobian → 影像空間方向 + 線寬 ----------
        principal_dirs = compute_junction_principal_directions(junction_idx)
        if len(principal_dirs) < 2:
            return {'vertices': [], 'confidence': 0.0,
                    'method': 'steger_offset',
                    'error': 'insufficient_directions'}

        pt_m = TEMPLATE_POINTS[junction_idx]
        J = HomographyUtils.compute_jacobian(H, pt_m)
        tA_m, tB_m = principal_dirs[0], principal_dirs[1]
        tA_px = HomographyUtils.transform_direction(J, tA_m)
        tB_px = HomographyUtils.transform_direction(J, tB_m)

        wA_px = HomographyUtils.compute_line_width_px(J, tA_m, self.line_width_m)
        wB_px = HomographyUtils.compute_line_width_px(J, tB_m, self.line_width_m)
        # 與 EdgeBasedVertexFinder 同樣的 clip，避免極端 Jacobian 造成偏移過大或過小
        wA_clipped = float(np.clip(wA_px, 3.5, 20))
        wB_clipped = float(np.clip(wB_px, 3.5, 20))
        halfA = 0.5 * wA_clipped
        halfB = 0.5 * wB_clipped

        # Step 2: 確定 ROI（中心 = center_px，半徑 = ratio * max(half)） ---
        wr = max(wA_clipped, wB_clipped)
        R = max(20, int(S4_STEGER_ROI_HALF_WIDTH_RATIO * wr))
        h_img, w_img = img_gray.shape[:2]
        cx_int = int(round(float(center_px[0])))
        cy_int = int(round(float(center_px[1])))
        rx1 = max(0, cx_int - R)
        ry1 = max(0, cy_int - R)
        rx2 = min(w_img, cx_int + R)
        ry2 = min(h_img, cy_int + R)
        # 若有 YOLO bbox 限制，只把 ROI 「擴展到」bbox 外圍 — 不要被 bbox 砍小，
        # 因為線一定會延伸到 bbox 外，掃描需要這個延伸區。維持與舊 edge scanner 同樣處理。
        if roi_bounds is not None:
            bx1, by1, bx2, by2 = roi_bounds
            rx1 = max(0, min(rx1, int(bx1) - int(2 * wr)))
            ry1 = max(0, min(ry1, int(by1) - int(2 * wr)))
            rx2 = min(w_img, max(rx2, int(bx2) + int(2 * wr)))
            ry2 = min(h_img, max(ry2, int(by2) + int(2 * wr)))
        if rx2 - rx1 < 6 or ry2 - ry1 < 6:
            return {'vertices': [], 'confidence': 0.0,
                    'method': 'steger_offset',
                    'error': f'invalid_roi:{rx2 - rx1}x{ry2 - ry1}'}

        roi = img_gray[ry1:ry2, rx1:rx2]

        # Step 3: 抽 Steger ridge points + 切線方向（ROI-local 座標）---
        # 注意：新版 _steger_ridge_points_simple 加了 return_tangents=True，
        # 同時拿到位置 + 切線方向 + ridge strength（給 direction-clustering 用）
        ridge_local, ridge_tangents, ridge_strengths = _steger_ridge_points_simple(
            roi,
            sigma=S4_STEGER_SIGMA,
            threshold_pct=float(S4_STEGER_THRESHOLD_PCT) / 100.0,
            bright_lines=S4_STEGER_BRIGHT_LINES,
            return_tangents=True,
        )
        if ridge_local is None or len(ridge_local) < S4_STEGER_MIN_RIDGE_POINTS:
            return {
                'vertices': [], 'confidence': 0.0,
                'method': 'steger_offset',
                'error': f'too_few_ridge_points:{0 if ridge_local is None else len(ridge_local)}',
                'line_widths': {'A': wA_px, 'B': wB_px},
                'directions': {'tA': tA_px, 'tB': tB_px},
                'debug': {
                    'ridge_pts_global': None,
                    'roi_bounds': (rx1, ry1, rx2, ry2),
                },
            }

        # 轉成 image-global 座標
        ridge_global = ridge_local + np.array([rx1, ry1], dtype=np.float32)

        # Step 4: Deterministic line-pair fit（新方法 — 跟 S1 同款）---------
        # 流程：doubled-angle 2-means → ρ-clustering → IRLS-Huber TLS → multi-criteria
        # scoring，全程 deterministic、不依賴 H prior 強約束。junction_type 從 template
        # 拿、center_prior 用 H projection center 當 soft prior。
        cp_arr = np.asarray(center_px, dtype=np.float64).reshape(2)
        jt_id = int(TEMPLATE_TYPES[junction_idx])
        jt_name = {0: 'L', 1: 'T', 2: 'X'}.get(jt_id, '?')
        lineA, lineB, idxA, idxB = _fit_line_pair_deterministic(
            ridge_global, ridge_tangents, ridge_strengths,
            junction_type=jt_name,
            center_prior=cp_arr,
            target_dir_A=tA_px,
            target_dir_B=tB_px,
            direction_tol_deg=S4_STEGER_DIRECTION_TOL_DEG,
            min_pts_per_line=S4_STEGER_RANSAC_MIN_INLIERS,
        )
        fit_method_used = 'deterministic'

        # Fallback 1: deterministic 失敗 → 用舊 direction-clustering（softer 約束）
        if lineA is None or lineB is None:
            max_center_dist = S4_STEGER_CENTER_DIST_RATIO * max(wA_clipped, wB_clipped)
            lA2, lB2, iA2, iB2 = _fit_lines_by_direction_clustering(
                ridge_global, ridge_tangents, ridge_strengths,
                tA_px, tB_px,
                direction_tol_deg=S4_STEGER_DIRECTION_TOL_DEG,
                expected_through_pt=cp_arr,
                max_dist_to_pt=max_center_dist,
                min_pts_per_line=S4_STEGER_RANSAC_MIN_INLIERS,
            )
            if lineA is None and lA2 is not None:
                lineA, idxA = lA2, iA2
            if lineB is None and lB2 is not None:
                lineB, idxB = lB2, iB2
            fit_method_used = 'direction_clustering_fallback'

        fitA = (lineA, idxA) if lineA is not None else None
        fitB = (lineB, idxB) if lineB is not None else None

        # Fallback 2: 仍失敗 → 給舊 RANSAC 試一次（最後 fallback）
        if fitA is None or fitB is None:
            max_center_dist = S4_STEGER_CENTER_DIST_RATIO * max(wA_clipped, wB_clipped)
            if fitA is None:
                fitA = _fit_line_with_direction_prior(
                    ridge_global, tA_px, seed=17,
                    expected_through_pt=cp_arr,
                    max_dist_to_pt=max_center_dist,
                )
            if fitB is None:
                fitB = _fit_line_with_direction_prior(
                    ridge_global, tB_px, seed=23,
                    expected_through_pt=cp_arr,
                    max_dist_to_pt=max_center_dist,
                )
            fit_method_used = 'ransac_fallback'

        # 失敗條件：兩條中線都 fit 不出來 → 整個 junction 給 H_refine 補
        if fitA is None and fitB is None:
            return {
                'vertices': [], 'confidence': 0.0,
                'method': 'steger_offset',
                'error': 'no_midline',
                'line_widths': {'A': wA_px, 'B': wB_px},
                'directions': {'tA': tA_px, 'tB': tB_px},
                'debug': {
                    'ridge_pts_global': ridge_global,
                    'roi_bounds': (rx1, ry1, rx2, ry2),
                },
            }

        lineA_mid, inA = (fitA if fitA is not None else (None, None))
        lineB_mid, inB = (fitB if fitB is not None else (None, None))

        # Step 5: 沿法向偏移 ±half 形成 4 條虛擬邊界線 -------------------
        lineA_plus = _shift_line_along_normal(lineA_mid, +halfA) if lineA_mid is not None else None
        lineA_minus = _shift_line_along_normal(lineA_mid, -halfA) if lineA_mid is not None else None
        lineB_plus = _shift_line_along_normal(lineB_mid, +halfB) if lineB_mid is not None else None
        lineB_minus = _shift_line_along_normal(lineB_mid, -halfB) if lineB_mid is not None else None

        # Step 6: 4 種交點 → 4 個 corner ---------------------------------
        # corner score = 兩條中線 inlier 數的調和平均，再正規化到 0~1
        if inA is not None and inB is not None and len(inA) > 0 and len(inB) > 0:
            base_score = 2.0 / (1.0 / len(inA) + 1.0 / len(inB))
            # 用 ROI 內最多可能的 ridge 點數當分母 → 0~1 範圍
            base_score = float(min(1.0, base_score / max(1.0, len(ridge_global) / 4.0)))
        else:
            base_score = 0.4

        all_corners = []
        combinations = [
            (lineA_plus, lineB_plus, '++'),
            (lineA_plus, lineB_minus, '+-'),
            (lineA_minus, lineB_plus, '-+'),
            (lineA_minus, lineB_minus, '--'),
        ]
        for la, lb, label in combinations:
            if la is None or lb is None:
                continue
            ipt = _line_intersection_param(la, lb)
            if ipt is None:
                continue
            ipt = np.asarray(ipt, dtype=np.float64).reshape(2)
            # 只接受落在 ROI（含些許餘裕）內的交點
            if not (rx1 - 4 <= ipt[0] <= rx2 + 4 and ry1 - 4 <= ipt[1] <= ry2 + 4):
                continue
            # 落在 ROI 邊角太遠的點分數打折
            dist_from_center = float(np.linalg.norm(ipt - np.asarray(center_px, dtype=np.float64)))
            decay = math.exp(-max(0.0, dist_from_center - 1.5 * wr) / 12.0)
            score = float(np.clip(base_score * decay, 0.0, 1.0))
            all_corners.append({
                'pos': ipt.astype(np.float32),
                'score': score,
                'type': label,  # 先存 pixel-space 配對給的 label，下面會重判
                '_pixel_label': label,  # 保留 debug 用
                'method': 'steger_offset',
            })

        if not all_corners:
            return {
                'vertices': [], 'confidence': 0.0,
                'method': 'steger_offset',
                'error': 'no_intersections',
                'line_widths': {'A': wA_px, 'B': wB_px},
                'directions': {'tA': tA_px, 'tB': tB_px},
                'debug': {
                    'ridge_pts_global': ridge_global,
                    'lineA_mid': lineA_mid, 'lineB_mid': lineB_mid,
                    'lineA_plus': lineA_plus, 'lineA_minus': lineA_minus,
                    'lineB_plus': lineB_plus, 'lineB_minus': lineB_minus,
                    'roi_bounds': (rx1, ry1, rx2, ry2),
                },
            }

        # Step 6.5: ── 用 H⁻¹ 投回 template 空間重新判 corner_type label ──
        # ──────────────────────────────────────────────────────────────
        # 為什麼要這樣做：
        #   pixel 空間的 lineA_plus / lineB_plus 用 perp(tA_px) / perp(tB_px) 偏移，
        #   但「+perp(tA_px)」未必跟「+nA_m 經 H 投影後的方向」一致。當 Jacobian
        #   det < 0（球場 y 軸朝上、影像 y 朝下，通常都是如此），兩個方向會差
        #   一個正負號。如果 Steger 用 pixel-space label 而 h_refine 用 template-
        #   space label，同一個 '++' 字串會指向不同物理角點 → cid 配對成功
        #   但 final_pos 拉到對角中央 → vertex 跑很多。
        #
        # 解法：所有交點都 H⁻¹ 投回 template 公尺空間，看 delta_m 落在
        #   (nA_m, nB_m) 的哪個象限，產生「template-space 權威 label」。這跟
        #   h_refine `_build_h_rectified_corners` 用同一套定義，配對就會對齊。
        nA_m_unit = np.array([-tA_m[1], tA_m[0]], dtype=np.float64)
        nB_m_unit = np.array([-tB_m[1], tB_m[0]], dtype=np.float64)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Hinv = None

        if Hinv is not None:
            p0_m_v = np.asarray(pt_m, dtype=np.float64).reshape(2)
            # 批次轉換
            pts_px = np.array([c['pos'] for c in all_corners],
                              dtype=np.float32).reshape(-1, 1, 2)
            pts_m = cv2.perspectiveTransform(pts_px, Hinv).reshape(-1, 2)
            for c, pm in zip(all_corners, pts_m):
                delta_m = np.asarray(pm, dtype=np.float64) - p0_m_v
                sa_sign = float(np.dot(delta_m, nA_m_unit))
                sb_sign = float(np.dot(delta_m, nB_m_unit))
                sa_char = '+' if sa_sign >= 0 else '-'
                sb_char = '+' if sb_sign >= 0 else '-'
                c['type'] = sa_char + sb_char  # template-space 權威 label

        # 同個 label 出現兩次（罕見，但理論上可能）→ 保留 score 較高者
        if all_corners:
            best_per_label = {}
            for c in all_corners:
                lb = c['type']
                if lb not in best_per_label or c['score'] > best_per_label[lb]['score']:
                    best_per_label[lb] = c
            all_corners = list(best_per_label.values())

        # Step 7: T/L 用 topology 約束篩到正確象限 -----------------------
        jt = int(TEMPLATE_TYPES[junction_idx])
        if jt == 2:  # X：4 corners 全收
            sel_corners = all_corners
        else:
            sel = self._topo_filter._apply_topology_constraints(
                all_corners, H, junction_idx,
                tA_m.astype(np.float64), tB_m.astype(np.float64),
                halfA, halfB,
            )
            sel_corners = sel if sel else sorted(
                all_corners, key=lambda c: c['score'], reverse=True,
            )[:(2 if jt == 1 else 1)]  # T:2, L:1

        # Step 8: 整理輸出 vertex list（與 EdgeBasedVertexFinder 同格式） -
        vertices = []
        for c in sel_corners:
            ct = c.get('type', 'outer')
            # corner_code：用 (junction_idx, corner_type, tA_m, tB_m) 算出
            # 8-bit 全域編碼。tA_m / tB_m 用 template 空間方向（同 _build_h_rectified_corners）
            try:
                ccode = corner_type_to_corner_code(
                    junction_idx=int(junction_idx),
                    corner_type=ct,
                    tA_m=np.asarray(tA_m, dtype=np.float64),
                    tB_m=np.asarray(tB_m, dtype=np.float64),
                )
            except Exception:
                ccode = -1
            vertices.append({
                'pos_px': np.asarray(c['pos'], dtype=np.float32),
                'pos_init': np.asarray(center_px, dtype=np.float32).copy(),
                'width_px': (wA_px + wB_px) / 2.0,
                'confidence': float(c.get('score', 0.5)),
                'junction_idx': int(junction_idx),
                'corner_type': ct,
                'corner_code': int(ccode),
                'score': float(c.get('score', 0.0)),
            })

        avg_conf = float(np.mean([v['confidence'] for v in vertices])) if vertices else 0.0

        return {
            'vertices': vertices,
            'confidence': avg_conf,
            'method': 'steger_offset',
            'line_widths': {'A': wA_px, 'B': wB_px},
            'directions': {'tA': tA_px, 'tB': tB_px},
            'debug': {
                'ridge_pts_global': ridge_global,
                'ridge_tangents': ridge_tangents,
                'ridge_strengths': ridge_strengths,
                'lineA_mid': lineA_mid, 'lineB_mid': lineB_mid,
                'lineA_plus': lineA_plus, 'lineA_minus': lineA_minus,
                'lineB_plus': lineB_plus, 'lineB_minus': lineB_minus,
                'roi_bounds': (rx1, ry1, rx2, ry2),
                'inlier_count_A': 0 if inA is None else int(len(inA)),
                'inlier_count_B': 0 if inB is None else int(len(inB)),
                'group_idx_A': inA,    # 屬於 A 線的 ridge point indices
                'group_idx_B': inB,    # 屬於 B 線的 ridge point indices
                'fit_method': fit_method_used,
            },
        }


__all__ = [
    'StegerVertexFinder',
    '_fit_line_with_direction_prior',
    '_fit_lines_by_direction_clustering',
    '_tls_line_weighted',
    '_shift_line_along_normal',
    '_line_intersection_param',
]
