"""
Vertex Quality Scorer — Gradient-Geometry Edition
==================================================
每個 vertex 給一個連續分數（0~1）

演算法流程（per ROI）：
  Stage 1  結構張量場
    Sxx, Syy, Sxy = Gσh * (∇I ∇Iᵀ)
    harris_R = Det(S) − k·Tr(S)²          （梯度版 Harris, 沿用輸出鍵名）
    En  = Tr(S) 正規化 ∈ [0,1]            （梯度能量）
    coh = (λ1−λ2)/(λ1+λ2) ∈ [0,1]         （同調度: 線→1, 角→低）
  Stage 2  稠密 Förstner 收斂殘差
    Rn(p) = Σ_w (∇I·(p−x))² / Σ_w |∇I|² / E_rand
    以矩核摺積一次算整張圖; 真交點處鄰域等照度約束線收斂 → Rn 低
    ridge 證據 = En·coh（取代 steger_strength）, ridge mask = 高coh∧高能量
  Stage 3  Feature-Ridge Diff Map（語意同前: >0 corner、<0 line、≈0 背景）
    diff = clip( min(En/e0,1) · (R0 − Rn)/R0 , −1, 1 ),  R0=0.6, e0=0.15
    （junction Rn≈0.45 → 正; 線 Rn≈0.76 → 負; 背景低能量 → ≈0）
  Stage 4  Adaptive NMS (Brown et al. 2005)   —— 不變
  Stage 5  Gradient Convergence Filter（取代 Ridge Proximity）
    對每個候選峰在局部窗解 Förstner p*; |p*−peak| ≤ max_dist 且
    能量支持像素數 ≥ min_area 才保留（峰必須真有梯度線收斂於此）

Vertex Scoring（per vertex, 無 grade）—— 公式不變：
  dist_score    = exp(-d / tau)            d = 到最近 final peak 的距離
                                           tau = peak_radius_px
  heatmap_score = clip(diff[vertex], 0, 1)   負值歸零（line-dominant）
  composite     = alpha*dist_score + (1-alpha)*heatmap_score   預設 alpha=0.5

對外介面：evaluate_vertex 方法。
"""
from __future__ import annotations

import math

import numpy as np
import cv2
from dataclasses import dataclass, field
from scipy.ndimage import (gaussian_filter, maximum_filter, binary_dilation,
                           label, distance_transform_edt)

from ..config import (
    VQ_HARRIS_K               as _HARRIS_K,
    VQ_HARRIS_SIGMA           as _HARRIS_SIGMA,
    VQ_HARRIS_THRESHOLD_PCT   as _HARRIS_THR_PCT,
    VQ_STEGER_SIGMA           as _STEGER_SIGMA,
    VQ_STEGER_THRESHOLD_PCT   as _STEGER_THR_PCT,
    VQ_STEGER_DILATION_RADIUS as _STEGER_DIL_R,
    VQ_STEGER_DARK_RIDGES     as _DARK_RIDGES,
    VQ_TOP_K                  as _TOP_K,
    VQ_PEAK_RADIUS_PX         as _PEAK_RADIUS,
    VQ_INSET                  as _INSET,
    VQ_ANMS_C                 as _ANMS_C,
    VQ_ANMS_CANDIDATE_POOL    as _ANMS_POOL,
    VQ_ANMS_LOOSE_NMS_RADIUS  as _ANMS_NMS_R,
    VQ_PROX_ENABLED           as _PROX_ENABLED,
    VQ_PROX_MIN_AREA          as _PROX_MIN_AREA,
    VQ_PROX_MAX_DIST          as _PROX_MAX_DIST,
    VQ_PROX_CLOSING_RADIUS    as _PROX_CLOSE_R,
)


# ═══════════════════════════════════════════════════════
#  Default scoring weights
# ═══════════════════════════════════════════════════════

DIST_WEIGHT_DEFAULT    = 0.5   # alpha in composite
HEATMAP_WEIGHT_DEFAULT = 0.5

_COH_RIDGE_THR = 0.65          # 同調度高於此視為線狀
_RN_SPLIT      = 0.55          # 收斂殘差分界: junction(≈0.27) < 0.55 ≈ 線中心 (平窗實測)
_EN_GATE       = 0.15          # 能量門控尺度 (壓低弱梯度背景)


# ═══════════════════════════════════════════════════════
#  Low-level gradient helpers
# ═══════════════════════════════════════════════════════

def _gradients(gray_f32: np.ndarray, sigma: float):
    """Gaussian 平滑後的 Sobel 梯度 (Ix, Iy)。sigma = 梯度尺度。"""
    f = cv2.GaussianBlur(gray_f32, (0, 0), max(float(sigma), 0.3))
    Ix = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return Ix, Iy


def _structure_tensor_field(Ix, Iy, sigma: float):
    """以 Gσ 平滑的結構張量分量。"""
    s = max(float(sigma), 0.5)
    Sxx = gaussian_filter(Ix * Ix, s)
    Syy = gaussian_filter(Iy * Iy, s)
    Sxy = gaussian_filter(Ix * Iy, s)
    return Sxx, Syy, Sxy


def _dense_conv_resid(Ix, Iy, win_radius: int):
    """稠密 Förstner 收斂殘差 (以視窗中心為查詢點), 正規化使隨機梯度 ≈ 1。

    R(p) = Σ_w [Ix²u² + 2·IxIy·uv + Iy²v²] / Σ_w (Ix²+Iy²) / E_rand,
      其中 (u,v) = x − p, w = (2r+1)² 平窗, E_rand = E[u²+v²]/2。
    實測 (W=21): junction Rn≈0.27 < 線中心 ≈0.55 < 背景 ≈0.76。
    """
    r = max(4, int(win_radius))
    n1 = 2 * r + 1
    u = np.arange(-r, r + 1, dtype=np.float32)
    ones = np.ones(n1, np.float32)
    u2 = u * u
    norm = 1.0 / (n1 * n1)
    e_rand = float(2.0 * n1 * u2.sum()) * norm / 2.0   # = E[u²+v²]/2
    gxx = Ix * Ix
    gyy = Iy * Iy
    gxy = Ix * Iy
    B = cv2.BORDER_REFLECT
    # 矩核皆可分離: Ku2 = u²⊗1, Kuv = u⊗v, Kv2 = 1⊗v² (×1/N)
    num = (cv2.sepFilter2D(gxx, -1, u2, ones, borderType=B)
           + 2.0 * cv2.sepFilter2D(gxy, -1, u, u, borderType=B)
           + cv2.sepFilter2D(gyy, -1, ones, u2, borderType=B)) * norm
    den = cv2.sepFilter2D(gxx + gyy, -1, ones, ones, borderType=B) * norm
    return (num / (den * e_rand + 1e-9)).astype(np.float32)


def _forstner_local(Ix, Iy, cy: int, cx: int, r: int):
    """在 (cy,cx) 半徑 r 的窗內解 Förstner p*。
    回傳 (dy, dx, n_support) — p* 相對窗中心的偏移與能量支持數; 無解回傳 None。"""
    h, w = Ix.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    ix = Ix[y0:y1, x0:x1].astype(np.float64)
    iy = Iy[y0:y1, x0:x1].astype(np.float64)
    g2 = ix * ix + iy * iy
    if g2.size == 0:
        return None
    n_support = int((g2 > 0.1 * g2.max()).sum()) if g2.max() > 0 else 0
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    A11 = float((ix * ix).sum()); A22 = float((iy * iy).sum())
    A12 = float((ix * iy).sum())
    det = A11 * A22 - A12 * A12
    if det <= 1e-9 or (A11 + A22) <= 1e-9:
        return None
    b1 = float((ix * ix * xx + ix * iy * yy).sum())
    b2 = float((ix * iy * xx + iy * iy * yy).sum())
    px = (A22 * b1 - A12 * b2) / det
    py = (A11 * b2 - A12 * b1) / det
    return (py - cy, px - cx, n_support)


def _disk_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(np.uint8)


def _topk_anms(diff: np.ndarray, top_k: int,
               c: float, candidate_pool: int, loose_nms_radius: int):
    """
    Adaptive Non-Maximal Suppression (Brown et al. 2005) on diff map.
    Returns (ys, xs, vals) sorted by suppression radius descending.
    """
    pos = np.clip(diff, 0.0, None).astype(np.float32)
    lm  = maximum_filter(pos, size=2 * loose_nms_radius + 1)
    cand = (pos == lm) & (pos > 0)
    ys, xs = np.where(cand)
    if len(ys) == 0:
        return np.array([], int), np.array([], int), np.array([], float)

    vals = diff[ys, xs].astype(np.float64)
    if len(vals) > candidate_pool:
        keep = np.argpartition(vals, -candidate_pool)[-candidate_pool:]
        ys, xs, vals = ys[keep], xs[keep], vals[keep]

    n = len(vals)
    if n == 1:
        return ys, xs, vals.astype(float)

    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    d2  = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    stronger = vals[:, None] < c * vals[None, :]
    np.fill_diagonal(stronger, False)
    d2_masked = np.where(stronger, d2, np.inf)
    radii     = np.min(d2_masked, axis=1)

    order = np.argsort(radii)[::-1]
    sel   = order[:min(top_k, n)]
    return ys[sel], xs[sel], vals[sel].astype(float)


def _significant_ridge_mask(ridge_mask: np.ndarray,
                            min_component_area: int,
                            closing_radius: int = 3) -> np.ndarray:
    """Keep only connected components of ridge_mask with area >= threshold.
    （與舊版相同, 供 sig_ridge_mask 輸出鍵）"""
    rm = ridge_mask.astype(bool)
    if min_component_area <= 1 or not rm.any():
        return rm.copy()

    if closing_radius > 0:
        struct = _disk_kernel(closing_radius)
        closed = binary_dilation(rm, structure=struct)
        closed = ~binary_dilation(~closed, structure=struct)
    else:
        closed = rm

    structure = np.ones((3, 3), dtype=bool)
    labelled, n = label(closed, structure=structure)
    if n == 0:
        return np.zeros_like(rm)
    counts = np.bincount(labelled.ravel(), minlength=n + 1)
    keep_ids = np.where(counts >= min_component_area)[0]
    keep_ids = keep_ids[keep_ids != 0]
    if keep_ids.size == 0:
        return np.zeros_like(rm)
    keep = np.zeros(n + 1, dtype=bool)
    keep[keep_ids] = True
    return keep[labelled]


def _convergence_filter(ys, xs, vals, Ix, Iy, max_dist, min_support, win_r):
    """[取代 ridge-proximity] 峰必須有梯度線收斂證據:
    局部 Förstner p* 與峰距離 ≤ max_dist, 且能量支持數 ≥ min_support。"""
    if len(ys) == 0:
        e = np.array([], int); ef = np.array([], float)
        return e, e, ef, e, e
    keep = np.zeros(len(ys), bool)
    for i, (y, x) in enumerate(zip(ys, xs)):
        f = _forstner_local(Ix, Iy, int(y), int(x), win_r)
        if f is None:
            continue
        dy, dx, n_sup = f
        if math.hypot(dy, dx) <= max_dist and n_sup >= min_support:
            keep[i] = True
    drop = ~keep
    return (ys[keep], xs[keep], vals[keep], ys[drop], xs[drop])


# ═══════════════════════════════════════════════════════
#  Main analysis function
# ═══════════════════════════════════════════════════════

def run_harris_steger_analysis(
    gray_roi:       np.ndarray,
    harris_k:       float = _HARRIS_K,
    harris_sigma:   float = _HARRIS_SIGMA,
    harris_thr_pct: int   = _HARRIS_THR_PCT,
    steger_sigma:   float = _STEGER_SIGMA,
    steger_thr_pct: int   = _STEGER_THR_PCT,
    steger_dil_r:   int   = _STEGER_DIL_R,
    dark_ridges:    bool  = _DARK_RIDGES,
    top_k:          int   = _TOP_K,
    inset:          int   = _INSET,
    anms_c:         float = _ANMS_C,
    anms_pool:      int   = _ANMS_POOL,
    anms_nms_r:     int   = _ANMS_NMS_R,
    prox_enabled:   bool  = _PROX_ENABLED,
    prox_min_area:  int   = _PROX_MIN_AREA,
    prox_max_dist:  float = _PROX_MAX_DIST,
    prox_close_r:   int   = _PROX_CLOSE_R,
) -> dict:
    """
    End-to-end gradient-geometry pipeline.（簽名與回傳鍵與舊版完全相同）

    參數對應（語意換新, 名稱保留以維持相容）:
      harris_sigma   : 結構張量平滑尺度 + conv_resid 視窗尺度
      steger_sigma   : 梯度計算前的平滑尺度
      steger_thr_pct : ridge 證據能量門檻 (%)
      dark_ridges    : 接受但忽略（梯度對亮暗線對稱）
      prox_max_dist  : 峰的 Förstner p* 最大允許偏移 (px)
      prox_min_area  : 峰局部窗內最少能量支持像素數
      prox_close_r   : 僅用於 sig_ridge_mask 形態學（沿用）

    Returns (all arrays padded back to original ROI size):
      harris_R         : (H,W) float32  結構張量 Harris 響應
      harris_mask      : (H,W) bool     NMS+threshold corners (reference)
      harris_thresh    : float
      steger_ridge     : (H,W) bool     線狀證據遮罩 (高同調∧高能量)
      steger_excl      : (H,W) bool     dilated exclusion zone
      steger_strength  : (H,W) float32  ridge 證據 = En·coh
      steger_eigenval  : (H,W) float32  λ1−λ2 (結構張量特徵值差)
      sig_ridge_mask   : (H,W) bool     significant-component ridge mask
      diff             : (H,W) float32  corner−ridge 證據 ∈ [-1, 1]
      peaks            : list[(row, col)]  FINAL peaks after ANMS + convergence
      peak_vals        : list[float]       diff value at each peak
      dropped_peaks    : list[(row, col)]  peaks dropped by convergence filter
      inset            : int

    Backward-compat aliases:
      combined         : same as diff
      threshold        : 0.0  (in_zone criterion is now diff > 0)
    """
    _e = np.zeros((1, 1), np.float32)
    if gray_roi is None or gray_roi.size == 0:
        return dict(harris_R=_e, harris_mask=np.zeros((1,1),bool),
                    harris_thresh=0.0, steger_ridge=np.zeros((1,1),bool),
                    steger_excl=np.zeros((1,1),bool), steger_strength=_e,
                    steger_eigenval=_e, sig_ridge_mask=np.zeros((1,1),bool),
                    diff=_e, combined=_e, threshold=0.0,
                    peaks=[], peak_vals=[], dropped_peaks=[], inset=0)

    roi_h, roi_w = gray_roi.shape[:2]
    inset = max(0, min(inset, (min(roi_h, roi_w) - 7) // 2))
    work  = gray_roi[inset:roi_h-inset, inset:roi_w-inset] if inset else gray_roi
    wh, ww = work.shape[:2]
    wf32  = np.clip(work, 0, 255).astype(np.float32)

    # Stage 1: 梯度 + 結構張量場
    Ix, Iy = _gradients(wf32, steger_sigma)
    Sxx, Syy, Sxy = _structure_tensor_field(Ix, Iy, harris_sigma)
    tr_S  = Sxx + Syy
    det_S = Sxx * Syy - Sxy * Sxy
    R     = (det_S - harris_k * tr_S * tr_S).astype(np.float32)
    h_thr = harris_thr_pct / 100.0 * float(max(R.max(), 1e-9))
    R_pos = np.where(R > 0, R, 0.0).astype(np.float32)
    lm_h  = maximum_filter(R_pos, size=2 * anms_nms_r + 1)
    h_mask = (R_pos == lm_h) & (R > h_thr)

    En  = (tr_S / (float(np.percentile(tr_S, 99)) + 1e-9)).astype(np.float32)
    disc = np.sqrt(np.maximum((Sxx - Syy) ** 2 + 4.0 * Sxy * Sxy, 0.0))
    coh  = (disc / (tr_S + 1e-9)).astype(np.float32)          # (λ1−λ2)/(λ1+λ2)
    eigdiff = disc.astype(np.float32)                          # λ1−λ2

    # ridge 證據 (取代 Steger; 供輸出鍵與遮罩, 不參與 diff)
    strength = (En * coh).astype(np.float32)
    smax = float(strength.max()) if strength.max() > 0 else 1.0
    ridge = (coh > _COH_RIDGE_THR) & (strength > steger_thr_pct / 100.0 * smax)
    excl  = binary_dilation(ridge, structure=_disk_kernel(steger_dil_r)) \
        if steger_dil_r > 0 else ridge.copy()

    # Stage 2: 稠密收斂殘差 (核心判別器)
    Rn = _dense_conv_resid(Ix, Iy, int(round(7.0 * max(float(harris_sigma), 1.0))))

    # Stage 3: diff map — 能量門控 × 收斂證據 × 非退化性, 維持舊語意
    #   (>0 corner, <0 line, ≈0 背景)
    #   (1−coh) 抑制 Förstner 退化: 邊緣上梯度共線 → 約束線退化 → Rn 假性低谷,
    #   此時 coh→1, 乘上 (1−coh) 將其壓回。junction coh≈0.5 → 保留。
    en_gate = np.clip(En / _EN_GATE, 0.0, 1.0)
    nondeg  = np.clip(2.0 * (1.0 - coh), 0.0, 1.0)
    diff = np.clip(en_gate * nondeg * (_RN_SPLIT - Rn) / _RN_SPLIT,
                   -1.0, 1.0).astype(np.float32)

    # Stage 4: ANMS peaks（不變）
    ys_a, xs_a, vals_a = _topk_anms(
        diff, top_k=top_k, c=anms_c,
        candidate_pool=anms_pool, loose_nms_radius=anms_nms_r)

    # Stage 5: gradient convergence filter（取代 ridge proximity）
    sig_mask = _significant_ridge_mask(ridge, prox_min_area, prox_close_r)
    if prox_enabled and len(ys_a) > 0:
        win_r = max(3, int(round(3.0 * harris_sigma)))
        ys_k, xs_k, vals_k, ys_d, xs_d = _convergence_filter(
            ys_a, xs_a, vals_a, Ix, Iy,
            max_dist=prox_max_dist, min_support=prox_min_area, win_r=win_r)
    else:
        ys_k, xs_k, vals_k = ys_a, xs_a, vals_a
        ys_d = np.array([], int);  xs_d = np.array([], int)

    # Convert (row, col) tuples; add inset offset back to original ROI coords
    peaks         = [(int(r) + inset, int(c) + inset) for r, c in zip(ys_k, xs_k)]
    peak_vals     = vals_k.tolist() if hasattr(vals_k, 'tolist') else list(vals_k)
    dropped_peaks = [(int(r) + inset, int(c) + inset) for r, c in zip(ys_d, xs_d)]

    def _pad(a, dt=np.float32):
        out = np.zeros((roi_h, roi_w), dt)
        out[inset:inset+wh, inset:inset+ww] = a
        return out

    diff_padded = _pad(diff)

    return dict(
        harris_R        = _pad(R),
        harris_mask     = _pad(h_mask, bool),
        harris_thresh   = h_thr,
        steger_ridge    = _pad(ridge, bool),
        steger_excl     = _pad(excl, bool),
        steger_strength = _pad(strength),
        steger_eigenval = _pad(eigdiff),
        sig_ridge_mask  = _pad(sig_mask, bool),
        diff            = diff_padded,
        combined        = diff_padded,      # backward-compat alias
        threshold       = 0.0,              # in_zone criterion: diff > 0
        peaks           = peaks,
        peak_vals       = peak_vals,
        dropped_peaks   = dropped_peaks,
        inset           = inset,
    )


# ═══════════════════════════════════════════════════════
#  Data classes (simplified — no grade fields)
# ═══════════════════════════════════════════════════════

@dataclass
class VertexQualityResult:
    """
    Per-vertex quality score (no grade).

    composite         : alpha*dist_score + (1-alpha)*heatmap_score  in [0,1]
    dist_score        : exp(-d/tau) where d = dist to nearest peak, tau = peak_radius_px
    heatmap_score     : clip(diff[vertex], 0, 1)
    nearest_peak_dist : Euclidean distance (px) from vertex to nearest final peak
    diff_value        : raw diff-map value at vertex (can be negative)
    num_peaks         : number of final peaks in the ROI
    """
    composite:         float = 0.0
    dist_score:        float = 0.0
    heatmap_score:     float = 0.0
    nearest_peak_dist: float = float("inf")
    diff_value:        float = 0.0
    num_peaks:         int   = 0

    # Backward-compat aliases
    dm_value:          float = float("inf")   # alias for nearest_peak_dist
    response_val:      float = 0.0            # alias for diff_value

    def to_dict(self):
        return dict(
            composite         = round(self.composite, 4),
            dist_score        = round(self.dist_score, 4),
            heatmap_score     = round(self.heatmap_score, 4),
            nearest_peak_dist = round(self.nearest_peak_dist, 3),
            diff_value        = round(self.diff_value, 4),
            num_peaks         = self.num_peaks,
        )


@dataclass
class JunctionScore:
    """Junction-level aggregation (optional for downstream use)."""
    vertex_scores: list  = field(default_factory=list)
    status:        str   = "unknown"
    composite:     float = 0.0
    issues:        list  = field(default_factory=list)

    def to_dict(self):
        return dict(
            composite     = round(self.composite, 4),
            status        = self.status,
            total_count   = len(self.vertex_scores),
            vertex_scores = [vs.to_dict() for vs in self.vertex_scores],
            issues        = self.issues,
        )


@dataclass
class PipelineScore:
    """Pipeline-level summary (optional)."""
    junction_scores:    dict  = field(default_factory=dict)
    composite:          float = 0.0
    avg_vertex_quality: float = 0.0
    active_junctions:   int   = 0
    total_vertices:     int   = 0

    def to_dict(self):
        return dict(
            composite          = round(self.composite, 4),
            avg_vertex_quality = round(self.avg_vertex_quality, 4),
            active_junctions   = self.active_junctions,
            total_vertices     = self.total_vertices,
            junction_scores    = {str(k): v.to_dict()
                                  for k, v in self.junction_scores.items()},
        )


# ═══════════════════════════════════════════════════════
#  VertexQualityScorer   —— 介面與公式
# ═══════════════════════════════════════════════════════

class VertexQualityScorer:
    """
    每個 vertex 獨立評分（加權和，無 grade）：
      composite = dist_weight * exp(-d/tau) + heatmap_weight * clip(diff[v], 0, 1)

    下游自己決定如何聚合（例如取 max / mean / 閾值）。
    """

    def __init__(self,
                 peak_radius_px:  float = _PEAK_RADIUS,
                 harris_k:        float = _HARRIS_K,
                 harris_sigma:    float = _HARRIS_SIGMA,
                 harris_thr_pct:  int   = _HARRIS_THR_PCT,
                 steger_sigma:    float = _STEGER_SIGMA,
                 steger_thr_pct:  int   = _STEGER_THR_PCT,
                 steger_dil_r:    int   = _STEGER_DIL_R,
                 dark_ridges:     bool  = _DARK_RIDGES,
                 top_k:           int   = _TOP_K,
                 inset:           int   = _INSET,
                 anms_c:          float = _ANMS_C,
                 anms_pool:       int   = _ANMS_POOL,
                 anms_nms_r:      int   = _ANMS_NMS_R,
                 prox_enabled:    bool  = _PROX_ENABLED,
                 prox_min_area:   int   = _PROX_MIN_AREA,
                 prox_max_dist:   float = _PROX_MAX_DIST,
                 prox_close_r:    int   = _PROX_CLOSE_R,
                 dist_weight:     float = DIST_WEIGHT_DEFAULT,
                 heatmap_weight:  float = HEATMAP_WEIGHT_DEFAULT,
                 **_):
        self.peak_radius_px = float(peak_radius_px)
        self.harris_k       = float(harris_k)
        self.harris_sigma   = float(harris_sigma)
        self.harris_thr_pct = int(max(1, harris_thr_pct))
        self.steger_sigma   = float(steger_sigma)
        self.steger_thr_pct = int(max(1, steger_thr_pct))
        self.steger_dil_r   = int(steger_dil_r)
        self.dark_ridges    = bool(dark_ridges)
        self.top_k          = int(top_k)
        self.inset          = int(inset)
        self.anms_c         = float(anms_c)
        self.anms_pool      = int(anms_pool)
        self.anms_nms_r     = int(anms_nms_r)
        self.prox_enabled   = bool(prox_enabled)
        self.prox_min_area  = int(prox_min_area)
        self.prox_max_dist  = float(prox_max_dist)
        self.prox_close_r   = int(prox_close_r)

        # Normalize weights to sum to 1 (使用者可自由調比例)
        w_sum = float(dist_weight) + float(heatmap_weight)
        if w_sum <= 0:
            self.dist_weight    = DIST_WEIGHT_DEFAULT
            self.heatmap_weight = HEATMAP_WEIGHT_DEFAULT
        else:
            self.dist_weight    = float(dist_weight)    / w_sum
            self.heatmap_weight = float(heatmap_weight) / w_sum

    def _ana(self, roi_gray):
        return run_harris_steger_analysis(
            roi_gray,
            harris_k=self.harris_k, harris_sigma=self.harris_sigma,
            harris_thr_pct=self.harris_thr_pct,
            steger_sigma=self.steger_sigma, steger_thr_pct=self.steger_thr_pct,
            steger_dil_r=self.steger_dil_r, dark_ridges=self.dark_ridges,
            top_k=self.top_k, inset=self.inset,
            anms_c=self.anms_c, anms_pool=self.anms_pool,
            anms_nms_r=self.anms_nms_r,
            prox_enabled=self.prox_enabled,
            prox_min_area=self.prox_min_area,
            prox_max_dist=self.prox_max_dist,
            prox_close_r=self.prox_close_r,
        )

    def evaluate_vertex(self, vertex_pos_px, roi_gray, roi_origin):
        """
        對單一 vertex 給分（無 grade）。（輸入/輸出）
        """
        res = VertexQualityResult()
        if roi_gray is None or roi_gray.size == 0: return res
        if roi_gray.shape[0] < 7 or roi_gray.shape[1] < 7: return res

        roi_h, roi_w = roi_gray.shape[:2]
        ana   = self._ana(roi_gray)
        diff  = ana["diff"]
        peaks = ana["peaks"]
        res.num_peaks = len(peaks)

        vx = float(vertex_pos_px[0]) - float(roi_origin[0])
        vy = float(vertex_pos_px[1]) - float(roi_origin[1])
        vc = int(np.clip(round(vx), 0, roi_w - 1))
        vr = int(np.clip(round(vy), 0, roi_h - 1))

        # heatmap_score = clip(diff[vertex], 0, 1)
        if -1 <= round(vx) < roi_w + 1 and -1 <= round(vy) < roi_h + 1:
            dv = float(diff[vr, vc])
            res.diff_value    = dv
            res.response_val  = dv
            res.heatmap_score = float(np.clip(dv, 0.0, 1.0))

        # dist_score = exp(-d / tau)
        if peaks:
            pk   = np.array(peaks, np.float32)
            dist = np.sqrt((pk[:, 0] - vy) ** 2 + (pk[:, 1] - vx) ** 2)
            d    = float(dist.min())
            res.nearest_peak_dist = d
            res.dm_value          = d
            tau = max(self.peak_radius_px, 1e-6)
            res.dist_score = float(np.exp(-d / tau))
        else:
            res.dist_score = 0.0

        # Composite
        res.composite = float(
            self.dist_weight    * res.dist_score +
            self.heatmap_weight * res.heatmap_score
        )
        res.composite = float(np.clip(res.composite, 0.0, 1.0))
        return res