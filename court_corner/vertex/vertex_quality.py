"""
Vertex Quality Scorer — Structure-Tensor (Shi-Tomasi) Edition
=============================================================
每個 vertex 給一個連續分數（0~1）。本版內部演算法改用「角點定位品質評估工具」
的方法，**對外介面（建構子、evaluate_vertex、VertexQualityResult 欄位、
run_harris_steger_analysis 回傳鍵）完全不變**。

方法（忠於文獻，無自訂啟發式）：
  - 局部結構張量  N = Gσi * (∇I ∇Iᵀ)，特徵值 λ1 ≥ λ2。
  - 定位品質 = 正規化最小特徵值 (Shi & Tomasi, CVPR 1994, "Good Features to Track")
        quality = clip(λ2 / λ2_ref, 0, 1)
    λ2 代表「最差方向」的可定位性（同時涵蓋邊緣與平坦的退化）；
    λ2_ref 為該 ROI 內「具二向結構」像素之 λ2 高百分位 —— 隨 ROI 自適應。
  - 不確定度協方差 C ∝ σ²·N⁻¹ (Ferraz 2014; Vakhitov 2021)：λ2 大 ⇒ 橢圓小而圓。
  - Förstner 梯度收斂次像素精修  p* = N⁻¹ Σ(∇I∇Iᵀ)x，位移 shift = |p*−vertex|。

Vertex Scoring（per vertex，公式骨架不變，子分數改由上法產生）：
  heatmap_score = quality = clip(λ2 / λ2_ref, 0, 1)         （取代 diff-map 角點證據）
  dist_score    = exp(-shift / tau),  tau = peak_radius_px  （shift 取代「到最近 peak 距離」）
  composite     = dist_weight * dist_score + heatmap_weight * heatmap_score

對外介面：evaluate_vertex 方法。
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter, maximum_filter

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

_REF_PCT_DEFAULT  = 98.0       # λ2_ref：ROI 內結構像素 λ2 之百分位
_STRUCT_REL_FLOOR = 1e-3       # 「具二向結構」像素門檻 (相對於 λ2 最大值)


# ═══════════════════════════════════════════════════════
#  Low-level structure-tensor helpers (純 numpy / scipy)
# ═══════════════════════════════════════════════════════

def _structure_fields(gray: np.ndarray, grad_sigma: float, int_sigma: float):
    """計算 ROI 的結構張量場與 Förstner 右端。

    回傳 dict：Sxx,Sxy,Syy（積分後張量分量）、bx,by（梯度收斂右端）、
              lam1,lam2（特徵值場，lam1≥lam2）。
    """
    g = gray.astype(np.float64)
    gs = max(float(grad_sigma), 0.3)
    si = max(float(int_sigma), 0.5)

    Ix = gaussian_filter(g, gs, order=(0, 1), mode="nearest")
    Iy = gaussian_filter(g, gs, order=(1, 0), mode="nearest")
    Jxx, Jxy, Jyy = Ix * Ix, Ix * Iy, Iy * Iy

    Sxx = gaussian_filter(Jxx, si, mode="nearest")
    Sxy = gaussian_filter(Jxy, si, mode="nearest")
    Syy = gaussian_filter(Jyy, si, mode="nearest")

    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]].astype(np.float64)
    bx = gaussian_filter(Jxx * xs + Jxy * ys, si, mode="nearest")
    by = gaussian_filter(Jxy * xs + Jyy * ys, si, mode="nearest")

    tr = Sxx + Syy
    det = Sxx * Syy - Sxy * Sxy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    return dict(Sxx=Sxx, Sxy=Sxy, Syy=Syy, bx=bx, by=by,
                lam1=lam1, lam2=lam2, tr=tr, disc=disc)


def _lam2_reference(lam2: np.ndarray, ref_pct: float) -> float:
    """自適應參考 λ2_ref：只在具二向結構的像素上取高百分位
    （排除平坦與純邊緣，兩者 λ2≈0）。"""
    mx = float(lam2.max())
    if mx <= 0:
        return 1e-12
    vals = lam2[lam2 > _STRUCT_REL_FLOOR * mx]
    if vals.size < 16:
        return max(mx, 1e-9)
    return max(float(np.percentile(vals, float(ref_pct))), 1e-9)


def _forstner_at(fields: dict, vr: int, vc: int, max_shift: float):
    """在 (vr,vc) 解 Förstner 收斂點 N p = b；回傳 (px, py, shift) 或 (vc,vr,nan)。"""
    Sxx = float(fields["Sxx"][vr, vc]); Sxy = float(fields["Sxy"][vr, vc])
    Syy = float(fields["Syy"][vr, vc])
    l1 = float(fields["lam1"][vr, vc]); l2 = float(fields["lam2"][vr, vc])
    if l2 <= 1e-9 * max(l1, 1e-9):
        return float(vc), float(vr), float("nan")
    N = np.array([[Sxx, Sxy], [Sxy, Syy]])
    b = np.array([float(fields["bx"][vr, vc]), float(fields["by"][vr, vc])])
    try:
        p = np.linalg.solve(N, b)
    except np.linalg.LinAlgError:
        return float(vc), float(vr), float("nan")
    shift = float(math.hypot(p[0] - vc, p[1] - vr))
    if shift > max_shift:
        return float(vc), float(vr), float("nan")
    return float(p[0]), float(p[1]), shift


# ═══════════════════════════════════════════════════════
#  Compatibility analysis function (同名、同回傳鍵；改用結構張量)
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
    ref_pct:        float = _REF_PCT_DEFAULT,
) -> dict:
    """以結構張量重寫，回傳鍵與舊版相同（語意換新，名稱沿用以維持相容）：

      harris_R        : λ2 (最小特徵值) 場 —— 新「角點響應」
      harris_mask     : λ2 之 NMS 峰且超過門檻
      harris_thresh   : 門檻值
      steger_ridge    : 高同調度遮罩 (coh > 0.65)，作為線狀證據
      steger_excl     : 同 steger_ridge（不再膨脹）
      steger_strength : 同調度 coh 場
      steger_eigenval : λ1−λ2
      sig_ridge_mask  : 同 steger_ridge
      diff            : quality = clip(λ2/λ2_ref) ∈ [0,1]（≥0；角點高、線/背景低）
      combined        : 同 diff
      threshold       : 0.0
      peaks           : λ2 局部極大（top_k）之 (row, col)
      peak_vals       : 各 peak 的 quality
      dropped_peaks   : []
      inset           : 套用之 inset
    """
    _e = np.zeros((1, 1), np.float32)
    if gray_roi is None or gray_roi.size == 0:
        return dict(harris_R=_e, harris_mask=np.zeros((1, 1), bool),
                    harris_thresh=0.0, steger_ridge=np.zeros((1, 1), bool),
                    steger_excl=np.zeros((1, 1), bool), steger_strength=_e,
                    steger_eigenval=_e, sig_ridge_mask=np.zeros((1, 1), bool),
                    diff=_e, combined=_e, threshold=0.0,
                    peaks=[], peak_vals=[], dropped_peaks=[], inset=0)

    roi_h, roi_w = gray_roi.shape[:2]
    inset = max(0, min(int(inset), (min(roi_h, roi_w) - 7) // 2))
    work = gray_roi[inset:roi_h - inset, inset:roi_w - inset] if inset else gray_roi
    wh, ww = work.shape[:2]

    f = _structure_fields(work, grad_sigma=steger_sigma, int_sigma=harris_sigma)
    lam1, lam2 = f["lam1"], f["lam2"]
    lam2_ref = _lam2_reference(lam2, ref_pct)

    quality = np.clip(lam2 / lam2_ref, 0.0, 1.0).astype(np.float32)
    coh = (f["disc"] / (f["tr"] + 1e-9)).astype(np.float32)      # (λ1−λ2)/(λ1+λ2)
    eigdiff = f["disc"].astype(np.float32)
    ridge = (coh > 0.65)

    # peaks：λ2 之局部極大且 quality 顯著
    nms_r = max(1, int(anms_nms_r))
    lm = maximum_filter(lam2, size=2 * nms_r + 1)
    thr = float(harris_thr_pct) / 100.0 * float(max(lam2.max(), 1e-9))
    peak_mask = (lam2 == lm) & (lam2 > thr) & (quality > 0.0)
    ys, xs = np.where(peak_mask)
    if len(ys) > int(top_k) and len(ys) > 0:
        order = np.argsort(lam2[ys, xs])[::-1][:int(top_k)]
        ys, xs = ys[order], xs[order]
    peaks = [(int(r) + inset, int(c) + inset) for r, c in zip(ys, xs)]
    peak_vals = [float(quality[r, c]) for r, c in zip(ys, xs)]

    def _pad(a, dt=np.float32):
        out = np.zeros((roi_h, roi_w), dt)
        out[inset:inset + wh, inset:inset + ww] = a
        return out

    diff_padded = _pad(quality)
    return dict(
        harris_R        = _pad(lam2.astype(np.float32)),
        harris_mask     = _pad(peak_mask, bool),
        harris_thresh   = thr,
        steger_ridge    = _pad(ridge, bool),
        steger_excl     = _pad(ridge, bool),
        steger_strength = _pad(coh),
        steger_eigenval = _pad(eigdiff),
        sig_ridge_mask  = _pad(ridge, bool),
        diff            = diff_padded,
        combined        = diff_padded,
        threshold       = 0.0,
        peaks           = peaks,
        peak_vals       = peak_vals,
        dropped_peaks   = [],
        inset           = inset,
    )


# ═══════════════════════════════════════════════════════
#  Data classes (欄位與舊版完全相同)
# ═══════════════════════════════════════════════════════

@dataclass
class VertexQualityResult:
    """
    Per-vertex quality score (no grade).

    composite         : dist_weight*dist_score + heatmap_weight*heatmap_score in [0,1]
    dist_score        : exp(-shift/tau)，shift = Förstner 次像素收斂位移，tau = peak_radius_px
    heatmap_score     : quality = clip(λ2/λ2_ref, 0, 1)  (Shi-Tomasi)
    nearest_peak_dist : shift（vertex 到其梯度收斂點的距離，px）
    diff_value        : quality（角點證據；此版 ∈ [0,1]）
    num_peaks         : 1 若 Förstner 成功收斂，否則 0
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
#  VertexQualityScorer   —— 介面不變，內部改用結構張量法
# ═══════════════════════════════════════════════════════

class VertexQualityScorer:
    """
    每個 vertex 獨立評分（加權和，無 grade）：
      composite = dist_weight * exp(-shift/tau) + heatmap_weight * clip(λ2/λ2_ref, 0, 1)

    建構子簽名與舊版相同（保留所有 VQ_* 參數以維持相容）；本版實際使用：
      harris_sigma  → 結構張量積分尺度 (σ_int)
      steger_sigma  → 梯度計算平滑尺度 (σ_grad)
      peak_radius_px→ dist_score 的 tau
      dist_weight / heatmap_weight → composite 權重
      ref_pct       → λ2_ref 百分位 (新增可選參數，預設 98)
    其餘參數接受但不使用（Harris/Steger/ANMS 啟發式已移除）。
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
                 ref_pct:         float = _REF_PCT_DEFAULT,
                 **_):
        self.peak_radius_px = float(peak_radius_px)
        # 使用中的參數（語意換新，名稱沿用）
        self.int_sigma  = float(harris_sigma)     # 結構張量積分尺度
        self.grad_sigma = float(steger_sigma)     # 梯度平滑尺度
        self.ref_pct    = float(ref_pct)
        # 仍保留以維持相容（本版不使用）
        self.harris_k       = float(harris_k)
        self.harris_thr_pct = int(max(1, harris_thr_pct))
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
        """相容用途：以結構張量法回傳與舊版相同鍵的分析結果。"""
        return run_harris_steger_analysis(
            roi_gray, harris_sigma=self.int_sigma, steger_sigma=self.grad_sigma,
            harris_thr_pct=self.harris_thr_pct, top_k=self.top_k,
            inset=self.inset, anms_nms_r=self.anms_nms_r, ref_pct=self.ref_pct)

    def evaluate_vertex(self, vertex_pos_px, roi_gray, roi_origin):
        """
        對單一 vertex 給分（無 grade）。輸入/輸出與舊版相同。
          vertex_pos_px : (x, y) 全域影像座標
          roi_gray      : ROI 灰階 patch
          roi_origin    : (x0, y0) ROI 在全域影像中的左上角
        """
        res = VertexQualityResult()
        if roi_gray is None or roi_gray.size == 0:
            return res
        if roi_gray.shape[0] < 7 or roi_gray.shape[1] < 7:
            return res

        roi_h, roi_w = roi_gray.shape[:2]
        vx = float(vertex_pos_px[0]) - float(roi_origin[0])
        vy = float(vertex_pos_px[1]) - float(roi_origin[1])
        vc = int(np.clip(round(vx), 0, roi_w - 1))
        vr = int(np.clip(round(vy), 0, roi_h - 1))

        # 結構張量場 + 自適應參考 λ2_ref（限定此 ROI）
        f = _structure_fields(roi_gray, grad_sigma=self.grad_sigma,
                              int_sigma=self.int_sigma)
        lam2_ref = _lam2_reference(f["lam2"], self.ref_pct)

        # heatmap_score = quality = clip(λ2 / λ2_ref)  (Shi-Tomasi)
        l2 = float(f["lam2"][vr, vc])
        quality = float(np.clip(l2 / lam2_ref, 0.0, 1.0))
        res.diff_value    = quality
        res.response_val  = quality
        res.heatmap_score = quality

        # dist_score = exp(-shift / tau)，shift = Förstner 收斂位移
        max_shift = max(4.0, 5.0 * self.int_sigma)
        _px, _py, shift = _forstner_at(f, vr, vc, max_shift)
        if math.isnan(shift):
            res.dist_score = 0.0
            res.nearest_peak_dist = float("inf")
            res.dm_value = float("inf")
            res.num_peaks = 0
        else:
            tau = max(self.peak_radius_px, 1e-6)
            res.dist_score = float(np.exp(-shift / tau))
            res.nearest_peak_dist = shift
            res.dm_value = shift
            res.num_peaks = 1

        # Composite（公式骨架不變）
        res.composite = float(np.clip(
            self.dist_weight * res.dist_score +
            self.heatmap_weight * res.heatmap_score, 0.0, 1.0))
        return res


__all__ = ["VertexQualityScorer", "VertexQualityResult",
           "JunctionScore", "PipelineScore", "run_harris_steger_analysis"]