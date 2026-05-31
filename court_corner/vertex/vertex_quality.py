"""
Vertex Quality Scorer — Harris + Steger + ANMS + Ridge Proximity Filter
========================================================================
每個 vertex 給一個連續分數（0~1），不再有 A/B/C/D/F Grade。
下游自己決定如何聚合或篩選。

演算法流程（per ROI）：
  Stage 1  Harris corner response
    R = Det(M) − k·Tr(M)²
  Stage 2  Steger (1998) ridge detector
    ridge_mask, exclusion_zone, strength (= |lambda_ridge|)
  Stage 3  Feature-Ridge Diff Map
    diff = norm(max(R, 0)) − norm(strength)   ∈ [-1, 1]
      > 0: corner 強於 ridge → 可靠 junction
      < 0: ridge 強於 corner → 純線段
  Stage 4  Adaptive NMS (Brown et al. 2005)
    per-candidate suppression radius r_i = min dist(p_i, p_j) over {j: v_i < c*v_j}
    取前 top_k 個（空間分佈感知，比固定 NMS 更均勻）
  Stage 5  Ridge Proximity Filter
    dist(peak, nearest significant-ridge pixel) ≤ max_dist 才保留
    （junction 必在白線附近，丟棄背景雜訊）

Vertex Scoring（per vertex, 無 grade）：
  dist_score    = exp(-d / tau)            d = 到最近 final peak 的距離
                                           tau = peak_radius_px
  heatmap_score = clip(diff[vertex], 0, 1)   負值歸零（ridge-dominant）
  composite     = alpha*dist_score + (1-alpha)*heatmap_score   預設 alpha=0.5

對外介面：evaluate_vertex 方法。
"""
from __future__ import annotations

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


# ═══════════════════════════════════════════════════════
#  Low-level algorithm helpers (ported from uploaded core.py)
# ═══════════════════════════════════════════════════════

def _structure_tensor(gray_f64: np.ndarray, sigma: float):
    """Gaussian-weighted outer product of grad I. Returns (Ixx, Iyy, Ixy)."""
    Ix = cv2.Sobel(gray_f64, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_f64, cv2.CV_64F, 0, 1, ksize=3)
    return (gaussian_filter(Ix * Ix, sigma),
            gaussian_filter(Iy * Iy, sigma),
            gaussian_filter(Ix * Iy, sigma))


def _harris_R_fn(Ixx, Iyy, Ixy, k):
    """R = Det(M) - k * Tr(M)^2"""
    return (Ixx * Iyy - Ixy ** 2 - k * (Ixx + Iyy) ** 2).astype(np.float32)


def _disk_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2 <= radius ** 2).astype(np.uint8)


def _steger_ridges(gray_u8: np.ndarray, sigma, thr_pct, dil_r, dark):
    """
    Steger (1998) ridge detector with sub-pixel verification.
    Returns (ridge_mask, exclusion_zone, strength, eigenval).
    """
    g   = gray_u8.astype(np.float64)
    Lx  = gaussian_filter(g, sigma, order=(0, 1))
    Ly  = gaussian_filter(g, sigma, order=(1, 0))
    Lxx = gaussian_filter(g, sigma, order=(0, 2))
    Lyy = gaussian_filter(g, sigma, order=(2, 0))
    Lxy = gaussian_filter(g, sigma, order=(1, 1))

    tr   = Lxx + Lyy
    disc = np.sqrt(np.maximum(((Lxx - Lyy) / 2) ** 2 + Lxy ** 2, 0.0))
    lam1 = tr / 2 + disc
    lam2 = tr / 2 - disc
    lam  = np.where(np.abs(lam2) > np.abs(lam1), lam2, lam1)

    nx_r = Lxy; ny_r = lam - Lxx
    nnorm = np.sqrt(nx_r ** 2 + ny_r ** 2) + 1e-12
    nx, ny = nx_r / nnorm, ny_r / nnorm
    t = -(Lx * nx + Ly * ny) / (Lxx * nx**2 + 2 * Lxy * nx * ny + Lyy * ny**2 + 1e-12)

    strength = np.abs(lam).astype(np.float32)
    smax     = float(strength.max()) if strength.max() > 0 else 1.0
    thresh   = thr_pct / 100.0 * smax
    sign_ok  = (lam < 0) if dark else (lam > 0)
    ridge    = (np.abs(t) <= 0.5) & (strength > thresh) & sign_ok
    excl     = binary_dilation(ridge, structure=_disk_kernel(dil_r)) if dil_r > 0 else ridge.copy()
    return ridge, excl, strength, lam.astype(np.float32)


def _feature_ridge_diff(harris_R: np.ndarray,
                        steger_strength: np.ndarray) -> np.ndarray:
    """diff = norm(max(R, 0)) - norm(strength)  in [-1, 1]"""
    feat  = np.clip(harris_R, 0, None).astype(np.float32)
    feat  = feat / (feat.max() + 1e-9)
    ridge = steger_strength / (steger_strength.max() + 1e-9)
    return (feat - ridge).astype(np.float32)


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
    """Keep only connected components of ridge_mask with area >= threshold."""
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


def _filter_by_ridge_proximity(ys, xs, vals, sig_mask, max_dist):
    """Drop peaks farther than max_dist from any significant-ridge pixel."""
    if len(ys) == 0:
        e = np.array([], int);  ef = np.array([], float)
        return e, e, ef, e, e
    if not sig_mask.any():
        e = np.array([], int);  ef = np.array([], float)
        return e, e, ef, ys.copy(), xs.copy()

    dist = distance_transform_edt(~sig_mask.astype(bool))
    peak_dists = dist[ys, xs]
    keep = peak_dists <= max_dist
    drop = ~keep
    return (ys[keep], xs[keep], vals[keep],
            ys[drop], xs[drop])


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
    End-to-end Harris + Steger + ANMS + ridge-proximity pipeline.

    Returns (all arrays padded back to original ROI size):
      harris_R         : (H,W) float32  Harris response
      harris_mask      : (H,W) bool     NMS+threshold corners (reference)
      harris_thresh    : float
      steger_ridge     : (H,W) bool     sub-pixel verified ridge pixels
      steger_excl      : (H,W) bool     dilated exclusion zone
      steger_strength  : (H,W) float32  |lambda_ridge|
      steger_eigenval  : (H,W) float32  signed lambda_ridge
      sig_ridge_mask   : (H,W) bool     significant-component ridge mask
      diff             : (H,W) float32  norm(R) - norm(strength) in [-1, 1]
      peaks            : list[(row, col)]  FINAL peaks after ANMS + proximity
      peak_vals        : list[float]       diff value at each peak
      dropped_peaks    : list[(row, col)]  ANMS peaks dropped by proximity
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
    wu8   = np.clip(work, 0, 255).astype(np.uint8)

    # Stage 1: Harris R
    Ixx, Iyy, Ixy = _structure_tensor(wu8.astype(np.float64), harris_sigma)
    R      = _harris_R_fn(Ixx, Iyy, Ixy, harris_k)
    h_thr  = harris_thr_pct / 100.0 * float(max(R.max(), 1e-9))
    R_pos  = np.where(R > 0, R, 0.0).astype(np.float32)
    lm_h   = maximum_filter(R_pos, size=2 * anms_nms_r + 1)
    h_mask = (R_pos == lm_h) & (R > h_thr)

    # Stage 2: Steger
    ridge, excl, strength, eigenval = _steger_ridges(
        wu8, steger_sigma, steger_thr_pct, steger_dil_r, dark_ridges)

    # Stage 3: diff map
    diff = _feature_ridge_diff(R, strength)

    # Stage 4: ANMS peaks
    ys_a, xs_a, vals_a = _topk_anms(
        diff, top_k=top_k, c=anms_c,
        candidate_pool=anms_pool, loose_nms_radius=anms_nms_r)

    # Stage 5: ridge proximity filter
    if prox_enabled and len(ys_a) > 0:
        sig_mask = _significant_ridge_mask(ridge, prox_min_area, prox_close_r)
        ys_k, xs_k, vals_k, ys_d, xs_d = _filter_by_ridge_proximity(
            ys_a, xs_a, vals_a, sig_mask, prox_max_dist)
    else:
        sig_mask = np.zeros_like(ridge, dtype=bool)
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
        steger_eigenval = _pad(eigenval),
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
#  VertexQualityScorer
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
        對單一 vertex 給分（無 grade）。
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


