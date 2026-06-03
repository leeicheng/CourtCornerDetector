"""
shared/steger.py — Steger (1998) 次像素脊線偵測原語
================================================================
原始程式碼由未隨附的 `s1_detection.steger_center` 提供下列原語，此處依
Steger 1998 的 Hessian 脊線數學（與 vertex_quality._steger_ridges 同源）
重建，使 Stage 2 的 H 精修與 Stage 3 的 Steger 角點搜尋可獨立運作。

對外提供（與原始 import 介面一致）：
  _steger_ridge_points_simple(roi, sigma, threshold_pct, bright_lines,
                              return_tangents=False)
  _line_intersection_simple(line1, line2)   # parametric (vx,vy,x0,y0)
  _axis_angle_deg(v)
  以及 deterministic-voting 後援所需的占位函式（回傳空，使 caller 自動
  落到 direction-clustering 路徑）。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────
# Steger ridge points（次像素 + 切線方向 + 強度）
# ──────────────────────────────────────────────────────────────
def _steger_ridge_points_simple(
    roi: np.ndarray,
    sigma: float = 1.2,
    threshold_pct: float = 0.15,
    bright_lines: bool = True,
    return_tangents: bool = False,
    mask: np.ndarray = None,
):
    """
    Steger 脊線偵測，回傳次像素脊點。

    統一實作：完全依賴 OpenCV（GaussianBlur + Sobel），不再使用
    scipy.ndimage.gaussian_filter，以降低卷積成本並消除與 court_lines.py
    的重複實作。數學與 Steger 1998 Hessian 脊線一致。

    Args:
        roi            : 灰階影像 (H, W)，uint8 或 float
        sigma          : 高斯導數尺度
        threshold_pct  : 強度門檻（佔最大強度的「比例」，非百分比；如 0.15）
        bright_lines   : True = 亮線（白線，λ<0）；False = 暗線（λ>0）
        return_tangents: True 則同時回傳切線方向與強度
        mask           : 同 roi 尺寸，非 0 處才接受脊點（YOLO 導引動態遮罩）

    Returns:
        return_tangents=False : pts (N,2) float32，影像座標 (x, y)
        return_tangents=True  : (pts (N,2), tangents (N,2), strengths (N,))
    """
    if roi is None or roi.size == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        if return_tangents:
            return empty, empty, np.zeros((0,), dtype=np.float32)
        return empty

    g = roi.astype(np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)

    # 高斯平滑 + Sobel 導數（OpenCV，C-level；取代 scipy gaussian_filter）
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    Lx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    Ly = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    Lxx = cv2.Sobel(g, cv2.CV_64F, 2, 0, ksize=3)
    Lyy = cv2.Sobel(g, cv2.CV_64F, 0, 2, ksize=3)
    Lxy = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)

    # Hessian 特徵值（取絕對值較大者為主曲率方向 = 脊線法向）
    tr = Lxx + Lyy
    disc = np.sqrt(np.maximum(((Lxx - Lyy) / 2.0) ** 2 + Lxy ** 2, 0.0))
    lam1 = tr / 2.0 + disc
    lam2 = tr / 2.0 - disc
    lam = np.where(np.abs(lam2) > np.abs(lam1), lam2, lam1)

    # 主曲率方向（脊線法向）的特徵向量。
    # 對稱矩陣 [[Lxx,Lxy],[Lxy,Lyy]] 的特徵向量有兩個等價表示式：
    #   v1 = (Lxy, lam - Lxx)      v2 = (lam - Lyy, Lxy)
    # 兩者成比例，但在軸對齊脊線（Lxy≈0）時其中一個會退化為零向量，
    # 故取範數較大者以保證數值穩定（修正垂直/水平線切線方向錯誤）。
    v1x, v1y = Lxy, lam - Lxx
    v2x, v2y = lam - Lyy, Lxy
    n1 = v1x ** 2 + v1y ** 2
    n2 = v2x ** 2 + v2y ** 2
    use_v1 = n1 >= n2
    nx_r = np.where(use_v1, v1x, v2x)
    ny_r = np.where(use_v1, v1y, v2y)
    nnorm = np.sqrt(nx_r ** 2 + ny_r ** 2) + 1e-12
    nx = nx_r / nnorm
    ny = ny_r / nnorm

    # 沿法向的次像素位移 t
    denom = (Lxx * nx ** 2 + 2.0 * Lxy * nx * ny + Lyy * ny ** 2)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t = -(Lx * nx + Ly * ny) / denom

    strength = np.abs(lam)
    smax = float(strength.max()) if strength.size and strength.max() > 0 else 1.0
    thresh = float(threshold_pct) * smax

    # 亮線：crest 處跨向二階導為負 → 主特徵值 λ<0；暗線相反
    sign_ok = (lam < 0) if bright_lines else (lam > 0)

    ridge = (np.abs(t) <= 0.5) & (strength > thresh) & sign_ok
    if mask is not None and mask.shape[:2] == g.shape[:2]:
        ridge &= (mask > 0)

    ys, xs = np.where(ridge)
    if len(ys) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        if return_tangents:
            return empty, empty, np.zeros((0,), dtype=np.float32)
        return empty

    # 次像素位置 = 整數座標 + t * 法向
    px = xs.astype(np.float64) + t[ys, xs] * nx[ys, xs]
    py = ys.astype(np.float64) + t[ys, xs] * ny[ys, xs]
    pts = np.stack([px, py], axis=1).astype(np.float32)

    if not return_tangents:
        return pts

    # 切線 = 法向旋轉 90°，並對齊半平面（±tangent 同一條線，符號不重要）
    tx = -ny[ys, xs]
    ty = nx[ys, xs]
    tangents = np.stack([tx, ty], axis=1).astype(np.float32)
    strengths = strength[ys, xs].astype(np.float32)
    return pts, tangents, strengths


# ──────────────────────────────────────────────────────────────
# 幾何小工具
# ──────────────────────────────────────────────────────────────
def _line_intersection_simple(line1, line2) -> Optional[np.ndarray]:
    """兩條 parametric 直線 (vx, vy, x0, y0) 的交點；近平行回傳 None。"""
    vx1, vy1, x1, y1 = [float(v) for v in line1[:4]]
    vx2, vy2, x2, y2 = [float(v) for v in line2[:4]]
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float64)
    if abs(float(np.linalg.det(A))) < 1e-9:
        return None
    b = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    try:
        t, _ = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float64)


def _axis_angle_deg(v) -> float:
    """向量在影像中的角度（度，mod 180）。"""
    a = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
    return a


# ──────────────────────────────────────────────────────────────
# Deterministic-voting 後援占位
#   原始 S1 的 voting 線對選擇器未隨附；回傳空集合即可讓
#   StegerVertexFinder 自動落到 direction-clustering（自包含、可重現）路徑。
# ──────────────────────────────────────────────────────────────
def _build_line_candidates_by_angle_offset(pts, tangents=None, strengths=None,
                                           min_points: int = 5, **_):
    return []


def _score_line_pair(*_args, **_kwargs):
    return 0.0


def _select_line_pair_by_voting(candidates, pts, bbox_center=None,
                                junction_type="X", min_angle_deg=35.0, **_):
    return None, None, None, None, {}


__all__ = [
    "_steger_ridge_points_simple",
    "_line_intersection_simple",
    "_axis_angle_deg",
    "_build_line_candidates_by_angle_offset",
    "_score_line_pair",
    "_select_line_pair_by_voting",
]
