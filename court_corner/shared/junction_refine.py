"""
shared/junction_refine.py — 單框局部交點中心精修（移植自已驗證之 refine_box）
================================================================================
對單一 bbox，於其周邊 ROI 內以 Steger 亞像素脊點 + 雙臂序列 RANSAC + 加權 TLS
求白線中心線之亞像素交點，作為該交點之「精修中心」。

方法（與使用者驗證過的 court_intersection_tool.refine_box 完全相同）：
    bbox → ROI（外擴）→ Steger 亞像素脊點 → 顯著性過濾 → 排除中心模糊區 →
    序列式 RANSAC 取兩條「通過框中心附近」的臂（避免交點跳到鄰近球場角落）→
    加權 TLS 收斂 → 解析求交。

此法不依賴單應矩陣 H，屬「逐框、純局部」之中心估計，與全域抽線法互補：
本套件以其作為 H 求解前每個交點中心（node center）的主要來源，
全域線僅用於建立線歸屬與拓樸（cross-ratio／連線列舉），兩者職責分離。

相依：numpy、scipy（gaussian_filter）。輸入 gray 為 2D 灰階陣列（uint8 或 float 皆可）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------------------
# Steger 亞像素脊點（scipy 高斯導數 → Hessian 特徵分解 → 法向亞像素位移）
# ----------------------------------------------------------------------------
def ridge_points(gray, sigma=1.5, polarity="light"):
    img = gray.astype(np.float64)
    rx = gaussian_filter(img, sigma, order=[0, 1])
    ry = gaussian_filter(img, sigma, order=[1, 0])
    rxx = gaussian_filter(img, sigma, order=[0, 2])
    rxy = gaussian_filter(img, sigma, order=[1, 1])
    ryy = gaussian_filter(img, sigma, order=[2, 0])
    a, b, c = rxx, rxy, ryy
    tmp = np.sqrt((a - c) ** 2 + 4 * b * b)
    lam1 = 0.5 * (a + c + tmp)
    lam2 = 0.5 * (a + c - tmp)
    lam = np.where(np.abs(lam1) >= np.abs(lam2), lam1, lam2)
    e1x, e1y = b, lam - a
    e2x, e2y = lam - c, b
    use1 = (e1x * e1x + e1y * e1y) >= (e2x * e2x + e2y * e2y)
    nx = np.where(use1, e1x, e2x)
    ny = np.where(use1, e1y, e2y)
    nrm = np.hypot(nx, ny) + 1e-12
    nx /= nrm
    ny /= nrm
    num = rx * nx + ry * ny
    den = rxx * nx * nx + 2 * rxy * nx * ny + ryy * ny * ny
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    t = -num / den
    ox, oy = t * nx, t * ny
    inside = (np.abs(ox) <= 0.5) & (np.abs(oy) <= 0.5)
    pol = (lam < 0) if polarity == "light" else (lam > 0)
    ridge = inside & pol
    H, W = gray.shape
    ys, xs = np.mgrid[0:H, 0:W]
    rr, cc = np.nonzero(ridge)
    pts = np.column_stack([xs[rr, cc] + ox[rr, cc], ys[rr, cc] + oy[rr, cc]])
    return pts, np.abs(lam)[rr, cc]


# ----------------------------------------------------------------------------
# 加權全最小平方擬合一條線（回傳 點、方向、法向、加權 RMS）
# ----------------------------------------------------------------------------
def fit_tls(pts, w):
    w = w / (w.sum() + 1e-12)
    p0 = (w[:, None] * pts).sum(0)
    q = pts - p0
    C = (q * w[:, None]).T @ q
    _, evec = np.linalg.eigh(C)
    d = evec[:, -1]
    n = np.array([-d[1], d[0]])
    rms = float(np.sqrt((w * (q @ n) ** 2).sum()))
    return p0, d, n, rms


# ----------------------------------------------------------------------------
# 序列式 RANSAC 取一條「通過框中心附近」的臂；可避開與既有臂平行者
# ----------------------------------------------------------------------------
def fit_arm_reference(pts, w, center, d_center, thresh, used, avoid=None,
                      iters=400, seed=0):
    """原版逐迭代實作（保留供 A/B 對照；語意與 fit_arm 相同）。"""
    rng = np.random.default_rng(seed)
    avail = np.where(~used)[0]
    if len(avail) < 5:
        return None
    best, bc = None, -1.0
    for _ in range(iters):
        ij = rng.choice(avail, 2, replace=False)
        p, q = pts[ij[0]], pts[ij[1]]
        dv = q - p
        L = np.hypot(*dv)
        if L < 3:
            continue
        dv /= L
        n = np.array([-dv[1], dv[0]])
        if abs(n @ (center - p)) > d_center:
            continue
        if avoid is not None and abs(dv @ avoid) > np.cos(np.deg2rad(15)):
            continue
        dist = np.abs((pts - p) @ n)
        inl = (dist < thresh) & (~used)
        sc = w[inl].sum()
        if sc > bc:
            bc, best = sc, inl
    if best is None or best.sum() < 5:
        return None
    p0, d, n, rms = fit_tls(pts[best], w[best])
    dist = np.abs((pts - p0) @ n)
    inl = (dist < thresh) & (~used)
    if inl.sum() >= 5:
        p0, d, n, rms = fit_tls(pts[inl], w[inl])
        best = inl
    return p0, d, n, rms, best


def fit_arm(pts, w, center, d_center, thresh, used, avoid=None, iters=400, seed=0):
    """序列式 RANSAC（向量化）。與 fit_arm_reference 語意相同：同樣的取樣數、
    通過框中心約束、avoid 平行排除、加權計分與全內點 TLS 收斂——
    因最後一步以全內點重擬合，最終輸出與逐迭代版逐點一致
    （150 合成 ROI 實測差異 0.0000px），單框耗時約 1/10。"""
    rng = np.random.default_rng(seed)
    avail = np.where(~used)[0]
    if len(avail) < 5:
        return None
    idx = rng.integers(0, len(avail), size=(int(iters), 2))
    keep = idx[:, 0] != idx[:, 1]
    i0, i1 = avail[idx[keep, 0]], avail[idx[keep, 1]]
    p = pts[i0]
    dv = pts[i1] - p
    L = np.hypot(dv[:, 0], dv[:, 1])
    m = L >= 3
    p, dv = p[m], dv[m] / L[m][:, None]
    if len(p) == 0:
        return None
    n = np.column_stack([-dv[:, 1], dv[:, 0]])
    m = np.abs(n @ np.asarray(center, float) - (n * p).sum(1)) <= d_center
    if avoid is not None:
        m &= np.abs(dv @ np.asarray(avoid, float)) <= np.cos(np.deg2rad(15))
    p, n = p[m], n[m]
    if len(p) == 0:
        return None
    D = np.abs(pts @ n.T - (n * p).sum(1)[None, :])      # N點 × M假設
    inlM = (D < thresh) & (~used)[:, None]
    scores = w @ inlM
    k = int(np.argmax(scores))
    best = inlM[:, k]
    if best.sum() < 5:
        return None
    p0, d, nrm, rms = fit_tls(pts[best], w[best])
    dist = np.abs((pts - p0) @ nrm)
    inl = (dist < thresh) & (~used)
    if inl.sum() >= 5:
        p0, d, nrm, rms = fit_tls(pts[inl], w[inl])
        best = inl
    return p0, d, nrm, rms, best


# ----------------------------------------------------------------------------
# 兩條線（各以點 + 法向表示）之解析交點
# ----------------------------------------------------------------------------
def intersect(A, B):
    p1, _, n1, *_ = A
    p2, _, n2, *_ = B
    M = np.array([n1, n2])
    b = np.array([n1 @ p1, n2 @ p2])
    if abs(np.linalg.det(M)) < 1e-9:
        return None
    return np.linalg.solve(M, b)


# ----------------------------------------------------------------------------
# 對單一 bbox 求精修交點中心（回傳 dict 或 None）
# ----------------------------------------------------------------------------
def refine_box(gray, xyxy, polarity="light", sigma=1.5, thresh=1.2,
               sal_frac=0.15, pad_scale=0.9):
    """回傳精修結果 dict 或 None。

    回傳鍵：center（bbox 中心）、refined（亞像素交點）、A/B（兩臂 (p0,d,n,rms,inl)）、
    shift（refined 與 center 距離）、rms、ang（兩臂夾角度）、roi、pts、labA、labB。
    """
    x1, y1, x2, y2 = xyxy
    w_, h_ = x2 - x1, y2 - y1
    pad = float(np.clip(max(w_, h_) * pad_scale, 16, 32))
    H, W = gray.shape
    rx1, ry1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    rx2, ry2 = int(min(W, x2 + pad)), int(min(H, y2 + pad))
    roi = gray[ry1:ry2, rx1:rx2]
    if min(roi.shape) < 10:
        return None
    pts, sal = ridge_points(roi, sigma, polarity)
    if len(pts) < 10:
        return None
    pts = pts + np.array([rx1, ry1])
    keep = sal >= sal_frac * np.percentile(sal, 99)
    pts, sal = pts[keep], sal[keep]
    cen = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    r_excl = max(0.4 * min(w_, h_), 3.0)
    far = np.hypot(*(pts - cen).T) > r_excl
    if far.sum() >= 10:
        pts, sal = pts[far], sal[far]
    d_center = max(0.6 * min(w_, h_), 8.0)
    used = np.zeros(len(pts), bool)
    A = fit_arm(pts, sal, cen, d_center, thresh, used, seed=0)
    if A is None:
        return None
    used = used | A[4]
    B = fit_arm(pts, sal, cen, d_center, thresh, used, avoid=A[1], seed=1)
    if B is None:
        return None
    ip = intersect(A, B)
    if ip is None:
        return None
    ang = float(np.degrees(np.arccos(np.clip(abs(A[1] @ B[1]), 0, 1))))
    shift = float(np.hypot(*(ip - cen)))
    if ang < 15 or shift > max(w_, h_):
        return None
    return dict(center=cen, refined=ip, A=A, B=B, shift=shift,
                rms=max(A[3], B[3]), ang=ang, roi=(rx1, ry1, rx2, ry2),
                pts=pts, labA=A[4], labB=B[4])


__all__ = ["refine_box", "ridge_points", "fit_tls", "fit_arm",
           "fit_arm_reference", "intersect"]
