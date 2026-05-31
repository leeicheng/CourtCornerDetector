"""
stage2_topology.py — 第二階段：拓樸求解
================================================================
依「偵測交點 + 模板」估計單應矩陣 H（template 公尺 → 影像像素），並建立
每個偵測交點 → 模板節點 id (0..29) 的拓樸對應關係。

設計：
  原始 court_homography_tool 以「球場線 cross-ratio 標號」為主、PROSAC 為輔，
  仰賴全域 Steger 抽線（未隨附）。本求解器改為**純點式**，不需抽線：

    1. 種子：取偵測點雲在四對角方向的極值點 → 近似球場四角，對應模板四個
       L 角 {idx 0,4,29,25}。窮舉 8 種 dihedral 排列（旋轉×鏡射）解出候選 H。
    2. 驗證/精修：對每個候選做 type-aware NN 指派（型別懲罰貪婪配對）+ DLT
       guided refit 迭代，並以 grid-twist（不折疊）、type-consistency（型別一致）、
       inlier 數、RMSE、orientation 慣例綜合評分，取最佳。
    3. 後援：種子失敗時，用型別約束的取樣 RANSAC（隨機四點 ↔ 模板四點）。
    4. （可選）Jacobian 導引的雙軸 Steger 次像素精修 H。

  以上驗證器（grid-twist / type-consistency / NN 指派 / guided refit / orient /
  DLT）皆移植自原始 court_homography_tool.py。
"""

from __future__ import annotations

import math
import itertools
from typing import List, Optional, Tuple

import numpy as np
import cv2

from ..shared.steger import _steger_ridge_points_simple, _line_intersection_simple
from .. import config


# ──────────────────────────────────────────────────────────────
# 模板常數（與 court_model 一致；junction_idx = row*5 + col）
# ──────────────────────────────────────────────────────────────
COL_X = [0.00, 0.46, 3.05, 5.64, 6.10]
ROW_Y = [13.40, 12.64, 8.68, 4.72, 0.76, 0.00]
N_COL, N_ROW = 5, 6
TEMPLATE_TYPES = [
    0, 1, 1, 1, 0,
    1, 2, 2, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 2, 2, 1,
    0, 1, 1, 1, 0,
]
TYPE_ID_TO_NAME = {0: "L", 1: "T", 2: "X"}
# 四角模板 index（順時針：TL, TR, BR, BL）
CORNER_IDS_CW = [0, 4, 29, 25]


def _tpl_type(r, c):
    return TYPE_ID_TO_NAME[TEMPLATE_TYPES[r * N_COL + c]]


def _tpl_xy(r, c):
    return (COL_X[c], ROW_Y[r])


# ──────────────────────────────────────────────────────────────
# DLT / 投影（移植自 court_homography_tool）
# ──────────────────────────────────────────────────────────────
def _normalize(pts):
    P = np.asarray(pts, float)
    c = P.mean(0)
    d = np.sqrt(((P - c) ** 2).sum(1)).mean()
    s = math.sqrt(2) / d if d > 1e-9 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    Ph = (T @ np.c_[P, np.ones(len(P))].T).T
    return Ph[:, :2], T


def _dlt(src, dst):
    if len(src) < 4:
        return None
    s, Ts = _normalize(src)
    d, Td = _normalize(dst)
    A = []
    for (X, Y), (x, y) in zip(s, d):
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x])
    try:
        _, _, vt = np.linalg.svd(np.asarray(A, float))
    except np.linalg.LinAlgError:
        return None
    H = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _proj(H, X):
    v = H @ np.array([X[0], X[1], 1.0])
    if abs(v[2]) < 1e-12:
        return (float("inf"), float("inf"))
    return (v[0] / v[2], v[1] / v[2])


def _signed_area(quad):
    x = [p[0] for p in quad]
    y = [p[1] for p in quad]
    return 0.5 * sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))


def _grid_twist_ok(H, max_ratio=None):
    """投影 30 點，檢查每個基本格有號面積同號（不折疊）、面積比不過大。"""
    if max_ratio is None:
        max_ratio = config.S2_GRID_TWIST_MAX_RATIO
    proj = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if not (math.isfinite(p[0]) and math.isfinite(p[1])):
                return False
            proj[(r, c)] = p
    signs, areas = [], []
    for r in range(N_ROW - 1):
        for c in range(N_COL - 1):
            quad = [proj[(r, c)], proj[(r, c + 1)], proj[(r + 1, c + 1)], proj[(r + 1, c)]]
            a = _signed_area(quad)
            if not math.isfinite(a):
                return False
            signs.append(0 if a == 0 else (1 if a > 0 else -1))
            areas.append(abs(a))
    if any(s == 0 for s in signs):
        return False
    if any(s > 0 for s in signs) and any(s < 0 for s in signs):
        return False
    arr = np.asarray(areas, float)
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))
    if p10 <= 1e-6 or p90 / p10 > max_ratio:
        return False
    return True


def _orient_score(H):
    """方向慣例：template +x → 影像右、+y → 影像下，分數越高越合慣例。"""
    cen = (3.05, 6.70)
    p0 = _proj(H, cen)
    px = _proj(H, (cen[0] + 1, cen[1]))
    py = _proj(H, (cen[0], cen[1] + 1))
    jx = np.subtract(px, p0)
    jy = np.subtract(py, p0)
    if not (np.all(np.isfinite(jx)) and np.all(np.isfinite(jy))):
        return -1e9
    jx = jx / (np.linalg.norm(jx) + 1e-9)
    jy = jy / (np.linalg.norm(jy) + 1e-9)
    return float(jx[0] + jy[1])


def _type_consistency(H, node_pts, node_types, subset=None, thr=None):
    """型別一致性：每個 inlier 以純位置 NN 指到最近投影格，看型別是否相符。"""
    cells = []
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                cells.append((_tpl_type(r, c), p))
    if not cells:
        return 0.0, 0, 0
    CT = [t for t, _ in cells]
    PP = np.array([p for _, p in cells], float)
    idxs = list(subset) if subset is not None else list(range(len(node_pts)))
    sx = [node_pts[i] for i in idxs]
    if not sx:
        return 0.0, 0, 0
    span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                       max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
    if thr is None:
        thr = max(8.0, 0.04 * span)
    match = tot = 0
    for i in idxs:
        x, y = node_pts[i]
        d = np.hypot(PP[:, 0] - x, PP[:, 1] - y)
        j = int(d.argmin())
        if d[j] > thr:
            continue
        tot += 1
        if CT[j] == node_types[i]:
            match += 1
    return (match / tot if tot else 0.0), match, tot


def _nn_assign(node_pts, node_types, H, thr, subset=None):
    """型別懲罰貪婪 NN 指派：回傳 (inlier idxs, rmse)。"""
    proj = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                proj[(r, c)] = (p, _tpl_type(r, c))
    idxs = subset if subset is not None else range(len(node_pts))
    pairs = []
    for di in idxs:
        pt = node_pts[di]
        t = node_types[di]
        for (rc, (pp, tt)) in proj.items():
            d = math.hypot(pp[0] - pt[0], pp[1] - pt[1])
            cost = d + (0.0 if tt == t else thr)
            if cost <= thr * 3:
                pairs.append((cost, di, rc, d))
    pairs.sort()
    used_d, used_t, inl, errs = set(), set(), [], []
    for cost, di, rc, d in pairs:
        if di in used_d or rc in used_t:
            continue
        used_d.add(di)
        used_t.add(rc)
        if d <= thr:
            inl.append(di)
            errs.append(d)
    rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf")
    return inl, rmse


def _nn_match(node_pts, node_types, H, thr, subset=None):
    """同 _nn_assign，但回傳 (node_idx, 對應 template metric 座標, dist)。"""
    proj = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                proj[(r, c)] = (p, _tpl_xy(r, c), _tpl_type(r, c))
    idxs = subset if subset is not None else range(len(node_pts))
    pairs = []
    for di in idxs:
        pt = node_pts[di]
        t = node_types[di]
        for (rc, (pp, mp, tt)) in proj.items():
            d = math.hypot(pp[0] - pt[0], pp[1] - pt[1])
            cost = d + (0.0 if tt == t else thr)
            if cost <= thr * 3:
                pairs.append((cost, di, rc, mp, d))
    pairs.sort()
    used_d, used_t, matches = set(), set(), []
    for cost, di, rc, mp, d in pairs:
        if di in used_d or rc in used_t:
            continue
        used_d.add(di)
        used_t.add(rc)
        if d <= thr:
            matches.append((di, mp, d))
    return matches


def _guided_refit(H, node_pts, node_types, sub, span, iters=None):
    """全 inlier DLT 重擬合迭代，直到 inlier 不增、rmse 不降。"""
    if iters is None:
        iters = config.S2_GUIDED_REFIT_ITERS
    thr = max(config.S2_NN_INLIER_MIN_PX, config.S2_NN_INLIER_RATIO * span)
    bi, brm = _nn_assign(node_pts, node_types, H, thr, subset=sub)
    best = (H, bi, brm)
    curH = H
    for _ in range(iters):
        m = _nn_match(node_pts, node_types, curH, thr * 1.3, subset=sub)
        if len(m) < 4:
            break
        newH = _dlt([mp for (_di, mp, _d) in m], [node_pts[di] for (di, _mp, _d) in m])
        if newH is None or not np.all(np.isfinite(newH)) or not _grid_twist_ok(newH):
            break
        inl, rmse = _nn_assign(node_pts, node_types, newH, thr, subset=sub)
        if len(inl) > len(best[1]) or (len(inl) == len(best[1]) and rmse < best[2] - 1e-9):
            best, curH = (newH, inl, rmse), newH
        else:
            break
    return best


# ──────────────────────────────────────────────────────────────
# Jacobian 導引雙軸 Steger 次像素 H 精修（移植自 court_homography_tool）
# ──────────────────────────────────────────────────────────────
def _assign_cells_simple(H, node_pts, node_types, subset, thr):
    cells = []
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                cells.append((r, c, p, _tpl_type(r, c)))
    pairs = []
    for i in subset:
        x, y = node_pts[i]
        t = node_types[i]
        for (r, c, p, tt) in cells:
            d = math.hypot(p[0] - x, p[1] - y)
            cost = d + (0.0 if tt == t else thr)
            if cost <= thr * 3:
                pairs.append((cost, i, (r, c), d))
    pairs.sort()
    used_i, used_c, out = set(), set(), {}
    for cost, i, rc, d in pairs:
        if i in used_i or rc in used_c:
            continue
        if d <= thr:
            used_i.add(i)
            used_c.add(rc)
            out[i] = rc
    return out


def refine_homography_steger(H, img_bgr, node_pts, node_types, subset,
                             bright_lines=True, iters=2, eps=0.05):
    """Jacobian 導引雙軸 Steger 次像素精修 H。回傳 (H_refined, n_refined)。"""
    if img_bgr is None:
        return np.asarray(H, float), 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    Himg, Wimg = gray.shape[:2]
    curH = np.asarray(H, float)
    n_ref = 0
    for _ in range(max(1, iters)):
        sx = [node_pts[i] for i in subset]
        if len(sx) < 4:
            break
        span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                           max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
        thr = max(6.0, 0.02 * span)
        label = _assign_cells_simple(curH, node_pts, node_types, subset, thr)
        if len(label) < 4:
            break
        corr, n_ref = [], 0
        for i, (r, c) in label.items():
            X, Y = _tpl_xy(r, c)
            p0 = np.array(_proj(curH, (X, Y)), float)
            pr = np.array(_proj(curH, (X + eps, Y)), float)
            pc = np.array(_proj(curH, (X, Y + eps)), float)
            a_row = (pr - p0) / eps
            a_col = (pc - p0) / eps
            nr = np.linalg.norm(a_row)
            nc = np.linalg.norm(a_col)
            fallback = (node_pts[i] if i < len(node_pts) else (float(p0[0]), float(p0[1])))
            if nr < 1e-6 or nc < 1e-6 or not all(np.isfinite(p0)):
                corr.append(((X, Y), fallback))
                continue
            ur = a_row / nr
            uc = a_col / nc
            ext_r = min(max(0.35 * nr, 10.0), 60.0)
            ext_c = min(max(0.35 * nc, 10.0), 60.0)
            corners = [p0 + sr * ext_r * ur + sc * ext_c * uc
                       for sr in (-1, 1) for sc in (-1, 1)]
            xs = [q[0] for q in corners]
            ys = [q[1] for q in corners]
            x0 = int(max(0, min(xs) - 3))
            x1 = int(min(Wimg, max(xs) + 3))
            y0 = int(max(0, min(ys) - 3))
            y1 = int(min(Himg, max(ys) + 3))
            if x1 - x0 < 6 or y1 - y0 < 6:
                corr.append(((X, Y), fallback))
                continue
            rp = _steger_ridge_points_simple(gray[y0:y1, x0:x1], sigma=1.2,
                                             threshold_pct=0.15, bright_lines=bright_lines)
            if len(rp) < 6:
                corr.append(((X, Y), fallback))
                continue
            rp = rp.astype(float) + np.array([x0, y0])
            rel = rp - p0
            n_row = np.array([-ur[1], ur[0]])
            n_col = np.array([-uc[1], uc[0]])
            d_row = np.abs(rel @ n_row)
            d_col = np.abs(rel @ n_col)
            al_row = rel @ ur
            al_col = rel @ uc
            ltol = max(3.0, 0.04 * min(nr, nc))
            row_pts = rp[(d_row < ltol) & (np.abs(al_row) > ltol)]
            col_pts = rp[(d_col < ltol) & (np.abs(al_col) > ltol)]
            Lr = Lc = None
            if len(row_pts) >= 4:
                ctr = row_pts.mean(0)
                _, _, vt = np.linalg.svd(row_pts - ctr)
                Lr = (vt[0][0], vt[0][1], ctr[0], ctr[1])
            if len(col_pts) >= 4:
                ctr = col_pts.mean(0)
                _, _, vt = np.linalg.svd(col_pts - ctr)
                Lc = (vt[0][0], vt[0][1], ctr[0], ctr[1])
            if Lr is not None and Lc is not None:
                Xp = _line_intersection_simple(Lr, Lc)
                if Xp is not None and all(math.isfinite(v) for v in Xp) and \
                        math.hypot(Xp[0] - p0[0], Xp[1] - p0[1]) < 0.6 * max(ext_r, ext_c):
                    corr.append(((X, Y), (float(Xp[0]), float(Xp[1]))))
                    n_ref += 1
                    continue
            corr.append(((X, Y), fallback))
        if len(corr) < 4:
            break
        newH = _dlt([s for s, _ in corr], [d for _, d in corr])
        if newH is not None and np.all(np.isfinite(newH)) and _grid_twist_ok(newH):
            curH = newH
        else:
            break
    return curH, n_ref


# ──────────────────────────────────────────────────────────────
# Stage 2 結果容器
# ──────────────────────────────────────────────────────────────
class TopologyResult:
    """拓樸求解結果。"""

    def __init__(self, status, H=None, assignment=None, inliers=None,
                 rmse=float("inf"), span=1.0, type_consistency=0.0,
                 confidence="low", reason="", n_steger_refined=0):
        self.status = status                 # 'ok' | 'fail'
        self.H = H                           # 3x3 ndarray (template m → image px) 或 None
        self.assignment = assignment or {}   # {detection_idx: template_id(0..29)}
        self.inliers = inliers or []         # list of detection_idx
        self.rmse = rmse
        self.span = span
        self.type_consistency = type_consistency
        self.confidence = confidence
        self.reason = reason
        self.n_steger_refined = n_steger_refined

    def visible_template_ids(self, H=None, img_shape=None, margin=0):
        """H 投影落在影像範圍內的模板節點 id 清單（含其影像座標）。"""
        H = self.H if H is None else H
        out = []
        if H is None:
            return out
        for tid in range(30):
            r, c = tid // 5, tid % 5
            p = _proj(H, _tpl_xy(r, c))
            if not (math.isfinite(p[0]) and math.isfinite(p[1])):
                continue
            if img_shape is not None:
                Himg, Wimg = img_shape[:2]
                if not (-margin <= p[0] <= Wimg + margin and -margin <= p[1] <= Himg + margin):
                    continue
            out.append((tid, (float(p[0]), float(p[1]))))
        return out


# ──────────────────────────────────────────────────────────────
# Stage 2 主類別
# ──────────────────────────────────────────────────────────────
class TopologySolver:
    """
    第二階段：由偵測交點 + 模板求 H 與拓樸對應。

    使用：
        solver = TopologySolver()
        result = solver.solve(node_pts, node_types, img_bgr=img)
    """

    def __init__(self,
                 steger_refine_h: bool = None,
                 steger_refine_iters: int = None,
                 ransac_fallback_iters: int = None,
                 min_inliers: int = None,
                 bright_lines: bool = True):
        self.steger_refine_h = (config.S2_STEGER_REFINE_H
                                if steger_refine_h is None else steger_refine_h)
        self.steger_refine_iters = (config.S2_STEGER_REFINE_ITERS
                                    if steger_refine_iters is None else steger_refine_iters)
        self.ransac_fallback_iters = (config.S2_RANSAC_FALLBACK_ITERS
                                      if ransac_fallback_iters is None else ransac_fallback_iters)
        self.min_inliers = config.S2_MIN_INLIERS if min_inliers is None else min_inliers
        self.bright_lines = bright_lines

    # --------------------------------------------------------------
    def solve(self, node_pts, node_types, img_bgr=None) -> TopologyResult:
        node_pts = [tuple(map(float, p)) for p in node_pts]
        node_types = list(node_types)
        n = len(node_pts)
        if n < 4:
            return TopologyResult("fail", reason=f"偵測交點不足（{n} < 4），無法估計單應矩陣")

        P = np.asarray(node_pts, float)
        span = float(math.hypot(P[:, 0].max() - P[:, 0].min(),
                                P[:, 1].max() - P[:, 1].min()) + 1e-6)
        subset = list(range(n))

        candidates = []
        candidates += self._seed_corner_quad(node_pts, node_types, subset, span)
        # 若四角種子已得到型別一致且 inlier 多的解，仍嘗試 RANSAC 補強（取最佳）
        best_pre = max(candidates, key=self._cand_key, default=None)
        if best_pre is None or not self._is_strong(best_pre):
            candidates += self._seed_ransac(node_pts, node_types, subset, span)

        if not candidates:
            return TopologyResult("fail", span=span,
                                  reason="四角種子與取樣 RANSAC 皆未求得有效 H")

        best = max(candidates, key=self._cand_key)
        H, inliers, rmse, tc = best["H"], best["inliers"], best["rmse"], best["tc"]

        if len(inliers) < self.min_inliers:
            return TopologyResult("fail", span=span,
                                  reason=f"最佳 H 之 inlier 數不足（{len(inliers)} < {self.min_inliers}）")

        # 可選：Steger 次像素精修
        n_ref = 0
        if self.steger_refine_h and img_bgr is not None:
            H1, n_ref = refine_homography_steger(
                H, img_bgr, node_pts, node_types, inliers,
                bright_lines=self.bright_lines, iters=self.steger_refine_iters)
            if n_ref >= 4 and _grid_twist_ok(H1):
                thr = max(config.S2_NN_INLIER_MIN_PX, config.S2_NN_INLIER_RATIO * span)
                inl1, rmse1 = _nn_assign(node_pts, node_types, H1, thr, subset=subset)
                if len(inl1) >= len(inliers):
                    H, inliers, rmse = H1, inl1, rmse1
                    tc, _, _ = _type_consistency(H, node_pts, node_types, subset)

        assignment = self._build_assignment(H, node_pts, node_types, subset, span)
        ratio = rmse / span
        if tc >= 0.95 and len(inliers) >= 9 and ratio < 0.012:
            conf = "high"
        elif tc >= 0.8 and ratio < 0.02:
            conf = "medium"
        else:
            conf = "low"

        return TopologyResult("ok", H=np.asarray(H, float), assignment=assignment,
                              inliers=sorted(inliers), rmse=rmse, span=span,
                              type_consistency=tc, confidence=conf,
                              n_steger_refined=n_ref)

    # --------------------------------------------------------------
    # 評分
    # --------------------------------------------------------------
    @staticmethod
    def _cand_key(cand):
        # 優先序：型別一致性 → inlier 數 → orientation 慣例 → RMSE。
        # orient 排在 RMSE 之前且量化（對稱解的 RMSE 皆≈0，僅浮點雜訊不同；
        # 若 RMSE 在前會被雜訊主導而選錯朝向）。量化到 0.1 使 ±1.999 的朝向
        # 差異主導、而同朝向內仍由 RMSE 細分。
        return (round(cand["tc"], 3), len(cand["inliers"]),
                round(cand["orient"], 1), -round(cand["rmse"] / max(cand["span"], 1.0), 4))

    @staticmethod
    def _is_strong(cand):
        return cand["tc"] >= 0.95 and len(cand["inliers"]) >= 9

    def _make_cand(self, H, node_pts, node_types, subset, span):
        if H is None or not np.all(np.isfinite(H)) or not _grid_twist_ok(H):
            return None
        Hb, inl, rmse = _guided_refit(H, node_pts, node_types, subset, span)
        if Hb is None or not _grid_twist_ok(Hb):
            return None
        tc, _, _ = _type_consistency(Hb, node_pts, node_types, subset)
        return {"H": Hb, "inliers": inl, "rmse": rmse, "span": span,
                "tc": tc, "orient": _orient_score(Hb)}

    # --------------------------------------------------------------
    # 種子 A：四角極值點 → 模板四角（8 種 dihedral）
    # --------------------------------------------------------------
    def _seed_corner_quad(self, node_pts, node_types, subset, span):
        P = np.asarray(node_pts, float)
        s = P[:, 0] + P[:, 1]
        d = P[:, 0] - P[:, 1]
        tl = int(np.argmin(s))   # top-left
        br = int(np.argmax(s))   # bottom-right
        tr = int(np.argmax(d))   # top-right
        bl = int(np.argmin(d))   # bottom-left
        quad_idx = [tl, tr, br, bl]
        if len(set(quad_idx)) != 4:
            return []
        Dq = [node_pts[i] for i in quad_idx]              # 影像四角（CW）
        Tq = [_tpl_xy(*divmod(t, 5)) for t in CORNER_IDS_CW]  # 模板四角 metric（CW）

        cands = []
        orders = [list(range(4)), list(range(4))[::-1]]   # 正/反序（鏡射）
        for order in orders:
            Dq_o = [Dq[i] for i in order]
            for rot in range(4):
                src = [Tq[(i + rot) % 4] for i in range(4)]
                H0 = _dlt(src, Dq_o)
                cand = self._make_cand(H0, node_pts, node_types, subset, span)
                if cand is not None:
                    cands.append(cand)
        return cands

    # --------------------------------------------------------------
    # 種子 B：型別約束取樣 RANSAC（後援）
    # --------------------------------------------------------------
    def _seed_ransac(self, node_pts, node_types, subset, span):
        n = len(node_pts)
        if n < 4:
            return []
        rng = np.random.default_rng(12345)
        P = np.asarray(node_pts, float)

        # 依型別把模板四點分群，作為取樣對應的目標
        tmpl_by_type = {"L": [], "T": [], "X": []}
        for tid in range(30):
            r, c = divmod(tid, 5)
            tmpl_by_type[_tpl_type(r, c)].append(tid)

        det_by_type = {"L": [], "T": [], "X": []}
        for i, t in enumerate(node_types):
            if t in det_by_type:
                det_by_type[t].append(i)

        best = None
        thr = max(config.S2_NN_INLIER_MIN_PX, config.S2_NN_INLIER_RATIO * span)
        iters = self.ransac_fallback_iters
        det_all = list(range(n))
        for _ in range(iters):
            # 取 4 個偵測點（盡量凸、面積夠大）
            di = rng.choice(n, 4, replace=False)
            dpts = P[di]
            if abs(_signed_area([tuple(p) for p in dpts])) < 0.02 * span * span:
                continue
            d_types = [node_types[k] for k in di]
            # 對應 4 個模板點：型別相容（若型別未知/None 則不限制）
            tcands = []
            ok = True
            for dt in d_types:
                pool = tmpl_by_type.get(dt) if dt in tmpl_by_type else None
                if not pool:
                    pool = list(range(30))
                tcands.append(pool)
            if not ok:
                continue
            # 隨機從各型別池抽一個模板點（不重複）
            chosen = []
            used = set()
            fail = False
            for pool in tcands:
                avail = [t for t in pool if t not in used]
                if not avail:
                    fail = True
                    break
                pick = int(rng.choice(avail))
                chosen.append(pick)
                used.add(pick)
            if fail:
                continue
            src = [_tpl_xy(*divmod(t, 5)) for t in chosen]
            dst = [tuple(p) for p in dpts]
            H0 = _dlt(src, dst)
            if H0 is None or not _grid_twist_ok(H0):
                continue
            inl, rmse = _nn_assign(node_pts, node_types, H0, thr, subset=subset)
            if best is None or len(inl) > best[1]:
                best = (H0, len(inl))
                if len(inl) >= max(9, int(0.6 * n)):
                    break

        if best is None:
            return []
        cand = self._make_cand(best[0], node_pts, node_types, subset, span)
        return [cand] if cand is not None else []

    # --------------------------------------------------------------
    # 建立 detection_idx → template_id 對應
    # --------------------------------------------------------------
    def _build_assignment(self, H, node_pts, node_types, subset, span):
        thr = max(config.S2_NN_INLIER_MIN_PX, config.S2_NN_INLIER_RATIO * span)
        cells = _assign_cells_simple(H, node_pts, node_types, subset, thr)
        return {int(i): int(r * 5 + c) for i, (r, c) in cells.items()}


__all__ = ["TopologySolver", "TopologyResult",
           "_proj", "_tpl_xy", "_tpl_type", "COL_X", "ROW_Y"]
