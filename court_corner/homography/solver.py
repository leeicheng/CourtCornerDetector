"""
homography/solver.py — 線為主單應求解（自 court_homography_tool 移植，去除 GUI）
================================================================
cross-ratio 線標號 → PROSAC → Steger 次像素精修，含 solve_image 入口與
courts_from_image。場地範本（COL_X/ROW_Y/編號/型別）與 shared.court_model 相同。
（原始來源：使用者的 court_homography_tool.py；此處為內嵌移植，不再外部引用。）
"""

import os, sys, math, json, itertools, time

import numpy as np
import cv2

from .court_lines import (
    _assign_junction_lines, _compute_court_lines,
    _line_intersection_simple, _steger_ridge_points_simple,
)


# ──────────────────────────────────────────────
# Template（30 點：6 列 × 5 行；x=col, y=row）
# ──────────────────────────────────────────────
COL_X = [0.00, 0.46, 3.05, 5.64, 6.10]                 # 5 條縱線
ROW_Y = [13.40, 12.64, 8.68, 4.72, 0.76, 0.00]         # 6 條橫線
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


def _tpl_type(r, c):
    return TYPE_ID_TO_NAME[TEMPLATE_TYPES[r * N_COL + c]]


def _tpl_xy(r, c):
    return (COL_X[c], ROW_Y[r])


# 演算法參數
CR_TOL      = 0.06    # cross-ratio 殘差門檻
MIN_LINES   = 4       # 每族至少線數（< 即無法處理）
MIN_CORR    = 6       # 最少對應點數
MIN_COURT_NODES = 6   # 一座球場至少節點數（< 視為碎片/雜訊，不當球場）
MIN_LC      = 0.5     # 最佳候選的線一致性下限（< 視為不可靠，判失敗而非畫爛 H）
MIN_TC      = 0.6     # 型別一致性下限（inlier 類型須與投影點類型相符，< 視為 H 錯位/翻轉）
RECOVER_TOL = 18.0    # 投影點與偵測點視為同一點的像素容忍


# ──────────────────────────────────────────────
# 幾何小工具
# ──────────────────────────────────────────────
def _merge_lines(members, fits, node_pts):
    """合併重複線：同族重複線會通過相同交點（共用 ≥2 成員）；
    真正相距 0.76m 的服務線通過不同交點，不會被合併。"""
    n = len(members)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    sets = [set(m) for m in members]
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets[i] & sets[j]) >= 2:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out_m, out_f = [], []
    for g in groups.values():
        mem = sorted(set().union(*[sets[i] for i in g]))
        out_m.append(mem)
        if len(mem) >= 2:                       # 以成員點 PCA 重擬合
            P = np.asarray([node_pts[k] for k in mem], float)
            ctr = P.mean(0)
            _, _, vt = np.linalg.svd(P - ctr)
            out_f.append((float(vt[0][0]), float(vt[0][1]),
                          float(ctr[0]), float(ctr[1])))
        else:
            out_f.append(fits[max(g, key=lambda i: len(members[i]))])
    return out_m, out_f


def _augment_line_members(members, fits, node_pts, allow=None, tol=4.0, min_existing=2):
    """共線強化：把『落在既有線上但 Steger 沒連到』的點補進該線（垂距 ≤ tol），再重擬合。
    只擴充既有線、不新增線——保持家族結構，避免偶然共線灌爆 cross-ratio 家族切分。
    allow：限定可加入的節點索引集合（通常為本球場節點，避免跨場誤併）。
    依手標分析：真實球場線點到線垂距 p99≈3.2px，故 tol=4px 既補得到漏點又幾乎不誤收。"""
    if not members:
        return members, fits
    P = np.asarray(node_pts, float)
    allow = set(allow) if allow is not None else set(range(len(node_pts)))
    out_m, out_f = [], []
    for mem, fit in zip(members, fits):
        if len(mem) < min_existing:
            out_m.append(mem); out_f.append(fit); continue
        vx, vy, x0, y0 = fit
        nrm = math.hypot(vx, vy)
        if nrm < 1e-9:
            out_m.append(mem); out_f.append(fit); continue
        nx, ny = -vy / nrm, vx / nrm                 # 線法向（單位）
        cur = set(mem)
        for di in allow:
            if di in cur:
                continue
            d = abs((P[di, 0] - x0) * nx + (P[di, 1] - y0) * ny)
            if d <= tol:
                cur.add(di)
        mem2 = sorted(cur)
        if len(mem2) > len(mem):                     # 有補到 → PCA 重擬合
            Q = P[mem2]; ctr = Q.mean(0)
            _, _, vt = np.linalg.svd(Q - ctr)
            out_f.append((float(vt[0][0]), float(vt[0][1]), float(ctr[0]), float(ctr[1])))
        else:
            out_f.append(fit)
        out_m.append(mem2)
    # 補點後可能讓兩條原本分開的線共用 ≥2 點 → 再合併一次（去重）
    return _merge_lines(out_m, out_f, node_pts)


def _fit_1d(t, o):
    """擬合 1D 射影 o = (a t + b)/(c t + 1)，回傳 (a,b,c)。"""
    t = np.asarray(t, float); o = np.asarray(o, float)
    A = np.c_[t, np.ones_like(t), -t * o]
    sol, *_ = np.linalg.lstsq(A, o, rcond=None)
    return sol


def _apply_1d(abc, t):
    a, b, c = abc
    den = c * t + 1.0
    return (a * t + b) / den if abs(den) > 1e-9 else 1e18


def _label_family(coords, tpl_positions):
    """把觀測線位置 coords 對到 tpl_positions（容許多餘/雜散線）。
    回傳 (matches{obs_idx->tpl_slot}, residual, n_match)。用 4 錨點 cross-ratio +
    1D 射影內點計數，全列舉、確定性。"""
    k, m = len(coords), len(tpl_positions)
    if k < MIN_LINES or m < MIN_LINES:
        return {}, float("inf"), 0
    O = list(coords)
    spread = max(O) - min(O)
    tol = max(1e-6, 0.05 * spread)              # 位置內點容忍（相對展幅）
    best = ({}, float("inf"), 0)
    for oi in itertools.combinations(range(k), 4):
        cro = _cross_ratio(O[oi[0]], O[oi[1]], O[oi[2]], O[oi[3]])
        if cro is None:
            continue
        for tj in itertools.combinations(range(m), 4):
            crt = _cross_ratio(*[tpl_positions[j] for j in tj])
            if crt is None or abs(cro - crt) > 0.12:
                continue
            abc = _fit_1d([tpl_positions[j] for j in tj],
                          [O[i] for i in oi])
            # 把每個 tpl slot 投到觀測軸，找最近且保序的觀測線
            matches, used, res = {}, set(), []
            for j in range(m):
                pred = _apply_1d(abc, tpl_positions[j])
                cand = [(abs(O[i] - pred), i) for i in range(k) if i not in used]
                if not cand:
                    continue
                dd, i = min(cand)
                if dd < tol:
                    matches[i] = j; used.add(i); res.append(dd)
            # 保序檢查（遞增或遞減皆可，方向之後由對稱步驟固定）
            mi = sorted(matches.items())
            slots = [s for _, s in mi]
            if slots != sorted(slots) and slots != sorted(slots, reverse=True):
                continue
            n = len(matches)
            rr = float(np.mean(res)) if res else float("inf")
            if n > best[2] or (n == best[2] and rr < best[1]):
                best = (dict(matches), rr, n)
    return best


def _circ_mean(angs):
    s = sum(math.sin(2 * a) for a in angs)
    c = sum(math.cos(2 * a) for a in angs)
    return 0.5 * math.atan2(s, c)


def _cross_ratio(a, b, c, d):
    den = (a - d) * (b - c)
    if abs(den) < 1e-9:
        return None
    return (a - c) * (b - d) / den


def _split_families_angle(dirs):
    """依方向角（mod π）的兩個最大間隙把線切成兩族。"""
    n = len(dirs)
    if n < 2:
        return list(range(n)), []
    ang = [math.atan2(d[1], d[0]) % math.pi for d in dirs]
    order = sorted(range(n), key=lambda i: ang[i])
    sa = [ang[i] for i in order]
    gaps = [((sa[(i + 1) % n] - sa[i]) % math.pi) for i in range(n)]
    g1, g2 = sorted(sorted(range(n), key=lambda i: -gaps[i])[:2])
    famA = order[g1 + 1:g2 + 1]
    famB = order[g2 + 1:] + order[:g1 + 1]
    return famA, famB


def _split_score(famA, famB, dirs):
    """族內角度一致性（圓變異越小越好）。"""
    def spread(fam):
        if not fam:
            return math.pi
        angs = [math.atan2(dirs[i][1], dirs[i][0]) % math.pi for i in fam]
        m = _circ_mean(angs)
        return sum(min(abs(a - m), math.pi - abs(a - m)) ** 2 for a in angs) / len(angs)
    return spread(famA) + spread(famB)


def _split_families_vp(dirs, line_fits, node_pts):
    """交叉性二分圖著色（對透視穩健）。回傳 (famA, famB) 或 None。"""
    n = len(dirs)
    P = np.asarray(node_pts, float)
    mnx, mny = P.min(0); mxx, mxy = P.max(0)
    mw, mh = (mxx - mnx) * 0.12 + 1, (mxy - mny) * 0.12 + 1
    mnx, mny, mxx, mxy = mnx - mw, mny - mh, mxx + mw, mxy + mh
    cross = [[False] * n for _ in range(n)]
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            Q = _line_intersection_simple(line_fits[i], line_fits[j])
            if Q is not None and mnx <= Q[0] <= mxx and mny <= Q[1] <= mxy:
                cross[i][j] = cross[j][i] = True
                deg[i] += 1; deg[j] += 1
    if max(deg) == 0:
        return None
    color = {}
    s = max(range(n), key=lambda i: deg[i])
    color[s] = 0; stack = [s]
    while stack:
        u = stack.pop()
        for v in range(n):
            if cross[u][v] and v not in color:
                color[v] = 1 - color[u]; stack.append(v)
    ang = [math.atan2(d[1], d[0]) % math.pi for d in dirs]
    A0 = [i for i in color if color[i] == 0]
    A1 = [i for i in color if color[i] == 1]
    if not A0 or not A1:
        return None
    m0 = _circ_mean([ang[i] for i in A0]); m1 = _circ_mean([ang[i] for i in A1])
    def _da(a, b):
        d = abs(a - b) % math.pi; return min(d, math.pi - d)
    famA, famB = list(A0), list(A1)
    for i in range(n):
        if i in color:
            continue
        (famA if _da(ang[i], m0) <= _da(ang[i], m1) else famB).append(i)
    return famA, famB


def _split_families(dirs, line_fits=None, node_pts=None):
    """兩種候選（方向角隙、交叉性 VP）取族內角度一致性較佳且兩族皆 ≥2 者；
    皆不可用則回傳角隙法結果。"""
    n = len(dirs)
    if n < 2:
        return list(range(n)), []
    cands = [_split_families_angle(dirs)]
    if line_fits is not None and node_pts:
        vp = _split_families_vp(dirs, line_fits, node_pts)
        if vp is not None:
            cands.append(vp)
    valid = [c for c in cands if min(len(c[0]), len(c[1])) >= 2]
    pool = valid if valid else cands
    return min(pool, key=lambda c: _split_score(c[0], c[1], dirs))


def _order_family(fam, line_fits, line_centroids):
    """族內依垂直於平均方向的位置排序（固定順序）。"""
    angs = [math.atan2(line_fits[i][1], line_fits[i][0]) % math.pi for i in fam]
    ma = _circ_mean(angs)
    nx, ny = -math.sin(ma), math.cos(ma)        # 法向
    pos = [(line_centroids[i][0] * nx + line_centroids[i][1] * ny, i) for i in fam]
    pos.sort()
    return [i for _, i in pos]


def _coords_on_transversal(ordered, line_fits, transv):
    """各線與 transversal 交點，沿 transversal 方向的 1D 座標。"""
    ox, oy, dx, dy = transv[2], transv[3], transv[0], transv[1]
    out = []
    for i in ordered:
        P = _line_intersection_simple(line_fits[i], transv)
        if P is None:
            return None
        out.append((P[0] - ox) * dx + (P[1] - oy) * dy)
    return out


def _cr_residual(s, t):
    """連續四元組 cross-ratio 的中位差。"""
    diffs = []
    for i in range(len(s) - 3):
        a = _cross_ratio(s[i], s[i + 1], s[i + 2], s[i + 3])
        b = _cross_ratio(t[i], t[i + 1], t[i + 2], t[i + 3])
        if a is None or b is None:
            continue
        diffs.append(abs(a - b))
    return float(np.median(diffs)) if diffs else float("inf")


def _best_subset(coords, tpl_positions):
    """把 k 條觀測線（已排序）對到 tpl 位置的保序子集，回傳 (subset_indices, residual)。"""
    k, m = len(coords), len(tpl_positions)
    if k < MIN_LINES or k > m:
        return None, float("inf")
    best, best_res = None, float("inf")
    for subset in itertools.combinations(range(m), k):
        t = [tpl_positions[j] for j in subset]
        res = _cr_residual(coords, t)
        if res < best_res:
            best_res, best = res, subset
    return best, best_res


# ──────────────────────────────────────────────
# DLT homography（template 公尺 → 影像像素）
# ──────────────────────────────────────────────
def _normalize(pts):
    P = np.asarray(pts, float)
    c = P.mean(0)
    d = np.sqrt(((P - c) ** 2).sum(1)).mean()
    s = math.sqrt(2) / d if d > 1e-9 else 1.0
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    Ph = (T @ np.c_[P, np.ones(len(P))].T).T
    return Ph[:, :2], T


def _dlt(src, dst):
    s, Ts = _normalize(src)
    d, Td = _normalize(dst)
    A = []
    for (X, Y), (x, y) in zip(s, d):
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:   # 退化解 → 回 None，交由上游丟棄
        return None
    return H / H[2, 2]


def _proj(H, X):
    v = H @ np.array([X[0], X[1], 1.0])
    if abs(v[2]) < 1e-12:
        return (float("inf"), float("inf"))
    return (v[0] / v[2], v[1] / v[2])


_SYMS = [
    ("identity", lambda r, c: (r, c)),
    ("flipC",    lambda r, c: (r, N_COL - 1 - c)),
    ("flipR",    lambda r, c: (N_ROW - 1 - r, c)),
    ("flipRC",   lambda r, c: (N_ROW - 1 - r, N_COL - 1 - c)),
]


def _signed_area(quad):
    x = [p[0] for p in quad]; y = [p[1] for p in quad]
    return 0.5 * sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))


def _grid_twist_ok(H, max_ratio=120.0):
    """投影 30 點，檢查每個基本格子有號面積同號（不折疊）、面積比不過大（移植自參考工具）。"""
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
        return False                                   # 混合號 → 折疊/扭轉
    arr = np.asarray(areas, float)
    p10 = float(np.percentile(arr, 10)); p90 = float(np.percentile(arr, 90))
    if p10 <= 1e-6 or p90 / p10 > max_ratio:
        return False
    return True


def _type_consistency(H, node_pts, node_types, subset=None, thr=None):
    """型別一致性（正確性檢查）：把每個 inlier 以「純位置」NN 指到最近投影格，
    再看偵測類型 (L/T/X) 是否等於該格的範本類型。回傳 (一致比例, 一致數, inlier 數)。
    錯的 H（例如整體位移一列）會把 X 交點對到 T/L 的位置 → 型別不符 → 比例下降。
    注意：用純位置而非帶型別懲罰的指派，才是真正的獨立檢查。"""
    cells = []
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                cells.append((_tpl_type(r, c), p))
    if not cells:
        return 0.0, 0, 0
    CT = [t for t, _ in cells]; PP = np.array([p for _, p in cells], float)
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
        if d[j] > thr:                 # 離任何格都太遠 → 不是這個 H 的 inlier，跳過
            continue
        tot += 1
        if CT[j] == node_types[i]:
            match += 1
    return (match / tot if tot else 0.0), match, tot


# ── VP-based 範本 id relabel（左下=0 → 右上=29），移植自 topology_id_relabel.py ──
def _line_from_two_pts(p1, p2):
    a = p1[1] - p2[1]; b = p2[0] - p1[0]; c = p1[0] * p2[1] - p2[0] * p1[1]
    n = math.hypot(a, b)
    return None if n < 1e-9 else (a / n, b / n, c / n)


def _vp_of_family(family_cells, proj):
    """family_cells: 每條線一串 (r,c)；用投影點建影像線、SVD 齊次最小平方求消失點。"""
    img_lines = []
    for tids in family_cells:
        pts = [proj[rc] for rc in tids if rc in proj]
        if len(pts) < 2:
            continue
        pa, pb, md = pts[0], pts[-1], -1.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d2 = (pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2
                if d2 > md:
                    md = d2; pa, pb = pts[i], pts[j]
        if md < 1.0:
            continue
        L = _line_from_two_pts(pa, pb)
        if L is not None:
            img_lines.append(L)
    if len(img_lines) < 2:
        return None
    M = np.asarray(img_lines, float)
    try:
        _, _, vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1, :]
    if abs(h[2]) < 1e-12:                       # 平行 → 無窮遠，回沿方向的極遠點
        nx, ny = float(h[0]), float(h[1]); nrm = math.hypot(nx, ny)
        return None if nrm < 1e-12 else (nx / nrm * 1e9, ny / nrm * 1e9)
    return (float(h[0] / h[2]), float(h[1] / h[2]))


def _vp_relabel_ids(proj):
    """proj: {(r,c):(x,y)} 投影 30 格。回傳 ({(r,c):id 0~29}, flip_y, flip_x, debug)。
    規則：長軸線(縱/同 col)的消失點 → 近側(離 VP 較遠)當 row0；近排 col0 應在影像左。"""
    long_lines  = [[(r, c) for r in range(N_ROW)] for c in range(N_COL)]   # 縱線(長軸)
    vp_long = _vp_of_family(long_lines, proj)
    def mid(cells):
        ps = [proj[rc] for rc in cells if rc in proj]
        return None if not ps else (sum(p[0] for p in ps)/len(ps), sum(p[1] for p in ps)/len(ps))
    r0 = mid([(0, c) for c in range(N_COL)]); r5 = mid([(N_ROW-1, c) for c in range(N_COL)])
    flip_y = False; dbg = []
    if vp_long and r0 and r5:
        d0 = math.hypot(r0[0]-vp_long[0], r0[1]-vp_long[1])
        d5 = math.hypot(r5[0]-vp_long[0], r5[1]-vp_long[1])
        flip_y = (d5 > d0)
        dbg.append(f"VP_long=({vp_long[0]:.0f},{vp_long[1]:.0f}) d0={d0:.0f} d5={d5:.0f}→{'flip_y' if flip_y else 'no flip_y'}")
    else:
        dbg.append("VP_long 不足，flip_y=False")
    near = (N_ROW - 1) if flip_y else 0
    nr = [proj.get((near, c)) for c in range(N_COL)]
    flip_x = False
    if all(p is not None for p in nr):
        flip_x = (nr[0][0] > nr[-1][0])
        dbg.append(f"近排 col0_x={nr[0][0]:.0f} col4_x={nr[-1][0]:.0f}→{'flip_x' if flip_x else 'no flip_x'}")
    idmap = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            r2 = (N_ROW - 1 - r) if flip_y else r
            c2 = (N_COL - 1 - c) if flip_x else c
            idmap[(r, c)] = r2 * N_COL + c2
    return idmap, flip_y, flip_x, "; ".join(dbg)


def _line_consistency(H, line_members, node_pts, subset=None):
    """線點專屬的強驗證（無影像也能用）：把節點 NN 指到投影格位後，
    每條偵測線的成員若全落在範本同列或同行＝一致。回傳一致線比例 [0,1]。
    錯的 H 會把真實線上的點散到不同列/行 → 比例掉下去（擋非退化假陽性，如 _5 的 165px）。"""
    proj = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                proj[(r, c)] = p
    if not proj:
        return 0.0
    cells = list(proj.keys()); PP = np.array([proj[c] for c in cells], float)
    cell_of = {}
    idxs = subset if subset is not None else range(len(node_pts))
    for i in idxs:
        x, y = node_pts[i]
        d = np.hypot(PP[:, 0] - x, PP[:, 1] - y)
        cell_of[i] = cells[int(d.argmin())]
    good = tot = 0
    for m in line_members:
        cs = [cell_of[i] for i in m if i in cell_of]
        if len(cs) < 2:
            continue
        tot += 1
        if len({c[0] for c in cs}) == 1 or len({c[1] for c in cs}) == 1:
            good += 1
    return good / tot if tot else 0.0


def _h_degenerate(H, span, centroid, img_wh=None):
    """退化防護：投影爆掉/塌陷/離質心過遠都判退化（擋近奇異等假陽性，如 H 爆炸成上萬 px）。
    便宜、不需影像、不重解。"""
    pts = []
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if not (math.isfinite(p[0]) and math.isfinite(p[1])):
                return True
            pts.append(p)
    P = np.asarray(pts, float)
    diag = math.hypot(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min())
    if diag > 10.0 * span or diag < 0.15 * span:
        return True
    corners = [_proj(H, _tpl_xy(0, 0)), _proj(H, _tpl_xy(0, N_COL - 1)),
               _proj(H, _tpl_xy(N_ROW - 1, N_COL - 1)), _proj(H, _tpl_xy(N_ROW - 1, 0))]
    if abs(_signed_area(corners)) < 0.02 * span * span:
        return True
    cx, cy = centroid
    if math.hypot(P[:, 0].mean() - cx, P[:, 1].mean() - cy) > 4.0 * span:
        return True
    if img_wh is not None:
        W, H_ = img_wh
        if not (-2 * span < P[:, 0].mean() < W + 2 * span and
                -2 * span < P[:, 1].mean() < H_ + 2 * span):
            return True
    return False


def _orient_score(H):
    """方向慣例：template +x → 影像右、+y → 影像下，分數越高越合慣例。"""
    cen = (3.05, 6.70)
    p0 = _proj(H, cen); px = _proj(H, (cen[0] + 1, cen[1])); py = _proj(H, (cen[0], cen[1] + 1))
    jx = np.subtract(px, p0); jy = np.subtract(py, p0)
    if not (np.all(np.isfinite(jx)) and np.all(np.isfinite(jy))):
        return -1e9
    jx = jx / (np.linalg.norm(jx) + 1e-9)
    jy = jy / (np.linalg.norm(jy) + 1e-9)
    return float(jx[0] + jy[1])


# ──────────────────────────────────────────────
# 單一球場：求 H
# ──────────────────────────────────────────────
def _node_rc_from_labels(line_label, line_members, node_pts, node_types):
    """依 line_label（global line -> ('V',col)/('H',row)）建立 (row,col,img_pt,type) 對應。"""
    node_lines = {i: [] for i in range(len(node_pts))}
    for li, m in enumerate(line_members):
        for k in m:
            node_lines[k].append(li)
    corr = []
    for i, lis in node_lines.items():
        col = row = None
        for li in lis:
            lab = line_label.get(li)
            if not lab:
                continue
            if lab[0] == "V":
                col = lab[1]
            else:
                row = lab[1]
        if col is not None and row is not None:
            corr.append((row, col, node_pts[i], node_types[i]))
    return corr


def _fit_vp(lines):
    """多條直線的最小平方交點（消失點）。line=(vx,vy,x0,y0)。"""
    rows, rhs = [], []
    for vx, vy, x0, y0 in lines:
        nrm = math.hypot(vx, vy)
        if nrm < 1e-9:
            continue
        a, b = -vy / nrm, vx / nrm
        rows.append([a, b]); rhs.append(-(a * x0 + b * y0))
    if len(rows) < 2:
        return None
    try:
        sol, *_ = np.linalg.lstsq(np.asarray(rows, float), np.asarray(rhs, float), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not (math.isfinite(sol[0]) and math.isfinite(sol[1])):
        return None
    return (float(sol[0]), float(sol[1]))


def _depth_vp(famA, famB, line_fits):
    """取較陡（影像偏垂直）的線族當『深度方向』，回傳其消失點（近/遠定向用）。"""
    def steep(fam):
        return float(np.mean([abs(line_fits[i][1]) /
                              (math.hypot(line_fits[i][0], line_fits[i][1]) + 1e-9)
                              for i in fam])) if fam else 0.0
    fam = famA if steep(famA) >= steep(famB) else famB
    if len(fam) < 2:
        return None
    return _fit_vp([line_fits[i] for i in fam])


def _vp_consistent(H, vp):
    """近/遠定向：固定約定 metric y=13.40 端為『遠端』，應投影到較靠近深度消失點處。
    回傳 +1（合）/ -1（不合）/ 0（無 VP）。"""
    if vp is None:
        return 0
    p_far = _proj(H, (3.05, 13.40)); p_near = _proj(H, (3.05, 0.0))
    if not all(math.isfinite(v) for v in (*p_far, *p_near)):
        return 0
    d_far = math.hypot(p_far[0] - vp[0], p_far[1] - vp[1])
    d_near = math.hypot(p_near[0] - vp[0], p_near[1] - vp[1])
    return 1 if d_far < d_near else -1


def _pick_H_sym(base_corr, vp=None):
    """對 base_corr 試 4 對稱各解一個 H。優先序：不折疊(grid-twist) > VP 近/遠一致 > 方向慣例。"""
    best = None
    for name, sym in _SYMS:
        src = [_tpl_xy(*sym(r, c)) for (r, c, _pt, _t) in base_corr]
        dst = [pt for (_r, _c, pt, _t) in base_corr]
        try:
            H = _dlt(src, dst)
        except Exception:
            continue
        if H is None or not np.all(np.isfinite(H)):
            continue
        reproj = [_proj(H, _tpl_xy(*sym(r, c))) for (r, c, _pt, _t) in base_corr]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in reproj):
            continue
        sc = _orient_score(H)
        if sc <= -1e8:
            continue
        cand = {"name": name, "H": H, "score": sc, "twist": _grid_twist_ok(H),
                "vp": _vp_consistent(H, vp), "sym": sym}
        if best is None or (cand["twist"], cand["vp"], cand["score"]) > \
                           (best["twist"], best["vp"], best["score"]):
            best = cand
    return best


def _finalize_homography(base_corr, node_pts, method, vp=None):
    """共用收尾：選 H＋對稱、算 RMSE、投影 30 點。"""
    best = _pick_H_sym(base_corr, vp=vp)
    if best is None:
        return None

    H, sym = best["H"], best["sym"]
    pts0 = np.asarray(node_pts, float)
    if len(pts0):
        span0 = math.hypot(pts0[:, 0].max() - pts0[:, 0].min(),
                           pts0[:, 1].max() - pts0[:, 1].min()) + 1e-6
        cen0 = (float(pts0[:, 0].mean()), float(pts0[:, 1].mean()))
        if _h_degenerate(H, span0, cen0):           # 退化（爆炸/塌陷）→ 不接受
            return None
    errs = []
    type_ok = 0
    for (r, c, pt, t) in base_corr:
        rr, cc = sym(r, c)
        px = _proj(H, _tpl_xy(rr, cc))
        errs.append(math.hypot(px[0] - pt[0], px[1] - pt[1]))
        if _tpl_type(rr, cc) == t:
            type_ok += 1
    rmse = float(np.sqrt(np.mean(np.square(errs))))

    pts = np.asarray(node_pts, float)
    span = (math.hypot(pts[:, 0].max() - pts[:, 0].min(),
                       pts[:, 1].max() - pts[:, 1].min()) + 1e-6) if len(pts) else 1.0
    projected = []
    for r in range(N_ROW):
        for c in range(N_COL):
            px = _proj(H, _tpl_xy(r, c))
            dmin = (float(np.min(np.hypot(pts[:, 0] - px[0], pts[:, 1] - px[1])))
                    if len(pts) else 1e9)
            projected.append({"row": r, "col": c, "type": _tpl_type(r, c),
                              "xy": [float(px[0]), float(px[1])],
                              "recovered": dmin > RECOVER_TOL})
    ratio = rmse / span
    if type_ok >= len(base_corr) and len(base_corr) >= 8 and ratio < 0.012:
        conf = "high"
    elif type_ok >= len(base_corr) - 1 and ratio < 0.02:
        conf = "medium"
    else:
        conf = "low"
    return {"status": "ok", "H": H.tolist(), "rmse": rmse, "span": float(span),
            "num_corr": len(base_corr), "sym": best["name"], "type_ok": type_ok,
            "confidence": conf, "method": method, "projected": projected}


def _col_types(c):
    return {_tpl_type(r, c) for r in range(N_ROW)}


def _row_types(r):
    return {_tpl_type(r, c) for c in range(N_COL)}


def _line_type_ok(member_types, kind, idx):
    """該線成員型別是否與被指派的 template 線相容（容忍 1 個偵測誤差）。"""
    allowed = _col_types(idx) if kind == "V" else _row_types(idx)
    bad = sum(1 for t in member_types if t not in allowed)
    return bad <= 1


def _ccw_order(pts):
    """回傳 4 點繞質心的逆時針順序索引。"""
    cx = sum(p[0] for p in pts) / 4.0; cy = sum(p[1] for p in pts) / 4.0
    return sorted(range(4), key=lambda i: math.atan2(pts[i][1] - cy, pts[i][0] - cx))


def _convex_quad_ok(pts, min_ang=22.0, max_ang=158.0):
    """4 點需構成凸四邊形且內角不過尖/過鈍（移植自 is_convex_quad_points 精神）。"""
    if len(pts) != 4:
        return False
    o = _ccw_order(pts)
    q = [pts[i] for i in o]
    if abs(_signed_area(q)) < 1e-6:
        return False
    for i in range(4):
        a = np.subtract(q[(i - 1) % 4], q[i]); b = np.subtract(q[(i + 1) % 4], q[i])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return False
        ang = math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1, 1))))
        if ang < min_ang or ang > max_ang:
            return False
    return True


_CAND_BY_TYPE = {nm: [r * N_COL + c for r in range(N_ROW) for c in range(N_COL)
                      if _tpl_type(r, c) == nm] for nm in ("L", "T", "X")}


def _nn_assign(node_pts, node_types, H, sym, thr, subset=None):
    """型別懲罰的貪婪 NN 指派（移植自 assign_detections_to_templates_nn）。回傳 inliers, rmse。"""
    proj = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            rr, cc = sym(r, c)
            p = _proj(H, _tpl_xy(rr, cc))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                proj[(r, c)] = (p, _tpl_type(rr, cc))
    idxs = subset if subset is not None else range(len(node_pts))
    pairs = []
    for di in idxs:
        pt = node_pts[di]; t = node_types[di]
        for (rc, (pp, tt)) in proj.items():
            d = math.hypot(pp[0] - pt[0], pp[1] - pt[1])
            cost = d + (0.0 if tt == t else thr)      # 型別不符加懲罰
            if cost <= thr * 3:
                pairs.append((cost, di, rc, d))
    pairs.sort()
    used_d, used_t, inl, errs = set(), set(), [], []
    for cost, di, rc, d in pairs:
        if di in used_d or rc in used_t:
            continue
        used_d.add(di); used_t.add(rc)
        if d <= thr:
            inl.append(di); errs.append(d)
    rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf")
    return inl, rmse


def _nn_match(node_pts, node_types, H, sym, thr, subset=None):
    """同 _nn_assign 的貪婪配對，但回傳每個 inlier 的 (節點idx, 對應範本metric座標, 距離)。"""
    proj = {}
    for r in range(N_ROW):
        for c in range(N_COL):
            rr, cc = sym(r, c)
            p = _proj(H, _tpl_xy(rr, cc))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                proj[(r, c)] = (p, _tpl_xy(rr, cc), _tpl_type(rr, cc))
    idxs = subset if subset is not None else range(len(node_pts))
    pairs = []
    for di in idxs:
        pt = node_pts[di]; t = node_types[di]
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
        used_d.add(di); used_t.add(rc)
        if d <= thr:
            matches.append((di, mp, d))
    return matches


def _guided_refit(H, sym, node_pts, node_types, sub, span, iters=4):
    """用『線族之外的點也算進來』的全部 inlier 重算 H（補位）：粗 H → NN 對應 →
    全 inlier DLT 重擬合 → 迭代，直到 inlier 不增、rmse 不降。回傳 (H, inl, rmse)。"""
    thr = max(6.0, 0.02 * span)
    bi, brm = _nn_assign(node_pts, node_types, H, sym, thr, subset=sub)
    best = (H, bi, brm)
    curH = H
    for _ in range(iters):
        m = _nn_match(node_pts, node_types, curH, sym, thr * 1.3, subset=sub)
        if len(m) < 4:
            break
        try:
            newH = _dlt([mp for (_di, mp, _d) in m], [node_pts[di] for (di, _mp, _d) in m])
        except Exception:
            break
        if newH is None or not np.all(np.isfinite(newH)) or not _grid_twist_ok(newH):
            break
        inl, rmse = _nn_assign(node_pts, node_types, newH, sym, thr, subset=sub)
        if len(inl) > len(best[1]) or (len(inl) == len(best[1]) and rmse < best[2] - 1e-9):
            best, curH = (newH, inl, rmse), newH
        else:
            break
    return best


def _build_prosac_result(H, inl, rmse, check_pts, span, method):
    """共用：投影 30 點、接受門檻、信心。check_pts 為本場節點（判 recovered）。"""
    if rmse > 0.03 * span:
        return None
    pts = np.asarray(check_pts, float)
    if len(pts):
        cen0 = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        if _h_degenerate(np.asarray(H, float), span, cen0):
            return None
    projected = []
    for r in range(N_ROW):
        for c in range(N_COL):
            px = _proj(H, _tpl_xy(r, c))
            dmin = (float(np.min(np.hypot(pts[:, 0] - px[0], pts[:, 1] - px[1])))
                    if len(pts) else 1e9)
            projected.append({"row": r, "col": c, "type": _tpl_type(r, c),
                              "xy": [float(px[0]), float(px[1])],
                              "recovered": dmin > RECOVER_TOL})
    ratio = rmse / span
    conf = "high" if (len(inl) >= 9 and ratio < 0.012) else ("medium" if ratio < 0.02 else "low")
    return {"status": "ok", "H": H.tolist(), "rmse": rmse, "span": float(span),
            "num_corr": len(inl), "sym": method, "type_ok": len(inl),
            "confidence": conf, "method": method, "projected": projected}


def _solve_seeded_prosac(node_pts, node_types, famA, famB, line_members, line_fits,
                         subset, vp=None):
    """缺線主力：以「兩條 famA 線 × 兩條 famB 線」的 4 個交點當局部 2×2 種子。
    交點優先取偵測到的節點（有型別）；該處漏檢時改用 line_fits 幾何交點補上（型別未知，
    型別檢查只套在有偵測的角點）。用局部區推 H，再 NN 投影驗證全部點。"""
    if len(famA) < 2 or len(famB) < 2:
        return None
    mem = [set(m) for m in line_members]
    sub = list(subset)
    if len(sub) < 6:
        return None
    sx = [node_pts[i] for i in sub]
    cx = float(np.mean([p[0] for p in sx])); cy = float(np.mean([p[1] for p in sx]))
    span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                       max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
    thr = max(6.0, 0.02 * span)

    def cross_corner(la, lb):
        """回傳 (point, type_or_None)；漏檢角點用幾何交點補。"""
        s = mem[la] & mem[lb]
        if len(s) == 1:
            k = next(iter(s))
            return (node_pts[k], node_types[k])
        P = _line_intersection_simple(line_fits[la], line_fits[lb])
        if P is None or not all(math.isfinite(v) for v in P):
            return None
        if math.hypot(P[0] - cx, P[1] - cy) > 2.5 * span:   # 近消失點/不穩 → 棄
            return None
        return ((float(P[0]), float(P[1])), None)

    best, best_key = None, (-1, 1.0)
    good_inl = max(MIN_CORR, min(int(0.5 * len(sub)), 12))
    good_rmse = 0.015 * span
    stop = False
    for la1, la2 in itertools.combinations(famA, 2):
        if stop:
            break
        for lb1, lb2 in itertools.combinations(famB, 2):
            if stop:
                break
            corners = [cross_corner(la1, lb1), cross_corner(la1, lb2),
                       cross_corner(la2, lb1), cross_corner(la2, lb2)]
            if any(x is None for x in corners):
                continue
            pimg = [c[0] for c in corners]
            tps = [c[1] for c in corners]
            if sum(1 for t in tps if t is not None) < 2:    # 至少 2 個有型別的角點錨定
                continue
            if len({(round(p[0], 1), round(p[1], 1)) for p in pimg}) < 4:
                continue
            if not _convex_quad_ok(pimg):
                continue
            for role in ("A=col", "A=row"):
                if stop:
                    break
                Aslots = range(N_COL) if role == "A=col" else range(N_ROW)
                Bslots = range(N_ROW) if role == "A=col" else range(N_COL)
                for ca in itertools.combinations(Aslots, 2):
                    if stop:
                        break
                    for cb in itertools.combinations(Bslots, 2):
                        if role == "A=col":
                            rc = [(cb[0], ca[0]), (cb[1], ca[0]),
                                  (cb[0], ca[1]), (cb[1], ca[1])]
                        else:
                            rc = [(ca[0], cb[0]), (ca[0], cb[1]),
                                  (ca[1], cb[0]), (ca[1], cb[1])]
                        # 型別檢查只套在有偵測的角點
                        if any(tps[k] is not None and _tpl_type(r, c) != tps[k]
                               for k, (r, c) in enumerate(rc)):
                            continue
                        corr = [(rc[k][0], rc[k][1], pimg[k],
                                 tps[k] if tps[k] is not None else _tpl_type(*rc[k]))
                                for k in range(4)]
                        bb = _pick_H_sym(corr, vp=vp)
                        if bb is None or not bb["twist"]:
                            continue
                        H, sym = bb["H"], bb["sym"]
                        inl, rmse = _nn_assign(node_pts, node_types, H, sym, thr, subset=sub)
                        if len(inl) < max(MIN_CORR, min(int(0.55 * len(sub)), 10)):  # 絕對下限~10
                            continue
                        key = (len(inl), -(rmse / span))
                        if key > best_key:
                            best_key, best = key, (H, sym, inl, rmse)
                            # 早停只在「夠強且線一致」時觸發（避免停在過不了下游門檻的解）
                            if len(inl) >= good_inl and rmse < good_rmse and \
                               _line_consistency(H, line_members, node_pts, sub) >= 0.95:
                                stop = True; break
    if best is None:
        return None
    H, sym, inl, rmse = best
    H, inl, rmse = _guided_refit(H, sym, node_pts, node_types, sub, span)  # 線外點補位
    return _build_prosac_result(H, inl, rmse, sx, span, "seeded-prosac")


def _solve_quad_prosac(sub_pts, sub_types, pool=None, same_line=None,
                       method="blind-prosac", max_quads=160, max_attempts=6000):
    """點式 PROSAC：在 pool（預設全部點）裡依型別辨識度取 4 點，型別相容地指派範本 id，
    估 H、過 grid-twist、NN 投影驗證。same_line 為共線點對（local idx）；同線兩點被指派到
    非同列也非同行的假設直接淘汰（用線上的共線資訊剪枝）。"""
    n = len(sub_pts)
    if pool is None:
        pool = list(range(n))
    if len(pool) < 4:
        return None
    same_line = same_line or set()
    span = (math.hypot(max(p[0] for p in sub_pts) - min(p[0] for p in sub_pts),
                       max(p[1] for p in sub_pts) - min(p[1] for p in sub_pts)) + 1e-6)
    thr = max(6.0, 0.02 * span)
    W = {"L": 3.0, "X": 2.0, "T": 1.0}
    order = sorted(pool, key=lambda i: -W[sub_types[i]])

    quads, seen = [], set()
    for psz in range(4, len(order) + 1):
        for q in itertools.combinations(order[:psz], 4):
            k = tuple(sorted(q))
            if k in seen:
                continue
            seen.add(k)
            if sum(1 for i in q if sub_types[i] in ("L", "X")) < 2:
                continue
            quads.append(q)
            if len(quads) >= max_quads:
                break
        if len(quads) >= max_quads:
            break

    def _collinear_ok(quad, tids):
        for a in range(4):
            for b in range(a + 1, 4):
                if frozenset((quad[a], quad[b])) in same_line:
                    ta, tb = tids[a], tids[b]
                    if ta // N_COL != tb // N_COL and ta % N_COL != tb % N_COL:
                        return False
        return True

    best, best_key, attempts = None, (-1, 1.0), 0
    for q in quads:
        pimg = [sub_pts[i] for i in q]
        if not _convex_quad_ok(pimg):
            continue
        oimg = _ccw_order(pimg)
        img_ccw = [q[i] for i in oimg]
        types_ccw = [sub_types[i] for i in img_ccw]
        cand_lists = [_CAND_BY_TYPE[t] for t in types_ccw]
        sgn_img = _signed_area([pimg[i] for i in oimg]) > 0
        for sym_name, sym in _SYMS:
            cnt = 0
            for tids in itertools.product(*cand_lists):
                if len(set(tids)) < 4:
                    continue
                if not _collinear_ok(img_ccw, tids):     # 共線硬剪枝
                    continue
                tpl = [_tpl_xy(*sym(t // N_COL, t % N_COL)) for t in tids]
                if abs(_signed_area(tpl)) < 0.2:
                    continue
                if (_signed_area(tpl) > 0) != sgn_img:
                    continue
                cnt += 1
                if cnt > 60:
                    break
                attempts += 1
                if attempts > max_attempts:
                    break
                try:
                    H = _dlt(tpl, [sub_pts[i] for i in img_ccw])
                except Exception:
                    continue
                if H is None or not np.all(np.isfinite(H)) or not _grid_twist_ok(H):
                    continue
                inl, rmse = _nn_assign(sub_pts, sub_types, H, sym, thr)
                if len(inl) < max(MIN_CORR, min(int(0.55 * n), 10)):  # 單座~15點即可被接受
                    continue
                key = (len(inl), -(rmse / span))
                if key > best_key:
                    best_key, best = key, (H, sym, inl, rmse)
            if attempts > max_attempts:
                break
        if attempts > max_attempts:
            break
    if best is None:
        return None
    H, sym, inl, rmse = best
    H, inl, rmse = _guided_refit(H, sym, sub_pts, sub_types,
                                 list(range(n)), span)   # 線外點補位
    return _build_prosac_result(H, inl, rmse, sub_pts, span, method)


def _solve_point_prosac(node_pts, node_types, famA, famB, line_members, line_fits, subset, vp=None):
    """缺線後援（取樣的點一律限定為「落在球場線上」的線點，不退到全部點盲抽）：
      ① 線交點局部 2×2 種子推 H（種子取自偵測角點，或兩線之幾何交點，皆為線點）；
      ② 找不到 2×2 → 只在「線上的點」做 RANSAC（共線剪枝）。
    線點不足 4 個時回傳 None（寧可失敗，也不以非線點亂湊出幾何擬合合理卻 ID 錯位之 H）。"""
    res = _solve_seeded_prosac(node_pts, node_types, famA, famB, line_members, line_fits, subset, vp=vp)
    if res is not None:
        return res

    sub = list(subset)
    loc = {g: i for i, g in enumerate(sub)}        # full idx -> local idx
    sub_pts = [node_pts[g] for g in sub]
    sub_types = [node_types[g] for g in sub]

    # 落在球場線上的點 + 共線點對（local idx）
    lined_local, same_line = set(), set()
    for m in line_members:
        ms = [loc[g] for g in m if g in loc]
        for a in ms:
            lined_local.add(a)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                same_line.add(frozenset((ms[a], ms[b])))

    # ② 只用「線上的點」做 RANSAC；線點不足 4 個則失敗，不退到全部點盲抽
    if len(lined_local) >= 4:
        return _solve_quad_prosac(sub_pts, sub_types, pool=sorted(lined_local),
                                  same_line=same_line, method="lined-prosac")
    return None


def _solve_cross_ratio(famA, famB, line_members, line_fits, node_pts, node_types, vp=None):
    """主解：每族 ≥4 條線，用 cross-ratio 線標號。"""
    transvB = max(famB, key=lambda i: len(line_members[i]))
    transvA = max(famA, key=lambda i: len(line_members[i]))
    csA = _coords_on_transversal(famA, line_fits, line_fits[transvB])
    csB = _coords_on_transversal(famB, line_fits, line_fits[transvA])
    if csA is None or csB is None:
        return None
    mAV, rAV, nAV = _label_family(csA, COL_X)
    mAH, rAH, nAH = _label_family(csA, ROW_Y)
    mBV, rBV, nBV = _label_family(csB, COL_X)
    mBH, rBH, nBH = _label_family(csB, ROW_Y)
    opt1 = (nAV + nBH, -(rAV + rBH), "A=V", mAV, mBH)
    opt2 = (nAH + nBV, -(rAH + rBV), "A=H", mAH, mBV)
    _s, _nr, tag, mA, mB = max(opt1, opt2)
    if min(len(mA), len(mB)) < MIN_LINES:
        return None
    line_label = {}
    A_kind, B_kind = ("V", "H") if tag == "A=V" else ("H", "V")
    for j, li in enumerate(famA):
        if j in mA: line_label[li] = (A_kind, mA[j])
    for j, li in enumerate(famB):
        if j in mB: line_label[li] = (B_kind, mB[j])
    base_corr = _node_rc_from_labels(line_label, line_members, node_pts, node_types)
    if len(base_corr) < MIN_CORR:
        return None
    return _finalize_homography(base_corr, node_pts, "cross-ratio", vp=vp)


def _solve_link_enum(famA, famB, line_members, line_fits, node_pts, node_types,
                     adjacency=None, vp=None):
    """後援（link-PROSAC）：cross-ratio 不可行時，沿連線族「依品質排序」確定性列舉線標號，
    各自解 H，type-aware 重投影 inlier 打分，命中高信心即提早停。
    adjacency: 可選 set of frozenset({i,j})，表示 i,j 在連線圖中相鄰（同一條線）；
    用於共線硬剪枝——相鄰兩點被指派到非同列也非同行者，該假設直接淘汰。"""
    kA, kB = len(famA), len(famB)
    A_types = [[node_types[k] for k in line_members[li]] for li in famA]
    B_types = [[node_types[k] for k in line_members[li]] for li in famB]
    # 線品質：成員多 + 含辨識度高的型別(L/X) 者優先
    def _lq(li):
        ts = [node_types[k] for k in line_members[li]]
        return len(ts) + 0.5 * sum(1 for t in ts if t in ("L", "X"))
    qA = [_lq(li) for li in famA]
    qB = [_lq(li) for li in famB]

    span = (math.hypot(max(p[0] for p in node_pts) - min(p[0] for p in node_pts),
                       max(p[1] for p in node_pts) - min(p[1] for p in node_pts)) + 1e-6)

    node_lines = {}
    for li, m in enumerate(line_members):
        for k in m:
            node_lines.setdefault(k, []).append(li)

    def _collinear_ok(line_label):
        if not adjacency:
            return True
        rc = {}
        for i, lis in node_lines.items():
            col = row = None
            for li in lis:
                lab = line_label.get(li)
                if lab and lab[0] == "V": col = lab[1]
                elif lab: row = lab[1]
            if col is not None and row is not None:
                rc[i] = (row, col)
        for fs in adjacency:
            a, b = tuple(fs)
            if a in rc and b in rc and rc[a][0] != rc[b][0] and rc[a][1] != rc[b][1]:
                return False        # 相鄰卻不同列也不同行 → 不合法
        return True

    # 列舉候選假設，以「總品質」排序（高品質先試 → 早停才有意義）
    cands = []
    for (Ak, Bk, Asl, Bsl) in (("V", "H", N_COL, N_ROW), ("H", "V", N_ROW, N_COL)):
        if kA > Asl or kB > Bsl:
            continue
        for subA in itertools.combinations(range(Asl), kA):
            if any(not _line_type_ok(A_types[j], Ak, subA[j]) for j in range(kA)):
                continue
            for subB in itertools.combinations(range(Bsl), kB):
                if any(not _line_type_ok(B_types[j], Bk, subB[j]) for j in range(kB)):
                    continue
                prio = sum(qA[j] * (subA[j] in (0, Asl - 1)) for j in range(kA)) \
                     + sum(qB[j] * (subB[j] in (0, Bsl - 1)) for j in range(kB))
                cands.append((-prio, Ak, Bk, subA, subB))
    cands.sort(key=lambda c: c[0])

    best, best_key = None, (-1, 1.0)
    for _p, Ak, Bk, subA, subB in cands:
        line_label = {}
        for j, li in enumerate(famA): line_label[li] = (Ak, subA[j])
        for j, li in enumerate(famB): line_label[li] = (Bk, subB[j])
        if not _collinear_ok(line_label):
            continue
        bc = _node_rc_from_labels(line_label, line_members, node_pts, node_types)
        if len(bc) < 5:                       # <5 點 H 不可靠（4 點必過擬合）
            continue
        res = _finalize_homography(bc, node_pts, "link-enum", vp=vp)
        if res is None:
            continue
        key = (res["type_ok"], -(res["rmse"] / span))
        if key > best_key:
            best_key, best = key, res
        if res["rmse"] < 1.0 and res["num_corr"] >= 8:   # 近乎完美才早停
            break
    if best is None:
        return None
    # 收緊接受門檻：型別需全一致、RMSE 須夠小，否則寧可不給
    if best["type_ok"] < best["num_corr"] or best["rmse"] > 0.025 * span:
        return None
    return best


def solve_court_homography(node_pts, node_types, line_members, line_fits,
                           adjacency=None, node_subset=None):
    """
    node_pts    : [(x,y), ...]            影像座標
    node_types  : ['L'/'T'/'X', ...]
    line_members: [[node_idx,...], ...]   每條球場線的成員交點
    line_fits   : [(vx,vy,x0,y0), ...]    對應每條線的無限直線（與 line_members 對齊）
    adjacency   : 可選 set of frozenset({i,j})，連線圖相鄰對（給後援做共線剪枝）
    主解 cross-ratio 線標號（每族 ≥4 線）；不足則改走 link-PROSAC 後援。
    """
    centroids = [tuple(np.mean([node_pts[k] for k in m], axis=0)) if m else (0, 0)
                 for m in line_members]
    dirs = [(f[0], f[1]) for f in line_fits]
    famA, famB = _split_families(dirs, line_fits, node_pts) if len(dirs) >= 2 else ([], [])
    subset = node_subset if node_subset is not None else list(range(len(node_pts)))
    sx = [node_pts[i] for i in subset] if subset else node_pts
    span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                       max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6) if sx else 1.0
    vp = None

    # 候選收集 + 線一致性挑選（無 chamfer）：便宜的線層若給出高一致性解就早停（維持原始版的快），
    # 否則收集各層候選、挑「線一致性最高」者——擋掉 link-enum 那種自洽卻錯的解（如 _5 lc=0.667）。
    cands = []

    def consider(res):
        if res is None or res.get("status") != "ok":
            return None
        Hc = np.asarray(res["H"], float)
        res["line_consistency"] = _line_consistency(Hc, line_members, node_pts, subset)
        tc, tm, tt = _type_consistency(Hc, node_pts, node_types, subset)
        res["type_consistency"] = tc
        res["type_match"] = (tm, tt)
        cands.append(res)
        return res

    def good(res):
        return (res is not None and res.get("line_consistency", 0.0) >= 0.95
                and res.get("type_consistency", 0.0) >= 0.8)

    tmr = {"cross_ratio_ms": 0.0, "link_enum_ms": 0.0, "point_prosac_ms": 0.0}
    cx = {"n_nodes": len(subset), "n_lines": len(line_members),
          "famA": len(famA), "famB": len(famB), "n_candidates": 0}

    if min(len(famA), len(famB)) >= 2:
        famA = _order_family(famA, line_fits, centroids)
        famB = _order_family(famB, line_fits, centroids)
        vp = _depth_vp(famA, famB, line_fits)
        cx["famA"], cx["famB"] = len(famA), len(famB)
        if len(famA) >= MIN_LINES and len(famB) >= MIN_LINES:
            _t = time.perf_counter()
            r = consider(_solve_cross_ratio(famA, famB, line_members, line_fits,
                                            node_pts, node_types, vp=vp))
            tmr["cross_ratio_ms"] += (time.perf_counter() - _t) * 1000
            if good(r):
                r["timing"] = tmr; r["complexity"] = {**cx, "n_candidates": len(cands)}
                return r
        _t = time.perf_counter()
        r = consider(_solve_link_enum(famA, famB, line_members, line_fits,
                                      node_pts, node_types, adjacency=adjacency, vp=vp))
        tmr["link_enum_ms"] += (time.perf_counter() - _t) * 1000
        if good(r):
            r["timing"] = tmr; r["complexity"] = {**cx, "n_candidates": len(cands)}
            return r

    # 便宜層沒有高一致性解 → 跑點式後援（種子 2×2 → 線上點；一律只用線點、不盲抽）
    if not any(good(c) for c in cands):
        _t = time.perf_counter()
        consider(_solve_point_prosac(node_pts, node_types, famA, famB, line_members,
                                     line_fits, subset, vp=vp))
        tmr["point_prosac_ms"] += (time.perf_counter() - _t) * 1000

    cx["n_candidates"] = len(cands)
    if not cands:
        return {"status": "fail",
                "reason": f"無法求得可靠 H（線族 A={len(famA)} / B={len(famB)}；"
                          f"cross-ratio / link-PROSAC / 點式 PROSAC 皆未通過）",
                "timing": tmr, "complexity": cx}

    best = max(cands, key=lambda c: (c.get("line_consistency", 0.0) + c.get("type_consistency", 0.0),
                                     c.get("num_corr", 0),
                                     -c.get("rmse", 1e9) / max(c.get("span", span), 1)))
    best["timing"] = tmr; best["complexity"] = cx
    if best.get("line_consistency", 0.0) < MIN_LC:   # 最佳候選都太不一致 → 寧可判失敗、不畫爛 H
        return {"status": "fail",
                "reason": f"最佳候選線一致性過低（lc={best.get('line_consistency',0):.2f}"
                          f" < {MIN_LC}；method={best.get('method')}，可能線太稀/球場被切碎）",
                "timing": tmr, "complexity": cx}
    if best.get("type_consistency", 1.0) < MIN_TC:   # 型別對不上 → H 八成錯位/翻轉
        return {"status": "fail",
                "reason": f"型別一致性過低（type={best.get('type_consistency',0):.2f}"
                          f" < {MIN_TC}，inlier 類型與投影點類型不符 → H 可能錯位/翻轉）",
                "timing": tmr, "complexity": cx}
    return best


# ──────────────────────────────────────────────
# 把 folder_yolo_tool 的線/掛點結果整理成各球場的輸入
# ──────────────────────────────────────────────
def _assign_cells_simple(H, node_pts, node_types, subset, thr):
    """型別懲罰貪婪 NN：把 subset 內每個偵測點指到一個範本格 (r,c)。回傳 {node_idx:(r,c)}。"""
    cells = []
    for r in range(N_ROW):
        for c in range(N_COL):
            p = _proj(H, _tpl_xy(r, c))
            if math.isfinite(p[0]) and math.isfinite(p[1]):
                cells.append((r, c, p, _tpl_type(r, c)))
    pairs = []
    for i in subset:
        x, y = node_pts[i]; t = node_types[i]
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
            used_i.add(i); used_c.add(rc); out[i] = rc
    return out


def refine_homography_steger(H, img_bgr, node_pts, node_types, subset,
                             dark=False, iters=2, eps=0.05):
    """Jacobian 導引的雙軸 Steger 次像素精修：
      對每個已標 (r,c) 的交點，用 H 算出該點影像位置 + 兩條球場線方向（H 的 Jacobian），
      沿兩軸把 bbox/窗依局部尺度擴大、重跑 Steger，脊點分到兩軸各擬一條線，交點即次像素交點；
      用這些精準交點重擬合 H，迭代收斂。回傳 (H_refined, n_refined)。"""
    if img_bgr is None:
        return np.asarray(H, float), 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    Himg, Wimg = gray.shape[:2]
    bright = not dark
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
            pr = np.array(_proj(curH, (X + eps, Y)), float)   # 沿 COL_X：橫線(row line)方向
            pc = np.array(_proj(curH, (X, Y + eps)), float)   # 沿 ROW_Y：縱線(col line)方向
            a_row = (pr - p0) / eps; a_col = (pc - p0) / eps
            nr = np.linalg.norm(a_row); nc = np.linalg.norm(a_col)
            fallback = (node_pts[i] if i < len(node_pts) else (float(p0[0]), float(p0[1])))
            if nr < 1e-6 or nc < 1e-6 or not all(np.isfinite(p0)):
                corr.append(((X, Y), fallback)); continue
            ur = a_row / nr; uc = a_col / nc
            ext_r = min(max(0.35 * nr, 10.0), 60.0)           # 依 Jacobian 局部尺度擴大窗
            ext_c = min(max(0.35 * nc, 10.0), 60.0)
            corners = [p0 + sr * ext_r * ur + sc * ext_c * uc
                       for sr in (-1, 1) for sc in (-1, 1)]
            xs = [q[0] for q in corners]; ys = [q[1] for q in corners]
            x0 = int(max(0, min(xs) - 3)); x1 = int(min(Wimg, max(xs) + 3))
            y0 = int(max(0, min(ys) - 3)); y1 = int(min(Himg, max(ys) + 3))
            if x1 - x0 < 6 or y1 - y0 < 6:
                corr.append(((X, Y), fallback)); continue
            rp = _steger_ridge_points_simple(gray[y0:y1, x0:x1], sigma=1.2,
                                                  threshold_pct=0.15, bright_lines=bright)
            if len(rp) < 6:
                corr.append(((X, Y), fallback)); continue
            rp = rp.astype(float) + np.array([x0, y0])
            rel = rp - p0
            n_row = np.array([-ur[1], ur[0]]); n_col = np.array([-uc[1], uc[0]])
            d_row = np.abs(rel @ n_row); d_col = np.abs(rel @ n_col)
            al_row = rel @ ur; al_col = rel @ uc
            ltol = max(3.0, 0.04 * min(nr, nc))
            row_pts = rp[(d_row < ltol) & (np.abs(al_row) > ltol)]   # 屬橫線、離中心夠遠
            col_pts = rp[(d_col < ltol) & (np.abs(al_col) > ltol)]
            Lr = Lc = None
            if len(row_pts) >= 4:
                ctr = row_pts.mean(0); _, _, vt = np.linalg.svd(row_pts - ctr)
                Lr = (vt[0][0], vt[0][1], ctr[0], ctr[1])
            if len(col_pts) >= 4:
                ctr = col_pts.mean(0); _, _, vt = np.linalg.svd(col_pts - ctr)
                Lc = (vt[0][0], vt[0][1], ctr[0], ctr[1])
            if Lr is not None and Lc is not None:
                Xp = _line_intersection_simple(Lr, Lc)
                if Xp is not None and all(math.isfinite(v) for v in Xp) and \
                        math.hypot(Xp[0] - p0[0], Xp[1] - p0[1]) < 0.6 * max(ext_r, ext_c):
                    corr.append(((X, Y), (float(Xp[0]), float(Xp[1])))); n_ref += 1
                    continue
            corr.append(((X, Y), fallback))                  # 兩軸不全 → 退回原偵測
        if len(corr) < 4:
            break
        try:
            newH = _dlt([s for s, _ in corr], [d for _, d in corr])
        except Exception:
            break
        if newH is not None and np.all(np.isfinite(newH)) and _grid_twist_ok(newH):
            curH = newH
        else:
            break
    return curH, n_ref


def _refit_lines_subset(line_members, line_fits, node_pts, keep):
    """把線群限縮到 keep 節點集合，重擬合。回傳 (members, fits)。"""
    out_m, out_f = [], []
    for m in line_members:
        mm = [k for k in m if k in keep]
        if len(mm) < 2:
            continue
        P = np.asarray([node_pts[k] for k in mm], float); ctr = P.mean(0)
        _, _, vt = np.linalg.svd(P - ctr)
        out_m.append(mm); out_f.append((float(vt[0][0]), float(vt[0][1]),
                                        float(ctr[0]), float(ctr[1])))
    return out_m, out_f


def _solve_multi_court(node_pts, node_types, line_members, line_fits, nodes,
                       adjacency=None, max_courts=4):
    """循序多球場擬合：在節點集合裡鎖定一座 → 取走其 inlier → 在剩下的點再找下一座。
    解決『兩座被合併成一個分量』（Steger 線跨座相連）的情形。回傳 list of result。"""
    ident = lambda r, c: (r, c)
    out = []
    remaining = set(nodes)
    for _ in range(max_courts):
        if len(remaining) < MIN_COURT_NODES:
            break
        sub = sorted(remaining)
        lm, lf = _refit_lines_subset(line_members, line_fits, node_pts, remaining)
        _t = time.perf_counter()
        res = solve_court_homography(node_pts, node_types, lm, lf,
                                     adjacency=adjacency, node_subset=sub)
        this_solve_ms = (time.perf_counter() - _t) * 1000     # 這一座自己的求解時間
        if res.get("status") != "ok":
            break
        res.setdefault("timing", {})["court_solve_ms"] = this_solve_ms
        sx = [node_pts[i] for i in sub]
        span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                           max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
        inl, _r = _nn_assign(node_pts, node_types, np.asarray(res["H"], float),
                             ident, max(6.0, 0.02 * span), subset=sub)
        if len(inl) < MIN_CORR:
            break
        res["inlier_nodes"] = sorted(inl)
        out.append(res)
        remaining -= set(inl)
        if len(remaining) <= 2:                 # 幾乎全覆蓋 → 收工（多為單座）
            break
    return out


def courts_from_image(img_bgr, anns, class_names, dark=False):
    """回傳 list of court dict：{node_idx, node_pts, node_types, line_members, line_fits}。
    以「共用球場線」連通分量分群。"""
    lines, _ridge = _compute_court_lines(img_bgr, anns, dark=dark)
    if not lines:
        return [], []
    gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if (img_bgr is not None and img_bgr.ndim == 3) else img_bgr)
    assign = _assign_junction_lines(anns, class_names, lines, gray=gray, dark=dark)
    node_pts   = [tuple(n["pt"]) for n in assign]
    node_types = [n["type"] for n in assign]
    node_lines = [list(n["lines"]) for n in assign]

    # 線 → 成員
    line_members = {li: [] for li in range(len(lines))}
    for ni, lis in enumerate(node_lines):
        for li in lis:
            line_members[li].append(ni)

    # 連通分量（node 間共用線即相連）
    parent = list(range(len(node_pts)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for m in line_members.values():
        for k in range(1, len(m)):
            union(m[0], m[k])
    comps = {}
    for i in range(len(node_pts)):
        comps.setdefault(find(i), []).append(i)

    courts = []
    for nodes in comps.values():
        if len(nodes) < MIN_COURT_NODES:          # 單點/碎片不是球場 → 跳過（避免假失敗污染）
            continue
        nodeset = set(nodes)
        loc_members, loc_fits = [], []
        for li, m in line_members.items():
            mm = [k for k in m if k in nodeset]
            if len(mm) >= 2:
                loc_members.append(mm)
                loc_fits.append(tuple(lines[li]["Limg"]))
        if not loc_members:                        # 沒有線 → 無法當球場
            continue
        loc_members, loc_fits = _merge_lines(loc_members, loc_fits, node_pts)
        loc_members, loc_fits = _augment_line_members(
            loc_members, loc_fits, node_pts, allow=nodeset, tol=4.0)
        courts.append({"nodes": nodes, "node_pts": node_pts,
                       "node_types": node_types,
                       "line_members": loc_members, "line_fits": loc_fits})
    return courts, assign


def solve_image(img_bgr, anns, class_names, dark=False, steger_refine=True):
    """對整張影像所有球場求 H。每個連通分量先循序多球場擬合（處理兩座被連成一個分量），
    再對每座做 Jacobian 導引的雙軸 Steger 次像素精修出最終 H（不退步才採用）。
    每座結果含 timing（各階段毫秒）與 complexity（節點/線/族/候選 數）。"""
    _t0 = time.perf_counter()
    courts, _ = courts_from_image(img_bgr, anns, class_names, dark=dark)
    line_ms = (time.perf_counter() - _t0) * 1000        # Steger 全域抽線（通常最重）
    out = []
    ci = 0
    for ct in courts:
        npts, ntypes = ct["node_pts"], ct["node_types"]
        _t = time.perf_counter()
        results = _solve_multi_court(npts, ntypes, ct["line_members"],
                                     ct["line_fits"], ct["nodes"])
        solve_ms = (time.perf_counter() - _t) * 1000
        if not results:
            res = solve_court_homography(npts, ntypes, ct["line_members"],
                                         ct["line_fits"], node_subset=ct["nodes"])
            _refine_inplace_hungarian(res, npts, ntypes, ct["nodes"], ct["line_members"])
            tm = res.setdefault("timing", {})
            tm["line_extract_ms"] = line_ms; tm["solve_ms"] = solve_ms
            tm["total_ms"] = line_ms + solve_ms
            res["court"] = ci; res["num_nodes"] = len(ct["nodes"]); ci += 1
            out.append(res); continue
        for res in results:
            ref_ms = relabel_ms = hun_ms = 0.0
            sub = res.get("inlier_nodes") or ct["nodes"]
            if steger_refine and img_bgr is not None:
                _t = time.perf_counter()
                _refine_inplace(res, img_bgr, npts, ntypes, sub,
                                ct["line_members"], dark)
                ref_ms = (time.perf_counter() - _t) * 1000
            # 最終精化：用『所有 inlier』以匈牙利演算法配 tid、全部對應重算 H（非僅 4 點）
            _t = time.perf_counter()
            _refine_inplace_hungarian(res, npts, ntypes, sub, ct["line_members"])
            hun_ms = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
            _apply_vp_relabel(res, npts, ntypes)
            relabel_ms = (time.perf_counter() - _t) * 1000
            tm = res.setdefault("timing", {})
            tm["line_extract_ms"] = line_ms; tm["solve_ms"] = solve_ms
            tm["steger_refine_ms"] = ref_ms; tm["hungarian_refit_ms"] = hun_ms
            tm["vp_relabel_ms"] = relabel_ms
            tm["total_ms"] = line_ms + solve_ms + ref_ms + hun_ms + relabel_ms
            res["court"] = ci; res["num_nodes"] = len(ct["nodes"]); ci += 1
            out.append(res)
    return out


def _apply_vp_relabel(res, node_pts, node_types):
    """用消失點把範本 id 定為 0~29（左下→右上），只改 label 不動 H/像素。
    為每個投影點加 template_id；為每個 inlier 加 (node_idx → template_id)。"""
    if res.get("status") != "ok" or "projected" not in res:
        return
    proj = {(p["row"], p["col"]): tuple(p["xy"]) for p in res["projected"]
            if all(math.isfinite(v) for v in p["xy"])}
    if len(proj) < 8:
        return
    idmap, flip_y, flip_x, dbg = _vp_relabel_ids(proj)
    for p in res["projected"]:
        p["template_id"] = idmap.get((p["row"], p["col"]))
    res["vp_relabel"] = {"flip_y": flip_y, "flip_x": flip_x, "debug": dbg}
    sub = res.get("inlier_nodes") or []
    if sub:
        cells = list(proj.keys()); PP = np.array([proj[c] for c in cells], float)
        sx = [node_pts[i] for i in sub]
        span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                           max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
        thr = max(8.0, 0.04 * span)
        inl_ids = {}
        for i in sub:
            x, y = node_pts[i]
            d = np.hypot(PP[:, 0] - x, PP[:, 1] - y); j = int(d.argmin())
            if d[j] <= thr:
                inl_ids[i] = idmap.get(cells[j])
        res["inlier_template_ids"] = inl_ids


def _refine_H_hungarian(H, node_pts, node_types, subset, iters=3):
    """最終精化：以匈牙利演算法（全域最優一對一指派）將 subset 內『所有』inlier 配到
    30 個模板交點，再用『全部』對應 DLT 重算 H（非僅 4 點），迭代至 H 穩定。

    成本 = 投影距離 + 型別不符懲罰（與貪婪 NN 同一成本，但此處求全域最小總成本而非逐一貪婪）；
    僅實際距離 ≤ thr 之指派計入內點。回傳 (H_refined, n_corr)。scipy 不可用時原樣回傳。
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return np.asarray(H, float), 0

    sub = list(subset) if subset is not None else list(range(len(node_pts)))
    if len(sub) < 4:
        return np.asarray(H, float), 0
    sx = [node_pts[i] for i in sub]
    span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                       max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
    thr = max(6.0, 0.02 * span)
    tids = [(r, c) for r in range(N_ROW) for c in range(N_COL)]      # 30 個模板交點
    BIG = thr * 100.0

    curH = np.asarray(H, float)
    n_best = 0
    for _ in range(max(1, iters)):
        proj = [(_proj(curH, _tpl_xy(r, c)), _tpl_type(r, c)) for (r, c) in tids]
        # 成本矩陣 (len(sub) × 30)
        cost = np.full((len(sub), len(tids)), BIG, float)
        for a, di in enumerate(sub):
            x, y = node_pts[di]
            t = node_types[di]
            for b, (pp, tt) in enumerate(proj):
                if not (math.isfinite(pp[0]) and math.isfinite(pp[1])):
                    continue
                d = math.hypot(pp[0] - x, pp[1] - y)
                if d <= thr * 3:                                    # 太遠者不納入候選
                    cost[a, b] = d + (0.0 if tt == t else thr)      # 型別不符加懲罰
        rows, cols = linear_sum_assignment(cost)                    # 全域最優一對一
        src, dst = [], []
        for a, b in zip(rows, cols):
            if cost[a, b] >= BIG:                                   # 未實際配上
                continue
            di = sub[a]
            r, c = tids[b]
            pp = proj[b][0]
            if math.hypot(pp[0] - node_pts[di][0], pp[1] - node_pts[di][1]) <= thr:
                src.append(_tpl_xy(r, c))
                dst.append(node_pts[di])
        if len(src) < 4:
            break
        try:
            newH = _dlt(src, dst)                                   # 全部對應重算 H
        except Exception:
            break
        if newH is None or not np.all(np.isfinite(newH)) or not _grid_twist_ok(newH):
            break
        n_best = len(src)
        if np.allclose(newH, curH, atol=1e-6):                      # 已收斂
            curH = newH
            break
        curH = newH
    return curH, n_best


def _refine_inplace_hungarian(res, node_pts, node_types, subset, line_members):
    """最終精化（就地）：以匈牙利演算法配所有 inlier 之 tid、全部對應重算 H。
    只在不退化、格子不折疊、線一致性不變差時採用；並更新 res 的 H/projected。"""
    if res is None or res.get("status") != "ok" or res.get("H") is None:
        return
    H0 = np.asarray(res["H"], float)
    H1, ncorr = _refine_H_hungarian(H0, node_pts, node_types, subset)
    if ncorr < 4 or not _grid_twist_ok(H1):
        return
    sx = [node_pts[i] for i in subset]
    if not sx:
        return
    span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                       max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
    cen = (float(np.mean([p[0] for p in sx])), float(np.mean([p[1] for p in sx])))
    if _h_degenerate(H1, span, cen):
        return
    lc0 = _line_consistency(H0, line_members, node_pts, subset)
    lc1 = _line_consistency(H1, line_members, node_pts, subset)
    if lc1 < lc0 - 1e-9:                                            # 線一致性變差 → 不採用
        return
    pts = np.asarray(sx, float)
    projected = []
    for r in range(N_ROW):
        for c in range(N_COL):
            px = _proj(H1, _tpl_xy(r, c))
            dmin = (float(np.min(np.hypot(pts[:, 0] - px[0], pts[:, 1] - px[1])))
                    if len(pts) else 1e9)
            projected.append({"row": r, "col": c, "type": _tpl_type(r, c),
                              "xy": [float(px[0]), float(px[1])],
                              "recovered": dmin > RECOVER_TOL})
    res["H"] = H1.tolist()
    res["projected"] = projected
    res["line_consistency"] = lc1
    res["method"] = res.get("method", "?") + "+hungarian-refit"
    res["hungarian_corr"] = ncorr


def _refine_inplace(res, img_bgr, node_pts, node_types, subset, line_members, dark):
    """對單座結果做雙軸 Steger 精修；只在不退化、格子不折疊、線一致性不變差時採用。"""
    H0 = np.asarray(res["H"], float)
    H1, nref = refine_homography_steger(H0, img_bgr, node_pts, node_types, subset, dark=dark)
    if nref < 4 or not _grid_twist_ok(H1):
        return
    sx = [node_pts[i] for i in subset]
    span = (math.hypot(max(p[0] for p in sx) - min(p[0] for p in sx),
                       max(p[1] for p in sx) - min(p[1] for p in sx)) + 1e-6)
    cen = (float(np.mean([p[0] for p in sx])), float(np.mean([p[1] for p in sx])))
    if _h_degenerate(H1, span, cen):
        return
    lc0 = _line_consistency(H0, line_members, node_pts, subset)
    lc1 = _line_consistency(H1, line_members, node_pts, subset)
    if lc1 < lc0 - 1e-9:
        return
    pts = np.asarray(sx, float)
    projected = []
    for r in range(N_ROW):
        for c in range(N_COL):
            px = _proj(H1, _tpl_xy(r, c))
            dmin = (float(np.min(np.hypot(pts[:, 0] - px[0], pts[:, 1] - px[1])))
                    if len(pts) else 1e9)
            projected.append({"row": r, "col": c, "type": _tpl_type(r, c),
                              "xy": [float(px[0]), float(px[1])],
                              "recovered": dmin > RECOVER_TOL})
    res["H"] = H1.tolist(); res["projected"] = projected
    res["line_consistency"] = lc1
    res["method"] = res.get("method", "?") + "+steger-refine"
    res["steger_refined"] = nref
    res.setdefault("complexity", {})["steger_rois"] = len(subset)


def _json_default(o):
    """讓 numpy 數值/陣列可被 json 序列化。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _image_timing(results, wall_ms):
    """由單張的各座 result 與實測 wall time，彙整誠實的影像層級時間（避免重複計）。
    line：整張共用、只計一次；refine/relabel：各座加總；solve：wall 扣掉其餘。"""
    line = 0.0; refine = 0.0; relabel = 0.0
    for r in results:
        tm = r.get("timing", {})
        line = max(line, tm.get("line_extract_ms", 0.0))    # 共用值，取一次
        if r.get("status") == "ok":
            refine += tm.get("steger_refine_ms", 0.0)
            relabel += tm.get("vp_relabel_ms", 0.0)
    if not results:
        line = wall_ms
    solve = max(0.0, wall_ms - line - refine - relabel)
    return {"line_extract_ms": round(line, 1), "solve_ms": round(solve, 1),
            "steger_refine_ms": round(refine, 1), "vp_relabel_ms": round(relabel, 1),
            "total_ms": round(wall_ms, 1)}
