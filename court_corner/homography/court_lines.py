"""
homography/court_lines.py — 球場白線抽取（自 folder_yolo_tool 移植，去除 GUI）
================================================================
全域 Steger 抽線、交點掛線、型別判讀等非 GUI 演算法，供線為主求 H 使用。
（原始來源：使用者的 folder_yolo_tool.py；此處為內嵌移植，不再外部引用。）
"""

import sys, os, random, math, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ──────────────────────────────────────────────
# 資料結構
# ──────────────────────────────────────────────
@dataclass
class ImageItem:
    id: int
    file_name: str        # 相對 image_root 的路徑（顯示用）
    abs_path: str
    width: int = 0
    height: int = 0

@dataclass
class Annotation:
    image_id: int
    class_id: int          # 直接就是 YOLO class id
    bbox: List[float]      # [x, y, w, h]  像素座標
    ann_id: int = -1

# ──────────────────────────────────────────────
# 顏色工具
# ──────────────────────────────────────────────
_COLOR_CACHE: Dict[int, Tuple[int,int,int]] = {}

def class_color(cls_id: int) -> Tuple[int,int,int]:
    if cls_id not in _COLOR_CACHE:
        rng = random.Random(cls_id + 42)
        _COLOR_CACHE[cls_id] = (rng.randint(80,255), rng.randint(80,255), rng.randint(80,255))
    return _COLOR_CACHE[cls_id]

def _steger_ridge_points_simple(
    roi_gray: np.ndarray,
    sigma: float = 1.2,
    threshold_pct: float = 0.18,
    bright_lines: bool = True,
) -> np.ndarray:
    if roi_gray is None or roi_gray.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    g = roi_gray.astype(np.float64)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    Lx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    Ly = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    Lxx = cv2.Sobel(g, cv2.CV_64F, 2, 0, ksize=3)
    Lyy = cv2.Sobel(g, cv2.CV_64F, 0, 2, ksize=3)
    Lxy = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)
    trace = Lxx + Lyy
    diff = Lxx - Lyy
    root = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * Lxy * Lxy))
    lam1 = 0.5 * (trace + root)
    lam2 = 0.5 * (trace - root)
    use_lam1 = np.abs(lam1) >= np.abs(lam2)
    ridge_lam = np.where(use_lam1, lam1, lam2)
    strength = np.abs(ridge_lam)
    max_strength = float(np.max(strength)) if strength.size else 0.0
    if max_strength <= 1e-9:
        return np.zeros((0, 2), dtype=np.float32)
    thr = threshold_pct * max_strength
    h, w = g.shape[:2]
    pts = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            lam = float(ridge_lam[y, x])
            if strength[y, x] < thr:
                continue
            if bright_lines and lam >= 0:
                continue
            if (not bright_lines) and lam <= 0:
                continue
            vx = float(Lxy[y, x])
            vy = float(lam - Lxx[y, x])
            nrm = math.hypot(vx, vy)
            if nrm < 1e-9:
                vx, vy = 1.0, 0.0
            else:
                vx, vy = vx / nrm, vy / nrm
            gx = float(Lx[y, x])
            gy = float(Ly[y, x])
            hnn = float(vx * vx * Lxx[y, x] + 2.0 * vx * vy * Lxy[y, x] + vy * vy * Lyy[y, x])
            if abs(hnn) < 1e-9:
                continue
            t = -float(vx * gx + vy * gy) / hnn
            if abs(t) <= 0.5:
                pts.append((x + t * vx, y + t * vy))
    return np.asarray(pts, dtype=np.float32)




def _line_intersection_simple(line1, line2):
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
    return (x1 + t * vx1, y1 + t * vy1)




def _infer_junction_type(name: Optional[str]) -> Optional[str]:
    """由類別名稱推斷交點形狀，回傳 'L' / 'T' / 'X'；無法判斷回傳 None。
    回傳 None 時 _draw_steger 會維持原本的雙軸（整條）畫法。
    若你的資料集類別命名規則不同，只需調整這個函式即可。"""
    if not name:
        return None
    s = str(name).strip().upper()
    if s in ("L", "T", "X"):
        return s
    if "CROSS" in s or "十字" in s:
        return "X"
    # 取名稱中第一個出現的英文字母作為形狀代號（如 "L_corner"、"T-junction"、"X3"）
    for ch in s:
        if ch in ("L", "T", "X"):
            return ch
        if ch.isalpha():
            break
    return None








# 類型 → 可連線方向數上限（L 角點 2、T 3、X 十字 4）
_TYPE_DEGREE = {"L": 2, "T": 3, "X": 4}


def _extract_global_court_lines(region_gray, *, sigma=1.2, threshold_pct=0.14,
                                dark=False, line_thr=2.0, min_inliers=40,
                                min_span=40.0, merge_ang_deg=4.0, merge_rho=5.0,
                                max_iter=60, seed=0):
    """在一塊大區域上跑一次 Steger，用多次 RANSAC 抽出整條球場線，
    再依 (方向角, 帶號垂距) 合併重複線。回傳 list of dict
    {L:(vx,vy,x0,y0) 區域座標, n:inlier數, span:線段長}。"""
    pts = _steger_ridge_points_simple(region_gray, sigma=sigma,
                                      threshold_pct=threshold_pct,
                                      bright_lines=not dark)
    if len(pts) < min_inliers:
        return [], pts
    P = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(max_iter):
        if len(P) < min_inliers:
            break
        best_n, best_ins = 0, None
        for _ in range(300):
            idx = rng.choice(len(P), 2, replace=False)
            d = P[idx[1]] - P[idx[0]]
            nrm = float(np.linalg.norm(d))
            if nrm < 5.0:
                continue
            vx, vy = d / nrm
            nx, ny = -vy, vx
            dist = np.abs((P[:, 0] - P[idx[0], 0]) * nx + (P[:, 1] - P[idx[0], 1]) * ny)
            ins = dist < line_thr
            c = int(ins.sum())
            if c > best_n:
                best_n, best_ins = c, ins
        if best_n < min_inliers or best_ins is None:
            break
        L = np.asarray(cv2.fitLine(P[best_ins], cv2.DIST_L2, 0, 0.01, 0.01),
                       dtype=np.float64).reshape(-1)
        proj = (P[best_ins, 0] - L[2]) * L[0] + (P[best_ins, 1] - L[3]) * L[1]
        span = float(proj.max() - proj.min())
        if span >= min_span:
            raw.append({"L": L, "n": best_n, "span": span})
        P = P[~best_ins]

    # 合併近重複線
    def feats(L):
        vx, vy = float(L[0]), float(L[1])
        th = math.atan2(vy, vx) % math.pi
        nx, ny = -vy, vx
        rho = float(L[2] * nx + L[3] * ny)
        if nx < 0 or (abs(nx) < 1e-6 and ny < 0):
            rho = -rho
        return th, rho

    merged = []
    for r in sorted(raw, key=lambda t: -t["n"]):
        th, rho = feats(r["L"])
        dup = False
        for m in merged:
            dth = abs(th - m["th"]); dth = min(dth, math.pi - dth)
            if math.degrees(dth) < merge_ang_deg and abs(rho - m["rho"]) < merge_rho:
                dup = True
                break
        if not dup:
            r["th"], r["rho"] = th, rho
            merged.append(r)
    return merged, pts




def _half_segment_support(pa, pb, ridge_xy, *, frac=0.55, lat_tol=2.5,
                          step=3.0, min_cover=0.55):
    """檢查線段「靠近 pa 那一半」是否有脊點（pa 這側是否真的有畫出的線）。
    用於判斷某一端是否有 Steger 線支撐這條連線。"""
    ax, ay = pa; bx, by = pb
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6 or ridge_xy is None or len(ridge_xy) == 0:
        return False
    dx, dy = dx / L, dy / L
    nx, ny = -dy, dx
    Lh = max(step, frac * L)
    s = (ridge_xy[:, 0] - ax) * dx + (ridge_xy[:, 1] - ay) * dy
    lat = np.abs((ridge_xy[:, 0] - ax) * nx + (ridge_xy[:, 1] - ay) * ny)
    on = (lat < lat_tol) & (s >= 0.0) & (s <= Lh)
    if not np.any(on):
        return False
    nb = max(1, int(Lh / step))
    bins = np.clip((s[on] / Lh * nb).astype(int), 0, nb - 1)
    return (len(np.unique(bins)) / nb) >= min_cover


def _compute_court_lines(img_bgr, anns, *, pad_frac=0.06, sigma=1.2,
                         threshold_pct=0.14, dark=False):
    """在所有 bbox 圍出的大範圍上跑一次 Steger，抽出整條球場線。
    回傳 (lines, ridge_img)，lines 內含 'Limg'（原圖座標的 (vx,vy,x0,y0)），
    ridge_img 為原圖座標的脊點。"""
    if not anns:
        return [], np.zeros((0, 2), dtype=np.float32)
    H, W = img_bgr.shape[:2]
    xs1 = min(a.bbox[0] for a in anns); ys1 = min(a.bbox[1] for a in anns)
    xs2 = max(a.bbox[0] + a.bbox[2] for a in anns)
    ys2 = max(a.bbox[1] + a.bbox[3] for a in anns)
    mx = (xs2 - xs1) * pad_frac; my = (ys2 - ys1) * pad_frac
    RX1 = max(0, int(xs1 - mx)); RY1 = max(0, int(ys1 - my))
    RX2 = min(W, int(xs2 + mx)); RY2 = min(H, int(ys2 + my))
    if RX2 - RX1 < 5 or RY2 - RY1 < 5:
        return [], np.zeros((0, 2), dtype=np.float32)
    region = cv2.cvtColor(img_bgr[RY1:RY2, RX1:RX2], cv2.COLOR_BGR2GRAY) \
        if img_bgr.ndim == 3 else img_bgr[RY1:RY2, RX1:RX2]
    lines, ridge = _extract_global_court_lines(region, sigma=sigma,
                                               threshold_pct=threshold_pct, dark=dark)
    for m in lines:
        m["Limg"] = (float(m["L"][0]), float(m["L"][1]),
                     float(m["L"][2] + RX1), float(m["L"][3] + RY1))
    ridge_img = np.asarray(ridge, dtype=np.float32).reshape(-1, 2).copy()
    if len(ridge_img):
        ridge_img[:, 0] += RX1
        ridge_img[:, 1] += RY1
    return lines, ridge_img


def _assign_junction_lines(anns, class_names, lines, *, on_line_tol=4.0):
    """把每個交點掛到最近通過它的（最多 2）條球場線。
    回傳 nodes：{ann_id, type, pt, lines(索引), axis_lines(Limg), cap}。
    pt 為兩條入射線的交點（若有 2 條且交點落在 bbox 內），否則 bbox 中心。"""
    def pt_line_dist(Limg, px, py):
        vx, vy, x0, y0 = Limg
        nx, ny = -vy, vx
        return abs((px - x0) * nx + (py - y0) * ny)

    nodes = []
    for a in anns:
        cx, cy = a.bbox[0] + a.bbox[2] / 2.0, a.bbox[1] + a.bbox[3] / 2.0
        jtype = _infer_junction_type(class_names.get(a.class_id, ""))
        cand = sorted(((pt_line_dist(m["Limg"], cx, cy), li)
                       for li, m in enumerate(lines)), key=lambda t: t[0])
        inc = [li for dpx, li in cand if dpx < on_line_tol][:2]
        pt = (cx, cy)
        if len(inc) == 2:
            ipt = _line_intersection_simple(lines[inc[0]]["Limg"], lines[inc[1]]["Limg"])
            if ipt is not None and abs(ipt[0] - cx) < a.bbox[2] and abs(ipt[1] - cy) < a.bbox[3]:
                pt = ipt
        nodes.append({"ann_id": a.ann_id, "type": jtype, "pt": pt,
                      "lines": inc,
                      "axis_lines": [lines[k]["Limg"] for k in inc],
                      "cap": _TYPE_DEGREE.get(jtype, 4)})
    return nodes


def _junction_axes_for_thumbs(img_bgr, anns, class_names, *, dark=False):
    """供縮圖使用：回傳 {ann_id: {'lines':[Limg,...], 'pt':(x,y)}}，
    其中的雙軸來自全域球場線（原圖座標）。"""
    lines, _ = _compute_court_lines(img_bgr, anns, dark=dark)
    nodes = _assign_junction_lines(anns, class_names, lines)
    return {n["ann_id"]: {"lines": n["axis_lines"], "pt": n["pt"]} for n in nodes}


def _global_line_link_graph(img_bgr, anns, class_names, *, pad_frac=0.06,
                            sigma=1.2, threshold_pct=0.14, dark=False,
                            on_line_tol=4.0, min_gap=6.0):
    """大範圍一次 Steger → 抽出整條球場線 → 把交點掛到通過它的線上 →
    同一條線上相鄰的交點相連（依端點脊點支撐分實線/虛線）。
    類型作為連線數上限（L=2/T=3/X=4）。回傳 (edges, nodes)。"""
    if not anns:
        return [], []
    lines, ridge_img = _compute_court_lines(
        img_bgr, anns, pad_frac=pad_frac, sigma=sigma,
        threshold_pct=threshold_pct, dark=dark)
    if not lines:
        return [], []
    nodes = _assign_junction_lines(anns, class_names, lines, on_line_tol=on_line_tol)

    # 同一條線上的交點，沿線排序後相鄰相連
    # 每端各自判斷是否有 Steger 線支撐：
    #   兩端都有 → 實線；只有一端有 → 虛線（無線那端只能被連，被動）；都沒有 → 不連
    cand_edges = {}
    for li, m in enumerate(lines):
        vx, vy = m["Limg"][0], m["Limg"][1]
        members = [k for k, n in enumerate(nodes) if li in n["lines"]]
        if len(members) < 2:
            continue
        members.sort(key=lambda k: nodes[k]["pt"][0] * vx + nodes[k]["pt"][1] * vy)
        for u, v in zip(members[:-1], members[1:]):
            pu, pv = nodes[u]["pt"], nodes[v]["pt"]
            dist = math.hypot(pu[0] - pv[0], pu[1] - pv[1])
            if dist < min_gap:
                continue
            supp_u = _half_segment_support(pu, pv, ridge_img)
            supp_v = _half_segment_support(pv, pu, ridge_img)
            if not supp_u and not supp_v:
                continue                                  # 兩端都無線 → 不連
            style = "solid" if (supp_u and supp_v) else "dashed"
            # 虛線時，無支撐的那端為被動端（只能被連）
            passive = None
            if style == "dashed":
                passive = nodes[v]["ann_id"] if supp_u else nodes[u]["ann_id"]
            key = frozenset((u, v))
            rec = {"dist": dist, "style": style, "passive": passive}
            if key not in cand_edges or dist < cand_edges[key]["dist"]:
                cand_edges[key] = rec

    deg = [0] * len(nodes)
    edges = []
    # 先連實線（高信心）再連虛線；同類再依長度
    order = sorted(cand_edges.items(),
                   key=lambda kv: (0 if kv[1]["style"] == "solid" else 1,
                                   kv[1]["dist"]))
    for key, rec in order:
        i, j = tuple(key)
        if deg[i] >= nodes[i]["cap"] or deg[j] >= nodes[j]["cap"]:
            continue
        deg[i] += 1; deg[j] += 1
        ni, nj = nodes[i], nodes[j]
        edges.append({
            "a": ni["ann_id"], "b": nj["ann_id"],
            "type_a": ni["type"], "type_b": nj["type"],
            "pa": [float(ni["pt"][0]), float(ni["pt"][1])],
            "pb": [float(nj["pt"][0]), float(nj["pt"][1])],
            "length": float(rec["dist"]),
            "style": rec["style"],
            "passive": rec["passive"],
        })
    out_nodes = [{"ann_id": n["ann_id"], "type": n["type"],
                  "pt": [float(n["pt"][0]), float(n["pt"][1])]} for n in nodes]
    return edges, out_nodes

# ──────────────────────────────────────────────
# 可互動 BBox 畫布
# ──────────────────────────────────────────────
HANDLE_SIZE = 8      # 角落縮放把手邊長 px
HIT_SLACK   = 6      # 邊線點擊容忍 px
