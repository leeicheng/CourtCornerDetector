"""
Vertex Reprojection
====================
把 S4 找到的 vertex（影像座標）用 H⁻¹ reproj 回 template 空間（公尺），
與 template ground truth 比較誤差。

關鍵概念：
  vertex 不是 template junction 中心，而是 junction 的角點。
  其「真值位置」是：
      gt_template = p0_m + sa * (LINE_WIDTH_M/2) * nA_m + sb * (LINE_WIDTH_M/2) * nB_m
  其中 sa, sb ∈ {+1, -1} 由 corner_type 字串解析得到。

對外介面：
  compute_vertex_reprojection(vertices, H) -> list of dict（每筆含 reproj 位置與誤差）
  summarize_reprojection(records)         -> dict（各種統計）
"""

import numpy as np
import cv2

from ..shared.court_model import (
    TEMPLATE_POINTS, TEMPLATE_TYPES, LINE_WIDTH_M,
    compute_junction_principal_directions,
)


_TYPE_NAME = {0: 'L', 1: 'T', 2: 'X'}


def _gt_template_position(
    junction_idx: int,
    corner_type: str,
    corner_code: int = -1,
) -> np.ndarray:
    """
    根據 junction_idx 與 corner_type ('++', '+-', '-+', '--') 計算
    該 vertex 在 template 空間（公尺）的真值位置。

    用同一套定義跟 h_refine._build_h_rectified_corners 對齊：
        gt_m = p0_m + sa·(LW/2)·nA_m + sb·(LW/2)·nB_m
    其中 nA_m = perp(tA_m), nB_m = perp(tB_m)
    (sa, sb) ∈ {±1}² 由 corner_type 字串解析。

    重要前提：vertex 的 corner_type 字串必須以 **template 空間定義**
    （而非 pixel 空間）。Steger v 在 Step 6.5 已經 H⁻¹ 投回 template
    重判 label，符合此前提；EdgeBasedVertexFinder 同樣用 template-space
    定義。

    `corner_code` 目前只當 metadata（output 時透傳），不用於計算 gt——
    因為 corner_type 字串如果上游算對，就跟 corner_code 等價（兩者皆從
    template-space delta dot nA/nB 算出）。保留參數是為了未來如果想加
    sanity check（corner_code 解出的 lcid 是否與 corner_type 一致）。

    Returns:
        np.array([x_m, y_m], dtype=float64)；無法解析時回傳 junction 中心。
    """
    p0_m = TEMPLATE_POINTS[junction_idx].astype(np.float64)
    if not isinstance(corner_type, str) or len(corner_type) != 2:
        return p0_m
    if corner_type[0] not in '+-' or corner_type[1] not in '+-':
        return p0_m

    pd = compute_junction_principal_directions(junction_idx)
    if len(pd) < 2:
        return p0_m

    tA_m = pd[0].astype(np.float64)
    tB_m = pd[1].astype(np.float64)
    nA_m = np.array([-tA_m[1], tA_m[0]], dtype=np.float64)
    nB_m = np.array([-tB_m[1], tB_m[0]], dtype=np.float64)
    half_lw = LINE_WIDTH_M / 2.0

    sa = +1.0 if corner_type[0] == '+' else -1.0
    sb = +1.0 if corner_type[1] == '+' else -1.0

    return p0_m + sa * half_lw * nA_m + sb * half_lw * nB_m


def compute_vertex_reprojection(vertices: list, H: np.ndarray) -> list:
    """
    對每個 vertex 算它在 template 空間的 reproj 位置 + 誤差。

    Args:
        vertices : list of vertex dict（pos_px, junction_idx, corner_type, ...）
        H        : Homography (template -> image)

    Returns:
        list of dict, 每筆含
            'junction_idx'   : int
            'junction_type'  : 'L' / 'T' / 'X'
            'corner_type'    : '++' / '+-' / '-+' / '--'
            'pos_px'         : 影像座標 (np.float32, 2)
            'reproj_m'       : reproj 後的 template 座標 (公尺, np.float64, 2)
            'gt_m'           : ground truth template 座標 (公尺, np.float64, 2)
            'err_vec_m'      : reproj_m - gt_m (公尺, 2)
            'err_m'          : np.linalg.norm(err_vec_m) (公尺, scalar)
            'h_refine_source': 透傳（'fused' / 'h_replaced' / 'h_filled' / None）
            'method_used'    : 透傳
    """
    if not vertices or H is None:
        return []

    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return []

    # 批次 reproj：所有 pos_px 一次轉
    pts_px = np.array([v['pos_px'] for v in vertices], dtype=np.float32)
    pts_px = pts_px.reshape(-1, 1, 2)
    pts_m = cv2.perspectiveTransform(pts_px, Hinv).reshape(-1, 2).astype(np.float64)

    records = []
    for v, p_m in zip(vertices, pts_m):
        mi = int(v.get('junction_idx', -1))
        if mi < 0 or mi >= 30:
            continue
        ct = v.get('corner_type', '')
        cc = v.get('corner_code', -1)
        gt_m = _gt_template_position(mi, ct, cc)
        err_vec = p_m - gt_m
        err = float(np.linalg.norm(err_vec))
        records.append({
            'junction_idx':     mi,
            'junction_type':    _TYPE_NAME.get(int(TEMPLATE_TYPES[mi]), '?'),
            'corner_type':      ct if isinstance(ct, str) else '',
            'corner_code':      int(cc) if isinstance(cc, (int, np.integer)) else -1,
            'pos_px':           np.asarray(v['pos_px'], dtype=np.float32).copy(),
            'reproj_m':         p_m.copy(),
            'gt_m':             gt_m.copy(),
            'err_vec_m':        err_vec.copy(),
            'err_m':            err,
            'h_refine_source':  v.get('h_refine_source'),
            'method_used':      v.get('method_used'),
        })
    return records


def summarize_reprojection(records: list) -> dict:
    """
    對 reproj records 做多維度統計。

    Returns:
        dict 含
          'overall'            : {count, mean, median, max, rmse}（公尺）
          'per_junction_type'  : {'X': {...}, 'T': {...}, 'L': {...}}
          'per_corner_type'    : {'++': {...}, ...}
          'per_h_refine_source': {'fused': {...}, 'h_replaced': {...}, 'h_filled': {...}, 'none': {...}}
    """
    def _stats(errs: list) -> dict:
        if not errs:
            return {'count': 0, 'mean': 0.0, 'median': 0.0, 'max': 0.0, 'rmse': 0.0}
        arr = np.asarray(errs, dtype=np.float64)
        return {
            'count':  int(arr.size),
            'mean':   float(np.mean(arr)),
            'median': float(np.median(arr)),
            'max':    float(np.max(arr)),
            'rmse':   float(np.sqrt(np.mean(arr ** 2))),
        }

    overall = [r['err_m'] for r in records]

    by_jtype = {'X': [], 'T': [], 'L': []}
    for r in records:
        jt = r['junction_type']
        if jt in by_jtype:
            by_jtype[jt].append(r['err_m'])

    by_ctype = {'++': [], '+-': [], '-+': [], '--': []}
    for r in records:
        ct = r['corner_type']
        if ct in by_ctype:
            by_ctype[ct].append(r['err_m'])

    by_src = {'fused': [], 'h_replaced': [], 'h_filled': [], 'none': []}
    for r in records:
        src = r.get('h_refine_source')
        key = src if src in by_src else 'none'
        by_src[key].append(r['err_m'])

    return {
        'overall':             _stats(overall),
        'per_junction_type':   {k: _stats(v) for k, v in by_jtype.items()},
        'per_corner_type':     {k: _stats(v) for k, v in by_ctype.items()},
        'per_h_refine_source': {k: _stats(v) for k, v in by_src.items()},
    }
