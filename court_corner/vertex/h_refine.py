"""
H-based Vertex Refinement (v3.2: H-rectified rectangle + cid-only matching)
============================================================================

設計思路：

  Step 1 — 用 H + 線寬重建 4 個 H-projected corner：
           * 4 個位置 = H × (junction_center ± 0.5·LW·nA ± 0.5·LW·nB)
           * 每個 corner 同時帶 8-bit corner_code (cid)，給後續配對用。

  Step 2 — 依 junction type + topology id 決定保留哪幾個 corner：
           * X (jt=2) → 全部 4 個
           * T (jt=1) → stem 那一側的 2 個 corner
           * L (jt=0) → 朝球場中心 + 朝球場外 2 個 corner

  Step 3 — 對每個保留下來的 corner，找 Steger raw 中**相同 cid** 的 v_s
           （唯 cid 配對，不 fallback 到 corner_type 局部 label，避免 template
           relabel/翻轉時 corner_type 語意改變造成誤配）：
           * Steger raw 沒帶 cid 的 vertex → 一律跳過
           * v_s 不存在               → 該 corner 不輸出
           * v_s 存在 + 距離 ≤ tol    → 加權融合 (h_weight·v_h + (1-h_weight)·v_s)
                                        source = 'fused'
           * v_s 存在 + 距離 > tol    → 用 v_h（捨棄 Steger 的 outlier）
                                        source = 'h_replaced'

對外介面：
  HomographyVertexRefiner.refine(vertices, H, junction_idx, cp) -> List[vertex_dict]

每個輸出 vertex 帶：
  pos_px, pos_init, width_px, confidence, score,
  junction_idx, corner_type, corner_code,
  h_refine_source     : 'fused' / 'h_replaced'
  h_anchor_px         : v_h (H projection 位置)
  h_dist_px           : ||final_pos - v_h||
  steger_pos_px       : v_s (Steger raw 位置)
  steger_dist_px      : ||v_s - v_h||
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from ..shared.court_model import (
    TEMPLATE_POINTS, TEMPLATE_TYPES, LINE_WIDTH_M,
    compute_junction_principal_directions, get_t_junction_stem_info,
    corner_type_to_corner_code,
)
from ..shared.homography import HomographyUtils

from ..config import (
    VF_H_REFINE_ENABLED,                     # noqa: F401  (used by caller)
    VF_H_REFINE_OUTLIER_PX as _DEFAULT_OUTLIER_PX,
    VF_H_REFINE_BLEND_WEIGHT as _DEFAULT_H_WEIGHT,
)


# 球場中心（template 空間，公尺）— 給 L junction inner/outer 排序用
_COURT_CENTER_M = np.array([3.05, 6.70], dtype=np.float64)


# ============================================================
#  Step 1: H-rectified 4-corner reconstruction
# ============================================================

def _build_h_rectified_corners(
    junction_idx: int, H: np.ndarray
) -> Tuple[List[dict], Tuple[float, float]]:
    """
    給定 junction_idx 和 H，回傳「H + 線寬重建出的 4 個 corner」。

    流程：
      p0_m, tA_m, tB_m = template 空間中心與兩主方向
      nA_m = perp(tA_m), nB_m = perp(tB_m)
      half = LINE_WIDTH_M / 2 = 0.02 m
      v_sa_sb_m = p0_m + sa·half·nA_m + sb·half·nB_m   (sa, sb ∈ ±1)
      v_sa_sb_px = H × v_sa_sb_m

    結果 4 個點在影像中構成「template 矩形」經 H 投影後的小四邊形。
    這個小四邊形：
      - 邊長 ≈ wA × wB (px)
      - 對角線方向 = tA / tB 的 H projection
      - 矩形相對位置 reflect 全圖透視（near 大、far 小）

    Returns:
        (corners, (wA_px, wB_px))
        corners: list of dict with
            'pos_px'      : (2,) np.float32 影像座標
            'corner_type' : '++' / '+-' / '-+' / '--'
            'sa', 'sb'    : ±1 (template 空間 nA/nB 的符號)
            'v_m'         : (2,) np.float64 template 公尺座標（給後續 inner/outer 比對用）
    """
    p0_m = TEMPLATE_POINTS[junction_idx].astype(np.float64)
    pd = compute_junction_principal_directions(junction_idx)
    if len(pd) < 2:
        return [], (0.0, 0.0)

    tA_m = pd[0].astype(np.float64)
    tB_m = pd[1].astype(np.float64)
    nA_m = np.array([-tA_m[1], tA_m[0]], dtype=np.float64)
    nB_m = np.array([-tB_m[1], tB_m[0]], dtype=np.float64)
    half = LINE_WIDTH_M / 2.0  # 0.02 m

    # Jacobian → line widths（給輸出當 width_px）
    J = HomographyUtils.compute_jacobian(H, p0_m)
    wA_px = float(HomographyUtils.compute_line_width_px(J, tA_m, LINE_WIDTH_M))
    wB_px = float(HomographyUtils.compute_line_width_px(J, tB_m, LINE_WIDTH_M))

    corners = []
    for sa, sb in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
        label = f'{"+" if sa > 0 else "-"}{"+" if sb > 0 else "-"}'
        v_m = (p0_m + sa * half * nA_m + sb * half * nB_m)
        try:
            v_px = cv2.perspectiveTransform(
                v_m.astype(np.float32).reshape(1, 1, 2), H,
            ).reshape(2)
        except Exception:
            continue
        if not np.all(np.isfinite(v_px)):
            continue
        # 8-bit corner_code：(junction_idx, corner_type='++/+-/-+/--', tA_m, tB_m)
        # → 落在 template grid 上的哪一個物理角點。用這個當 H↔Steger 配對的 key
        # 比直接用 corner_type 局部 label 穩定（不會被 relabel/翻轉影響語意）。
        try:
            ccode = corner_type_to_corner_code(
                junction_idx=int(junction_idx),
                corner_type=label,
                tA_m=tA_m, tB_m=tB_m,
            )
        except Exception:
            ccode = -1
        corners.append({
            'pos_px': v_px.astype(np.float32),
            'corner_type': label,
            'corner_code': int(ccode),
            'sa': sa,
            'sb': sb,
            'v_m': v_m,
        })
    return corners, (wA_px, wB_px)


# ============================================================
#  Step 2: topology-id filter
# ============================================================

def _filter_corners_by_topology(
    corners: List[dict], junction_idx: int
) -> List[dict]:
    """
    依 junction type + topology id，從 4 個 H-rectified corner 中選出該保留的：
      X (jt=2) → 全部 4 個
      T (jt=1) → stem 側 2 個（用 get_t_junction_stem_info 決定）
      L (jt=0) → 朝球場中心 + 朝球場外 2 個

    Args:
        corners      : _build_h_rectified_corners 的輸出
        junction_idx : topology id（決定 stem 朝向 / 球場中心方向）

    Returns:
        篩選後的 corner list（keep 順序穩定 — '++','+-','-+','--' 順序）
    """
    if not corners:
        return []
    jt = int(TEMPLATE_TYPES[junction_idx])
    p0_m = TEMPLATE_POINTS[junction_idx].astype(np.float64)
    pd = compute_junction_principal_directions(junction_idx)
    if len(pd) < 2:
        return list(corners)
    tA_m = pd[0].astype(np.float64)
    tB_m = pd[1].astype(np.float64)
    nA_m = np.array([-tA_m[1], tA_m[0]], dtype=np.float64)
    nB_m = np.array([-tB_m[1], tB_m[0]], dtype=np.float64)

    # X：全部 4
    if jt == 2:
        return list(corners)

    # T：stem 側 2 個
    if jt == 1:
        stem_info = get_t_junction_stem_info(junction_idx)
        if stem_info is None:
            # 沒有 stem 資訊 → 退化：取前 2 個（不該發生，但保險）
            return list(corners[:2])
        ss = stem_info['stem_sign']      # ±1
        ax = stem_info['stem_axis']      # 'A' or 'B'
        # stem 方向（影像中 T 的「腿」朝向）
        # 對 T 而言：stem axis 那一側才有實體外緣 corner，bar axis 那一側是「橫桿」沒外緣
        # 我們要保留 stem axis 上「ss 號正號的那一側」+ 旁邊兩個
        # 這跟 v1/v2 在 _CORNER_LIB 篩 T 時用的邏輯一致
        if ax == 'B':
            stem_dir = ss * tB_m
            dot_val = float(np.dot(nA_m, stem_dir))
            n_bar_stem = nA_m if dot_val >= 0 else -nA_m
            n_stem = nB_m
            # 應保留：所有 sa = sign(n_bar_stem · nA) 的 corner
            keep_sa = +1 if dot_val >= 0 else -1
            kept = [c for c in corners if c['sa'] == keep_sa]
        else:  # ax == 'A'
            stem_dir = ss * tA_m
            dot_val = float(np.dot(nB_m, stem_dir))
            n_bar_stem = nB_m if dot_val >= 0 else -nB_m
            n_stem = nA_m
            keep_sb = +1 if dot_val >= 0 else -1
            kept = [c for c in corners if c['sb'] == keep_sb]
        if len(kept) == 2:
            return kept
        # 退化：理論上不會發生（T 一邊永遠剛好 2 個 corner），但保險
        return list(corners[:2])

    # L：inner（朝球場中心）+ outer（朝球場外）
    # 我們不能像 T 那樣用 sa/sb 直接過濾 — L 的 nA/nB 跟 court_center 沒固定關係。
    # 改用 4 個 corner 對 (court_center - p0_m) 的內積：max=inner、min=outer。
    vC = _COURT_CENTER_M - p0_m
    n_C = float(np.linalg.norm(vC))
    if n_C < 1e-9:
        return list(corners[:2])
    vals = []
    for c in corners:
        d = c['v_m'] - p0_m
        vals.append(float(np.dot(d, vC) / n_C))  # 沿 vC 方向的距離
    inner_i = int(np.argmax(vals))
    outer_i = int(np.argmin(vals))
    if inner_i == outer_i:
        return list(corners[:2])
    return [corners[inner_i], corners[outer_i]]


# ============================================================
#  v1 介面相容：compute_h_vertex_candidates
# ============================================================
# Junction Detail tab 仍用這個函式畫 H projection 比對 panel。

def compute_h_vertex_candidates(junction_idx: int, H: np.ndarray
                                ) -> Tuple[List[dict], Tuple[float, float]]:
    """
    給單一 junction，回傳「H 投影 + topology 篩選後的 vertex candidates」。

    回傳 (candidates, (wA_px, wB_px))，每個 candidate dict 含：
      'pos_px', 'corner_type', 'corner_code', 'sa', 'sb'
    """
    raw_corners, (wA_px, wB_px) = _build_h_rectified_corners(junction_idx, H)
    if not raw_corners:
        return [], (wA_px, wB_px)
    selected = _filter_corners_by_topology(raw_corners, junction_idx)
    out = [{
        'pos_px': c['pos_px'],
        'corner_type': c['corner_type'],
        'corner_code': c.get('corner_code', -1),
        'sa': c['sa'],
        'sb': c['sb'],
    } for c in selected]
    return out, (wA_px, wB_px)


# ============================================================
#  HomographyVertexRefiner v3
# ============================================================

class HomographyVertexRefiner:
    """
    v3.1 — H-rectified rectangle + per-corner fuse-or-replace.

    Step 1  用 H + 線寬重建 4 個 H-projected corner（template 矩形 → 影像）
    Step 2  用 junction type + topology id 篩 4 → 2 (T/L) 或 4 (X)
    Step 3  對每個保留的 corner_type:
              v_s = Steger raw 中相同 corner_type 的 vertex
              v_h = H projection 位置
              ├─ v_s 不存在            → 該 corner 不輸出
              ├─ ||v_s - v_h|| ≤ tol → 加權融合（'fused'）
              └─ ||v_s - v_h|| > tol  → 用 v_h（'h_replaced'）

    輸出 corner 數 ≤ 篩選後數量（X 最多 4、T/L 最多 2）。
    若所有 corner 的 Steger 都漏掉 → 整個 junction 不輸出。
    """

    def __init__(self,
                 img_gray: Optional[np.ndarray] = None,
                 outlier_px: float = _DEFAULT_OUTLIER_PX,
                 h_weight: float = _DEFAULT_H_WEIGHT):
        # img_gray 留參數但 v3.1 不用（跟 v2 介面一致）
        self.img_gray = img_gray
        self.outlier_px = float(outlier_px)
        self.h_weight = float(np.clip(h_weight, 0.0, 1.0))

    def refine(self, vertices: list, H: np.ndarray,
               junction_idx: int, center_px: np.ndarray) -> list:
        """
        對單一 junction 做 H refine（v3.2 流程，唯 cid 配對）。

        配對 key：corner_code（8-bit 全域編碼，cid）。完全不 fallback 到
        corner_type 字串，因為 cid 才能避免 template-id relabel / 翻轉時
        誤配。Steger raw 中沒帶合法 cid 的 vertex 一律跳過；H projection
        corner 算不出 cid 的也跳過。

        Args:
            vertices    : Steger raw vertices（必須帶 corner_code 欄位）
            H           : Homography (template -> image)
            junction_idx: topology id
            center_px   : junction 中心的影像座標

        Returns:
            refined_vertices: list of dict, 數量 ≤ topology-filter 後的數量。
            Steger 漏的 corner（沒配對到 cid）不輸出。
        """
        # Step 1: H-rectified 4 corners
        h_corners, (wA_px, wB_px) = _build_h_rectified_corners(junction_idx, H)
        if not h_corners:
            logging.warning("[H-refine] H projection failed at junction %d", junction_idx)
            return []

        # Step 2: topology-id filter
        kept = _filter_corners_by_topology(h_corners, junction_idx)

        # Step 3: 對每個 kept corner 做 fuse-or-replace 或 skip
        width_px = (wA_px + wB_px) / 2.0
        # 預先 index Steger raw by corner_code
        steger_by_code: Dict[int, dict] = {}
        for v in vertices:
            cc = v.get('corner_code')
            if not isinstance(cc, (int, np.integer)) or int(cc) < 0:
                continue  # 沒帶合法 cid 一律跳過（不 fallback）
            pos = v.get('pos_px')
            if pos is None:
                continue
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
            if pos_arr.size != 2 or not np.all(np.isfinite(pos_arr)):
                continue
            steger_by_code.setdefault(int(cc), v)

        out = []
        for c in kept:
            cc = int(c.get('corner_code', -1))
            if cc < 0:
                # H corner 算不出 cid → 沒辦法跟 Steger 配對 → 跳過
                continue
            v_h = np.asarray(c['pos_px'], dtype=np.float64).reshape(2)
            ct = c['corner_type']

            steger_v = steger_by_code.get(cc)
            if steger_v is None:
                # Steger 沒抓到這個 cid → 不輸出
                continue

            v_s = np.asarray(steger_v['pos_px'], dtype=np.float64).reshape(2)
            dist = float(np.linalg.norm(v_s - v_h))

            if dist > self.outlier_px:
                # outlier：直接用 H projection
                final_pos = v_h
                src = 'h_replaced'
                base_conf = float(steger_v.get('confidence', 0.5)) * 0.5
            else:
                # 加權融合：H 主導，Steger 提供 sub-pixel 修正
                final_pos = self.h_weight * v_h + (1.0 - self.h_weight) * v_s
                src = 'fused'
                # confidence：依距離微幅折減
                base_conf = float(steger_v.get('confidence', 0.5))
                fade = 0.2 * (dist / max(self.outlier_px, 1e-6))
                base_conf = float(np.clip(base_conf * (1.0 - fade), 0.0, 1.0))

            v_out = {
                'pos_px': final_pos.astype(np.float32),
                'pos_init': np.asarray(center_px, dtype=np.float32).copy(),
                'width_px': float(width_px),
                'confidence': float(base_conf),
                'score': float(base_conf),
                'junction_idx': int(junction_idx),
                'corner_type': ct,
                'corner_code': cc,
                'h_refine_source': src,           # 'fused' | 'h_replaced'
                'h_anchor_px': v_h.astype(np.float32),
                'h_dist_px': float(np.linalg.norm(final_pos - v_h)),
                'steger_pos_px': v_s.astype(np.float32),
                'steger_dist_px': dist,
            }
            out.append(v_out)

        return out


__all__ = [
    'HomographyVertexRefiner',
    'compute_h_vertex_candidates',
    '_build_h_rectified_corners',
    '_filter_corners_by_topology',
]