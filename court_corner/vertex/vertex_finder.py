"""
Vertex Finder - 角點偵測核心邏輯
==================================
包含:
- SubpixelUtils: Subpixel 精度工具
- EdgeScanner: 邊緣掃描器
- LineRANSAC: RANSAC 直線擬合
- TopologyConstraints: 拓撲語意約束
- EdgeBasedVertexFinder: 主要角點搜尋器
"""

import logging
import time
import numpy as np
import cv2

from ..shared.court_model import (
    TEMPLATE_POINTS, TEMPLATE_TYPES, LINE_WIDTH_M,
    JUNCTION_INCIDENT, compute_junction_principal_directions,
    corner_type_to_corner_code,
)
from ..shared.homography import HomographyUtils
from ..config import VQ_PEAK_RADIUS_PX as _PEAK_RADIUS_PX, VF_RANSAC_ITERATIONS as _RANSAC_ITER, VF_RANSAC_THRESHOLD as _RANSAC_THRESH


# ================= Subpixel 精度工具 =================

class SubpixelUtils:
    """Subpixel 精度工具類別"""

    @staticmethod
    def subpixel_peak_1d(arr: np.ndarray, i: int) -> float:
        """
        對 arr 的 i 做 3 點拋物線插值，回傳 subpixel index (i + delta)

        Args:
            arr: 1D array
            i: peak index (需要 1 <= i <= n-2)

        Returns:
            subpixel index
        """
        if i < 1 or i >= len(arr) - 1:
            return float(i)

        y0, y1, y2 = float(arr[i-1]), float(arr[i]), float(arr[i+1])
        denom = (y0 - 2*y1 + y2)
        if abs(denom) < 1e-12:
            return float(i)
        delta = 0.5 * (y0 - y2) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        return float(i) + delta

    @staticmethod
    def weighted_median_1d(x: np.ndarray, w: np.ndarray) -> float:
        """
        計算 1D weighted median
        比 weighted average 更 robust，不易被 outlier 拉走
        """
        if len(x) == 0:
            return 0.0
        if len(x) == 1:
            return float(x[0])

        idx = np.argsort(x)
        x_sorted = x[idx]
        w_sorted = w[idx]
        cum = np.cumsum(w_sorted)
        cutoff = 0.5 * np.sum(w_sorted)
        pos = np.searchsorted(cum, cutoff)
        pos = min(pos, len(x_sorted) - 1)
        return float(x_sorted[pos])

    @staticmethod
    def weighted_median_2d(pts: np.ndarray, w: np.ndarray) -> np.ndarray:
        """計算 2D weighted median (逐軸)"""
        if len(pts) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.array([
            SubpixelUtils.weighted_median_1d(pts[:, 0], w),
            SubpixelUtils.weighted_median_1d(pts[:, 1], w)
        ], dtype=np.float32)


# ================= RANSAC 直線擬合 =================

class LineRANSAC:
    """RANSAC 直線擬合工具類別"""

    @staticmethod
    def fit_line(points: np.ndarray, n_iterations: int = _RANSAC_ITER,
                 threshold: float = _RANSAC_THRESH) -> tuple:
        """
        RANSAC 直線擬合

        Returns:
            (a, b, c) where ax + by + c = 0，或 None
        """
        if len(points) < 2:
            return None

        best_line = None
        best_inliers = 0

        for _ in range(n_iterations):
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                continue

            a, b = -dy, dx
            norm = np.sqrt(a*a + b*b)
            a, b = a / norm, b / norm
            c = -(a * p1[0] + b * p1[1])

            distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
            inliers = np.sum(distances < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_line = (a, b, c)

        # 用所有 inliers 重新 fit (SVD)
        if best_line is not None and best_inliers >= 3:
            a, b, c = best_line
            distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
            inlier_mask = distances < threshold
            inlier_points = points[inlier_mask]

            if len(inlier_points) >= 2:
                centroid = np.mean(inlier_points, axis=0)
                centered = inlier_points - centroid
                _, _, Vt = np.linalg.svd(centered)
                direction = Vt[0]

                a, b = -direction[1], direction[0]
                norm = np.hypot(a, b) + 1e-12
                a, b = a / norm, b / norm
                c = -(a * centroid[0] + b * centroid[1])
                best_line = (a, b, c)

        return best_line


# ================= 邊緣掃描器 =================

class EdgeScanner:
    """邊緣掃描器類別"""

    def __init__(self):
        self.subpixel = SubpixelUtils()

    def scan_for_edges(self, Ix, Iy, center_px, tangent, normal,
                       line_width_px, R, L, Ns, h, w) -> tuple:
        """
        沿法向掃描找雙邊 edge
        改進: 使用 bilinear interpolation (cv2.remap) 取代 int(round())

        Returns:
            (edges_plus, edges_minus): 兩側邊界的點雲
        """
        edges_plus = []
        edges_minus = []

        half_w = line_width_px / 2
        tol = max(1.5, 0.35 * line_width_px)

        S = int(R * 0.8)
        half_L = L // 2
        center_skip = max(half_w * 0.8, 2.0)

        for s_idx in range(Ns):
            s = -S + (2 * S * s_idx) / (Ns - 1) if Ns > 1 else 0

            if abs(s) < center_skip:
                continue

            p_sample = center_px + s * tangent

            # 建立 float 座標 grid
            xs = np.zeros(L, dtype=np.float32)
            ys = np.zeros(L, dtype=np.float32)
            positions = []

            for r_idx in range(L):
                r = -half_L + r_idx
                p = p_sample + r * normal
                xs[r_idx] = p[0]
                ys[r_idx] = p[1]
                positions.append((r, p.copy()))

            # 檢查邊界
            valid = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
            if np.sum(valid) < 5:
                continue

            # 使用 cv2.remap 做 bilinear interpolation
            # 顯式轉換為 float32 確保類型一致
            map_x = xs.reshape(1, -1).astype(np.float32)
            map_y = ys.reshape(1, -1).astype(np.float32)

            Ix_sampled = cv2.remap(Ix, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0).flatten()
            Iy_sampled = cv2.remap(Iy, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0).flatten()

            # 計算 directional gradient
            profile = Ix_sampled * normal[0] + Iy_sampled * normal[1]

            # 過濾無效點
            profile[~valid] = 0

            edge_pair = self.find_edge_pair_signed(profile, positions, half_w, tol)

            if edge_pair:
                r1, p1, r2, p2 = edge_pair
                if r1 < r2:
                    edges_minus.append(p1)
                    edges_plus.append(p2)
                else:
                    edges_plus.append(p1)
                    edges_minus.append(p2)

        return edges_plus, edges_minus

    def find_edge_pair_signed(self, profile: np.ndarray, positions: list,
                              half_w: float, tol: float) -> tuple:
        """
        用 signed gradient 找 edge pair + subpixel interpolation
        
        改進:
        1. MAD-based threshold (更 robust 於地板紋理/反光)
        2. 兩階段 expected_dist 配對 (降低對 Homography 的敏感度)
        """
        n = len(profile)
        if n < 3:
            return None

        # 改用 MAD (Median Absolute Deviation) 設定閾值
        # sigma = 1.4826 * median(|p - median(p)|)
        abs_p = np.abs(profile)
        median_p = np.median(profile)
        mad = np.median(np.abs(profile - median_p))
        sigma = 1.4826 * mad
        thr = 2.5 * sigma  # k = 2.5
        thr = max(thr, 3.0)  # 最小閾值，避免雜訊

        # 找正梯度峰值 (允許 plateau peak)
        pos_peaks = []
        for i in range(1, n - 1):
            is_peak = (profile[i] >= profile[i-1] and profile[i] >= profile[i+1])
            is_strict = (profile[i] > profile[i-1]) or (profile[i] > profile[i+1])
            if is_peak and is_strict and profile[i] > thr:
                i_sub = SubpixelUtils.subpixel_peak_1d(profile, i)
                delta = i_sub - i
                r_sub = positions[i][0] + delta

                p_curr = np.array(positions[i][1], dtype=np.float64)
                if delta >= 0 and i + 1 < len(positions):
                    p_nbr = np.array(positions[i+1][1], dtype=np.float64)
                elif delta < 0 and i - 1 >= 0:
                    p_nbr = np.array(positions[i-1][1], dtype=np.float64)
                else:
                    p_nbr = p_curr
                p_sub = p_curr + delta * (p_nbr - p_curr)

                pos_peaks.append((i, profile[i], r_sub, p_sub))

        # 找負梯度峰值
        neg_peaks = []
        neg_profile = -profile
        for i in range(1, n - 1):
            is_peak = (profile[i] <= profile[i-1] and profile[i] <= profile[i+1])
            is_strict = (profile[i] < profile[i-1]) or (profile[i] < profile[i+1])
            if is_peak and is_strict and profile[i] < -thr:
                i_sub = SubpixelUtils.subpixel_peak_1d(neg_profile, i)
                delta = i_sub - i
                r_sub = positions[i][0] + delta

                p_curr = np.array(positions[i][1], dtype=np.float64)
                if delta >= 0 and i + 1 < len(positions):
                    p_nbr = np.array(positions[i+1][1], dtype=np.float64)
                elif delta < 0 and i - 1 >= 0:
                    p_nbr = np.array(positions[i-1][1], dtype=np.float64)
                else:
                    p_nbr = p_curr
                p_sub = p_curr + delta * (p_nbr - p_curr)

                neg_peaks.append((i, -profile[i], r_sub, p_sub))

        if len(pos_peaks) == 0 or len(neg_peaks) == 0:
            return None

        expected_dist = 2 * half_w

        # 兩階段配對：降低對 Homography/Jacobian 的敏感度
        loose_tol = 2.0 * tol  # 第一階段放寬容忍度

        # Stage 1: 收集候選 pair 距離
        candidate_dists = []
        for idx1, mag1, r1, p1 in pos_peaks:
            for idx2, mag2, r2, p2 in neg_peaks:
                dist = abs(r2 - r1)
                if abs(dist - expected_dist) < loose_tol:
                    candidate_dists.append(dist)

        # 取 median 當 measured expected_dist
        if len(candidate_dists) >= 3:
            expected_dist_measured = np.median(candidate_dists)
        else:
            expected_dist_measured = expected_dist

        # Stage 2: 用 measured 做精確配對
        best_pair = None
        best_score = -1

        for idx1, mag1, r1, p1 in pos_peaks:
            for idx2, mag2, r2, p2 in neg_peaks:
                dist = abs(r2 - r1)

                if abs(dist - expected_dist_measured) < tol:
                    dist_score = 1.0 - abs(dist - expected_dist_measured) / tol
                    edge_score = (mag1 + mag2) / (2 * np.max(abs_p) + 1e-6)
                    score = 0.5 * dist_score + 0.5 * edge_score

                    if score > best_score:
                        best_score = score
                        best_pair = (r1, p1, r2, p2)

        return best_pair


# ================= 拓撲語意約束 =================

class TopologyConstraints:
    """拓撲語意約束類別"""

    @staticmethod
    def ray_signs_template(junction_idx: int, t_m: np.ndarray, eps: float = 1e-6) -> set:
        """對某方向取 ray signs（template space）"""
        p0 = TEMPLATE_POINTS[junction_idx]
        signs = set()
        for nb in JUNCTION_INCIDENT[junction_idx]:
            v = TEMPLATE_POINTS[nb] - p0
            s = float(np.dot(v, t_m))
            if s > eps:
                signs.add(+1)
            elif s < -eps:
                signs.add(-1)
        return signs

    @staticmethod
    def px_to_template(Hinv: np.ndarray, p_px: np.ndarray) -> np.ndarray:
        """將影像座標轉換到 template 座標"""
        pt = np.array(p_px, dtype=np.float32).reshape(1, 1, 2)
        pm = cv2.perspectiveTransform(pt, Hinv).reshape(2)
        return pm.astype(np.float32)

    @staticmethod
    def corner_quadrant_signs(pm: np.ndarray, p0m: np.ndarray,
                              tA_m: np.ndarray, tB_m: np.ndarray,
                              eps: float = 1e-6) -> tuple:
        """
        計算角點在 (tA, tB) 象限的符號（template space）

        Returns:
            (sign_A, sign_B): 每個方向的符號 (+1, -1, 或 0)
        """
        dv = pm - p0m
        a = float(np.dot(dv, tA_m))
        b = float(np.dot(dv, tB_m))
        sa = 0 if abs(a) < eps else (+1 if a > 0 else -1)
        sb = 0 if abs(b) < eps else (+1 if b > 0 else -1)
        return sa, sb


# ================= 主要角點搜尋器 =================

class EdgeBasedVertexFinder:
    """
    基於雙邊 edge 約束的外緣角點搜尋器

    核心流程：
    1. 沿法向掃描找雙邊 edge
    2. 用線寬約束過濾有效 edge pairs
    3. 直線擬合得到四條邊界線
    4. 交點 = 外緣角點候選
    """

    def __init__(self, line_width_m: float = LINE_WIDTH_M):
        self.line_width_m = line_width_m
        self.debug_info = {}
        self.edge_scanner = EdgeScanner()
        self.topo_constraints = TopologyConstraints()

    def find_vertices_for_junction(self, img_gray: np.ndarray, H: np.ndarray,
                                    junction_idx: int, center_px: np.ndarray,
                                    do_multi_shift: bool = True,
                                    Ix: np.ndarray = None, Iy: np.ndarray = None,
                                    roi_bounds: tuple = None) -> dict:
        """
        為單一 junction 尋找外緣角點

        Args:
            img_gray    : 灰階影像
            H           : Homography matrix (template -> image)
            junction_idx: Junction 在 template 中的 index
            center_px   : Junction 中心的影像座標
            do_multi_shift: 是否執行多次平移搜尋
            Ix, Iy      : 預先計算好的梯度 (可選)
            roi_bounds  : (x1,y1,x2,y2) YOLO bbox 範圍；若提供，R 不超出 bbox 半寬高

        Returns:
            dict with 'vertices', 'confidence', 'debug_info'
        """
        pt_m = TEMPLATE_POINTS[junction_idx]
        J = HomographyUtils.compute_jacobian(H, pt_m)

        # Step 1: 取得兩條主方向
        principal_dirs = compute_junction_principal_directions(junction_idx)
        if len(principal_dirs) < 2:
            return {'vertices': [], 'confidence': 0.0, 'error': 'insufficient_directions'}

        tA_m, tB_m = principal_dirs[0], principal_dirs[1]

        # Step 2: 轉換到影像空間
        tA_px = HomographyUtils.transform_direction(J, tA_m)
        tB_px = HomographyUtils.transform_direction(J, tB_m)
        nA_px = HomographyUtils.perp(tA_px)
        nB_px = HomographyUtils.perp(tB_px)

        wA_px = HomographyUtils.compute_line_width_px(J, tA_m, self.line_width_m)
        wB_px = HomographyUtils.compute_line_width_px(J, tB_m, self.line_width_m)
        halfA = 0.5 * wA_px
        halfB = 0.5 * wB_px

        # Fix②: R 對齊 S3 的 ROI 大小（4 * max_line_width）
        # 舊版 8*halfA = 4*wA，與 S3 的 4*wr 相同，
        # 但 S3 有 clip(wA, 3.5, 20)，vertex_finder 沒有 clip。
        # 統一 clip 後再算 R，確保 edge scanner 不超出 S3 分配的 ROI。
        # 注意：edge scanner 的 R/L 不受 YOLO bbox 約束——bbox 只用於
        # 結構張量 ROI（quality scoring）；掃描需要沿線段延伸，必然超出 bbox。
        wA_clipped = float(np.clip(wA_px, 3.5, 20))
        wB_clipped = float(np.clip(wB_px, 3.5, 20))
        wr = max(wA_clipped, wB_clipped)
        R = max(20, int(4 * wr))
        L = int(6 * max(halfA, halfB) + 10)
        Ns = 21

        h, w = img_gray.shape[:2]

        if Ix is None or Iy is None:
            blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
            Ix = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            Iy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        # 兩階段搜尋策略：
        # 1. 先跑單點試算
        # 2. 若角點不足或分數太低，才進 multi-shift
        
        jt = int(TEMPLATE_TYPES[junction_idx])
        target_corners = 4 if jt == 2 else 2  # X 需要 4，L/T 需要 2
        
        # DEBUG LOG 1: junction 參數
        shift_range = min(R // 3, 20)
        logging.debug("[Junction %d] R=%d L=%d shift_range=%d wA=%.1f wB=%.1f",
                      junction_idx, R, L, shift_range, wA_px, wB_px)
        
        # Stage 1: 單點試算
        corners_result = self._find_corners_at_center(
            Ix, Iy, center_px, R, L, Ns,
            tA_px, tB_px, nA_px, nB_px,
            wA_px, wB_px, halfA, halfB, h, w
        )
        
        all_corners = []
        if corners_result and corners_result['corners']:
            # 檢查是否足夠
            single_corners = corners_result['corners']
            avg_score = np.mean([c['score'] for c in single_corners])
            
            # 條件：角點數量足夠且平均分數 > 0.4
            if len(single_corners) >= target_corners and avg_score > 0.4:
                # 直接用單點結果，加上 confidence
                all_corners = [{**c, 'confidence': avg_score} for c in single_corners]
            elif do_multi_shift:
                # Stage 2: multi-shift 補強
                all_corners = self._multi_shift_search(
                    Ix, Iy, center_px, R, L, Ns,
                    tA_px, tB_px, nA_px, nB_px,
                    wA_px, wB_px, halfA, halfB, h, w
                )
        elif do_multi_shift:
            # 單點完全失敗，用 multi-shift
            all_corners = self._multi_shift_search(
                Ix, Iy, center_px, R, L, Ns,
                tA_px, tB_px, nA_px, nB_px,
                wA_px, wB_px, halfA, halfB, h, w
            )

        # 拓撲語意約束過濾
        all_corners = self._apply_topology_constraints(
            all_corners, H, junction_idx, tA_m, tB_m, halfA, halfB
        )

        # 整理輸出
        vertices = []
        for corner in all_corners:
            ct = corner.get('type', 'outer')
            try:
                ccode = corner_type_to_corner_code(
                    junction_idx=int(junction_idx),
                    corner_type=ct,
                    tA_m=np.asarray(tA_m, dtype=np.float64),
                    tB_m=np.asarray(tB_m, dtype=np.float64),
                )
            except Exception:
                ccode = -1
            vertices.append({
                'pos_px': corner['pos'],
                'pos_init': center_px.copy(),
                'width_px': (wA_px + wB_px) / 2,
                'confidence': corner.get('confidence', 0.5),
                'junction_idx': junction_idx,
                'corner_type': ct,
                'corner_code': int(ccode),
                'score': corner.get('score', 0.0)
            })

        avg_conf = np.mean([v['confidence'] for v in vertices]) if vertices else 0.0

        return {
            'vertices': vertices,
            'confidence': avg_conf,
            'line_widths': {'A': wA_px, 'B': wB_px},
            'directions': {'tA': tA_px, 'tB': tB_px}
        }

    def _apply_topology_constraints(self, all_corners, H, junction_idx,
                                     tA_m, tB_m, halfA, halfB) -> list:
        """應用拓撲語意約束過濾角點"""
        jt = int(TEMPLATE_TYPES[junction_idx])
        p0m = TEMPLATE_POINTS[junction_idx]
        fixed_selection = False

        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Hinv = None

        if Hinv is not None and len(all_corners) > 0:
            signA = self.topo_constraints.ray_signs_template(junction_idx, tA_m)
            signB = self.topo_constraints.ray_signs_template(junction_idx, tB_m)

            def is_bar(s):
                return (+1 in s) and (-1 in s)

            isA_bar = is_bar(signA)
            isB_bar = is_bar(signB)

            if jt == 1:  # T junction
                if isA_bar and (not isB_bar):
                    stem_axis = 'B'
                    stem_signs = signB
                elif isB_bar and (not isA_bar):
                    stem_axis = 'A'
                    stem_signs = signA
                else:
                    stem_axis = None
                    stem_signs = None

                if stem_axis is not None and stem_signs:
                    filtered = []
                    for c in all_corners:
                        pm = self.topo_constraints.px_to_template(Hinv, c['pos'])
                        sa, sb = self.topo_constraints.corner_quadrant_signs(pm, p0m, tA_m, tB_m)
                        stem_s = sa if stem_axis == 'A' else sb
                        if stem_s in stem_signs:
                            filtered.append(c)

                    if len(filtered) >= 2:
                        bar_axis = 'A' if stem_axis == 'B' else 'B'
                        group = {+1: [], -1: []}
                        for c in filtered:
                            pm = self.topo_constraints.px_to_template(Hinv, c['pos'])
                            sa, sb = self.topo_constraints.corner_quadrant_signs(pm, p0m, tA_m, tB_m)
                            bar_s = sa if bar_axis == 'A' else sb
                            if bar_s in group:
                                group[bar_s].append(c)

                        def best_one(lst):
                            if not lst:
                                return None
                            return sorted(
                                lst,
                                key=lambda c: (c.get('confidence', 0.0), c.get('count', 0),
                                             -c.get('spread', 999.0), c.get('score', 0.0)),
                                reverse=True
                            )[0]

                        c1 = best_one(group[+1])
                        c2 = best_one(group[-1])
                        picked2 = [c for c in [c1, c2] if c is not None]

                        if len(picked2) == 2:
                            all_corners = picked2
                            fixed_selection = True
                        else:
                            all_corners = filtered

            elif jt == 0:  # L junction
                if len(all_corners) >= 2:
                    C = np.array([3.05, 6.70], dtype=np.float32)
                    vC = C - p0m

                    vals = []
                    for c in all_corners:
                        pm = self.topo_constraints.px_to_template(Hinv, c['pos'])
                        vals.append(float(np.dot(pm - p0m, vC)))

                    inner_idx = int(np.argmax(vals))
                    outer_idx = int(np.argmin(vals))

                    if inner_idx != outer_idx:
                        all_corners = [all_corners[inner_idx], all_corners[outer_idx]]
                        fixed_selection = True

        if not fixed_selection:
            target_n = 4 if jt == 2 else 2

            all_corners = sorted(
                all_corners,
                key=lambda c: (
                    c.get('confidence', 0.0),
                    c.get('count', 0),
                    -c.get('spread', 999.0),
                    c.get('score', 0.0)
                ),
                reverse=True
            )

            picked = []
            min_sep = max(2.0, 0.6 * max(halfA, halfB))
            for c in all_corners:
                if len(picked) >= target_n:
                    break
                p = c['pos']
                if all(np.linalg.norm(p - pc['pos']) > min_sep for pc in picked):
                    picked.append(c)

            all_corners = picked

        return all_corners

    def _multi_shift_search(self, Ix, Iy, center_px, R, L, Ns,
                            tA_px, tB_px, nA_px, nB_px,
                            wA_px, wB_px, halfA, halfB, h, w) -> list:
        """
        多次平移搜尋，收集穩定的角點
        
        改進：使用隨機 Gaussian 抽樣取代全網格暴力掃描
        - 限制 shift_range 最大值
        - 限制總嘗試次數
        - 早停機制
        """
        # 限制 shift_range 避免爆炸
        shift_range = min(R // 3, 20)
        
        # 最大嘗試次數
        max_trials = 120
        
        # 0-hit 早停閾值
        zero_hit_threshold = 20
        
        # 時間預算 (秒)
        time_budget = 0.10  # 100ms
        start_time = time.perf_counter()
        
        # 早停：當每個象限都收集到足夠候選時
        min_candidates_per_quadrant = 5
        
        all_candidates = {'++': [], '+-': [], '-+': [], '--': []}
        
        # 使用 Gaussian 分佈抽樣（中心密集、外圍稀疏）
        sigma = max(shift_range / 2, 3.0)
        
        trial_count = 0
        early_stop_reason = None
        
        for _ in range(max_trials):
            trial_count += 1
            
            # Gaussian 抽樣
            dx = int(np.clip(np.random.normal(0, sigma), -shift_range, shift_range))
            dy = int(np.clip(np.random.normal(0, sigma), -shift_range, shift_range))
            
            shifted_center = center_px + np.array([dx, dy], dtype=np.float32)

            result = self._find_corners_at_center(
                Ix, Iy, shifted_center, R, L, Ns,
                tA_px, tB_px, nA_px, nB_px,
                wA_px, wB_px, halfA, halfB, h, w
            )

            if result and result['corners']:
                for corner in result['corners']:
                    lbl = corner.get('type')
                    if lbl in all_candidates:
                        all_candidates[lbl].append({
                            'pos': corner['pos'],
                            'score': corner['score'],
                            'shift': (dx, dy)
                        })
            
            total_candidates = sum(len(c) for c in all_candidates.values())
            
            # 早停 1: 所有象限都有足夠候選
            all_sufficient = all(
                len(candidates) >= min_candidates_per_quadrant
                for candidates in all_candidates.values()
            )
            if all_sufficient:
                early_stop_reason = 'sufficient'
                break
            
            # 早停 2: 0-hit (跑了 N 次但完全沒候選)
            if trial_count >= zero_hit_threshold and total_candidates == 0:
                early_stop_reason = 'zero_hit'
                break
            
            # 早停 3: 時間預算
            if time.perf_counter() - start_time > time_budget:
                early_stop_reason = 'timeout'
                break
        
        # DEBUG LOG
        total_candidates = sum(len(c) for c in all_candidates.values())
        quadrant_counts = {k: len(v) for k, v in all_candidates.items()}
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logging.debug(
            "[multi-shift] %s trials=%d candidates=%d %.1fms",
            early_stop_reason or 'max_trials', trial_count, total_candidates, elapsed_ms
        )

        # Clustering
        stable_corners = []
        for lbl, candidates in all_candidates.items():
            if len(candidates) < 3:
                continue

            positions = np.array([c['pos'] for c in candidates])
            scores = np.array([c['score'] for c in candidates])

            weights = np.maximum(scores, 0.01)
            weighted_pos = SubpixelUtils.weighted_median_2d(positions, weights)

            distances = np.linalg.norm(positions - weighted_pos, axis=1)
            spread = np.std(distances)

            cluster_threshold = max(2.0, (wA_px + wB_px) / 4)

            if spread < cluster_threshold:
                confidence = 1.0 - min(1.0, spread / cluster_threshold)
                stable_corners.append({
                    'pos': weighted_pos,
                    'score': np.mean(scores),
                    'confidence': confidence,
                    'type': lbl,
                    'spread': spread,
                    'count': len(candidates)
                })

        return stable_corners

    def _find_corners_at_center(self, Ix, Iy, center_px, R, L, Ns,
                                 tA_px, tB_px, nA_px, nB_px,
                                 wA_px, wB_px, halfA, halfB, h, w) -> dict:
        """在給定中心點尋找四個角點"""
        cx, cy = center_px
        if cx < R or cx >= w - R or cy < R or cy >= h - R:
            return None

        edgesA_plus, edgesA_minus = self.edge_scanner.scan_for_edges(
            Ix, Iy, center_px, tA_px, nA_px, wA_px, R, L, Ns, h, w
        )
        edgesB_plus, edgesB_minus = self.edge_scanner.scan_for_edges(
            Ix, Iy, center_px, tB_px, nB_px, wB_px, R, L, Ns, h, w
        )

        # 自適應 min_points (與 Ns 相關)
        min_points = max(3, Ns // 6)
        if (len(edgesA_plus) < min_points or len(edgesA_minus) < min_points or
            len(edgesB_plus) < min_points or len(edgesB_minus) < min_points):
            return None

        # 自適應 RANSAC threshold (與線寬相關)
        max_line_width = max(wA_px, wB_px)
        ransac_threshold = max(1.5, 0.35 * max_line_width)

        lineA_plus = LineRANSAC.fit_line(np.array(edgesA_plus), threshold=ransac_threshold)
        lineA_minus = LineRANSAC.fit_line(np.array(edgesA_minus), threshold=ransac_threshold)
        lineB_plus = LineRANSAC.fit_line(np.array(edgesB_plus), threshold=ransac_threshold)
        lineB_minus = LineRANSAC.fit_line(np.array(edgesB_minus), threshold=ransac_threshold)

        if any(line is None for line in [lineA_plus, lineA_minus, lineB_plus, lineB_minus]):
            return None

        corners = []
        combinations = [
            (lineA_plus, lineB_plus, '++'),
            (lineA_plus, lineB_minus, '+-'),
            (lineA_minus, lineB_plus, '-+'),
            (lineA_minus, lineB_minus, '--'),
        ]

        for lineA, lineB, label in combinations:
            intersection = HomographyUtils.line_intersection(lineA, lineB)
            if intersection is not None:
                dist = np.linalg.norm(intersection - center_px)
                score = max(0, 1.0 - dist / (2 * R))
                corners.append({
                    'pos': intersection,
                    'score': score,
                    'type': label
                })

        return {'corners': corners}
