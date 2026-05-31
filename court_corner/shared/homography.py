"""
Homography Utils - Homography 與 Jacobian 計算工具
==================================================
包含:
- Jacobian 計算
- 方向轉換
- 線寬計算
"""

import numpy as np
import cv2
from .court_model import LINE_WIDTH_M, TEMPLATE_POINTS


class HomographyUtils:
    """Homography 相關計算工具"""

    @staticmethod
    def compute_jacobian(H: np.ndarray, pt_m: np.ndarray) -> np.ndarray:
        """
        計算 homography 在 template 點 (X, Y) 的 Jacobian (2x2)
        J 代表「template 的小位移 → image 的小位移」
        """
        X, Y = pt_m
        h = H.flatten()

        u = h[0] * X + h[1] * Y + h[2]
        v = h[3] * X + h[4] * Y + h[5]
        d = h[6] * X + h[7] * Y + h[8]

        if abs(d) < 1e-10:
            d = 1e-10

        d2 = d * d

        J = np.array([
            [(h[0] * d - u * h[6]) / d2, (h[1] * d - u * h[7]) / d2],
            [(h[3] * d - v * h[6]) / d2, (h[4] * d - v * h[7]) / d2]
        ])

        return J

    @staticmethod
    def transform_direction(J: np.ndarray, dir_m: np.ndarray) -> np.ndarray:
        """將 template 方向轉換到影像空間並正規化"""
        dir_px = J @ dir_m
        norm = np.linalg.norm(dir_px)
        if norm < 1e-10:
            return np.array([1.0, 0.0])
        return dir_px / norm

    @staticmethod
    def compute_line_width_px(J: np.ndarray, tangent_m: np.ndarray,
                               line_width_m: float = LINE_WIDTH_M) -> float:
        """
        計算線在影像中的寬度 (pixel)
        線寬沿法向量量度
        """
        normal_m = np.array([-tangent_m[1], tangent_m[0]])
        normal_px = J @ normal_m
        scale = np.linalg.norm(normal_px)
        return line_width_m * scale

    @staticmethod
    def perp(v: np.ndarray) -> np.ndarray:
        """旋轉 90 度"""
        return np.array([-v[1], v[0]])

    @staticmethod
    def line_intersection(line1: tuple, line2: tuple) -> np.ndarray:
        """計算兩直線交點 (ax + by + c = 0)，平行時回傳 None"""
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return np.array([x, y], dtype=np.float32)
