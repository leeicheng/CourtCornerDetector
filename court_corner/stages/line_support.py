"""
stages/line_support.py — 單應的「白線支持度」驗證
================================================================
把求得的 H 投影出球場真實格線（48 條邊，省略中段中線），沿每條線取樣，
量測影像上是否真有白線證據（亮脊：線中心比兩側亮，且 white-tophat 響應夠強），
回傳整體支持度 [0,1] 與每條邊的支持度。

用途：
  - 驗證 H 是否「在影像上有線支持」——把格線投到空地的錯解（求解跑掉）支持度低。
  - 作為信心調整與雙球場挑選的依據。

限制（重要）：
  - 線支持是「必要非充分」。羽球場平行線多，錯位的對應仍可能踩在真白線上而得到
    不低的支持度；此情況需靠方向/對應邏輯處理，非線支持所能分辨。
  - 在 D2 對稱翻轉下，格線投影落在同一批白線，支持度相同——線支持無法分辨翻轉。

暗線球場（dark=True）：白底暗線，脊性測試反向（中心比兩側暗）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2

from ..shared.court_model import build_grid_connections, _tpl_xy, N_COL


def _make_odd(v: int) -> int:
    v = int(round(v))
    return v if v % 2 == 1 else v + 1


class LineSupportScorer:
    def __init__(self,
                 dark: bool = False,
                 k_ridge: float = 10.0,
                 tophat_thresh: float = 12.0,
                 min_edge_samples: int = 4,
                 margin: float = 20.0):
        self.dark = dark
        self.k_ridge = float(k_ridge)
        self.tophat_thresh = float(tophat_thresh)
        self.min_edge_samples = int(min_edge_samples)
        self.margin = float(margin)
        self._edges = build_grid_connections()

    # ----------------------------------------------------------------
    @staticmethod
    def _bilinear(g, x, y):
        h, w = g.shape
        if x < 0 or y < 0 or x > w - 1 or y > h - 1:
            return None
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        dx, dy = x - x0, y - y0
        return float((g[y0, x0] * (1 - dx) + g[y0, x1] * dx) * (1 - dy) +
                     (g[y1, x0] * (1 - dx) + g[y1, x1] * dx) * dy)

    # ----------------------------------------------------------------
    def score(self, gray: np.ndarray, H: np.ndarray) -> dict:
        """回傳 {support, n_edges, per_edge:{(a,b):frac}, n_samples}。"""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        h, w = gray.shape
        s = max(h, w) / 640.0                       # 依解析度縮放
        perp = max(3.0, 4.0 * s)
        far = max(7.0, 9.0 * s)
        ksize = _make_odd(max(7, 9 * s))

        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if self.dark:
            th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker)   # 暗線：black-tophat
        else:
            th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker)     # 亮線：white-tophat

        H = np.asarray(H, dtype=np.float64)

        def proj(idx):
            r, c = divmod(idx, N_COL)
            X = _tpl_xy(r, c)
            v = H @ np.array([X[0], X[1], 1.0])
            if abs(v[2]) < 1e-9:
                return None
            return np.array([v[0] / v[2], v[1] / v[2]])

        per_edge = {}
        sup_list = []
        n_samp = 0
        m = self.margin
        for a, b in self._edges:
            pa, pb = proj(a), proj(b)
            if pa is None or pb is None:
                continue
            # 兩端都在畫面外（含邊界外緣）則略過
            def outside(p):
                return p[0] < -m or p[1] < -m or p[0] > w + m or p[1] > h + m
            if outside(pa) and outside(pb):
                continue
            d = pb - pa
            L = float(np.hypot(*d))
            if L < 2:
                continue
            n = np.array([-d[1], d[0]]) / L
            N = max(6, int(L / 4))
            hit = 0
            tot = 0
            for t in np.linspace(0.08, 0.92, N):
                p = pa + d * t
                cv_ = self._bilinear(gray, p[0], p[1])
                lv = self._bilinear(gray, p[0] + n[0] * perp, p[1] + n[1] * perp)
                rv = self._bilinear(gray, p[0] - n[0] * perp, p[1] - n[1] * perp)
                lf = self._bilinear(gray, p[0] + n[0] * far, p[1] + n[1] * far)
                rf = self._bilinear(gray, p[0] - n[0] * far, p[1] - n[1] * far)
                tv = self._bilinear(th, p[0], p[1])
                if None in (cv_, lv, rv, lf, rf, tv):
                    continue
                tot += 1
                bg = float(np.median([lv, rv, lf, rf]))
                if self.dark:
                    ridge = (bg - cv_) > self.k_ridge and cv_ <= lv + 1 and cv_ <= rv + 1
                else:
                    ridge = (cv_ - bg) > self.k_ridge and cv_ >= lv - 1 and cv_ >= rv - 1
                if ridge and tv > self.tophat_thresh:
                    hit += 1
            if tot >= self.min_edge_samples:
                frac = hit / tot
                per_edge[(a, b)] = frac
                sup_list.append(frac)
                n_samp += tot
        support = float(np.mean(sup_list)) if sup_list else 0.0
        return {"support": support, "n_edges": len(sup_list),
                "per_edge": per_edge, "n_samples": n_samp}


__all__ = ["LineSupportScorer"]
