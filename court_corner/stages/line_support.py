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
    def score(self, gray: np.ndarray, H: np.ndarray) -> dict:
        """回傳 {support, n_edges, per_edge:{(a,b):frac}, n_samples}。

        向量化版本：先把所有邊、所有採樣點的 5 組座標（中心 cv、近側 lv/rv、
        遠側 lf/rf）一次收集成連續陣列，再用單次 cv2.remap 對 gray 與 th
        進行批次雙線性取樣，最後以 NumPy 布林運算取代逐點 if-else 判斷。
        """
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
        m = self.margin

        def proj(idx):
            r, c = divmod(idx, N_COL)
            X = _tpl_xy(r, c)
            v = H @ np.array([X[0], X[1], 1.0])
            if abs(v[2]) < 1e-9:
                return None
            return np.array([v[0] / v[2], v[1] / v[2]])

        # ── 1) 先把每條邊的採樣點座標收集成連續陣列 ──────────────
        # 為每條有效邊記錄其在攤平陣列中的 [start, end) 區段，最後再切回。
        edge_keys = []          # [(a, b), ...]
        edge_spans = []         # [(start, end), ...]
        cx_all, cy_all = [], []  # 中心
        lx_all, ly_all = [], []  # 近側 +n*perp
        rx_all, ry_all = [], []  # 近側 -n*perp
        fx_all, fy_all = [], []  # 遠側 +n*far
        gx_all, gy_all = [], []  # 遠側 -n*far
        cursor = 0
        for a, b in self._edges:
            pa, pb = proj(a), proj(b)
            if pa is None or pb is None:
                continue
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
            ts = np.linspace(0.08, 0.92, N)
            px = pa[0] + d[0] * ts
            py = pa[1] + d[1] * ts
            cx_all.append(px);              cy_all.append(py)
            lx_all.append(px + n[0] * perp); ly_all.append(py + n[1] * perp)
            rx_all.append(px - n[0] * perp); ry_all.append(py - n[1] * perp)
            fx_all.append(px + n[0] * far);  fy_all.append(py + n[1] * far)
            gx_all.append(px - n[0] * far);  gy_all.append(py - n[1] * far)
            edge_keys.append((a, b))
            edge_spans.append((cursor, cursor + N))
            cursor += N

        if cursor == 0:
            return {"support": 0.0, "n_edges": 0, "per_edge": {}, "n_samples": 0}

        cx = np.concatenate(cx_all).astype(np.float32)
        cy = np.concatenate(cy_all).astype(np.float32)
        lx = np.concatenate(lx_all).astype(np.float32)
        ly = np.concatenate(ly_all).astype(np.float32)
        rx = np.concatenate(rx_all).astype(np.float32)
        ry = np.concatenate(ry_all).astype(np.float32)
        fx = np.concatenate(fx_all).astype(np.float32)
        fy = np.concatenate(fy_all).astype(np.float32)
        gx = np.concatenate(gx_all).astype(np.float32)
        gy = np.concatenate(gy_all).astype(np.float32)

        # ── 2) 單次 cv2.remap 全局批次取樣（C-level 雙線性） ──────
        # 把 5 組採樣點水平串接成一張 (1, 5K) 的 map，一次取樣即可。
        K = cx.shape[0]
        map_x = np.concatenate([cx, lx, rx, fx, gx]).reshape(1, -1)
        map_y = np.concatenate([cy, ly, ry, fy, gy]).reshape(1, -1)
        # 邊界以 NaN 標記（越界視同無效採樣，對應原 _bilinear 回傳 None）
        sampled_g = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=float("nan")).reshape(5, K)
        sampled_t = cv2.remap(th, map_x[:, :K], map_y[:, :K],
                              interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=float("nan")).reshape(K)

        cv_v = sampled_g[0]
        lv = sampled_g[1]; rv = sampled_g[2]
        lf = sampled_g[3]; rf = sampled_g[4]
        tv = sampled_t

        # 任一採樣越界（NaN）→ 該點無效
        valid = np.isfinite(cv_v) & np.isfinite(lv) & np.isfinite(rv) \
            & np.isfinite(lf) & np.isfinite(rf) & np.isfinite(tv)

        bg = np.median(np.stack([lv, rv, lf, rf], axis=0), axis=0)
        # ── 3) 向量化脊性判斷（取代 if-else） ─────────────────────
        if self.dark:
            ridge = ((bg - cv_v) > self.k_ridge) & (cv_v <= lv + 1) & (cv_v <= rv + 1)
        else:
            ridge = ((cv_v - bg) > self.k_ridge) & (cv_v >= lv - 1) & (cv_v >= rv - 1)
        hit_mask = valid & ridge & (tv > self.tophat_thresh)

        # ── 4) 依每條邊的區段切回，計 hit/tot ───────────────────
        per_edge = {}
        sup_list = []
        n_samp = 0
        for key, (st, en) in zip(edge_keys, edge_spans):
            vseg = valid[st:en]
            tot = int(vseg.sum())
            if tot >= self.min_edge_samples:
                hit = int(hit_mask[st:en].sum())
                frac = hit / tot
                per_edge[key] = frac
                sup_list.append(frac)
                n_samp += tot
        support = float(np.mean(sup_list)) if sup_list else 0.0
        return {"support": support, "n_edges": len(sup_list),
                "per_edge": per_edge, "n_samples": n_samp}


__all__ = ["LineSupportScorer"]
