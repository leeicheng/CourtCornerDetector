"""
stage4_quality.py — 第四階段：品質評估與輸出
================================================================
對第三階段產生的角點候選，做幾何與影像證據評估，決定最終可輸出的角點
集合 (cid, x, y, conf)。

影像證據：以 VertexQualityScorer 在每個角點的局部 ROI 計算
  composite = dist_weight·exp(-d/τ) + heatmap_weight·clip(diff,0,1)
其中 d 為到最近 Harris/Steger peak 的距離、diff 為（正規化 Harris-R 減
正規化 Steger 脊強度）之差異圖在該點的值。composite∈[0,1] 即角點信心。

幾何證據：以 H⁻¹ 把角點投回 template 空間，與該 cid 的理論真值比較重投影
誤差（reprojection.py），作為幾何一致性診斷。

最終以 corner_conf 門檻過濾 composite，輸出 (cid, x, y, conf)。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .. import config
from ..vertex.vertex_quality import VertexQualityScorer
from ..vertex.reprojection import compute_vertex_reprojection, summarize_reprojection


class FinalCorner:
    """最終輸出角點。"""

    def __init__(self, cid, x, y, conf, junction_idx=-1, corner_type="",
                 source="", reproj_err_m=None):
        self.cid = int(cid)
        self.x = float(x)
        self.y = float(y)
        self.conf = float(conf)
        self.junction_idx = int(junction_idx)
        self.corner_type = corner_type
        self.source = source
        self.reproj_err_m = reproj_err_m

    def as_tuple(self):
        return (self.cid, self.x, self.y, self.conf)

    def as_dict(self):
        d = {"cid": self.cid, "x": round(self.x, 3), "y": round(self.y, 3),
             "conf": round(self.conf, 4)}
        if self.junction_idx >= 0:
            d["junction_idx"] = self.junction_idx
        if self.corner_type:
            d["corner_type"] = self.corner_type
        if self.source:
            d["source"] = self.source
        if self.reproj_err_m is not None:
            d["reproj_err_m"] = round(float(self.reproj_err_m), 4)
        return d


class QualityEvaluator:
    """
    第四階段：品質評估與輸出。

    使用：
        qe = QualityEvaluator(corner_conf=0.6)
        corners = qe.evaluate(img_gray, candidates, H=H)
    """

    def __init__(self,
                 corner_conf: float = None,
                 roi_half_ratio: float = None,
                 roi_min_half: int = None,
                 scorer: VertexQualityScorer = None):
        self.corner_conf = (config.CORNER_CONF_DEFAULT
                            if corner_conf is None else float(corner_conf))
        self.roi_half_ratio = (config.VQ_ROI_HALF_WIDTH_RATIO
                               if roi_half_ratio is None else roi_half_ratio)
        self.roi_min_half = (config.VQ_ROI_MIN_HALF
                             if roi_min_half is None else roi_min_half)
        self.geom_tau_m = config.VQ_GEOM_TAU_M
        self.img_line_weight = config.VQ_IMG_LINE_WEIGHT
        self.topo_quality_weight = dict(config.VQ_TOPO_QUALITY_WEIGHT)
        self.line_support_ratio = config.VQ_LINE_SUPPORT_RADIUS_RATIO
        self.line_support_min_r = config.VQ_LINE_SUPPORT_MIN_RADIUS
        self.scorer = scorer or VertexQualityScorer(
            peak_radius_px=config.VQ_PEAK_RADIUS_PX,
            harris_k=config.VQ_HARRIS_K,
            harris_sigma=config.VQ_HARRIS_SIGMA,
            harris_thr_pct=config.VQ_HARRIS_THRESHOLD_PCT,
            steger_sigma=config.VQ_STEGER_SIGMA,
            steger_thr_pct=config.VQ_STEGER_THRESHOLD_PCT,
            steger_dil_r=config.VQ_STEGER_DILATION_RADIUS,
            dark_ridges=config.VQ_STEGER_DARK_RIDGES,
            top_k=config.VQ_TOP_K,
            inset=config.VQ_INSET,
            anms_c=config.VQ_ANMS_C,
            anms_pool=config.VQ_ANMS_CANDIDATE_POOL,
            anms_nms_r=config.VQ_ANMS_LOOSE_NMS_RADIUS,
            prox_enabled=config.VQ_PROX_ENABLED,
            prox_min_area=config.VQ_PROX_MIN_AREA,
            prox_max_dist=config.VQ_PROX_MAX_DIST,
            prox_close_r=config.VQ_PROX_CLOSING_RADIUS,
            dist_weight=config.VQ_DIST_WEIGHT,
            heatmap_weight=config.VQ_HEATMAP_WEIGHT,
        )

    # --------------------------------------------------------------
    def _roi_for(self, pos_px, width_px, img_shape):
        half = max(int(round(self.roi_half_ratio * max(width_px, 1.0))),
                   int(self.roi_min_half))
        Himg, Wimg = img_shape[:2]
        cx, cy = float(pos_px[0]), float(pos_px[1])
        x0 = int(np.clip(round(cx - half), 0, Wimg - 1))
        x1 = int(np.clip(round(cx + half), 1, Wimg))
        y0 = int(np.clip(round(cy - half), 0, Himg - 1))
        y1 = int(np.clip(round(cy + half), 1, Himg))
        return x0, y0, x1, y1

    # --------------------------------------------------------------
    def _line_support(self, img_gray, pos_px, width_px, roi):
        """
        白線亮度支持（遮蔽偵測）：角點鄰域是否確實壓在亮線上。
        取角點小鄰域的高百分位亮度，相對於 ROI 背景做正規化。
        亮線→接近 1；被遮蔽 / 出界 / 草地→接近 0。對乾淨合成與真實影像皆有效。
        """
        Himg, Wimg = img_gray.shape[:2]
        rad = max(int(round(self.line_support_ratio * max(width_px, 1.0))),
                  int(self.line_support_min_r))
        cx, cy = int(round(pos_px[0])), int(round(pos_px[1]))
        x0 = max(0, cx - rad); x1 = min(Wimg, cx + rad + 1)
        y0 = max(0, cy - rad); y1 = min(Himg, cy + rad + 1)
        patch = img_gray[y0:y1, x0:x1]
        if patch.size == 0 or roi.size == 0:
            return 0.0
        local_hi = float(np.percentile(patch, 80))
        bg = float(np.percentile(roi, 25))
        hi = float(np.percentile(roi, 97))
        denom = max(hi - bg, 1e-6)
        return float(np.clip((local_hi - bg) / denom, 0.0, 1.0))

    # --------------------------------------------------------------
    def evaluate(self, img_gray: np.ndarray, candidates: List,
                 H: np.ndarray = None, geom_quality: str = "high",
                 return_all: bool = False):
        """
        評估角點候選並輸出最終集合。

        信心融合（同時用幾何與影像證據）：
            conf = g × image_support
            g            = topo_quality_weight[geom_quality] × exp(-reproj_err_m / τ_g)
            image_support = max(VertexQualityScorer.composite,
                                VQ_IMG_LINE_WEIGHT × 白線亮度支持)

        Args:
            img_gray    : 灰階影像
            candidates  : List[CornerCandidate]（第三階段輸出）
            H           : Homography（供重投影幾何證據；可省略，省略時 g 取 topo 權重）
            geom_quality: 第二階段 H 信心（'high'/'medium'/'low'）→ 幾何證據基準
            return_all  : True 則回傳所有候選（含未過門檻者）

        Returns:
            (final_corners: List[FinalCorner], report: dict)
        """
        img_shape = img_gray.shape
        topo_w = self.topo_quality_weight.get(geom_quality, 1.0)

        # 幾何證據：批次重投影誤差（以 cid 對照）
        reproj_by_cid = {}
        if H is not None and candidates:
            verts = [c.as_vertex() for c in candidates]
            recs = compute_vertex_reprojection(verts, H)
            for r in recs:
                reproj_by_cid[int(r.get("corner_code", -1))] = r["err_m"]

        scored: List[FinalCorner] = []
        for c in candidates:
            x0, y0, x1, y1 = self._roi_for(c.pos_px, c.width_px, img_shape)
            roi = img_gray[y0:y1, x0:x1]

            # 影像證據 1：Harris/Steger 角點 composite（真實影像紋理較有效）
            if roi.size == 0 or roi.shape[0] < 7 or roi.shape[1] < 7:
                composite = 0.0
            else:
                res = self.scorer.evaluate_vertex(
                    (float(c.pos_px[0]), float(c.pos_px[1])), roi, (x0, y0))
                composite = float(res.composite)
            # 影像證據 2：白線亮度支持（遮蔽偵測；合成與真實皆有效）
            line_sup = self._line_support(img_gray, c.pos_px, c.width_px, roi)
            image_support = max(composite, self.img_line_weight * line_sup)

            # 幾何證據
            err_m = reproj_by_cid.get(c.corner_code)
            if err_m is None:
                g = topo_w
            else:
                g = topo_w * float(np.exp(-float(err_m) / max(self.geom_tau_m, 1e-6)))

            conf = float(np.clip(g * image_support, 0.0, 1.0))
            scored.append(FinalCorner(
                cid=c.corner_code, x=float(c.pos_px[0]), y=float(c.pos_px[1]),
                conf=conf, junction_idx=c.junction_idx, corner_type=c.corner_type,
                source=c.source, reproj_err_m=err_m))

        kept = [fc for fc in scored if fc.conf >= self.corner_conf]
        kept.sort(key=lambda fc: (-fc.conf, fc.cid))

        report = {
            "n_candidates": len(candidates),
            "n_passed": len(kept),
            "corner_conf": self.corner_conf,
            "geom_quality": geom_quality,
            "mean_conf_passed": (round(float(np.mean([fc.conf for fc in kept])), 4)
                                 if kept else 0.0),
        }
        if reproj_by_cid:
            errs = [v for v in reproj_by_cid.values()]
            report["reproj_err_m"] = {
                "mean": round(float(np.mean(errs)), 4),
                "max": round(float(np.max(errs)), 4),
            }

        return (scored if return_all else kept), report


__all__ = ["QualityEvaluator", "FinalCorner"]
