"""
stage3_corners.py — 第三階段：角點生成
================================================================
利用第二階段的拓樸對應（H + template_id），於每個交點的局部區域推導白線
外緣角點。

每個交點依其型別（X→4 角、T→2 stem 側角、L→2 inner+outer 角）由 H 投影
得幾何候選角點（_build_h_rectified_corners + _filter_corners_by_topology），
再以 Steger 中線偏移法（StegerVertexFinder）在局部 ROI 萃取白線、做次像素
精修，並透過 corner_code(cid) 配對融合（HomographyVertexRefiner）。

輸出為完整候選集：每個幾何上有效的角點都保留為候選（帶 cid），Steger 有
確認者用融合/取代後位置（精度較高），未確認者保留 H 投影位置（標記
'h_only'）。最終是否輸出由第四階段以影像 + 幾何證據評估決定。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import config
from ..vertex.steger_vertex_finder import StegerVertexFinder
from ..vertex.h_refine import (
    HomographyVertexRefiner, _build_h_rectified_corners, _filter_corners_by_topology,
)


class CornerCandidate:
    """單一角點候選。"""

    def __init__(self, corner_code, junction_idx, corner_type, pos_px,
                 width_px, source, prelim_conf, h_pos_px=None):
        self.corner_code = int(corner_code)      # cid
        self.junction_idx = int(junction_idx)
        self.corner_type = corner_type           # '++'/'+-'/'-+'/'--'
        self.pos_px = np.asarray(pos_px, dtype=np.float32)        # 最終位置（精修後若有）
        self.h_pos_px = np.asarray(h_pos_px if h_pos_px is not None else pos_px,
                                   dtype=np.float32)               # H 投影位置（永遠保留）
        self.width_px = float(width_px)
        self.source = source                     # 'fused'|'h_replaced'|'h_only'
        self.prelim_conf = float(prelim_conf)

    def as_vertex(self):
        """轉成 vertex dict（供 reprojection / quality 共用）。"""
        return {
            "pos_px": self.pos_px.copy(),
            "junction_idx": self.junction_idx,
            "corner_type": self.corner_type,
            "corner_code": self.corner_code,
            "width_px": self.width_px,
            "confidence": self.prelim_conf,
            "h_refine_source": self.source,
        }


class CornerGenerator:
    """
    第三階段：角點生成器。

    使用：
        gen = CornerGenerator()
        candidates = gen.generate(img_gray, H, junctions)
        # junctions: List[(junction_idx, center_px)]，center_px 為偵測中心或 H 投影中心
    """

    def __init__(self,
                 line_width_m: float = None,
                 h_refine_enabled: bool = None,
                 outlier_px: float = None,
                 blend_weight: float = None,
                 bright_lines: bool = True):
        self.line_width_m = config.LINE_WIDTH_M if line_width_m is None else line_width_m
        self.h_refine_enabled = (config.VF_H_REFINE_ENABLED
                                 if h_refine_enabled is None else h_refine_enabled)
        self.outlier_px = config.VF_H_REFINE_OUTLIER_PX if outlier_px is None else outlier_px
        self.blend_weight = (config.VF_H_REFINE_BLEND_WEIGHT
                             if blend_weight is None else blend_weight)
        self.bright_lines = bright_lines
        self._steger = StegerVertexFinder(line_width_m=self.line_width_m)

    # --------------------------------------------------------------
    def generate(self, img_gray: np.ndarray, H: np.ndarray,
                 junctions: List) -> List[CornerCandidate]:
        """
        對一組交點生成角點候選。

        Args:
            img_gray : 灰階影像 (H, W)
            H        : Homography (template m → image px)
            junctions: List[(junction_idx, center_px)]

        Returns:
            List[CornerCandidate]
        """
        refiner = HomographyVertexRefiner(
            img_gray=img_gray, outlier_px=self.outlier_px, h_weight=self.blend_weight)

        out: List[CornerCandidate] = []
        for junction_idx, center_px in junctions:
            junction_idx = int(junction_idx)
            center_px = np.asarray(center_px, dtype=np.float32)

            # (1) H 投影幾何候選（完整集合，topology filtered）
            h_corners, (wA_px, wB_px) = _build_h_rectified_corners(junction_idx, H)
            if not h_corners:
                continue
            kept = _filter_corners_by_topology(h_corners, junction_idx)
            if not kept:
                continue
            width_px = (wA_px + wB_px) / 2.0

            # (2) Steger 次像素精修 + cid 融合
            refined_by_cid = {}
            if self.h_refine_enabled:
                try:
                    sv = self._steger.find_vertices_for_junction(
                        img_gray, H, junction_idx, center_px)
                    refined = refiner.refine(sv.get("vertices", []), H,
                                             junction_idx, center_px)
                    for rv in refined:
                        cc = int(rv.get("corner_code", -1))
                        if cc >= 0:
                            refined_by_cid[cc] = rv
                except Exception:
                    refined_by_cid = {}

            # (3) 合併：以 H 候選為基底，Steger 確認者改用精修位置
            for c in kept:
                cc = int(c.get("corner_code", -1))
                if cc < 0:
                    continue
                ct = c.get("corner_type", "outer")
                v_h = np.asarray(c["pos_px"], dtype=np.float32)
                rv = refined_by_cid.get(cc)
                if rv is not None:
                    pos = np.asarray(rv["pos_px"], dtype=np.float32)
                    src = rv.get("h_refine_source", "fused")
                    prelim = float(rv.get("confidence", 0.5))
                else:
                    pos = v_h
                    src = "h_only"
                    prelim = 0.3   # 純幾何，待第四階段以影像證據確認
                out.append(CornerCandidate(
                    corner_code=cc, junction_idx=junction_idx, corner_type=ct,
                    pos_px=pos, width_px=width_px, source=src, prelim_conf=prelim,
                    h_pos_px=v_h))

        # 去重：同 cid 可能由不同 junction 重複生成（理論上不會，但保險），保留信心高者
        best = {}
        for cand in out:
            key = cand.corner_code
            if key not in best or cand.prelim_conf > best[key].prelim_conf:
                best[key] = cand
        return list(best.values())


__all__ = ["CornerGenerator", "CornerCandidate"]
