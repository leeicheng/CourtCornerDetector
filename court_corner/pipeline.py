"""
pipeline.py — 四階段角點定位管線編排
================================================================
串接：
  Stage 1  JunctionDetector  交點偵測（YOLO）
  Stage 2  TopologySolver    拓樸求解（H + template_id 對應）
  Stage 3  CornerGenerator   角點生成（H 投影 + Steger 精修）
  Stage 4  QualityEvaluator  品質評估與輸出（cid, x, y, conf）

第三階段所需的交點清單由 H 投影產生：列出所有投影中心落在影像內的模板
節點（共 ≤30），有對應到偵測中心者優先採用偵測中心（較準），其餘採 H
投影中心。如此即使 YOLO 漏抓某些交點，只要 H 夠準仍能補出其角點，並由
第四階段以影像證據裁決是否輸出。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import cv2

from . import config
from .stages.detection import JunctionDetector, DetectionResult
from .stages.topology import TopologySolver, TopologyResult, _proj, _tpl_xy
from .stages.corners import CornerGenerator
from .stages.quality import QualityEvaluator, FinalCorner


class PipelineResult:
    """整體管線輸出。"""

    def __init__(self):
        self.status = "fail"
        self.corners: List[FinalCorner] = []
        self.H: Optional[np.ndarray] = None
        self.detection: Optional[DetectionResult] = None
        self.topology: Optional[TopologyResult] = None
        self.report: dict = {}
        self.message: str = ""

    def corners_as_tuples(self):
        return [c.as_tuple() for c in self.corners]

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "H": (self.H.tolist() if self.H is not None else None),
            "n_detections": (len(self.detection) if self.detection else 0),
            "topology": ({
                "status": self.topology.status,
                "confidence": self.topology.confidence,
                "type_consistency": round(self.topology.type_consistency, 4),
                "n_inliers": len(self.topology.inliers),
                "rmse_px": round(self.topology.rmse, 4),
                "n_steger_refined": self.topology.n_steger_refined,
            } if self.topology else None),
            "report": self.report,
            "corners": [c.as_dict() for c in self.corners],
        }


class CourtCornerPipeline:
    """
    羽球場角點四階段定位管線。

    使用：
        pipe = CourtCornerPipeline("best.pt", yolo_conf=0.25, corner_conf=0.6)
        result = pipe.run("court.jpg")
        for cid, x, y, conf in result.corners_as_tuples():
            ...
    """

    def __init__(self,
                 yolo_weight: str,
                 yolo_conf: float = None,
                 corner_conf: float = None,
                 bright_lines: bool = True,
                 steger_refine_h: bool = None,
                 verbose: bool = True):
        self.verbose = verbose
        self.bright_lines = bright_lines
        # Stage 1
        self.detector = JunctionDetector(
            yolo_weight, conf=yolo_conf, verbose=verbose)
        # Stage 2
        self.solver = TopologySolver(
            steger_refine_h=steger_refine_h, bright_lines=bright_lines)
        # Stage 3
        self.generator = CornerGenerator(bright_lines=bright_lines)
        # Stage 4
        self.evaluator = QualityEvaluator(corner_conf=corner_conf)

    # --------------------------------------------------------------
    def _log(self, *a):
        if self.verbose:
            print(*a)

    # --------------------------------------------------------------
    def _build_junction_list(self, H, detection: DetectionResult, topo: TopologyResult,
                             img_shape):
        """
        產生 Stage 3 的交點清單 [(junction_idx, center_px)]。
        所有 H 投影在影像內的模板節點都納入；對應到偵測的用偵測中心。
        """
        # template_id -> detection center（由 Stage 2 assignment 反查）
        tid_to_det = {}
        if detection is not None:
            for det_idx, tid in topo.assignment.items():
                if 0 <= det_idx < len(detection.node_pts):
                    tid_to_det[int(tid)] = detection.node_pts[det_idx]

        junctions = []
        for tid, p_img in topo.visible_template_ids(H, img_shape=img_shape, margin=4):
            center = tid_to_det.get(tid, p_img)   # 偵測中心優先
            junctions.append((tid, np.asarray(center, dtype=np.float32)))
        return junctions

    # --------------------------------------------------------------
    def run(self, img_path: str) -> PipelineResult:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取影像：{img_path}")
        return self.run_image(img)

    # --------------------------------------------------------------
    def run_image(self, img_bgr: np.ndarray,
                  detection: Optional[DetectionResult] = None) -> PipelineResult:
        """
        對 BGR 影像執行完整管線。

        Args:
            img_bgr   : 輸入影像
            detection : 可選，外部已算好的第一階段結果（測試/重跑時可跳過 YOLO）
        """
        out = PipelineResult()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

        # ---- Stage 1 ----
        if detection is None:
            detection = self.detector.detect(img_bgr)
        out.detection = detection
        if len(detection) < 4:
            out.message = f"第一階段交點不足（{len(detection)} < 4），無法求解單應矩陣。"
            self._log("[Pipeline]", out.message)
            return out

        # ---- Stage 2 ----
        topo = self.solver.solve(detection.node_pts, detection.node_types, img_bgr=img_bgr)
        out.topology = topo
        if topo.status != "ok" or topo.H is None:
            out.message = f"第二階段拓樸求解失敗：{topo.reason}"
            self._log("[Pipeline]", out.message)
            return out
        out.H = topo.H
        self._log(f"[Stage2] H 求解成功  conf={topo.confidence}  "
                  f"type_consistency={topo.type_consistency:.3f}  "
                  f"inliers={len(topo.inliers)}  rmse={topo.rmse:.3f}px  "
                  f"steger_refined={topo.n_steger_refined}")

        # ---- Stage 3 ----
        junctions = self._build_junction_list(topo.H, detection, topo, gray.shape)
        candidates = self.generator.generate(gray, topo.H, junctions)
        self._log(f"[Stage3] 由 {len(junctions)} 個交點生成 {len(candidates)} 個角點候選")

        # ---- Stage 4 ----
        corners, report = self.evaluator.evaluate(
            gray, candidates, H=topo.H, geom_quality=topo.confidence)
        out.corners = corners
        out.report = report
        out.status = "ok"
        out.message = (f"完成：輸出 {len(corners)} 個角點"
                       f"（候選 {report['n_candidates']}，門檻 conf≥{report['corner_conf']}）")
        self._log("[Stage4]", out.message)
        return out


__all__ = ["CourtCornerPipeline", "PipelineResult"]
