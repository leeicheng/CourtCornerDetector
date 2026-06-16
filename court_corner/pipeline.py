"""
pipeline.py — 四階段角點定位管線編排（線為主求 H）
================================================================
串接：
  Stage 1  JunctionDetector       交點偵測（YOLO）
  Stage 2  LineHomographySolver   單應求解（線為主）：橋接使用者的
             court_homography_tool.py（相依 folder_yolo_tool.py）——
             全域 Steger 抽線 → cross-ratio 線標號 → PROSAC → Steger 精修。
             需 court_homography_tool.py 與 folder_yolo_tool.py 可被 import
             （與本程式同層、或在 cwd / PYTHONPATH）。
  Stage 3  CornerGenerator        角點生成（H 投影 + Steger 精修）
  Stage 4  QualityEvaluator       品質評估與輸出（cid, x, y, conf）

交點清單直接採用 court_homography_tool 已精修的投影點（row*5+col = junction_idx，
與本套件 shared.court_model 範本一致）。
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import cv2

from . import config
from .stages.detection import JunctionDetector, DetectionResult
from .stages.corners import CornerGenerator
from .stages.quality import QualityEvaluator, FinalCorner


class PipelineResult:
    """整體管線輸出。"""

    def __init__(self):
        self.status = "fail"
        self.method = "line"
        self.confidence = "low"                # 整體 H 信心（high/medium/low）
        self.corners: List[FinalCorner] = []
        self.H: Optional[np.ndarray] = None
        self.detection: Optional[DetectionResult] = None
        self.line = None                        # LineHomographyResult
        self.report: dict = {}
        self.message: str = ""
        self.elapsed_s: float = 0.0             # 整體處理時間（秒）
        self.stage_times: dict = {}             # 各階段耗時（秒）

    def corners_as_tuples(self):
        return [c.as_tuple() for c in self.corners]

    def _homography_dict(self):
        if self.line is not None:
            lr = self.line
            return {
                "method": self.method,
                "confidence": self.confidence,
                "line_consistency": round(lr.line_consistency, 4),
                "type_consistency": round(lr.type_consistency, 4),
                "line_support": round(lr.line_support, 4),
                "line_support_ok": bool(lr.line_support_ok),
                "solver_method": lr.method,
                "attempt": lr.attempt,
                "n_steger_refined": lr.n_steger_refined,
                "n_courts": lr.n_courts,
                "n_junctions": len(lr.junctions),
            }
        return {"method": self.method, "confidence": self.confidence}

    def to_dict(self):
        rep = dict(self.report)
        candidates = rep.pop("corner_candidates", [])   # 提到頂層,與 corners 並列
        return {
            "status": self.status,
            "message": self.message,
            "elapsed_s": round(self.elapsed_s, 3),
            "stage_times": {k: round(v, 3) for k, v in self.stage_times.items()},
            "H": (self.H.tolist() if self.H is not None else None),
            "n_detections": (len(self.detection) if self.detection else 0),
            "homography": self._homography_dict(),
            "report": rep,
            "corners": [c.as_dict() for c in self.corners],
            "corner_candidates": candidates,
        }


class CourtCornerPipeline:
    """
    羽球場角點四階段定位管線（線為主求 H）。

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
                 dark: bool = False,
                 bright_lines: bool = None,
                 min_line_support: float = 0.45,
                 device: str = None,
                 verbose: bool = True):
        self.verbose = verbose
        self.dark = dark
        self.method = "line"
        self.bright_lines = (not dark) if bright_lines is None else bright_lines

        # Stage 1
        self.detector = JunctionDetector(yolo_weight, conf=yolo_conf,
                                         device=device, verbose=verbose)
        # Stage 2（線為主）
        from .stages.topology_line import LineHomographySolver
        self.line_solver = LineHomographySolver(
            dark=dark, steger_refine=True, min_line_support=min_line_support)
        # Stage 3 / 4
        self.generator = CornerGenerator(bright_lines=self.bright_lines)
        self.evaluator = QualityEvaluator(corner_conf=corner_conf)

    # --------------------------------------------------------------
    def _log(self, *a):
        if self.verbose:
            print(*a)

    # --------------------------------------------------------------
    def run(self, img_path: str) -> PipelineResult:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取影像：{img_path}")
        return self.run_image(img)

    # --------------------------------------------------------------
    def run_image(self, img_bgr: np.ndarray,
                  detection: Optional[DetectionResult] = None,
                  cache: Optional[dict] = None,
                  cache_key: Optional[str] = None) -> PipelineResult:
        """
        對 BGR 影像執行完整管線。

        Args:
            img_bgr   : 輸入影像
            detection : 可選，外部已算好的第一階段結果（測試/重跑時可跳過 YOLO）
            cache     : 可選，呼叫端持有的 dict；連同 cache_key 快取 Stage 2（抽線+求解），
                        供同一張圖的 sweep（corner_conf / min_line_support 等）重跑時跳過重算
            cache_key : 快取鍵（通常用影像路徑或唯一 id）
        """
        out = PipelineResult()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        t0 = time.perf_counter()

        # ---- Stage 1 ----
        ts = time.perf_counter()
        if detection is None:
            detection = self.detector.detect(img_bgr)
        out.stage_times["detect"] = time.perf_counter() - ts
        out.detection = detection
        if len(detection) < 4:
            out.message = f"第一階段交點不足（{len(detection)} < 4），無法求解單應矩陣。"
            out.elapsed_s = time.perf_counter() - t0
            self._log("[Pipeline]", out.message)
            return out

        # ---- Stage 2（線為主）----
        ts = time.perf_counter()
        from .stages.topology_line import detection_to_anns
        anns, class_names = detection_to_anns(detection)
        line_res = self.line_solver.solve(img_bgr, anns, class_names,
                                          cache=cache, cache_key=cache_key)
        out.stage_times["solve_H"] = time.perf_counter() - ts
        out.line = line_res
        if line_res.status != "ok" or line_res.H is None:
            out.message = f"第二階段（線為主）求解失敗：{line_res.message}"
            out.elapsed_s = time.perf_counter() - t0
            self._log("[Pipeline]", out.message)
            return out
        out.H = line_res.H
        out.confidence = line_res.confidence
        junctions = line_res.junctions
        self._log(f"[Stage2/line] H 求解成功  method={line_res.method}  "
                  f"lc={line_res.line_consistency:.3f}  tc={line_res.type_consistency:.3f}  "
                  f"線支持={line_res.line_support:.2f}{'' if line_res.line_support_ok else '(不足)'}  "
                  f"steger_refined={line_res.n_steger_refined}  交點={len(junctions)}")

        # ---- Stage 3 ----
        ts = time.perf_counter()
        candidates = self.generator.generate(gray, out.H, junctions)
        out.stage_times["corners"] = time.perf_counter() - ts
        self._log(f"[Stage3] 由 {len(junctions)} 個交點生成 {len(candidates)} 個角點候選")

        # ---- Stage 4 ----
        ts = time.perf_counter()
        corners, report = self.evaluator.evaluate(
            gray, candidates, H=out.H, geom_quality=out.confidence)
        out.stage_times["quality"] = time.perf_counter() - ts
        out.corners = corners
        out.report = report
        out.status = "ok"
        out.elapsed_s = time.perf_counter() - t0
        out.message = (f"完成：輸出 {len(corners)} 個角點 "
                       f"（strong {report['n_strong']} + weak {report['n_weak']}，"
                       f"hidden {report['n_hidden']}；候選 {report['n_candidates']}，"
                       f"門檻 conf≥{report['corner_conf']}）")
        self._log("[Stage4]", out.message,
                  f"｜處理時間 {out.elapsed_s:.2f}s")
        return out


__all__ = ["CourtCornerPipeline", "PipelineResult"]
