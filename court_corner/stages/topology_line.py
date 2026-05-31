"""
stages/topology_line.py — 線為主的單應求解（使用內嵌移植的求解器）
================================================================
以移植進本套件的線為主求解器（court_corner.homography）求 H：
  全域 Steger 抽線 → cross-ratio 線標號 → PROSAC → Steger 次像素精修。

本模組是橋接層：把 YOLO 偵測整理成求解器需要的 Annotation 清單、呼叫
solve_image()，再把結果整理成下游 Stage 3 / 4 需要的 (junction_idx, center_px)
交點清單與單應矩陣 H。

求解器演算法已直接移植至 court_corner/homography/（solver.py 與 court_lines.py），
不再外部引用 court_homography_tool / folder_yolo_tool。求解器的場地範本
（COL_X / ROW_Y / 編號 r*5+c / 型別）與 shared.court_model 完全相同，故其 H 可
直接供 Stage 3 使用，且 res['projected'] 中每點的 row*5+col 即等於 junction_idx。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..homography import solver as _solver
from ..homography.court_lines import Annotation


# ════════════════════════════════════════════════════════════════
#  YOLO 偵測 → Annotation 清單
# ════════════════════════════════════════════════════════════════
def detection_to_anns(detection, image_id: int = 0):
    """
    把 Stage 1 的 DetectionResult 轉成求解器的 Annotation 清單。
    bbox 格式 [x, y, w, h]（左上角 + 寬高），class_id 用 YOLO 原始類別 id。
    回傳 (anns, class_names)。
    """
    anns = []
    bboxes = getattr(detection, "bboxes", []) or []
    raw_ids = getattr(detection, "raw_class_ids", []) or []
    for k, (x1, y1, x2, y2) in enumerate(bboxes):
        cid = int(raw_ids[k]) if k < len(raw_ids) else 0
        anns.append(Annotation(image_id=image_id, class_id=cid,
                               bbox=[float(x1), float(y1),
                                     float(x2 - x1), float(y2 - y1)], ann_id=k))
    class_names = dict(getattr(detection, "class_names", {}) or {})
    return anns, class_names


# ════════════════════════════════════════════════════════════════
#  結果容器
# ════════════════════════════════════════════════════════════════
class LineHomographyResult:
    def __init__(self):
        self.status = "fail"
        self.H: Optional[np.ndarray] = None
        self.junctions: List[Tuple[int, np.ndarray]] = []   # [(junction_idx, (x,y))]
        self.confidence = "low"                              # high / medium / low
        self.line_consistency = 0.0
        self.type_consistency = 0.0
        self.method = ""
        self.n_steger_refined = 0
        self.n_courts = 0
        self.message = ""
        self.raw = None


# ════════════════════════════════════════════════════════════════
#  線為主求解器（橋接內嵌移植的 solve_image）
# ════════════════════════════════════════════════════════════════
class LineHomographySolver:
    """
    以內嵌移植的線為主求解器求 H。

    使用：
        solver = LineHomographySolver(dark=False, steger_refine=True)
        res = solver.solve(img_bgr, anns, class_names)
        # res.H, res.junctions=[(junction_idx, (x,y)), ...]
    """

    def __init__(self, dark: bool = False, steger_refine: bool = True,
                 image_margin: int = 4):
        self.dark = dark
        self.steger_refine = steger_refine
        self.image_margin = image_margin

    # ----------------------------------------------------------------
    @staticmethod
    def _grade(lc: float, tc: float) -> str:
        if lc >= 0.95 and tc >= 0.90:
            return "high"
        if lc >= 0.80 and tc >= 0.75:
            return "medium"
        return "low"

    # ----------------------------------------------------------------
    def solve(self, img_bgr, anns, class_names) -> LineHomographyResult:
        """對單張影像求 H（給定 YOLO anns 與 class_names）。"""
        out = LineHomographyResult()

        if not anns or len(anns) < 4:
            out.message = f"交點不足（{len(anns) if anns else 0} < 4），無法求解 H。"
            return out

        courts = _solver.solve_image(img_bgr, anns, class_names,
                                     dark=self.dark, steger_refine=self.steger_refine)
        out.n_courts = len(courts) if courts else 0
        oks = [c for c in (courts or []) if c.get("status") == "ok" and c.get("H") is not None]
        if not oks:
            reason = ""
            if courts:
                reason = courts[0].get("reason", "")
            out.message = "線為主求解未得到可靠 H。" + (f"（{reason}）" if reason else "")
            return out

        # 多球場時挑最佳：線一致性 + 型別一致性 → 對應點數
        best = max(oks, key=lambda c: (c.get("line_consistency", 0.0)
                                       + c.get("type_consistency", 0.0),
                                       c.get("num_corr", 0)))
        H = np.asarray(best["H"], dtype=np.float64)
        out.H = H
        out.raw = best
        out.line_consistency = float(best.get("line_consistency", 0.0))
        out.type_consistency = float(best.get("type_consistency", 0.0))
        out.method = str(best.get("method", ""))
        out.n_steger_refined = int(best.get("steger_refined", 0) or 0)
        out.confidence = self._grade(out.line_consistency, out.type_consistency)

        # 交點清單：用 res['projected']（row*5+col = junction_idx），取影像內者，
        # center_px 用 H 投影位置（已經過 Steger 精修，對齊實際白線）。
        Himg, Wimg = img_bgr.shape[:2]
        m = self.image_margin
        junctions = []
        projected = best.get("projected", [])
        if projected:
            for p in projected:
                xy = p.get("xy")
                if not xy or not all(np.isfinite(v) for v in xy):
                    continue
                x, y = float(xy[0]), float(xy[1])
                if -m <= x <= Wimg + m and -m <= y <= Himg + m:
                    jid = int(p["row"]) * 5 + int(p["col"])
                    junctions.append((jid, np.array([x, y], dtype=np.float32)))
        else:
            from ..shared.court_model import TEMPLATE_POINTS
            for jid, (X, Y) in enumerate(TEMPLATE_POINTS):
                v = H @ np.array([X, Y, 1.0])
                if abs(v[2]) < 1e-9:
                    continue
                x, y = v[0] / v[2], v[1] / v[2]
                if -m <= x <= Wimg + m and -m <= y <= Himg + m:
                    junctions.append((jid, np.array([x, y], dtype=np.float32)))

        out.junctions = junctions
        out.status = "ok"
        out.message = (f"線為主求解成功（method={out.method}，lc={out.line_consistency:.3f}，"
                       f"tc={out.type_consistency:.3f}，交點 {len(junctions)}）")
        return out


__all__ = ["LineHomographySolver", "LineHomographyResult", "detection_to_anns"]
