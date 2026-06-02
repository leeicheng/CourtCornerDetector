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
import cv2

from .. import config
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
        self.line_support = 0.0                              # 投影格線的白線支持度 [0,1]
        self.line_support_ok = False                         # 是否達線支持門檻
        self.attempt = ""                                    # 最終採用的嘗試（strict/relaxed/masked）
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
                 image_margin: int = 4, min_line_support: float = 0.45):
        self.dark = dark
        self.steger_refine = steger_refine
        self.image_margin = image_margin
        self.min_line_support = float(min_line_support)
        from .line_support import LineSupportScorer
        self._scorer = LineSupportScorer(dark=dark)
        # 多階段重試設定
        self.retry_enabled = bool(config.S2_RETRY_ENABLED)
        self.relaxed_params = dict(config.S2_RELAXED_LINE_PARAMS)
        self.retry_lc_ok = float(config.S2_RETRY_LC_OK)
        self.mask_dilate_ratio = float(config.S2_MASK_DILATE_RATIO)
        self.topk = int(config.S2_TOPK_CANDIDATES)
        self.fail_support_floor = float(config.S2_FAIL_SUPPORT_FLOOR)
        self.fail_lc_floor = float(config.S2_FAIL_LC_FLOOR)
        self.fail_tc_floor = float(config.S2_FAIL_TC_FLOOR)

    # ----------------------------------------------------------------
    def _build_court_mask(self, img_shape, anns, dilate_ratio):
        """YOLO 導引動態遮罩：交點凸包(+各框)外擴。把柱子/人/觀眾等場外結構排除在抽線之外。"""
        Himg, Wimg = img_shape[:2]
        mask = np.zeros((Himg, Wimg), np.uint8)
        centers, sizes = [], []
        for a in anns:
            x, y, w, h = a.bbox
            centers.append((x + w / 2.0, y + h / 2.0)); sizes.append(min(w, h))
        med = float(np.median(sizes)) if sizes else 10.0
        # 凸包（≥3 點）覆蓋所有交點間的線；不足則以圓覆蓋
        if len(centers) >= 3:
            hull = cv2.convexHull(np.asarray(centers, np.float32).reshape(-1, 1, 2)).astype(np.int32)
            cv2.fillConvexPoly(mask, hull, 255)
        for (cx, cy) in centers:
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(max(6, med)), 255, -1)
        r = max(6, int(round(dilate_ratio * med)))        # 外擴以含外圈球場線
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(mask, k)

    # ----------------------------------------------------------------
    @staticmethod
    def _grade(lc: float, tc: float, support: float, support_ok: bool) -> str:
        # 線支持不足 → 直接視為低信心（H 在影像上沒有足夠白線支持）
        if not support_ok:
            return "low"
        if lc >= 0.95 and tc >= 0.90:
            base = "high"
        elif lc >= 0.80 and tc >= 0.75:
            base = "medium"
        else:
            base = "low"
        # 支持度偏低時不給 high
        if base == "high" and support < 0.60:
            base = "medium"
        return base

    # ----------------------------------------------------------------
    def solve(self, img_bgr, anns, class_names) -> LineHomographyResult:
        """對單張影像求 H（給定 YOLO anns 與 class_names）。"""
        out = LineHomographyResult()

        if not anns or len(anns) < 4:
            out.message = f"交點不足（{len(anns) if anns else 0} < 4），無法求解 H。"
            return out

        # ── 多階段嘗試：strict → relaxed → masked，累積所有 ok 候選 ──
        attempts = [("strict", None, False)]
        if self.retry_enabled:
            attempts.append(("relaxed", self.relaxed_params, False))
            attempts.append(("masked", self.relaxed_params, True))

        gate = self.min_line_support
        mask = None
        cand = []                       # [{c: court_result, support: float, attempt: str}]
        n_attempts = 0
        for name, lp, use_mask in attempts:
            n_attempts += 1
            if use_mask and mask is None:
                mask = self._build_court_mask(img_bgr.shape, anns, self.mask_dilate_ratio)
            courts = _solver.solve_image(
                img_bgr, anns, class_names, dark=self.dark,
                steger_refine=self.steger_refine,
                line_params=lp, mask=(mask if use_mask else None))
            for c in (courts or []):
                if c.get("status") != "ok" or c.get("H") is None:
                    continue
                sup = float(self._scorer.score(
                    img_bgr, np.asarray(c["H"], dtype=np.float64))["support"])
                cand.append({"c": c, "support": sup, "attempt": name})
            # 提前結束：本輪已有「白線支持足 + line_consistency 夠」的候選
            if any(d["support"] >= gate
                   and float(d["c"].get("line_consistency", 0.0)) >= self.retry_lc_ok
                   for d in cand):
                break

        out.n_courts = n_attempts
        if not cand:
            out.message = ("線為主求解未得到可靠 H"
                           "（strict / relaxed / masked 皆無候選）。")
            return out

        # ── Attempt 4：保留 white-line support 前 K 個候選，再以
        #    (是否過線支持門檻, lc+tc, 線支持, court 序) 重排挑最佳 ──
        cand.sort(key=lambda d: -d["support"])
        topk = cand[: self.topk]

        def _rank(d):
            c = d["c"]
            lc = float(c.get("line_consistency", 0.0))
            tc = float(c.get("type_consistency", 0.0))
            return (1 if d["support"] >= gate else 0,
                    round(lc + tc, 3), d["support"], -int(c.get("court", 0)))

        best_d = max(topk, key=_rank)
        best, best_sup = best_d["c"], best_d["support"]
        support_ok = best_sup >= gate

        # ── Attempt 5：只有當最佳候選 線/型/白線支持 全都低於底線才算徹底失敗 ──
        lc0 = float(best.get("line_consistency", 0.0))
        tc0 = float(best.get("type_consistency", 0.0))
        if (best_sup < self.fail_support_floor and lc0 < self.fail_lc_floor
                and tc0 < self.fail_tc_floor):
            out.raw = best
            out.attempt = best_d["attempt"]
            out.message = (f"線為主求解所有候選證據皆不足（最佳：線支持={best_sup:.2f}，"
                           f"lc={lc0:.2f}，tc={tc0:.2f}；試 {n_attempts} 階段）。")
            return out

        out.attempt = best_d["attempt"]
        H = np.asarray(best["H"], dtype=np.float64)
        out.H = H
        out.raw = best
        out.line_consistency = float(best.get("line_consistency", 0.0))
        out.type_consistency = float(best.get("type_consistency", 0.0))
        out.method = str(best.get("method", ""))
        out.n_steger_refined = int(best.get("steger_refined", 0) or 0)
        out.line_support = float(best_sup)
        out.line_support_ok = bool(support_ok)
        out.confidence = self._grade(out.line_consistency, out.type_consistency,
                                     best_sup, support_ok)

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
        tag = "" if out.line_support_ok else "，⚠ 白線支持不足，結果可能不可靠"
        out.message = (f"線為主求解成功（嘗試={out.attempt}，method={out.method}，"
                       f"lc={out.line_consistency:.3f}，tc={out.type_consistency:.3f}，"
                       f"線支持={out.line_support:.2f}，交點 {len(junctions)}{tag}）")
        return out


__all__ = ["LineHomographySolver", "LineHomographyResult", "detection_to_anns"]
