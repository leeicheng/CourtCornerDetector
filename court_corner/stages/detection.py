"""
stage1_detection.py — 第一階段：交點偵測
================================================================
以 YOLO（ultralytics）由輸入影像偵測球場交點，輸出每個交點的：
  類別（L / T / X）、中心座標（bbox 中心）、信心值、bbox。

ultralytics 採延遲載入（僅在實際執行偵測時 import），避免在無需偵測的
情境下載入 torch。class → junction type 對應會自動判讀模型的 class 名稱
（含 'x'/'cross'→X、't'→T、'l'/'corner'→L），純數字索引則退回 config 表
（0=L,1=T,2=X）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import config


class DetectionResult:
    """第一階段偵測結果。"""

    def __init__(self, node_pts, node_types, confidences, bboxes,
                 class_names=None, raw_class_ids=None):
        self.node_pts = node_pts            # List[(x, y)]  bbox 中心
        self.node_types = node_types        # List['L'|'T'|'X']
        self.confidences = confidences      # List[float]
        self.bboxes = bboxes                # List[(x1, y1, x2, y2)]
        self.class_names = class_names or {}
        self.raw_class_ids = raw_class_ids or []

    def __len__(self):
        return len(self.node_pts)

    def counts_by_type(self):
        out = {"L": 0, "T": 0, "X": 0}
        for t in self.node_types:
            if t in out:
                out[t] += 1
        return out


class JunctionDetector:
    """
    第一階段：YOLO 交點偵測器。

    使用：
        det = JunctionDetector("best.pt", conf=0.25)
        result = det.detect(img_bgr)            # 或 det.detect_path("a.jpg")
    """

    def __init__(self,
                 weight_path: str,
                 conf: float = None,
                 iou: float = None,
                 max_det: int = None,
                 classid_to_type: dict = None,
                 device: Optional[str] = None,
                 verbose: bool = True):
        self.weight_path = weight_path
        self.conf = config.YOLO_DEFAULT_CONF if conf is None else float(conf)
        self.iou = config.YOLO_IOU if iou is None else float(iou)
        self.max_det = config.YOLO_MAX_DET if max_det is None else int(max_det)
        self.classid_to_type = classid_to_type or dict(config.YOLO_CLASSID_TO_TYPE)
        self.device = device
        self.verbose = verbose
        self._model = None          # 延遲載入
        self._class_names = None

    # --------------------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO   # 延遲載入
        except ImportError as e:
            raise ImportError(
                "需要 ultralytics 套件才能執行第一階段交點偵測。"
                "請先安裝：pip install ultralytics"
            ) from e
        self._model = YOLO(self.weight_path)
        # class id -> name
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            self._class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self._class_names = {i: str(v) for i, v in enumerate(names)}
        else:
            self._class_names = {}
        if self.verbose and self._class_names:
            print(f"[Stage1] YOLO 模型 class 名稱: {self._class_names}")

    # --------------------------------------------------------------
    def _map_class_to_type(self, cls_id: int) -> str:
        """class id → 'L'/'T'/'X'。先看名稱關鍵字，否則退回索引表。"""
        name = (self._class_names or {}).get(int(cls_id), "")
        low = str(name).strip().lower()
        if low:
            if "x" in low or "cross" in low or low in ("4", "+"):
                return "X"
            if low.startswith("t") or "tee" in low or "t_" in low:
                return "T"
            if low.startswith("l") or "corner" in low or "elbow" in low:
                return "L"
            # 純數字名稱
            if low.isdigit():
                return self.classid_to_type.get(int(low), "X")
        return self.classid_to_type.get(int(cls_id), "X")

    # --------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray) -> DetectionResult:
        """對 BGR 影像執行偵測。"""
        self._ensure_model()
        kw = dict(conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        results = self._model(img_bgr, **kw)
        if not results:
            return DetectionResult([], [], [], [], self._class_names, [])
        r0 = results[0]

        node_pts, node_types, confs, bboxes, raw_ids = [], [], [], [], []
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            cf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cid, cv in zip(xyxy, cls, cf):
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                node_pts.append((cx, cy))
                node_types.append(self._map_class_to_type(int(cid)))
                confs.append(float(cv))
                bboxes.append((float(x1), float(y1), float(x2), float(y2)))
                raw_ids.append(int(cid))

        res = DetectionResult(node_pts, node_types, confs, bboxes,
                              self._class_names, raw_ids)
        if self.verbose:
            cnt = res.counts_by_type()
            print(f"[Stage1] 偵測到 {len(res)} 個交點  "
                  f"(L={cnt['L']}, T={cnt['T']}, X={cnt['X']})  conf≥{self.conf}")
        return res

    # --------------------------------------------------------------
    def detect_path(self, img_path: str) -> DetectionResult:
        """讀檔後執行偵測（保留 EXIF 方向）。"""
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取影像：{img_path}")
        return self.detect(img)


__all__ = ["JunctionDetector", "DetectionResult"]
