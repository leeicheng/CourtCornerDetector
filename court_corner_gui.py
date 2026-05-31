#!/usr/bin/env python3
"""
court_corner_gui.py — 羽球場角點定位 GUI（PyQt6）
================================================================
以圖形介面操作四階段角點定位管線：
  - 載入 YOLO 權重（.pt）
  - 載入單張影像，或整個資料夾（可在清單間切換瀏覽）
  - 執行管線，將角點資訊（cid / 信心值）畫在影像上
  - 可縮放檢視、角點表格、批次處理整個資料夾並存檔

需求：PyQt6、numpy、opencv-python、scipy，以及 ultralytics（第一階段 YOLO）。
本程式需與 court_corner 套件同目錄（或套件可被 import）。

執行：
    python court_corner_gui.py
"""

import os
import sys
import json
import glob
import traceback

import numpy as np
import cv2

# 確保可 import 同目錄下的 court_corner 套件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPainter, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QDoubleSpinBox, QCheckBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QPlainTextEdit, QSplitter,
    QGroupBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QProgressBar,
    QMessageBox, QHeaderView, QSizePolicy,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ════════════════════════════════════════════════════════════════
#  繪圖：把管線結果畫到影像上（worker 批次存檔與主視窗顯示共用）
# ════════════════════════════════════════════════════════════════
def conf_color_bgr(conf: float):
    """依信心值給顏色（BGR）。"""
    if conf >= 0.8:
        return (0, 200, 0)        # 綠
    if conf >= 0.65:
        return (0, 210, 210)      # 黃
    return (0, 140, 255)          # 橙


def annotate_image(img_bgr, result, opts: dict):
    """
    將結果畫在影像副本上並回傳。
    opts: { 'grid':bool, 'detections':bool, 'corners':bool, 'labels':bool, 'conf':bool }
    """
    vis = img_bgr.copy()
    H = getattr(result, "H", None)

    # H 投影球場格線（淡藍）
    if opts.get("grid", True) and H is not None:
        try:
            from court_corner.stages.topology import _proj, _tpl_xy, N_ROW, N_COL

            def Pt(r, c):
                x, y = _proj(H, _tpl_xy(r, c))
                return (int(round(x)), int(round(y)))

            for r in range(N_ROW):
                for c in range(N_COL):
                    if c + 1 < N_COL:
                        cv2.line(vis, Pt(r, c), Pt(r, c + 1), (255, 170, 60), 1, cv2.LINE_AA)
                    if r + 1 < N_ROW:
                        cv2.line(vis, Pt(r, c), Pt(r + 1, c), (255, 170, 60), 1, cv2.LINE_AA)
        except Exception:
            pass

    # 偵測交點（黃點）
    if opts.get("detections", False) and getattr(result, "detection", None) is not None:
        for (x, y) in result.detection.node_pts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 220, 220), -1, cv2.LINE_AA)

    # 最終角點（依信心值上色）+ 標籤
    if opts.get("corners", True):
        for c in getattr(result, "corners", []):
            pt = (int(round(c.x)), int(round(c.y)))
            col = conf_color_bgr(c.conf)
            cv2.circle(vis, pt, 4, col, -1, cv2.LINE_AA)
            cv2.circle(vis, pt, 4, (30, 30, 30), 1, cv2.LINE_AA)
            label = ""
            if opts.get("labels", True):
                label = str(c.cid)
            if opts.get("conf", False):
                label = (label + " " if label else "") + f"{c.conf:.2f}"
            if label:
                cv2.putText(vis, label, (pt[0] + 6, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(vis, label, (pt[0] + 6, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return vis


# ════════════════════════════════════════════════════════════════
#  可縮放影像檢視器
# ════════════════════════════════════════════════════════════════
class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform |
                            QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self._has_img = False

    def set_image_bgr(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        first = not self._has_img
        self._item.setPixmap(QPixmap.fromImage(qimg))
        self._scene.setSceneRect(0, 0, w, h)
        self._has_img = True
        if first:
            self.fit()

    def fit(self):
        if not self._has_img:
            return
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, e):
        if not self._has_img:
            return
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


# ════════════════════════════════════════════════════════════════
#  背景工作執行緒：持有 pipeline（模型只載入一次），處理單張 / 批次
# ════════════════════════════════════════════════════════════════
class Worker(QObject):
    sig_status = pyqtSignal(str)
    sig_error = pyqtSignal(str)
    sig_single_done = pyqtSignal(object, object, object)      # path, img_bgr, result
    sig_result_only = pyqtSignal(object, object)             # path, result（批次用，不帶影像）
    sig_batch_progress = pyqtSignal(int, int, str)           # done, total, path
    sig_batch_done = pyqtSignal(str, int)                    # out_dir, count

    def __init__(self):
        super().__init__()
        self._pipeline = None
        self._weights = None

    # ----------------------------------------------------------------
    def _ensure_pipeline(self, weights, yolo_conf, corner_conf):
        """需要時建立 / 重建 pipeline；權重不變則沿用（模型不重載），僅更新參數。"""
        from court_corner.pipeline import CourtCornerPipeline
        if self._pipeline is None or self._weights != weights:
            self.sig_status.emit(f"載入權重並初始化管線：{os.path.basename(weights)} …")
            self._pipeline = CourtCornerPipeline(
                yolo_weight=weights, yolo_conf=yolo_conf,
                corner_conf=corner_conf, verbose=False)
            self._weights = weights
        else:
            # 僅更新參數，不重載模型
            self._pipeline.detector.conf = float(yolo_conf)
            self._pipeline.evaluator.corner_conf = float(corner_conf)
        return self._pipeline

    # ----------------------------------------------------------------
    @pyqtSlot(str, str, float, float)
    def run_single(self, path, weights, yolo_conf, corner_conf):
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                self.sig_error.emit(f"無法讀取影像：{path}")
                return
            pipe = self._ensure_pipeline(weights, yolo_conf, corner_conf)
            self.sig_status.emit(f"執行中：{os.path.basename(path)} …")
            result = pipe.run_image(img)
            self.sig_single_done.emit(path, img, result)
            self.sig_status.emit(_result_summary(path, result))
        except ImportError as e:
            self.sig_error.emit(
                "缺少 ultralytics 套件，無法執行第一階段 YOLO 偵測。\n"
                "請先安裝：pip install ultralytics\n\n" + str(e))
        except Exception as e:
            self.sig_error.emit("執行發生例外：\n" + "".join(
                traceback.format_exception(type(e), e, e.__traceback__)))

    # ----------------------------------------------------------------
    @pyqtSlot(list, str, str, float, float, dict)
    def run_batch(self, paths, out_dir, weights, yolo_conf, corner_conf, opts):
        try:
            pipe = self._ensure_pipeline(weights, yolo_conf, corner_conf)
            os.makedirs(out_dir, exist_ok=True)
            n = len(paths)
            ok = 0
            for i, path in enumerate(paths, 1):
                self.sig_batch_progress.emit(i, n, path)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                try:
                    result = pipe.run_image(img)
                except Exception as e:
                    self.sig_status.emit(f"  跳過 {os.path.basename(path)}：{e}")
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                vis = annotate_image(img, result, opts)
                cv2.imwrite(os.path.join(out_dir, stem + "_annotated.png"), vis)
                with open(os.path.join(out_dir, stem + "_corners.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                self.sig_result_only.emit(path, result)
                ok += 1
            self.sig_batch_done.emit(out_dir, ok)
        except ImportError as e:
            self.sig_error.emit(
                "缺少 ultralytics 套件，無法執行第一階段 YOLO 偵測。\n"
                "請先安裝：pip install ultralytics\n\n" + str(e))
        except Exception as e:
            self.sig_error.emit("批次執行發生例外：\n" + "".join(
                traceback.format_exception(type(e), e, e.__traceback__)))


def _result_summary(path, result):
    name = os.path.basename(path)
    if getattr(result, "status", "") == "ok":
        rep = result.report or {}
        return (f"{name}：輸出 {len(result.corners)} 角點"
                f"（候選 {rep.get('n_candidates', '?')}，conf≥{rep.get('corner_conf', '?')}）"
                f"，拓樸 {result.topology.confidence if result.topology else '?'}")
    return f"{name}：{getattr(result, 'message', '失敗')}"


# ════════════════════════════════════════════════════════════════
#  主視窗
# ════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    # 主執行緒 → worker
    sig_req_single = pyqtSignal(str, str, float, float)
    sig_req_batch = pyqtSignal(list, str, str, float, float, dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("羽球場角點定位 GUI")
        self.resize(1320, 820)

        self.weights_path = None
        self.image_paths = []           # 目前清單的影像路徑
        self.results = {}               # path -> PipelineResult（快取）
        self.cur_path = None
        self.cur_img = None             # 目前影像的原始 BGR（用於即時重繪）
        self._busy = False

        self._build_ui()
        self._start_worker()
        self._log("就緒。請先載入權重（.pt）與影像或資料夾。")

    # ----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # ---- 控制列 ----
        ctrl = QGroupBox("控制")
        cl = QGridLayout(ctrl)

        self.btn_weights = QPushButton("載入權重 (.pt)")
        self.btn_weights.clicked.connect(self.on_load_weights)
        self.lbl_weights = QLabel("（未載入）")
        self.lbl_weights.setStyleSheet("color:#555;")

        self.btn_image = QPushButton("載入影像")
        self.btn_image.clicked.connect(self.on_load_image)
        self.btn_folder = QPushButton("載入資料夾")
        self.btn_folder.clicked.connect(self.on_load_folder)

        self.spin_yolo = QDoubleSpinBox()
        self.spin_yolo.setRange(0.01, 1.0); self.spin_yolo.setSingleStep(0.05)
        self.spin_yolo.setValue(0.25); self.spin_yolo.setDecimals(2)
        self.spin_yolo.valueChanged.connect(self.on_param_changed)

        self.spin_corner = QDoubleSpinBox()
        self.spin_corner.setRange(0.0, 1.0); self.spin_corner.setSingleStep(0.05)
        self.spin_corner.setValue(0.60); self.spin_corner.setDecimals(2)
        self.spin_corner.valueChanged.connect(self.on_param_changed)

        self.btn_run = QPushButton("執行（目前影像）")
        self.btn_run.clicked.connect(self.on_run_current)
        self.btn_batch = QPushButton("批次處理資料夾…")
        self.btn_batch.clicked.connect(self.on_run_batch)

        self.btn_save_img = QPushButton("儲存標註圖")
        self.btn_save_img.clicked.connect(self.on_save_image)
        self.btn_save_json = QPushButton("儲存 JSON")
        self.btn_save_json.clicked.connect(self.on_save_json)

        cl.addWidget(self.btn_weights, 0, 0)
        cl.addWidget(self.lbl_weights, 0, 1, 1, 3)
        cl.addWidget(self.btn_image, 0, 4)
        cl.addWidget(self.btn_folder, 0, 5)

        cl.addWidget(QLabel("yolo_conf"), 1, 0)
        cl.addWidget(self.spin_yolo, 1, 1)
        cl.addWidget(QLabel("corner_conf"), 1, 2)
        cl.addWidget(self.spin_corner, 1, 3)
        cl.addWidget(self.btn_run, 1, 4)
        cl.addWidget(self.btn_batch, 1, 5)

        # 顯示選項
        self.chk_grid = QCheckBox("格線"); self.chk_grid.setChecked(True)
        self.chk_det = QCheckBox("偵測交點"); self.chk_det.setChecked(False)
        self.chk_corner = QCheckBox("角點"); self.chk_corner.setChecked(True)
        self.chk_label = QCheckBox("cid 標籤"); self.chk_label.setChecked(True)
        self.chk_conf = QCheckBox("信心值"); self.chk_conf.setChecked(False)
        self.chk_auto = QCheckBox("選取後自動執行"); self.chk_auto.setChecked(True)
        for chk in (self.chk_grid, self.chk_det, self.chk_corner, self.chk_label, self.chk_conf):
            chk.stateChanged.connect(self.on_display_opts_changed)
        opt_row = QHBoxLayout()
        for w in (QLabel("顯示："), self.chk_grid, self.chk_det, self.chk_corner,
                  self.chk_label, self.chk_conf, self.chk_auto,
                  self.btn_save_img, self.btn_save_json):
            opt_row.addWidget(w)
        opt_row.addStretch(1)
        cl.addLayout(opt_row, 2, 0, 1, 6)

        outer.addWidget(ctrl)

        # ---- 主區：左清單 | 中影像 | 右(表格+log) ----
        split = QSplitter(Qt.Orientation.Horizontal)

        # 左：影像清單
        left = QWidget(); ll = QVBoxLayout(left)
        ll.addWidget(QLabel("影像清單"))
        self.list_imgs = QListWidget()
        self.list_imgs.currentRowChanged.connect(self.on_list_selected)
        ll.addWidget(self.list_imgs)
        left.setMaximumWidth(280)
        split.addWidget(left)

        # 中：影像檢視
        self.viewer = ImageViewer()
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        split.addWidget(self.viewer)

        # 右：角點表格 + log
        right = QWidget(); rl = QVBoxLayout(right)
        btn_fit = QPushButton("符合視窗")
        btn_fit.clicked.connect(self.viewer.fit)
        rl.addWidget(btn_fit)
        rl.addWidget(QLabel("角點 (cid, x, y, conf, type, source)"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["cid", "x", "y", "conf", "type", "source"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_table_selected)
        rl.addWidget(self.table, 3)
        rl.addWidget(QLabel("訊息"))
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(500)
        rl.addWidget(self.log, 1)
        right.setMaximumWidth(420)
        split.addWidget(right)

        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)

        # ---- 底部：進度 + 狀態 ----
        bottom = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setVisible(False)
        self.lbl_status = QLabel("就緒")
        bottom.addWidget(self.lbl_status, 1)
        bottom.addWidget(self.progress)
        outer.addLayout(bottom)

    # ----------------------------------------------------------------
    def _start_worker(self):
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.sig_req_single.connect(self.worker.run_single)
        self.sig_req_batch.connect(self.worker.run_batch)
        self.worker.sig_status.connect(self.on_status)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_single_done.connect(self.on_single_done)
        self.worker.sig_result_only.connect(self.on_result_only)
        self.worker.sig_batch_progress.connect(self.on_batch_progress)
        self.worker.sig_batch_done.connect(self.on_batch_done)
        self.thread.start()

    # ================= 載入 =================
    def on_load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 YOLO 權重", "", "PyTorch 權重 (*.pt)")
        if not path:
            return
        self.weights_path = path
        self.lbl_weights.setText(path)
        self.results.clear()            # 權重變更 → 快取失效
        self._log(f"已選擇權重：{path}")

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影像", "", "影像 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        self._set_image_list([path])

    def on_load_folder(self):
        d = QFileDialog.getExistingDirectory(self, "選擇影像資料夾")
        if not d:
            return
        paths = []
        for ext in IMG_EXTS:
            paths += glob.glob(os.path.join(d, "*" + ext))
            paths += glob.glob(os.path.join(d, "*" + ext.upper()))
        paths = sorted(set(paths))
        if not paths:
            QMessageBox.information(self, "無影像", "此資料夾沒有支援的影像檔。")
            return
        self._set_image_list(paths)
        self._log(f"載入資料夾：{d}（{len(paths)} 張影像）")

    def _set_image_list(self, paths):
        self.image_paths = paths
        self.results.clear()
        self.list_imgs.blockSignals(True)
        self.list_imgs.clear()
        for p in paths:
            self.list_imgs.addItem(QListWidgetItem(os.path.basename(p)))
        self.list_imgs.blockSignals(False)
        if paths:
            self.list_imgs.setCurrentRow(0)   # 觸發 on_list_selected

    # ================= 清單選取 =================
    def on_list_selected(self, row):
        if row < 0 or row >= len(self.image_paths):
            return
        path = self.image_paths[row]
        self.cur_path = path
        self.cur_img = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.cur_img is None:
            self._log(f"無法讀取：{path}")
            return
        if path in self.results:
            self._render_current()
        else:
            self.viewer.set_image_bgr(self.cur_img)   # 先顯示原圖
            self._fill_table(None)
            if self.chk_auto.isChecked() and self.weights_path and not self._busy:
                self.on_run_current()

    # ================= 執行 =================
    def _opts(self):
        return {
            "grid": self.chk_grid.isChecked(),
            "detections": self.chk_det.isChecked(),
            "corners": self.chk_corner.isChecked(),
            "labels": self.chk_label.isChecked(),
            "conf": self.chk_conf.isChecked(),
        }

    def _check_ready(self):
        if not self.weights_path:
            QMessageBox.warning(self, "尚未載入權重", "請先載入 YOLO 權重（.pt）。")
            return False
        if not self.cur_path:
            QMessageBox.warning(self, "尚未載入影像", "請先載入影像或資料夾。")
            return False
        return True

    def on_run_current(self):
        if self._busy or not self._check_ready():
            return
        self._set_busy(True)
        self.sig_req_single.emit(self.cur_path, self.weights_path,
                                 self.spin_yolo.value(), self.spin_corner.value())

    def on_run_batch(self):
        if self._busy:
            return
        if not self.weights_path:
            QMessageBox.warning(self, "尚未載入權重", "請先載入 YOLO 權重（.pt）。")
            return
        if not self.image_paths:
            QMessageBox.warning(self, "尚未載入影像", "請先載入資料夾。")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾（標註圖與 JSON）")
        if not out_dir:
            return
        self._set_busy(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(self.image_paths))
        self.progress.setValue(0)
        self.sig_req_batch.emit(list(self.image_paths), out_dir, self.weights_path,
                                self.spin_yolo.value(), self.spin_corner.value(), self._opts())

    def on_param_changed(self, _):
        # 參數變更 → 快取失效（重新執行才會套用新參數）
        self.results.clear()

    def on_display_opts_changed(self, _):
        # 僅重繪，不重跑
        if self.cur_path in self.results:
            self._render_current()

    # ================= worker 回呼 =================
    @pyqtSlot(str)
    def on_status(self, msg):
        self.lbl_status.setText(msg)
        self._log(msg)

    @pyqtSlot(str)
    def on_error(self, msg):
        self._set_busy(False)
        self.progress.setVisible(False)
        self._log("[錯誤] " + msg)
        QMessageBox.critical(self, "錯誤", msg)

    @pyqtSlot(object, object, object)
    def on_single_done(self, path, img, result):
        self.results[path] = result
        if path == self.cur_path:
            self.cur_img = img
            self._render_current()
        self._set_busy(False)

    @pyqtSlot(object, object)
    def on_result_only(self, path, result):
        self.results[path] = result

    @pyqtSlot(int, int, str)
    def on_batch_progress(self, done, total, path):
        self.progress.setValue(done)
        self.lbl_status.setText(f"批次處理 {done}/{total}：{os.path.basename(path)}")

    @pyqtSlot(str, int)
    def on_batch_done(self, out_dir, count):
        self._set_busy(False)
        self.progress.setVisible(False)
        self._log(f"批次完成：{count} 張，輸出於 {out_dir}")
        QMessageBox.information(self, "批次完成",
                                f"已處理 {count} 張影像。\n標註圖與 JSON 已存至：\n{out_dir}")
        if self.cur_path in self.results:
            self._render_current()

    # ================= 繪製 / 表格 =================
    def _render_current(self):
        if self.cur_img is None:
            return
        result = self.results.get(self.cur_path)
        if result is None:
            self.viewer.set_image_bgr(self.cur_img)
            self._fill_table(None)
            return
        vis = annotate_image(self.cur_img, result, self._opts())
        self.viewer.set_image_bgr(vis)
        self._fill_table(result)

    def _fill_table(self, result):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if result is not None:
            for c in getattr(result, "corners", []):
                r = self.table.rowCount()
                self.table.insertRow(r)
                vals = [str(c.cid), f"{c.x:.1f}", f"{c.y:.1f}", f"{c.conf:.3f}",
                        c.corner_type, c.source]
                for j, v in enumerate(vals):
                    self.table.setItem(r, j, QTableWidgetItem(v))
        self.table.blockSignals(False)

    def on_table_selected(self):
        # 點表格列 → 在影像上以較大圈標出該角點
        items = self.table.selectedItems()
        if not items or self.cur_img is None:
            return
        result = self.results.get(self.cur_path)
        if result is None:
            return
        row = items[0].row()
        if row >= len(result.corners):
            return
        c = result.corners[row]
        vis = annotate_image(self.cur_img, result, self._opts())
        cv2.circle(vis, (int(round(c.x)), int(round(c.y))), 11, (0, 0, 255), 2, cv2.LINE_AA)
        self.viewer.set_image_bgr(vis)

    # ================= 儲存 =================
    def on_save_image(self):
        if self.cur_img is None or self.cur_path not in self.results:
            QMessageBox.information(self, "無結果", "目前影像尚無結果可儲存。請先執行。")
            return
        stem = os.path.splitext(os.path.basename(self.cur_path))[0]
        path, _ = QFileDialog.getSaveFileName(self, "儲存標註圖",
                                              stem + "_annotated.png", "PNG (*.png)")
        if not path:
            return
        vis = annotate_image(self.cur_img, self.results[self.cur_path], self._opts())
        cv2.imwrite(path, vis)
        self._log(f"已儲存標註圖：{path}")

    def on_save_json(self):
        if self.cur_path not in self.results:
            QMessageBox.information(self, "無結果", "目前影像尚無結果可儲存。請先執行。")
            return
        stem = os.path.splitext(os.path.basename(self.cur_path))[0]
        path, _ = QFileDialog.getSaveFileName(self, "儲存 JSON",
                                              stem + "_corners.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results[self.cur_path].to_dict(), f, ensure_ascii=False, indent=2)
        self._log(f"已儲存 JSON：{path}")

    # ================= 雜項 =================
    def _set_busy(self, busy):
        self._busy = busy
        for w in (self.btn_run, self.btn_batch, self.btn_weights,
                  self.btn_image, self.btn_folder):
            w.setEnabled(not busy)
        self.lbl_status.setText("處理中…" if busy else "就緒")

    def _log(self, msg):
        self.log.appendPlainText(str(msg))

    def closeEvent(self, e):
        try:
            self.thread.quit()
            self.thread.wait(2000)
        except Exception:
            pass
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
