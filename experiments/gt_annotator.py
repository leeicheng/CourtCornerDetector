#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gt_annotator.py — 羽球場角點 GT 標註工具 (PyQt6)

輸出格式（與 experiments/common.load_gt 完全相容）:
    <stem>.gt.json
    {
      "image": "xxx.jpg",
      "scene": "single" | "multi",
      "corners": [
        {"cid": int, "x": float, "y": float, "visibility": "visible"|"occluded",
         "node": [nx, ny], "lcid": "NW|NE|SW|SE", "manual": bool}
      ],
      "annot": {  // 工具自用，供續標
        "correspondences": [{"junction": j_idx, "x": px, "y": px}],
        "H": [[...3x3...]] | null
      }
    }

標註流程（H 蓋章）:
  1. 開資料夾 → 選影像
  2. 「對應點模式」: 在右側小地圖點一個 junction，再到影像上點對應位置
     （重複 ≥4 組，建議取分散的外圍交點）
  3. 按「解 H 並蓋章」→ 自動投影全部 80 個物理角點（依 NODE_TABLE 有效角遮罩）
  4. 切到「編輯模式」微調: 拖曳 / 方向鍵 0.25px 微調（Shift=1px）
     V 切換可見性、Delete 刪除、雙擊空白處於選定 cid 補點
  5. Ctrl+S 存檔（或開啟自動存檔，切換影像時自動寫出）

座標說明:
  cid = corner_code = (ny<<5)|(nx<<2)|lcid，物理角點 = junction ± 0.02m（線寬一半）
  與管線輸出的 cid 完全一致，可直接被 run_main_eval.py / match_by_cid 使用。
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import (QAction, QBrush, QColor, QFont, QImage, QKeySequence,
                         QPainter, QPen, QPixmap, QShortcut)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QGraphicsEllipseItem,
                             QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
                             QHBoxLayout, QHeaderView, QLabel, QListWidget,
                             QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                             QRadioButton, QSplitter, QStatusBar, QTableWidget,
                             QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
                             QButtonGroup, QSizePolicy)

# ---- court_corner 模型 ----
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from court_corner.shared.court_model import (  # noqa: E402
    TEMPLATE_POINTS, GRID_CONNECTIONS, NODE_TABLE, LINE_WIDTH_M,
    encode_corner, nx_ny_to_junction_idx, junction_idx_to_nx_ny,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
HALF_W = LINE_WIDTH_M / 2.0  # 0.02 m
LCID_NAMES = {0: "NW", 1: "NE", 2: "SW", 3: "SE"}

# 顏色
COL_VISIBLE = QColor(0, 200, 80)
COL_OCCLUDED = QColor(250, 170, 0)
COL_MANUAL_RING = QColor(255, 255, 255)
COL_SELECT = QColor(70, 150, 255)
COL_CORR = QColor(255, 60, 200)
COL_GRID = QColor(120, 200, 255, 110)


# =====================================================================
# 模板角點枚舉
# =====================================================================

def enumerate_valid_corners():
    """回傳 [{cid, junction_idx, nx, ny, lcid, world(tpl m, np2)}] 共 80 筆。"""
    out = []
    for (nx, ny), info in sorted(NODE_TABLE.items()):
        j = nx_ny_to_junction_idx(nx, ny)
        p0 = TEMPLATE_POINTS[j].astype(np.float64)
        for lcid in range(4):
            if not (info["valid_corner_mask"] >> lcid) & 1:
                continue
            # bit1: N(0)/S(1) → N = template y 變大；bit0: W(0)/E(1) → E = template x 變大
            dy = +HALF_W if (lcid >> 1) == 0 else -HALF_W
            dx = +HALF_W if (lcid & 1) == 1 else -HALF_W
            out.append({
                "cid": encode_corner(nx, ny, lcid),
                "junction_idx": j, "nx": nx, "ny": ny, "lcid": lcid,
                "world": np.array([p0[0] + dx, p0[1] + dy], dtype=np.float64),
            })
    return out


ALL_CORNERS = enumerate_valid_corners()
CORNER_BY_CID = {c["cid"]: c for c in ALL_CORNERS}


T180 = np.array([[-1.0, 0.0, 6.10], [0.0, -1.0, 13.40], [0.0, 0.0, 1.0]])


def rot180_cid(cid):
    c = CORNER_BY_CID.get(int(cid))
    if c is None:
        return int(cid)
    return encode_corner(4 - c["nx"], 6 - c["ny"], 3 - c["lcid"])


def row0_is_near(H):
    """以 5 條縱線的深度消失點判斷 row0（y=13.40）端在影像中是近或遠。
    回傳 True（近）/ False（遠）/ None（無法判定：近頂視或退化）。"""
    H = np.asarray(H, np.float64)

    def proj(x, y):
        v = H @ np.array([x, y, 1.0])
        return None if abs(v[2]) < 1e-12 else np.array([v[0] / v[2], v[1] / v[2]])

    A, b = [], []
    for cx in (0.0, 0.46, 3.05, 5.64, 6.10):
        p, q = proj(cx, 13.40), proj(cx, 0.0)
        if p is None or q is None:
            continue
        d = q - p
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        n = np.array([-d[1] / L, d[0] / L])
        A.append(n); b.append(float(n @ p))
    if len(A) < 2:
        return None
    A = np.asarray(A); b = np.asarray(b)
    try:
        vp, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 2:
        return None
    r0, r5 = proj(3.05, 13.40), proj(3.05, 0.0)
    if r0 is None or r5 is None:
        return None
    d0 = float(np.hypot(*(r0 - vp))); d5 = float(np.hypot(*(r5 - vp)))
    if max(d0, d5) <= 0 or abs(d0 - d5) < 0.02 * max(d0, d5):
        return None   # 近頂視等：方向本質上不明確，不強行翻
    return d0 > d5


def cid_label(cid):
    c = CORNER_BY_CID.get(cid)
    if c is None:
        return str(cid)
    return f'({c["nx"]},{c["ny"]}){LCID_NAMES[c["lcid"]]}'


# =====================================================================
# 球場小地圖（junction 選擇器）
# =====================================================================

class CourtMiniMap(QWidget):
    junctionClicked = pyqtSignal(int)

    PAD = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = -1          # 目前選定 junction
        self.done = set()           # 已有對應點的 junction
        self.setMinimumSize(190, 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # template (m) → widget px。預設標準視圖：row0（J0–J4）在上、J0 左上，
    # 上排由左至右為 0 1 2 3 4；rotated=True 時為 180° 旋轉視圖。
    # 兩種視圖皆為合法掌性（旋轉非鏡像），對應方向不影響存檔結果：
    # 存檔時會以消失點自動規範化為 row0=近端（見 normalize_orientation）。
    def _scale(self):
        w = self.width() - 2 * self.PAD
        h = self.height() - 2 * self.PAD
        s = min(w / 6.10, h / 13.40)
        ox = (self.width() - 6.10 * s) / 2
        oy = (self.height() - 13.40 * s) / 2
        return s, ox, oy

    def _to_px(self, p):
        s, ox, oy = self._scale()
        if getattr(self, "rotated", False):
            return QPointF(ox + (6.10 - p[0]) * s, oy + p[1] * s)
        return QPointF(ox + p[0] * s, oy + (13.40 - p[1]) * s)

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor(30, 70, 45))
        qp.setPen(QPen(QColor(230, 230, 230), 2))
        for i1, i2 in GRID_CONNECTIONS:
            a = self._to_px(TEMPLATE_POINTS[i1])
            b = self._to_px(TEMPLATE_POINTS[i2])
            qp.drawLine(a, b)
        for j in range(30):
            c = self._to_px(TEMPLATE_POINTS[j])
            r = 6.0
            if j == self.selected:
                qp.setBrush(QBrush(COL_SELECT)); qp.setPen(QPen(Qt.GlobalColor.white, 2)); r = 8.0
            elif j in self.done:
                qp.setBrush(QBrush(COL_CORR)); qp.setPen(QPen(Qt.GlobalColor.white, 1))
            else:
                qp.setBrush(QBrush(QColor(245, 245, 245))); qp.setPen(QPen(QColor(60, 60, 60), 1))
            qp.drawEllipse(c, r, r)
        # 四角 junction 編號（位置依目前視圖自動調整）
        qp.setPen(QColor(255, 235, 130))
        f = qp.font(); f.setPointSizeF(8.5); f.setBold(True); qp.setFont(f)
        for j in (0, 4, 25, 29):
            c = self._to_px(TEMPLATE_POINTS[j])
            dx = 8 if c.x() < self.width() / 2 else -26
            dy = -8 if c.y() > self.height() / 2 else 16
            qp.drawText(QPointF(c.x() + dx, c.y() + dy), f"J{j}")
        qp.setPen(QColor(220, 220, 220))
        f.setBold(False); f.setPointSizeF(7.5); qp.setFont(f)
        qp.drawText(6, self.height() - 18, "對應方向自由：存檔時自動規範化 row0=近端")
        qp.drawText(6, self.height() - 6, "粉=已有對應點  藍=待點影像")
        qp.end()

    def mousePressEvent(self, ev):
        pos = ev.position()
        best, best_d = -1, 1e9
        for j in range(30):
            c = self._to_px(TEMPLATE_POINTS[j])
            d = math.hypot(c.x() - pos.x(), c.y() - pos.y())
            if d < best_d:
                best, best_d = j, d
        if best_d <= 14:
            self.selected = best
            self.junctionClicked.emit(best)
            self.update()


# =====================================================================
# 影像場景中的圖元
# =====================================================================

class PointItem(QGraphicsEllipseItem):
    """GT 角點：可拖曳，圓心 = 次像素座標。"""
    R = 5.0

    def __init__(self, cid, x, y, visibility="visible", manual=False, win=None):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.cid = int(cid)
        self.visibility = visibility
        self.manual = bool(manual)
        self.win = win
        self.setPos(QPointF(x, y))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(10)
        self._label = QGraphicsSimpleTextItem(cid_label(self.cid), self)
        self._label.setPos(7, -16)
        f = QFont(); f.setPointSizeF(8.0)
        self._label.setFont(f)
        self._restyle()

    def set_cid(self, cid):
        self.cid = int(cid)
        self._label.setText(cid_label(self.cid))

    def _restyle(self):
        base = COL_VISIBLE if self.visibility == "visible" else COL_OCCLUDED
        pen = QPen(COL_SELECT if self.isSelected() else base, 2)
        self.setPen(pen)
        br = QColor(base); br.setAlpha(70)
        self.setBrush(QBrush(br))
        self._label.setBrush(QBrush(COL_MANUAL_RING if self.manual else base))
        self._label.setVisible(self.win.show_labels if self.win else True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.manual = True
            if self.win:
                self.win.on_point_moved(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._restyle()
            if self.win and bool(value):
                self.win.on_point_selected(self)
        return super().itemChange(change, value)

    def paint(self, qp, opt, w=None):
        super().paint(qp, opt, w)
        qp.setPen(QPen(self.pen().color(), 1))
        qp.drawLine(QPointF(-self.R - 3, 0), QPointF(self.R + 3, 0))
        qp.drawLine(QPointF(0, -self.R - 3), QPointF(0, self.R + 3))
        if self.manual:
            qp.setPen(QPen(COL_MANUAL_RING, 1))
            qp.drawEllipse(QRectF(-self.R - 2, -self.R - 2, 2 * self.R + 4, 2 * self.R + 4))


class CorrItem(QGraphicsEllipseItem):
    """對應點標記（junction ↔ 影像座標），可拖曳。"""
    R = 7.0

    def __init__(self, junction_idx, x, y, win=None):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.junction_idx = int(junction_idx)
        self.win = win
        self.setPos(QPointF(x, y))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(12)
        self.setPen(QPen(COL_CORR, 2))
        self._label = QGraphicsSimpleTextItem(f"J{self.junction_idx}", self)
        self._label.setBrush(QBrush(COL_CORR)); self._label.setPos(8, -18)
        f = QFont(); f.setPointSizeF(8.0); self._label.setFont(f)

    def set_junction(self, j):
        self.junction_idx = int(j)
        self._label.setText(f"J{self.junction_idx}")

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.win:
            self.win.mark_dirty()
        return super().itemChange(change, value)

    def paint(self, qp, opt, w=None):
        qp.setPen(self.pen())
        qp.drawLine(QPointF(-self.R, -self.R), QPointF(self.R, self.R))
        qp.drawLine(QPointF(-self.R, self.R), QPointF(self.R, -self.R))
        qp.drawEllipse(self.rect())


# =====================================================================
# 影像檢視器
# =====================================================================

MODE_EDIT, MODE_CORR = 0, 1


class ImageView(QGraphicsView):
    clickedAt = pyqtSignal(float, float)      # 場景座標（對應點模式下的點擊）
    mouseAt = pyqtSignal(float, float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.mode = MODE_EDIT
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self._panning = False
        self._last = None

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton or \
           (ev.button() == Qt.MouseButton.LeftButton and
                ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._panning = True
            self._last = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self.mode == MODE_CORR and ev.button() == Qt.MouseButton.LeftButton:
            # 點在既有圖元上則讓其拖曳；點空白則新增對應點
            if self.itemAt(ev.position().toPoint()) is None or \
               isinstance(self.itemAt(ev.position().toPoint()), QGraphicsPixmapItem):
                p = self.mapToScene(ev.position().toPoint())
                self.clickedAt.emit(p.x(), p.y())
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning:
            d = ev.position() - self._last
            self._last = ev.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        p = self.mapToScene(ev.position().toPoint())
        self.mouseAt.emit(p.x(), p.y())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(ev)


# =====================================================================
# 主視窗
# =====================================================================

class GTAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("羽球場角點 GT 標註工具")
        self.resize(1480, 920)

        self.img_dir = None
        self.images = []
        self.cur_idx = -1
        self.cur_img_path = None
        self.img_w = self.img_h = 0
        self.H = None
        self.dirty = False
        self.show_labels = True
        self.show_grid = True
        self.points = {}      # cid → PointItem
        self.corrs = []       # CorrItem list
        self.grid_items = []
        self._loading = False

        # --- 場景與檢視器 ---
        self.scene = QGraphicsScene(self)
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setZValue(0)
        self.scene.addItem(self.pix_item)
        self.view = ImageView(self.scene)
        self.view.clickedAt.connect(self.on_image_clicked)
        self.view.mouseAt.connect(self.on_mouse_at)

        # --- 左側：檔案清單 ---
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(4, 4, 4, 4)
        self.btn_open = QPushButton("開啟資料夾…")
        self.btn_open.clicked.connect(self.open_folder)
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_select_image)
        self.chk_autosave = QCheckBox("切換影像時自動存檔"); self.chk_autosave.setChecked(True)
        lv.addWidget(self.btn_open); lv.addWidget(self.file_list, 1); lv.addWidget(self.chk_autosave)

        # --- 右側：小地圖 + 控制 + 點表 ---
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(4, 4, 4, 4)
        self.minimap = CourtMiniMap()
        self.minimap.junctionClicked.connect(self.on_junction_picked)
        rv.addWidget(QLabel("① 小地圖選 junction → ② 點影像對應位置"))
        rv.addWidget(self.minimap, 2)
        self.btn_rot_map = QPushButton("旋轉小地圖 180°（僅顯示，不影響存檔）")
        def _rot():
            self.minimap.rotated = not getattr(self.minimap, "rotated", False)
            self.minimap.update()
        self.btn_rot_map.clicked.connect(_rot)
        rv.addWidget(self.btn_rot_map)

        hb = QHBoxLayout()
        self.btn_solve = QPushButton("解 H 並蓋章 (≥4 組)")
        self.btn_solve.clicked.connect(self.solve_and_stamp)
        self.btn_clear_corr = QPushButton("清空對應點")
        self.btn_clear_corr.clicked.connect(self.clear_corrs)
        hb.addWidget(self.btn_solve); hb.addWidget(self.btn_clear_corr)
        rv.addLayout(hb)

        hb2 = QHBoxLayout()
        self.rb_single = QRadioButton("single"); self.rb_multi = QRadioButton("multi")
        self.rb_single.setChecked(True)
        self.scene_group = QButtonGroup(self)
        self.scene_group.addButton(self.rb_single); self.scene_group.addButton(self.rb_multi)
        self.rb_single.toggled.connect(lambda *_: self.mark_dirty())
        hb2.addWidget(QLabel("場景:")); hb2.addWidget(self.rb_single); hb2.addWidget(self.rb_multi)
        hb2.addStretch(1)
        rv.addLayout(hb2)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["cid", "位置", "x", "y", "可見"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self.on_table_clicked)
        rv.addWidget(self.table, 3)
        self.stats_lbl = QLabel("—")
        rv.addWidget(self.stats_lbl)

        split = QSplitter()
        split.addWidget(left); split.addWidget(self.view); split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setStretchFactor(2, 0)
        split.setSizes([230, 900, 330])
        self.setCentralWidget(split)

        # --- 工具列 ---
        tb = QToolBar("tools"); self.addToolBar(tb)
        self.act_edit = QAction("編輯模式 (E)", self, checkable=True, checked=True)
        self.act_corr = QAction("對應點模式 (C)", self, checkable=True)
        self.act_edit.triggered.connect(lambda: self.set_mode(MODE_EDIT))
        self.act_corr.triggered.connect(lambda: self.set_mode(MODE_CORR))
        tb.addAction(self.act_edit); tb.addAction(self.act_corr); tb.addSeparator()
        a_save = QAction("存檔 (Ctrl+S)", self); a_save.triggered.connect(self.save_gt)
        tb.addAction(a_save)
        a_restamp = QAction("重蓋非手動點", self); a_restamp.triggered.connect(self.restamp_auto_points)
        tb.addAction(a_restamp); tb.addSeparator()
        self.act_labels = QAction("顯示標籤", self, checkable=True, checked=True)
        self.act_labels.triggered.connect(self.toggle_labels)
        self.act_grid = QAction("顯示格線", self, checkable=True, checked=True)
        self.act_grid.triggered.connect(self.toggle_grid)
        tb.addAction(self.act_labels); tb.addAction(self.act_grid)
        a_fit = QAction("適合視窗 (F)", self); a_fit.triggered.connect(self.fit_view)
        tb.addAction(a_fit)

        # --- 快捷鍵 ---
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_gt)
        QShortcut(QKeySequence("E"), self, activated=lambda: self.set_mode(MODE_EDIT))
        QShortcut(QKeySequence("C"), self, activated=lambda: self.set_mode(MODE_CORR))
        QShortcut(QKeySequence("F"), self, activated=self.fit_view)
        QShortcut(QKeySequence("V"), self, activated=self.toggle_visibility_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected)
        QShortcut(QKeySequence("PgDown"), self, activated=lambda: self.step_image(+1))
        QShortcut(QKeySequence("PgUp"), self, activated=lambda: self.step_image(-1))
        for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            QShortcut(QKeySequence(key), self,
                      activated=lambda dx=dx, dy=dy: self.nudge_selected(dx * 0.25, dy * 0.25))
            QShortcut(QKeySequence("Shift+" + key), self,
                      activated=lambda dx=dx, dy=dy: self.nudge_selected(dx * 1.0, dy * 1.0))

        self.setStatusBar(QStatusBar())
        self.status("開啟資料夾開始標註。流程：對應點模式點 ≥4 組 → 解 H 蓋章 → 編輯模式微調 → Ctrl+S")

    # ------------------------------------------------------------------
    def status(self, msg, ms=0):
        self.statusBar().showMessage(msg, ms)

    def mark_dirty(self):
        if not self._loading:
            self.dirty = True
            self.update_title()

    def update_title(self):
        name = self.cur_img_path.name if self.cur_img_path else "—"
        star = " *" if self.dirty else ""
        self.setWindowTitle(f"GT 標註 — {name}{star}")

    # ------------------------------------------------------------------
    # 檔案
    # ------------------------------------------------------------------
    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "選擇影像資料夾")
        if d:
            self.load_folder(d)

    def load_folder(self, d):
        self.img_dir = Path(d)
        self.images = sorted(p for p in self.img_dir.iterdir()
                             if p.suffix.lower() in IMG_EXTS and p.is_file())
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for p in self.images:
            it = QListWidgetItem(self._file_label(p))
            self.file_list.addItem(it)
        self.file_list.blockSignals(False)
        if self.images:
            self.file_list.setCurrentRow(0)

    def _file_label(self, p):
        gt = p.with_suffix(".gt.json")
        mark = "✓ " if gt.exists() else "　 "
        return mark + p.name

    def refresh_file_label(self):
        if 0 <= self.cur_idx < len(self.images):
            self.file_list.item(self.cur_idx).setText(self._file_label(self.images[self.cur_idx]))

    def step_image(self, d):
        if not self.images:
            return
        r = max(0, min(len(self.images) - 1, self.file_list.currentRow() + d))
        self.file_list.setCurrentRow(r)

    def on_select_image(self, row):
        if row < 0 or row >= len(self.images):
            return
        if self.dirty and self.cur_img_path is not None:
            if self.chk_autosave.isChecked():
                self.save_gt()
            else:
                rc = QMessageBox.question(self, "未儲存", "目前影像尚未儲存，要存檔嗎？",
                                          QMessageBox.StandardButton.Yes |
                                          QMessageBox.StandardButton.No)
                if rc == QMessageBox.StandardButton.Yes:
                    self.save_gt()
        self.load_image(row)

    def load_image(self, row):
        self._loading = True
        try:
            self.cur_idx = row
            self.cur_img_path = self.images[row]
            img = cv2.imread(str(self.cur_img_path))
            if img is None:
                self.status(f"無法讀取 {self.cur_img_path.name}")
                return
            self.img_h, self.img_w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, self.img_w, self.img_h, rgb.strides[0],
                          QImage.Format.Format_RGB888).copy()
            self.pix_item.setPixmap(QPixmap.fromImage(qimg))
            self.scene.setSceneRect(QRectF(0, 0, self.img_w, self.img_h))

            self.clear_annotations()
            self.load_gt_if_exists()
            self.fit_view()
            self.dirty = False
            self.update_title()
            self.refresh_table()
            self.update_minimap_done()
        finally:
            self._loading = False

    # ------------------------------------------------------------------
    # GT 載入 / 儲存
    # ------------------------------------------------------------------
    def gt_path(self):
        return self.cur_img_path.with_suffix(".gt.json") if self.cur_img_path else None

    def load_gt_if_exists(self):
        p = self.gt_path()
        if p is None or not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.status(f"GT 讀取失敗: {e}")
            return
        scene = data.get("scene", "single")
        (self.rb_multi if scene == "multi" else self.rb_single).setChecked(True)
        for c in data.get("corners", []):
            try:
                self.add_point(int(c["cid"]), float(c["x"]), float(c["y"]),
                               c.get("visibility", "visible"), bool(c.get("manual", False)))
            except (KeyError, ValueError, TypeError):
                continue
        annot = data.get("annot", {})
        for cr in annot.get("correspondences", []):
            try:
                self.add_corr(int(cr["junction"]), float(cr["x"]), float(cr["y"]))
            except (KeyError, ValueError, TypeError):
                continue
        Hm = annot.get("H")
        if Hm:
            self.H = np.asarray(Hm, dtype=np.float64)
            self.draw_grid()

    def save_gt(self):
        if self.cur_img_path is None:
            return
        self.normalize_orientation()
        corners = []
        for cid in sorted(self.points):
            it = self.points[cid]
            spec = CORNER_BY_CID.get(cid, {})
            corners.append({
                "cid": cid,
                "x": round(float(it.pos().x()), 3),
                "y": round(float(it.pos().y()), 3),
                "visibility": it.visibility,
                "node": [spec.get("nx"), spec.get("ny")],
                "lcid": LCID_NAMES.get(spec.get("lcid"), ""),
                "manual": it.manual,
            })
        data = {
            "image": self.cur_img_path.name,
            "scene": "multi" if self.rb_multi.isChecked() else "single",
            "convention": "row0_near",
            "corners": corners,
            "annot": {
                "correspondences": [
                    {"junction": c.junction_idx,
                     "x": round(float(c.pos().x()), 3),
                     "y": round(float(c.pos().y()), 3)} for c in self.corrs],
                "H": self.H.tolist() if self.H is not None else None,
            },
        }
        with open(self.gt_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        self.dirty = False
        self.update_title()
        self.refresh_file_label()
        self.status(f"已存檔 {self.gt_path().name}（{len(corners)} 角點）", 4000)

    # ------------------------------------------------------------------
    # 標註物件管理
    # ------------------------------------------------------------------
    def clear_annotations(self):
        for it in list(self.points.values()) + self.corrs + self.grid_items:
            self.scene.removeItem(it)
        self.points.clear(); self.corrs.clear(); self.grid_items.clear()
        self.H = None
        self.minimap.done = set(); self.minimap.selected = -1; self.minimap.update()

    def add_point(self, cid, x, y, visibility="visible", manual=False):
        if cid in self.points:
            it = self.points[cid]
            it.setPos(QPointF(x, y)); it.visibility = visibility; it._restyle()
            return it
        it = PointItem(cid, x, y, visibility, manual, win=self)
        self.scene.addItem(it)
        self.points[cid] = it
        return it

    def add_corr(self, j, x, y):
        it = CorrItem(j, x, y, win=self)
        self.scene.addItem(it)
        self.corrs.append(it)
        self.update_minimap_done()
        return it

    def update_minimap_done(self):
        self.minimap.done = {c.junction_idx for c in self.corrs}
        self.minimap.update()

    def clear_corrs(self):
        for c in self.corrs:
            self.scene.removeItem(c)
        self.corrs.clear()
        self.update_minimap_done()
        self.mark_dirty()

    # ------------------------------------------------------------------
    # 模式與互動
    # ------------------------------------------------------------------
    def set_mode(self, m):
        self.view.mode = m
        self.act_edit.setChecked(m == MODE_EDIT)
        self.act_corr.setChecked(m == MODE_CORR)
        self.status("編輯模式：拖曳/方向鍵微調，V 切換可見性，Delete 刪除"
                    if m == MODE_EDIT else
                    "對應點模式：先在右側小地圖選 junction，再點影像位置")

    def on_junction_picked(self, j):
        if self.view.mode != MODE_CORR:
            self.set_mode(MODE_CORR)
        self.status(f"已選 junction {j}（nx,ny={junction_idx_to_nx_ny(j)}），請點影像中的對應位置")

    def on_image_clicked(self, x, y):
        j = self.minimap.selected
        if j < 0:
            self.status("請先在右側小地圖選一個 junction")
            return
        # 標號框吸附：自動規範化後，session 的標號框可能與使用者的對應方向
        # 相差 180°。若已有 H，檢查使用者選的 j 與其 180° 對映 (29−j) 何者
        # 與點擊位置一致，自動採用一致者——避免 corrs 混入兩種框使 H 歪掉。
        if self.H is not None:
            H = np.asarray(self.H, np.float64)
            def _d(jj):
                p = cv2.perspectiveTransform(
                    TEMPLATE_POINTS[jj].astype(np.float32).reshape(1, 1, 2), H).reshape(2)
                return float(np.hypot(p[0] - x, p[1] - y)) if np.all(np.isfinite(p)) else 1e18
            dj, dr = _d(j), _d(29 - j)
            if dr < 0.5 * dj and dj > 25.0:
                self.status(f"對應方向與現有標註相差 180°，已自動修正：J{j} → J{29 - j}", 6000)
                j = 29 - j
            elif min(dj, dr) > 80.0:
                self.status(f"提醒：點擊位置離 J{j} 的投影 {dj:.0f}px，請確認沒點錯", 6000)
        # 同 junction 已存在 → 更新位置
        for c in self.corrs:
            if c.junction_idx == j:
                c.setPos(QPointF(x, y))
                self.mark_dirty()
                self.status(f"junction {j} 對應點已更新")
                return
        self.add_corr(j, x, y)
        self.mark_dirty()
        self.minimap.selected = -1
        self.minimap.update()
        self.status(f"junction {j} 對應點已加入（目前 {len(self.corrs)} 組）")

    def on_mouse_at(self, x, y):
        if 0 <= x < self.img_w and 0 <= y < self.img_h:
            self.statusBar().showMessage(f"({x:.2f}, {y:.2f})", 800)

    def selected_points(self):
        return [it for it in self.scene.selectedItems() if isinstance(it, PointItem)]

    def on_point_moved(self, it):
        self.mark_dirty()
        self.refresh_table_row(it)

    def on_point_selected(self, it):
        for r in range(self.table.rowCount()):
            if int(self.table.item(r, 0).text()) == it.cid:
                self.table.blockSignals(True)
                self.table.selectRow(r)
                self.table.blockSignals(False)
                break

    def toggle_visibility_selected(self):
        for it in self.selected_points():
            it.visibility = "occluded" if it.visibility == "visible" else "visible"
            it._restyle()
            self.refresh_table_row(it)
            self.mark_dirty()

    def delete_selected(self):
        changed = False
        for it in self.selected_points():
            self.scene.removeItem(it)
            self.points.pop(it.cid, None)
            changed = True
        for it in [c for c in self.scene.selectedItems() if isinstance(c, CorrItem)]:
            self.scene.removeItem(it)
            self.corrs.remove(it)
            self.update_minimap_done()
            changed = True
        if changed:
            self.mark_dirty()
            self.refresh_table()

    def nudge_selected(self, dx, dy):
        for it in self.selected_points():
            it.setPos(it.pos() + QPointF(dx, dy))

    def on_table_clicked(self, row, _col):
        cid = int(self.table.item(row, 0).text())
        it = self.points.get(cid)
        if it:
            self.scene.clearSelection()
            it.setSelected(True)
            self.view.centerOn(it)

    def fit_view(self):
        if self.img_w:
            self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def toggle_labels(self):
        self.show_labels = self.act_labels.isChecked()
        for it in self.points.values():
            it._restyle()

    def toggle_grid(self):
        self.show_grid = self.act_grid.isChecked()
        self.draw_grid()

    # ------------------------------------------------------------------
    # 方向規範化：保證輸出 GT 永遠 row0=近端（任意對應方向皆可標）
    # ------------------------------------------------------------------
    def normalize_orientation(self):
        """若目前標註的 row0 端在影像中為『遠』，整體做 180° 重映射
        （cid / 對應點 junction / H；座標不動）。回傳是否有翻轉。"""
        H = self.H
        if H is None and len(self.points) >= 6:
            src = np.array([CORNER_BY_CID[c]["world"] for c in self.points],
                           np.float64).reshape(-1, 1, 2)
            dst = np.array([[it.pos().x(), it.pos().y()]
                            for it in self.points.values()],
                           np.float64).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return False
        near = row0_is_near(H)
        if near is None or near:
            return False
        # row0 在遠端 → 180° 重映射
        new_points = {}
        for cid, it in self.points.items():
            it.set_cid(rot180_cid(cid))
            new_points[it.cid] = it
        self.points = new_points
        for c in self.corrs:
            c.set_junction(29 - c.junction_idx)
        if self.H is not None:
            Hn = np.asarray(self.H, np.float64) @ T180
            self.H = Hn / Hn[2, 2]
            self.draw_grid()
        self.update_minimap_done()
        self.refresh_table()
        self.mark_dirty()
        self.status("已自動規範化方向：row0 = 影像近端（cid/對應點已 180° 重映射）", 6000)
        return True

    # ------------------------------------------------------------------
    # H 求解與蓋章
    # ------------------------------------------------------------------
    def solve_and_stamp(self):
        # 對應來源 = 手點對應點 + 手動微調過的角點（後者為高倍率精修，品質更高）
        src, dst, tags = [], [], []
        for c in self.corrs:
            src.append(TEMPLATE_POINTS[c.junction_idx])
            dst.append([c.pos().x(), c.pos().y()])
            tags.append(f"J{c.junction_idx}")
        n_corr = len(src)
        for cid, it in self.points.items():
            if it.manual and cid in CORNER_BY_CID:
                src.append(CORNER_BY_CID[cid]["world"])
                dst.append([it.pos().x(), it.pos().y()])
                tags.append(cid_label(cid))
        n_manual = len(src) - n_corr
        if len(src) < 4:
            QMessageBox.warning(self, "對應不足",
                                "至少需要 4 組對應（對應點 + 手動調整過的角點合計）。")
            return
        src = np.asarray(src, dtype=np.float64).reshape(-1, 1, 2)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 1, 2)
        # 人工標註只有點擊噪聲、沒有粗大外點：用全點最小平方（正規化 DLT + LM 精修），
        # 噪聲隨點數 ~1/√n 平均掉——「標越多越準」。RANSAC 的固定門檻會把正常
        # 手點誤判為外點、只用一小撮共識集，加點無感且不穩，故不用。
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            QMessageBox.warning(self, "求解失敗", "findHomography 失敗，請檢查對應點是否退化（共線）。")
            return
        self.H = H

        # 殘差回報：個別偏大者提示重點（人工修正勝過默默剔除）
        proj = cv2.perspectiveTransform(src.astype(np.float32), H).reshape(-1, 2)
        res = np.hypot(*(proj - dst.reshape(-1, 2)).T)
        worst = ""
        if res.max() > 6.0:
            k = int(np.argmax(res))
            worst = f"⚠ {tags[k]} 殘差 {res[k]:.1f}px，建議重新點該處。"
        n_new = n_upd = n_skip = 0
        margin = 2.0
        for spec in ALL_CORNERS:
            p = cv2.perspectiveTransform(
                spec["world"].astype(np.float32).reshape(1, 1, 2), H).reshape(2)
            if not np.all(np.isfinite(p)):
                continue
            inside = (-margin <= p[0] <= self.img_w + margin and
                      -margin <= p[1] <= self.img_h + margin)
            cid = spec["cid"]
            old = self.points.get(cid)
            if not inside:
                if old is None:
                    n_skip += 1
                continue
            if old is not None and old.manual:
                n_skip += 1
                continue
            if old is None:
                self.add_point(cid, float(p[0]), float(p[1]), "visible", manual=False)
                n_new += 1
            else:
                old.setPos(QPointF(float(p[0]), float(p[1])))
                old.manual = False
                old._restyle()
                n_upd += 1
        self.draw_grid()
        flipped = self.normalize_orientation()
        self.mark_dirty()
        self.refresh_table()
        tag = "（已自動規範化 row0=近端）" if flipped else ""
        self.status(f"蓋章完成：新增 {n_new}、更新 {n_upd}、略過 {n_skip}（手動/框外）{tag}。"
                    f" 對應 {n_corr} 點＋手動角點 {n_manual} 點，"
                    f"殘差 中位 {np.median(res):.2f}px / 最大 {res.max():.2f}px。{worst}", 9000)

    def restamp_auto_points(self):
        """以目前 H 重新投影所有非手動點。"""
        if self.H is None:
            self.status("尚未求解 H")
            return
        self.solve_and_stamp() if self.corrs else None

    def draw_grid(self):
        for g in self.grid_items:
            self.scene.removeItem(g)
        self.grid_items.clear()
        if self.H is None or not self.show_grid:
            return
        pen = QPen(COL_GRID, 0)
        for i1, i2 in GRID_CONNECTIONS:
            seg = np.array([TEMPLATE_POINTS[i1], TEMPLATE_POINTS[i2]],
                           dtype=np.float32).reshape(-1, 1, 2)
            q = cv2.perspectiveTransform(seg, self.H).reshape(-1, 2)
            if not np.all(np.isfinite(q)):
                continue
            ln = QGraphicsLineItem(q[0, 0], q[0, 1], q[1, 0], q[1, 1])
            ln.setPen(pen); ln.setZValue(5)
            self.scene.addItem(ln)
            self.grid_items.append(ln)

    # ------------------------------------------------------------------
    # 點表
    # ------------------------------------------------------------------
    def refresh_table(self):
        self.table.setRowCount(0)
        for cid in sorted(self.points):
            it = self.points[cid]
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [str(cid), cid_label(cid),
                    f"{it.pos().x():.2f}", f"{it.pos().y():.2f}",
                    "✓" if it.visibility == "visible" else "遮"]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        n = len(self.points)
        nv = sum(1 for p in self.points.values() if p.visibility == "visible")
        self.stats_lbl.setText(f"角點 {n}（可見 {nv} / 遮蔽 {n - nv}）  對應點 {len(self.corrs)} 組")

    def refresh_table_row(self, it):
        for r in range(self.table.rowCount()):
            if int(self.table.item(r, 0).text()) == it.cid:
                self.table.item(r, 2).setText(f"{it.pos().x():.2f}")
                self.table.item(r, 3).setText(f"{it.pos().y():.2f}")
                self.table.item(r, 4).setText("✓" if it.visibility == "visible" else "遮")
                break
        n = len(self.points)
        nv = sum(1 for p in self.points.values() if p.visibility == "visible")
        self.stats_lbl.setText(f"角點 {n}（可見 {nv} / 遮蔽 {n - nv}）  對應點 {len(self.corrs)} 組")

    # ------------------------------------------------------------------
    def closeEvent(self, ev):
        if self.dirty and self.chk_autosave.isChecked():
            self.save_gt()
        super().closeEvent(ev)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="羽球場角點 GT 標註工具")
    ap.add_argument("img_dir", nargs="?", help="影像資料夾（可選，啟動後也能開啟）")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = GTAnnotator()
    win.show()
    if args.img_dir:
        win.load_folder(args.img_dir)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
