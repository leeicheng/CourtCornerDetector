# -*- coding: utf-8 -*-
"""
results_viewer.py — 實驗結果觀察工具（PyQt6）
================================================================
載入 results/ 目錄後可：
  - 「摘要」分頁：以表格檢視各實驗的彙整數據，一鍵複製成 Markdown
    （可直接貼回論文第五章表格）。
  - 「圖表」分頁：誤差 CDF / 直方圖、信心—誤差散佈與分箱曲線、
    判別性 ROC、各階段耗時、消融比較長條圖。
  - 「逐圖檢視」分頁：疊圖比較 GT（橘十字）與輸出角點（依信心上色圓圈）
    與誤差連線；右側表格列出每個配對角點之誤差，點選列會在影像上紅圈標示；
    滾輪縮放、拖曳平移。

使用：
  python -m experiments.results_viewer            # 之後在介面中選 results/
  python -m experiments.results_viewer --results results --img_dir data/test_imgs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QFont,
                         QGuiApplication)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QPushButton, QFileDialog, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QMessageBox, QHeaderView)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
for f in ("Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC",
          "Heiti TC", "Arial Unicode MS"):
    try:
        from matplotlib import font_manager
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = f
            break
    except Exception:
        break
matplotlib.rcParams["axes.unicode_minus"] = False

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def fmt(v, n=3, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.{n}f}"


# ================================================================
#  摘要表格建構：每種實驗 JSON → (標題, headers, rows) 清單
# ================================================================

def build_tables(name: str, data: dict):
    tables = []
    if name.startswith("main_eval") and "summary" in data:
        s = data["summary"]
        o = s["overall"]
        tables.append(("整體誤差（表 5.5 本方法列）",
                       ["中位數", "平均", "P90", "≤1px", "≤2px", "n"],
                       [[fmt(o["median"]), fmt(o["mean"]), fmt(o["p90"]),
                         fmt(o.get("succ@1px"), pct=True),
                         fmt(o.get("succ@2px"), pct=True), o["n"]]]))
        for key, title in (("by_type", "依交點類型"), ("by_tier", "依 tier"),
                           ("by_visibility", "依可見性")):
            rows = [[k, fmt(v["median"]), fmt(v["p90"]),
                     fmt(v.get("succ@2px"), pct=True), v["n"]]
                    for k, v in s.get(key, {}).items()]
            if rows:
                tables.append((title, ["分層", "中位數", "P90", "≤2px", "n"],
                               rows))
        rows = [[b["range"], fmt(b["fraction"], pct=True), fmt(b["median"]),
                 fmt(b["p90"]), b["n"]] for b in s.get("conf_bins", [])]
        tables.append(("信心分組（表 5.8）",
                       ["信心區間", "比例", "中位數", "P90", "n"], rows))
        st = s.get("stage_times_ms", {})
        tables.append(("各階段耗時（表 5.11）",
                       ["階段", "平均 (ms)", "佔比"],
                       [[k, f"{v['mean_ms']:.1f}", fmt(v["share"], pct=True)]
                        for k, v in st.items()]))
        tables.append(("其他", ["項目", "值"],
                       [["Spearman(conf, −err)", fmt(s.get("spearman_conf_err"))],
                        ["H confidence 分佈", json.dumps(
                            s.get("h_confidence_counts", {}))],
                        ["平均線支持度", fmt(s.get("mean_line_support"), 2)],
                        ["輸出率", fmt(s["counts"].get("output_rate"),
                                       pct=True)]]))
    elif name.startswith("baselines"):
        rows = []
        for m, st in data.get("summary", {}).items():
            if st.get("median") is None:
                continue
            rows.append([m, fmt(st["median"]), fmt(st["mean"]), fmt(st["p90"]),
                         fmt(st.get("succ@1px"), pct=True),
                         fmt(st.get("succ@2px"), pct=True), st["n"]])
        tables.append(("基準方法誤差（表 5.5）",
                       ["方法", "中位數", "平均", "P90", "≤1px", "≤2px", "n"],
                       rows))
    elif name.startswith("conf_sweep"):
        rows = [[r["threshold"], f"{r['candidates_per_img']:.1f}",
                 fmt(r["solve_ok_rate"], pct=True),
                 fmt(r["line_support_mean"], 2),
                 fmt(r["mapping_correct_single"], pct=True),
                 fmt(r["mapping_correct_multi"], pct=True),
                 fmt(r["solve_time_mean_s"], 2)]
                for r in data.get("table", [])]
        tables.append(("信心門檻掃描（表 5.3b）",
                       ["門檻", "候選/圖", "求解成功", "線支持",
                        "對應正確(單)", "對應正確(多)", "求解耗時s"], rows))
    elif name.startswith("ablation"):
        rows = [[v, fmt(st["median"]), fmt(st["p90"]),
                 fmt(st["output_rate"], pct=True),
                 fmt(st["cid_correct_rate"], pct=True)]
                for v, st in data.get("summary", {}).items()]
        tables.append(("方法消融（表 5.9）",
                       ["變體", "中位數", "P90", "輸出率", "編號正確率"],
                       rows))
    elif name.startswith("quality_discrim"):
        rows = []
        for m, t in data.get("table_5_6", {}).items():
            rows.append([m, fmt(t["auc_vs_online"]), fmt(t["auc_vs_shifted"]),
                         fmt(t["auc_vs_bg"]), fmt(t["pos_median"]),
                         f"{fmt(t['time_ms_per_point'], 2)} ms"])
        tables.append(("證據判別能力（表 5.6）",
                       ["方法", "AUC(線上)", "AUC(偏移)", "AUC(背景)",
                        "pos 中位數", "單點耗時"], rows))
        if data.get("table_5_7"):
            rows = [[v, fmt(t["auc_vs_bg"]), fmt(t["auc_vs_online"])]
                    for v, t in data["table_5_7"].items()]
            tables.append(("證據組成消融（表 5.7）",
                           ["設定", "AUC(背景)", "AUC(線上)"], rows))
    else:
        tables.append((name, ["鍵", "值（JSON）"],
                       [[k, json.dumps(v, ensure_ascii=False)[:120]]
                        for k, v in data.items() if k != "per_image"]))
    return tables


def tables_to_markdown(tables):
    out = []
    for title, headers, rows in tables:
        out.append(f"### {title}\n")
        out.append("| " + " | ".join(headers) + " |")
        out.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            out.append("| " + " | ".join(str(x) for x in r) + " |")
        out.append("")
    return "\n".join(out)


# ================================================================
#  影像檢視（縮放 / 平移 / 疊圖）
# ================================================================

class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._highlight = None

    def wheelEvent(self, ev):
        s = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(s, s)

    def fit(self):
        if self.scene_.items():
            self.fitInView(self.scene_.itemsBoundingRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def show_image(self, qimg: QImage, overlays):
        """overlays: list of dict(kind, x, y, [x2,y2], color, label)"""
        self.scene_.clear()
        self._highlight = None
        self.scene_.addPixmap(QPixmap.fromImage(qimg))
        font = QFont()
        font.setPointSizeF(8)
        for o in overlays:
            col = QColor(*o["color"])
            pen = QPen(col)
            pen.setWidthF(o.get("width", 1.6))
            pen.setCosmetic(True)
            k = o["kind"]
            if k == "circle":
                r = o.get("r", 4)
                self.scene_.addEllipse(o["x"] - r, o["y"] - r, 2 * r, 2 * r,
                                       pen)
            elif k == "cross":
                r = o.get("r", 6)
                self.scene_.addLine(o["x"] - r, o["y"], o["x"] + r, o["y"], pen)
                self.scene_.addLine(o["x"], o["y"] - r, o["x"], o["y"] + r, pen)
            elif k == "line":
                self.scene_.addLine(o["x"], o["y"], o["x2"], o["y2"], pen)
            if o.get("label"):
                t = self.scene_.addText(str(o["label"]), font)
                t.setDefaultTextColor(col)
                t.setPos(o["x"] + 4, o["y"] + 2)
        self.fit()

    def highlight(self, x, y, r=10):
        if self._highlight is not None:
            self.scene_.removeItem(self._highlight)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidthF(2.5)
        pen.setCosmetic(True)
        self._highlight = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        self._highlight.setPen(pen)
        self.scene_.addItem(self._highlight)
        self.centerOn(QPointF(x, y))


# ================================================================
#  主視窗
# ================================================================

class Viewer(QMainWindow):
    def __init__(self, results_dir=None, img_dir=None):
        super().__init__()
        self.setWindowTitle("羽球場角點實驗結果觀察工具")
        self.resize(1400, 880)
        self.results_dir = Path(results_dir) if results_dir else None
        self.img_dir = Path(img_dir) if img_dir else None
        self.data = {}          # name -> json dict
        self.current = None     # 目前選擇的實驗名稱
        self._build_ui()
        if self.results_dir:
            self.load_results_dir(self.results_dir)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QSplitter()
        self.setCentralWidget(root)

        # 左側：結果清單
        left = QWidget()
        lv = QVBoxLayout(left)
        b1 = QPushButton("開啟 results 目錄…")
        b1.clicked.connect(self.pick_results_dir)
        lv.addWidget(b1)
        self.exp_list = QListWidget()
        self.exp_list.currentItemChanged.connect(self.on_select_exp)
        lv.addWidget(QLabel("實驗結果："))
        lv.addWidget(self.exp_list, 1)
        b2 = QPushButton("設定影像目錄…（逐圖檢視用）")
        b2.clicked.connect(self.pick_img_dir)
        lv.addWidget(b2)
        self.img_dir_label = QLabel("影像目錄：未設定")
        self.img_dir_label.setWordWrap(True)
        lv.addWidget(self.img_dir_label)
        root.addWidget(left)

        # 右側：分頁
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        root.setStretchFactor(1, 1)

        # --- 分頁 1：摘要 ---
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.copy_btn = QPushButton("複製為 Markdown")
        self.copy_btn.clicked.connect(self.copy_markdown)
        top.addWidget(self.copy_btn)
        top.addStretch(1)
        v.addLayout(top)
        self.summary_host = QVBoxLayout()
        host = QWidget()
        host.setLayout(self.summary_host)
        from PyQt6.QtWidgets import QScrollArea
        sc = QScrollArea()
        sc.setWidget(host)
        sc.setWidgetResizable(True)
        v.addWidget(sc, 1)
        self.tabs.addTab(w, "摘要")

        # --- 分頁 2：圖表 ---
        w = QWidget()
        v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(["誤差 CDF", "誤差直方圖", "信心—誤差",
                                  "信心分佈（依 tier）", "ROC（判別性）",
                                  "各階段耗時", "消融比較"])
        self.plot_combo.currentIndexChanged.connect(self.update_plot)
        top.addWidget(self.plot_combo)
        top.addStretch(1)
        v.addLayout(top)
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        v.addWidget(self.canvas, 1)
        self.tabs.addTab(w, "圖表")

        # --- 分頁 3：逐圖檢視 ---
        w = QWidget()
        h = QHBoxLayout(w)
        lw = QWidget()
        lv2 = QVBoxLayout(lw)
        lv2.addWidget(QLabel("影像（main_eval / ablation）："))
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_select_image)
        lv2.addWidget(self.image_list, 1)
        h.addWidget(lw)

        self.view = ImageView()
        h.addWidget(self.view, 1)

        rw = QWidget()
        rv = QVBoxLayout(rw)
        self.fit_btn = QPushButton("符合視窗")
        self.fit_btn.clicked.connect(self.view.fit)
        rv.addWidget(self.fit_btn)
        self.img_info = QLabel("")
        self.img_info.setWordWrap(True)
        rv.addWidget(self.img_info)
        self.corner_table = QTableWidget(0, 6)
        self.corner_table.setHorizontalHeaderLabels(
            ["cid", "err(px)", "conf", "tier", "type", "vis"])
        self.corner_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.corner_table.itemSelectionChanged.connect(self.on_corner_select)
        rv.addWidget(self.corner_table, 1)
        h.addWidget(rw)
        self.tabs.addTab(w, "逐圖檢視")

        if self.img_dir:
            self.img_dir_label.setText(f"影像目錄：{self.img_dir}")

    # ---------------- 載入 ----------------
    def pick_results_dir(self):
        d = QFileDialog.getExistingDirectory(self, "選擇 results 目錄")
        if d:
            self.load_results_dir(Path(d))

    def pick_img_dir(self):
        d = QFileDialog.getExistingDirectory(self, "選擇影像目錄")
        if d:
            self.img_dir = Path(d)
            self.img_dir_label.setText(f"影像目錄：{self.img_dir}")
            self.populate_image_list()

    def load_results_dir(self, d: Path):
        self.results_dir = d
        self.data.clear()
        self.exp_list.clear()
        for p in sorted(d.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    self.data[p.stem] = json.load(f)
                self.exp_list.addItem(QListWidgetItem(p.stem))
            except Exception as e:
                print(f"[skip] {p.name}: {e}")
        if self.exp_list.count():
            self.exp_list.setCurrentRow(0)

    # ---------------- 摘要 ----------------
    def on_select_exp(self, item, _prev=None):
        if item is None:
            return
        self.current = item.text()
        self.render_summary()
        self.update_plot()
        self.populate_image_list()

    def render_summary(self):
        while self.summary_host.count():
            it = self.summary_host.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if not self.current:
            return
        tables = build_tables(self.current, self.data[self.current])
        for title, headers, rows in tables:
            lab = QLabel(f"<b>{title}</b>")
            self.summary_host.addWidget(lab)
            t = QTableWidget(len(rows), len(headers))
            t.setHorizontalHeaderLabels(headers)
            for i, r in enumerate(rows):
                for j, x in enumerate(r):
                    t.setItem(i, j, QTableWidgetItem(str(x)))
            t.resizeColumnsToContents()
            t.setMinimumHeight(min(60 + 26 * len(rows), 360))
            t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.summary_host.addWidget(t)
        self.summary_host.addStretch(1)

    def copy_markdown(self):
        if not self.current:
            return
        md = tables_to_markdown(build_tables(self.current,
                                             self.data[self.current]))
        QGuiApplication.clipboard().setText(md)
        QMessageBox.information(self, "已複製",
                                "已將表格複製為 Markdown，可直接貼上。")

    # ---------------- 圖表 ----------------
    def _matched_rows(self):
        d = self.data.get(self.current or "", {})
        rows = []
        for p in d.get("per_image", []):
            for r in p.get("matched", []):
                rows.append(r)
        return rows

    def update_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        kind = self.plot_combo.currentText()
        rows = self._matched_rows()
        d = self.data.get(self.current or "", {})

        if kind == "誤差 CDF" and rows:
            e = np.sort([r["err_px"] for r in rows])
            ax.plot(e, np.arange(1, len(e) + 1) / len(e))
            ax.set_xlabel("誤差 (px)"); ax.set_ylabel("累積比例")
            ax.set_xlim(0, min(10, e.max())); ax.grid(alpha=.3)
        elif kind == "誤差直方圖" and rows:
            e = np.array([r["err_px"] for r in rows])
            ax.hist(e, bins=40, range=(0, min(10, e.max())))
            ax.set_xlabel("誤差 (px)"); ax.set_ylabel("數量")
        elif kind == "信心—誤差" and rows:
            c = np.array([r.get("conf", 0) for r in rows])
            e = np.array([r["err_px"] for r in rows])
            ax.scatter(c, e, s=8, alpha=.35)
            bins = np.linspace(0, 1, 11)
            idx = np.digitize(c, bins)
            bx = [0.5 * (bins[i - 1] + bins[i]) for i in range(1, 11)
                  if (idx == i).any()]
            by = [np.median(e[idx == i]) for i in range(1, 11)
                  if (idx == i).any()]
            ax.plot(bx, by, "r-o", lw=2, ms=4, label="分箱中位數")
            ax.set_xlabel("conf"); ax.set_ylabel("誤差 (px)")
            ax.set_ylim(0, np.percentile(e, 98) if len(e) else 1)
            ax.legend(); ax.grid(alpha=.3)
        elif kind == "信心分佈（依 tier）" and rows:
            tiers = sorted({r.get("tier", "") for r in rows})
            for t in tiers:
                c = [r.get("conf", 0) for r in rows if r.get("tier") == t]
                ax.hist(c, bins=20, range=(0, 1), alpha=.55, label=t or "?")
            ax.set_xlabel("conf"); ax.set_ylabel("數量"); ax.legend()
        elif kind == "ROC（判別性）":
            qd = d if "table_5_6" in d else self.data.get("quality_discrim", {})
            for m, t in qd.get("table_5_6", {}).items():
                fpr, tpr = t.get("roc_vs_online", ([], []))
                if fpr:
                    ax.plot(fpr, tpr,
                            label=f"{m} AUC={fmt(t['auc_vs_online'])}")
            ax.plot([0, 1], [0, 1], "k--", lw=.8)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(); ax.grid(alpha=.3)
        elif kind == "各階段耗時":
            md = d if "summary" in d and "stage_times_ms" in d.get(
                "summary", {}) else self.data.get("main_eval", {})
            st = md.get("summary", {}).get("stage_times_ms", {})
            if st:
                ks = list(st)
                ax.barh(ks, [st[k]["mean_ms"] for k in ks])
                ax.set_xlabel("平均耗時 (ms)")
        elif kind == "消融比較":
            ab = d if d.get("meta", {}).get("experiment", "").startswith(
                "ablation") else self.data.get("ablation", {})
            s = ab.get("summary", {})
            if s:
                ks = list(s)
                x = np.arange(len(ks))
                ax.bar(x - .2, [s[k]["median"] or 0 for k in ks], .4,
                       label="中位數")
                ax.bar(x + .2, [s[k]["p90"] or 0 for k in ks], .4,
                       label="P90")
                ax.set_xticks(x, ks, rotation=20)
                ax.set_ylabel("誤差 (px)"); ax.legend()
        else:
            ax.text(.5, .5, "此實驗無對應資料", ha="center", va="center")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------------- 逐圖檢視 ----------------
    def populate_image_list(self):
        self.image_list.clear()
        d = self.data.get(self.current or "", {})
        for p in d.get("per_image", []):
            if isinstance(p, dict) and "image" in p:
                n_m = p.get("n_matched")
                errs = [r["err_px"] for r in p.get("matched", [])]
                med = f"  med={np.median(errs):.2f}px" if errs else ""
                self.image_list.addItem(
                    f"{p['image']}  [{p.get('status', '?')}]"
                    f"{'' if n_m is None else f'  配對{n_m}'}{med}")

    def _find_image_record(self, name):
        d = self.data.get(self.current or "", {})
        for p in d.get("per_image", []):
            if isinstance(p, dict) and p.get("image") == name:
                return p
        return None

    def on_select_image(self, item, _prev=None):
        if item is None:
            return
        name = item.text().split("  ")[0]
        rec = self._find_image_record(name)
        if rec is None:
            return
        if not self.img_dir:
            self.img_info.setText("請先設定影像目錄。")
            return
        ip = self.img_dir / name
        if not ip.exists():
            self.img_info.setText(f"找不到影像：{ip}")
            return
        qimg = QImage(str(ip))
        overlays = []
        matched = rec.get("matched", [])
        gt_seen = set()
        self._rows = matched
        for r in matched:
            col = (0, 200, 0) if r.get("conf", 0) >= .8 else \
                  (230, 200, 0) if r.get("conf", 0) >= .65 else (255, 140, 0)
            overlays.append(dict(kind="line", x=r["x"], y=r["y"],
                                 x2=r["gt_x"], y2=r["gt_y"],
                                 color=(255, 80, 80), width=1.0))
            overlays.append(dict(kind="circle", x=r["x"], y=r["y"],
                                 color=col, r=4, label=r["cid"]))
            overlays.append(dict(kind="cross", x=r["gt_x"], y=r["gt_y"],
                                 color=(255, 128, 0), r=6))
            gt_seen.add(int(r["cid"]))
        for c in rec.get("corners", []):
            if int(c["cid"]) not in gt_seen:
                overlays.append(dict(kind="circle", x=c["x"], y=c["y"],
                                     color=(160, 160, 160), r=3,
                                     label=c["cid"]))
        self.view.show_image(qimg, overlays)

        errs = [r["err_px"] for r in matched]
        self.img_info.setText(
            f"{name}\n狀態 {rec.get('status')}\n"
            f"偵測 {rec.get('n_detections', '—')}  輸出 {rec.get('n_pred', '—')}"
            f"  配對 {len(matched)}\n"
            + (f"誤差 med {np.median(errs):.2f} / P90 "
               f"{np.percentile(errs, 90):.2f} px\n" if errs else "")
            + f"漏報 cid：{rec.get('missed_cids', [])}\n"
            + f"H：{json.dumps(rec.get('homography', {}), ensure_ascii=False)}")

        self.corner_table.setRowCount(len(matched))
        for i, r in enumerate(sorted(matched, key=lambda x: -x["err_px"])):
            vals = [r["cid"], f"{r['err_px']:.2f}",
                    f"{r.get('conf', 0):.3f}", r.get("tier", ""),
                    r.get("corner_type", ""), r.get("visibility", "")]
            for j, v in enumerate(vals):
                self.corner_table.setItem(i, j, QTableWidgetItem(str(v)))
        self.corner_table.setProperty(
            "rows_sorted", sorted(matched, key=lambda x: -x["err_px"]))

    def on_corner_select(self):
        rows = self.corner_table.property("rows_sorted") or []
        i = self.corner_table.currentRow()
        if 0 <= i < len(rows):
            self.view.highlight(rows[i]["x"], rows[i]["y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--img_dir", default=None)
    args = ap.parse_args()
    app = QApplication(sys.argv)
    v = Viewer(args.results, args.img_dir)
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
