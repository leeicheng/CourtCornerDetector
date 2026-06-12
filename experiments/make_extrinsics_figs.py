#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_extrinsics_figs.py — 外參實驗分佈圖（圖 5.7a / 5.7b）

雙峰混合分佈下，中位數與 P90 會落在兩峰之間的無人區，低估系統真實表現；
本腳本以 CDF（log 軸）與重投影—位置誤差散點呈現分佈結構。

用法：
    python experiments/make_extrinsics_figs.py \
        --inputs results/extrinsics.json results_spread/extrinsics.json \
        --labels sorted spread --out results/figs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "junction":      ("Refined junctions (line intersection)", "#1a7a3a", "-"),
    "box_center":    ("YOLO box centers", "#555555", "-."),
    "corners_top6":  ("Corners top-6", "#7d3c98", "-"),
    "corners_top10": ("Corners top-10", "#1f5fa8", "-"),
    "corners_all":   ("Corners (all)", "#0e7c7b", "-"),
}


def load(path):
    return json.load(open(path, encoding="utf-8"))["per_image"]


def series(per, src, field="pos_err_cm"):
    return np.array([p[src][field] for p in per if p.get(src)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="各輸入檔之標籤（如 sorted spread），預設用檔名")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--gross_px", type=float, default=10.0)
    args = ap.parse_args()
    runs = [(lab, load(p)) for p, lab in zip(
        args.inputs,
        args.labels or [Path(p).stem for p in args.inputs])]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── 圖 a：位置誤差 CDF（log x）──
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    drawn_junction = False
    for run_lab, per in runs:
        for src, (lab, col, ls) in STYLE.items():
            v = series(per, src)
            if len(v) == 0:
                continue
            if src == "junction":
                if drawn_junction:
                    continue           # junction 與選取策略無關，畫一次即可
                drawn_junction = True
                full_lab = f"{lab} (n={len(v)})"
            else:
                full_lab = f"{lab}, {run_lab} (n={len(v)})"
                ls = "--" if "sort" in run_lab.lower() else "-"
            vv = np.sort(v); y = np.arange(1, len(vv) + 1) / len(vv)
            ax.step(vv, y, where="post", label=full_lab, color=col, ls=ls, lw=1.8)
    ax.axvspan(60, 400, color="0.92", zorder=0)
    ax.text(155, 0.97, "empty band", ha="center", va="top",
            fontsize=8, color="0.45")
    ax.axvline(30, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("Camera position error (cm, log scale)")
    ax.set_ylabel("Cumulative fraction of solved images")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.2, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "fig_extrinsics_cdf.png")

    # ── 圖 b：重投影 vs 位置誤差散點（log–log）──
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=150)
    run_lab, per = runs[-1]
    marks = {"junction": ("o", "#1a7a3a"), "corners_top10": ("^", "#1f5fa8"),
             "corners_top6": ("s", "#7d3c98"), "box_center": ("x", "#555555")}
    for src, (mk, col) in marks.items():
        x = series(per, src, "rmse_gt_px"); y = series(per, src)
        if len(x) == 0:
            continue
        ax.scatter(x, y, s=14, alpha=0.65, c=col, marker=mk,
                   label=STYLE[src][0], edgecolors="none")
    ax.axvline(args.gross_px, color="0.3", lw=1, ls="--")
    ax.text(args.gross_px * 1.1, ax.get_ylim()[0] * 1.5 if ax.get_ylim()[0] > 0
            else 0.2, f"{args.gross_px:.0f} px threshold",
            fontsize=8, rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Reprojection RMSE vs GT pixels (px, log)")
    ax.set_ylabel("Camera position error (cm, log)")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_extrinsics_scatter.png")

    # ── 達成率表（取代被雙峰污染的中位/平均）──
    print("達成率（成功求解之影像中，位置誤差 ≤ 門檻之比例）：")
    hdr = ["來源/run"] + [f"≤{t}cm" for t in (5, 10, 30, 100)] + ["正確模式中位(cm)"]
    print(" | ".join(hdr))
    for run_lab, per in runs:
        for src in STYLE:
            v = series(per, src)
            if len(v) == 0:
                continue
            cells = [f"{STYLE[src][0]} [{run_lab}] (n={len(v)})"]
            for t in (5, 10, 30, 100):
                cells.append(f"{np.mean(v <= t) * 100:.1f}%")
            good = v[v <= 30]
            cells.append(f"{np.median(good):.2f}" if len(good) else "—")
            print(" | ".join(cells))
    print(f"\n圖輸出至 {out}/")


if __name__ == "__main__":
    main()
