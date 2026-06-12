#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_yolopoint_eval.py — 以與 run_baselines 相同之協定評估 YOLOPoint（表 5.5）

輸入為 yolopoint_predict_local.py 之 YOLOPoint_full.csv（img_name, corner_x, corner_y），
協定：對每個「可見」GT 角點，於以 GT 為中心、邊長 --win 之方形視窗內取最近 keypoint，
計算歐氏誤差；視窗內無 keypoint 者計為未覆蓋（coverage 另報）。
與 Harris/Förstner 相同，此為 oracle 定位設定（提供正確鄰域），偏向有利於被測方法。

用法：
    python experiments/run_yolopoint_eval.py \
        --csv predict_out/YOLOPoint_full.csv \
        --img_dir datasets/gt --gt_dir datasets/gt --win 21 --out results
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from experiments.common import load_gt, find_images, error_stats, fmt  # noqa: E402


def load_csv(path):
    by_img = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["img_name"]
            pt = (float(row["corner_x"]), float(row["corner_y"]))
            by_img[name].append(pt)
            # 攤平命名（dir__file.png）也以末段檔名建索引
            base = name.split("__")[-1]
            if base != name:
                by_img[base].append(pt)
    return {k: np.asarray(v, np.float64) for k, v in by_img.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="YOLOPoint_full.csv")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--win", type=int, default=21, help="搜尋視窗邊長 (px)，與基準一致")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    preds = load_csv(args.csv)
    half = args.win / 2.0
    errs, n_gt_vis, n_covered = [], 0, 0
    per_image = []
    for ip in find_images(args.img_dir):
        gt = load_gt(ip, args.gt_dir)
        if not gt:
            continue
        kps = preds.get(ip.name)
        if kps is None:
            kps = preds.get(ip.stem + ip.suffix.lower())
        img_errs = []
        for cid, g in gt.items():
            if g["visibility"] != "visible":
                continue
            n_gt_vis += 1
            if kps is None or len(kps) == 0:
                continue
            dx = np.abs(kps[:, 0] - g["x"]); dy = np.abs(kps[:, 1] - g["y"])
            in_win = (dx <= half) & (dy <= half)
            if not in_win.any():
                continue
            d = np.hypot(kps[in_win, 0] - g["x"], kps[in_win, 1] - g["y"])
            e = float(d.min())
            errs.append(e); img_errs.append(e); n_covered += 1
        per_image.append({"image": ip.name, "n_gt_visible":
                          sum(1 for g in gt.values() if g["visibility"] == "visible"),
                          "n_covered": len(img_errs),
                          "median": float(np.median(img_errs)) if img_errs else None})

    s = error_stats(errs)
    coverage = n_covered / n_gt_vis if n_gt_vis else 0.0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = {"method": "yolopoint", "win": args.win, "coverage": coverage,
              "n_gt_visible": n_gt_vis, "n_covered": n_covered,
              "stats": s, "per_image": per_image}
    with open(out / "yolopoint_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n========== YOLOPoint（表 5.5 列）==========")
    print(f"可見 GT {n_gt_vis}，視窗內有 keypoint（覆蓋率）{fmt(coverage, pct=True)}  n={n_covered}")
    print(f"中位數 {fmt(s['median'])}  平均 {fmt(s['mean'])}  P90 {fmt(s['p90'])}  "
          f"≤1px {fmt(s['succ@1px'], pct=True)}  ≤2px {fmt(s['succ@2px'], pct=True)}")
    print("（註：與 Harris/Förstner 同為 GT 視窗 oracle 定位、僅可見點；"
          "覆蓋率為 YOLOPoint 特有欄位——其 keypoint 為稀疏輸出，視窗內可能無點。）")


if __name__ == "__main__":
    main()
