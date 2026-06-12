# -*- coding: utf-8 -*-
"""
run_baselines.py — 局部基準方法（對應論文表 5.5 之 Harris / Förstner 列）
================================================================
依論文 5.4 節設定：局部影像方法以「GT 角點為中心之固定視窗」提供搜尋
區域（偏向有利於基準方法），於視窗內取最強響應作為輸出位置：

  - Harris   ：cv2.cornerHarris → 取視窗內最大響應 → cornerSubPix 次像素精修
  - Förstner ：以視窗內梯度解收斂點  p* = (Σ ∇I∇Iᵀ)⁻¹ Σ (∇I∇Iᵀ x)

YOLOPoint 屬學習式方法，需另行以其模型推論後將輸出存成與 GT 相同格式，
再以 --extern_pred 載入計分（每張影像一個 <stem>.pred.json）。

使用：
  python -m experiments.run_baselines \\
      --img_dir data/test_imgs --gt_dir data/gt --win 21 --out results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (find_images, load_gt, error_stats,
                                save_result, fmt, load_result)


# ---------------- Harris ----------------

def harris_locate(gray, cx, cy, half):
    h, w = gray.shape
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < 7 or roi.shape[1] < 7:
        return None
    R = cv2.cornerHarris(np.float32(roi), blockSize=3, ksize=3, k=0.04)
    r, c = np.unravel_index(int(np.argmax(R)), R.shape)
    if R[r, c] <= 0:
        return None
    pt = np.array([[c, r]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(roi, pt, (3, 3), (-1, -1), crit)
    except cv2.error:
        pass
    return float(pt[0, 0] + x0), float(pt[0, 1] + y0)


# ---------------- Förstner ----------------

def forstner_locate(gray, cx, cy, half, sigma=1.5):
    h, w = gray.shape
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
    roi = np.float32(gray[y0:y1, x0:x1])
    if roi.shape[0] < 7 or roi.shape[1] < 7:
        return None
    roi = cv2.GaussianBlur(roi, (0, 0), sigma)
    Ix = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float64)
    Ix2, Iy2, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    A = np.array([[Ix2.sum(), Ixy.sum()], [Ixy.sum(), Iy2.sum()]])
    b = np.array([(Ix2 * xs + Ixy * ys).sum(), (Ixy * xs + Iy2 * ys).sum()])
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    p = np.linalg.solve(A, b)
    # 解可能落在視窗外（退化情形），夾回視窗範圍
    p[0] = np.clip(p[0], 0, roi.shape[1] - 1)
    p[1] = np.clip(p[1], 0, roi.shape[0] - 1)
    return float(p[0] + x0), float(p[1] + y0)


# ---------------- 主流程 ----------------

def run(args):
    imgs = find_images(args.img_dir)
    half = args.win // 2
    methods = {"harris": harris_locate, "forstner": forstner_locate}
    errors = {m: [] for m in methods}
    errors_by_type = {m: {} for m in methods}
    per_image = []

    # 若 main_eval 結果存在，可取得每個 cid 的 corner_type 供分層
    cid_type = {}
    if args.main_result and Path(args.main_result).exists():
        mr = load_result(args.main_result)
        for p in mr.get("per_image", []):
            for r in p.get("matched", []):
                cid_type[(p["image"], int(r["cid"]))] = r.get("corner_type", "")

    for ip in imgs:
        gt = load_gt(ip, args.gt_dir)
        if not gt:
            continue
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        rec = {"image": ip.name, "n_gt": len(gt), "results": {}}
        for m, fn in methods.items():
            rows = []
            for cid, g in gt.items():
                if g["visibility"] != "visible":
                    continue   # 局部方法無法處理遮蔽點，僅評可見點
                p = fn(gray, g["x"], g["y"], half)
                if p is None:
                    continue
                err = float(np.hypot(p[0] - g["x"], p[1] - g["y"]))
                rows.append({"cid": cid, "x": p[0], "y": p[1], "err_px": err})
                errors[m].append(err)
                t = cid_type.get((ip.name, cid), "")
                errors_by_type[m].setdefault(t, []).append(err)
            rec["results"][m] = rows
        per_image.append(rec)

    summary = {}
    for m in methods:
        summary[m] = error_stats(errors[m])
        summary[m]["by_type"] = {k: error_stats(v)
                                 for k, v in sorted(errors_by_type[m].items())}

    # 外部方法（如 YOLOPoint）之預測
    if args.extern_pred:
        ext_err = []
        for ip in imgs:
            gt = load_gt(ip, args.gt_dir)
            pp = Path(args.extern_pred) / f"{ip.stem}.pred.json"
            if not gt or not pp.exists():
                continue
            import json
            with open(pp, encoding="utf-8") as f:
                pred = json.load(f)
            pred = pred.get("corners", pred)
            for c in pred:
                g = gt.get(int(c["cid"]))
                if g:
                    ext_err.append(float(np.hypot(c["x"] - g["x"],
                                                  c["y"] - g["y"])))
        summary["extern"] = error_stats(ext_err)

    payload = {"args": vars(args), "summary": summary, "per_image": per_image}
    save_result(args.out, args.name, payload)

    print("\n========== 基準方法（表 5.5）==========")
    for m, st in summary.items():
        if "median" not in st:
            continue
        print(f"{m:>9}: 中位數 {fmt(st['median'])}  平均 {fmt(st['mean'])}  "
              f"P90 {fmt(st['p90'])}  ≤1px {fmt(st.get('succ@1px'), pct=True)}  "
              f"≤2px {fmt(st.get('succ@2px'), pct=True)}  (n={st['n']})")


def build_parser():
    ap = argparse.ArgumentParser(description="Harris / Förstner 基準")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--win", type=int, default=21, help="搜尋視窗邊長 (px)")
    ap.add_argument("--main_result", default="results/main_eval.json",
                    help="主實驗結果（用來取 corner_type 分層，可省略）")
    ap.add_argument("--extern_pred", default=None,
                    help="外部方法預測目錄（如 YOLOPoint），<stem>.pred.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="baselines")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
