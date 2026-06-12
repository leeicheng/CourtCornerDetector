#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_gt.py — GT 體檢：殘差 / 掌性 / 慣例 / 離群點報表 + 疊圖渲染

重要：殘差表抓不到「被歪 H 重蓋」的損壞（那些點對歪 H 自洽、殘差仍小），
唯一可靠判別是疊圖——看橘色格線是否貼住影像中的白線。
綠十字=可見、橘十字=遮蔽、紅十字=對擬合 H 殘差 >3px 的離群點（附 cid:誤差）。

用法：
    python experiments/audit_gt.py <影像資料夾> [--gt_dir <GT資料夾>] [--out gt_audit]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from court_corner.shared.court_model import TEMPLATE_POINTS, GRID_CONNECTIONS  # noqa: E402
from experiments.common import load_gt, find_images                            # noqa: E402
from experiments.diagnose_flip import WORLD_BY_CID, fit_h_gt                    # noqa: E402
from experiments.gt_annotator import row0_is_near                               # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def chirality(H):
    def proj(x, y):
        v = np.asarray(H, float) @ np.array([x, y, 1.0])
        return (v[0] / v[2], v[1] / v[2])
    q0, q1, q2 = proj(0, 0), proj(6.10, 0), proj(0, 13.40)
    cr = ((q1[0] - q0[0]) * (q2[1] - q0[1]) - (q1[1] - q0[1]) * (q2[0] - q0[0]))
    return "-1" if cr < 0 else "+1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_dir")
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--out", default="gt_audit", help="疊圖輸出資料夾")
    ap.add_argument("--outlier_px", type=float, default=3.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in sorted(Path(args.img_dir).iterdir())
            if p.suffix.lower() in IMG_EXTS]
    print(f"{'影像':42s} {'點數':>10s} {'慣例':>10s} {'殘差中位':>8s} "
          f"{'掌性':>4s} {'row0':>5s}  離群點(>{args.outlier_px}px)")
    for ip in imgs:
        gt = load_gt(ip, args.gt_dir)
        if not gt:
            print(f"{ip.name:42s} {'(無GT)':>10s}")
            continue
        gp = (Path(args.gt_dir) if args.gt_dir else ip.parent) / (ip.stem + ".gt.json")
        try:
            conv = json.load(open(gp, encoding="utf-8")).get("convention", "(無)")
        except Exception:
            conv = "?"
        n = len(gt); nv = sum(1 for g in gt.values() if g["visibility"] == "visible")
        H, res_med = fit_h_gt(gt)
        if H is None:
            print(f"{ip.name:42s} {f'{n}({nv}可見)':>10s} {conv:>10s} {'擬合失敗':>8s}")
            continue
        errs = {}
        for cid, g in gt.items():
            w = WORLD_BY_CID.get(cid)
            if w is None:
                continue
            p = cv2.perspectiveTransform(
                w.astype(np.float32).reshape(1, 1, 2), H).reshape(2)
            errs[cid] = float(np.hypot(p[0] - g["x"], p[1] - g["y"]))
        outliers = [(c, round(v, 1)) for c, v in
                    sorted(errs.items(), key=lambda t: -t[1])
                    if v > args.outlier_px][:5]
        near = row0_is_near(H)
        near_s = "近" if near else ("遠" if near is not None else "?")
        print(f"{ip.name:42s} {f'{n}({nv}可見)':>10s} {conv:>10s} "
              f"{res_med:8.2f} {chirality(H):>4s} {near_s:>5s}  "
              + (", ".join(f"cid{c}:{v}px" for c, v in outliers) or "無"))

        # 疊圖
        img = cv2.imread(str(ip))
        if img is None:
            continue
        for i1, i2 in GRID_CONNECTIONS:
            seg = np.array([TEMPLATE_POINTS[i1], TEMPLATE_POINTS[i2]],
                           np.float32).reshape(-1, 1, 2)
            q = cv2.perspectiveTransform(seg, H).reshape(-1, 2)
            if np.all(np.isfinite(q)):
                cv2.line(img, tuple(q[0].astype(int)), tuple(q[1].astype(int)),
                         (255, 200, 80), 1, cv2.LINE_AA)
        r0 = cv2.perspectiveTransform(np.array([[[3.05, 13.40]]], np.float32), H).reshape(2)
        if np.all(np.isfinite(r0)):
            cv2.putText(img, "row0", (int(r0[0]) - 20, int(r0[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
        for cid, g in gt.items():
            e = errs.get(cid, 0.0)
            col = (0, 0, 255) if e > args.outlier_px else \
                  ((0, 200, 0) if g["visibility"] == "visible" else (0, 165, 255))
            x, y = int(round(g["x"])), int(round(g["y"]))
            cv2.drawMarker(img, (x, y), col, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
            if e > args.outlier_px:
                cv2.putText(img, f"{cid}:{e:.1f}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(out / (ip.stem + "_gt.jpg")), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n疊圖輸出至 {out}/ ——重點看橘色格線是否貼住影像白線；"
          "格線歪 = 該張 GT 受損（清對應點重蓋），紅點 = 個別點要微調。")


if __name__ == "__main__":
    main()
