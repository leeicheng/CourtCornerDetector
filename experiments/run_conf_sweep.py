# -*- coding: utf-8 -*-
"""
run_conf_sweep.py — YOLO 信心門檻掃描（對應論文表 5.3b）
================================================================
對每張影像只執行一次 YOLO（以極低門檻 0.05 取得全部候選），之後依各
門檻過濾偵測結果、重跑 Stage 2–4，比較：

  門檻 | 候選數/圖 | 求解成功率 | H 信心分佈 | 線支持度 | 角點誤差中位數 | 求解耗時

「對應正確率」嚴格定義需要交點層級 GT；此處以兩個可行替代指標近似：
  (a) solve_ok 且 line_support_ok（管線自評）
  (b) 若提供角點 GT：配對角點誤差中位數 ≤ args.correct_thresh px 視為該圖
      模板對應正確（錯誤對應會造成大幅誤差，此判準在實務上足夠銳利）。
單球場 / 多球場分列：GT 檔可加 "scene": "single" | "multi"，或以
--multi_list 提供多球場影像清單（每行一個檔名）。

使用：
  python -m experiments.run_conf_sweep \\
      --img_dir data/test_imgs --gt_dir data/gt --weights best.pt \\
      --thresholds 0.8 0.6 0.4 0.25 --out results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (find_images, load_gt, match_by_cid,
                                match_by_cid_oriented,
                                error_stats, save_result, fmt)


def filter_detection(det, thr):
    from court_corner.stages.detection import DetectionResult
    keep = [i for i, c in enumerate(det.confidences) if c >= thr]
    return DetectionResult(
        node_pts=[det.node_pts[i] for i in keep],
        node_types=[det.node_types[i] for i in keep],
        confidences=[det.confidences[i] for i in keep],
        bboxes=[det.bboxes[i] for i in keep],
        class_names=det.class_names,
        raw_class_ids=[det.raw_class_ids[i] for i in keep]
        if det.raw_class_ids else [],
    )


def load_scene_map(args, imgs):
    """image name → 'single' | 'multi'。"""
    multi = set()
    if args.multi_list and Path(args.multi_list).exists():
        multi = {l.strip() for l in open(args.multi_list, encoding="utf-8")
                 if l.strip()}
    scene = {}
    for ip in imgs:
        s = "multi" if ip.name in multi else None
        gp = ip.with_suffix(".gt.json")
        if s is None and args.gt_dir:
            gp2 = Path(args.gt_dir) / f"{ip.stem}.gt.json"
            gp = gp2 if gp2.exists() else gp
        if s is None and gp.exists():
            try:
                with open(gp, encoding="utf-8") as f:
                    s = json.load(f).get("scene")
            except Exception:
                s = None
        scene[ip.name] = s or "single"
    return scene


def run(args):
    from court_corner.pipeline import CourtCornerPipeline

    pipe = CourtCornerPipeline(args.weights, yolo_conf=0.05,
                               corner_conf=args.corner_conf,
                               dark=args.dark, verbose=False)
    imgs = find_images(args.img_dir)
    scene = load_scene_map(args, imgs)
    thresholds = sorted(args.thresholds, reverse=True)

    rows_by_thr = {t: [] for t in thresholds}
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gt = load_gt(ip, args.gt_dir)
        det_full = pipe.detector.detect(img)          # YOLO 只跑一次
        for thr in thresholds:
            det = filter_detection(det_full, thr)
            t0 = time.perf_counter()
            try:
                res = pipe.run_image(img, detection=det)
            except Exception as e:
                res = None
                print(f"[warn] {ip.name} thr={thr}: {e}")
            dt = time.perf_counter() - t0
            row = {"image": ip.name, "scene": scene[ip.name],
                   "n_candidates": len(det), "elapsed_s": dt,
                   "solve_ok": bool(res and res.status == "ok")}
            if res and res.status == "ok":
                d = res.to_dict()
                hm = d["homography"]
                row.update(h_confidence=hm.get("confidence"),
                           line_support=hm.get("line_support"),
                           line_support_ok=hm.get("line_support_ok"),
                           solve_time_s=d["stage_times"].get("solve_H"))
                if gt:
                    m, _oflip = match_by_cid_oriented(d["corners"], gt)
                    errs = [r["err_px"] for r in m]
                    row["err_median"] = float(np.median(errs)) if errs else None
                    row["n_matched"] = len(m)
                    row["mapping_correct"] = (
                        bool(errs) and
                        float(np.median(errs)) <= args.correct_thresh)
            rows_by_thr[thr].append(row)
        print(f"[done] {ip.name}  全候選 {len(det_full)}")

    # 彙整
    table = []
    for thr in thresholds:
        rows = rows_by_thr[thr]
        n = max(len(rows), 1)
        def rate(key, subset=None):
            sel = [r for r in rows if subset is None or r["scene"] == subset]
            if not sel:
                return None
            vals = [r.get(key) for r in sel if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        errs = [r["err_median"] for r in rows if r.get("err_median") is not None]
        table.append({
            "threshold": thr,
            "candidates_per_img": float(np.mean([r["n_candidates"] for r in rows])),
            "solve_ok_rate": float(np.mean([r["solve_ok"] for r in rows])),
            "line_support_mean": rate("line_support"),
            "mapping_correct_single": rate("mapping_correct", "single"),
            "mapping_correct_multi": rate("mapping_correct", "multi"),
            "err_median_mean": float(np.mean(errs)) if errs else None,
            "solve_time_mean_s": rate("solve_time_s"),
        })

    payload = {"args": vars(args), "table": table,
               "rows_by_threshold": {str(k): v for k, v in rows_by_thr.items()}}
    save_result(args.out, args.name, payload)

    print("\n========== 信心門檻掃描（表 5.3b）==========")
    print(f"{'門檻':>6} {'候選/圖':>8} {'求解成功':>8} {'線支持':>7} "
          f"{'對應正確(單)':>11} {'對應正確(多)':>11} {'求解耗時s':>9}")
    for r in table:
        print(f"{r['threshold']:>6} {r['candidates_per_img']:>8.1f} "
              f"{fmt(r['solve_ok_rate'], pct=True):>8} "
              f"{fmt(r['line_support_mean'], 2):>7} "
              f"{fmt(r['mapping_correct_single'], pct=True):>11} "
              f"{fmt(r['mapping_correct_multi'], pct=True):>11} "
              f"{fmt(r['solve_time_mean_s'], 2):>9}")


def build_parser():
    ap = argparse.ArgumentParser(description="YOLO 信心門檻掃描")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.8, 0.6, 0.4, 0.25])
    ap.add_argument("--corner_conf", type=float, default=0.6)
    ap.add_argument("--correct_thresh", type=float, default=5.0,
                    help="角點誤差中位數 ≤ 此值視為模板對應正確")
    ap.add_argument("--multi_list", default=None,
                    help="多球場影像清單（每行一個檔名）")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="conf_sweep")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
