# -*- coding: utf-8 -*-
"""
run_ablation.py — 管線消融實驗（對應論文表 5.9）
================================================================
以同一資料集評估各變體之角點誤差、輸出率與信心分佈：

  full          完整方法
  no_steger     無 Steger 精修（CornerGenerator.h_refine_enabled=False，
                角點直接取 H 投影位置）
  no_jacobian   無 Jacobian 線寬（monkeypatch
                HomographyUtils.compute_line_width_px → 全圖固定線寬，
                以場地中心估一次，不隨透視變化）
  no_quality    無品質過濾（corner_conf=0，全部候選照單全收）
  no_topology   無拓樸交點推估（僅保留與 YOLO 偵測中心距離 ≤ r px 的
                交點，喪失遮蔽 / 未偵測交點之推估；r 預設 25px）

使用：
  python -m experiments.run_ablation \\
      --img_dir data/test_imgs --gt_dir data/gt --weights best.pt --out results
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (find_images, load_gt, match_by_cid,
                                match_by_cid_oriented,
                                error_stats, save_result, fmt,
                                visible_rows)


# ---------------- 變體開關 ----------------

@contextmanager
def patch_fixed_line_width(H_ref_provider):
    """
    無 Jacobian 線寬：把 compute_line_width_px 換成「以目前影像 H 在場地
    中心估一次的固定值」。H_ref_provider() 回傳目前影像的 H。
    """
    from court_corner.shared.homography import HomographyUtils
    orig = HomographyUtils.compute_line_width_px
    orig_J = HomographyUtils.compute_jacobian

    def fixed(J, tangent_m, line_width_m=0.04):
        H = H_ref_provider()
        if H is None:
            return orig(J, tangent_m, line_width_m)
        center = np.array([3.05, 6.70])           # 場地中心 (m)
        Jc = orig_J(H, center)
        sx = float(np.linalg.norm(Jc[:, 0]))
        sy = float(np.linalg.norm(Jc[:, 1]))
        return line_width_m * 0.5 * (sx + sy)     # 等向、全圖一致

    HomographyUtils.compute_line_width_px = staticmethod(fixed)
    try:
        yield
    finally:
        HomographyUtils.compute_line_width_px = orig


def filter_junctions_near_detection(junctions, detection, radius):
    """無拓樸推估：只留靠近 YOLO 偵測中心的交點。"""
    if detection is None or len(detection) == 0:
        return junctions
    pts = np.asarray(detection.node_pts, np.float32)
    kept = []
    for jidx, center in junctions:
        c = np.asarray(center, np.float32)
        if np.min(np.linalg.norm(pts - c, axis=1)) <= radius:
            kept.append((jidx, center))
    return kept


# ---------------- 主流程 ----------------

VARIANTS = ["full", "no_steger", "no_jacobian", "no_quality", "no_topology"]


def run_variant(variant, args, imgs, gt_cache, det_cache):
    from court_corner.pipeline import CourtCornerPipeline

    corner_conf = 0.0 if variant == "no_quality" else args.corner_conf
    pipe = CourtCornerPipeline(args.weights, device=args.device, yolo_conf=args.yolo_conf,
                               corner_conf=corner_conf, dark=args.dark,
                               verbose=False)
    if variant == "no_steger":
        pipe.generator.h_refine_enabled = False

    # no_topology：包裝 generator.generate，先過濾交點
    if variant == "no_topology":
        orig_gen = pipe.generator.generate
        state = {"det": None}

        def gen(gray, H, junctions):
            j = filter_junctions_near_detection(junctions, state["det"],
                                                args.topo_radius)
            return orig_gen(gray, H, j)
        pipe.generator.generate = gen

    # no_jacobian：以 contextmanager 攔截線寬
    h_holder = {"H": None}

    rows, per_image = [], []
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        gt = gt_cache[ip.name]
        det = det_cache.get(ip.name)
        if det is None:
            det = pipe.detector.detect(img)
            det_cache[ip.name] = det
        if variant == "no_topology":
            state["det"] = det

        def _go():
            return pipe.run_image(img, detection=det)

        try:
            if variant == "no_jacobian":
                # 先求 H（Stage2 不受線寬 patch 影響的部分照常），
                # 但 Stage2 內部也會用到線寬 → 以前一次 full H 也可。
                # 簡化：patch 期間 H 由 pipeline 求出後再被使用（Stage3/4），
                # h_holder 在 solve 完成後由 res 補上，第一次呼叫 fallback 原實作。
                with patch_fixed_line_width(lambda: h_holder["H"]):
                    res = _go()
                    h_holder["H"] = res.H
                    if res.status == "ok":      # 以正確 H 重跑一次 Stage3/4
                        res = _go()
            else:
                res = _go()
        except Exception as e:
            per_image.append({"image": ip.name, "status": "error",
                              "message": str(e)})
            continue

        d = res.to_dict()
        rec = {"image": ip.name, "status": d["status"],
               "n_pred": len(d.get("corners", []))}
        if gt and d["status"] == "ok":
            m, _oflip = match_by_cid_oriented(d["corners"], gt)
            rec["n_matched"] = len(m)
            rows.extend(m)
        per_image.append(rec)
    return rows, per_image


def run(args):
    imgs = find_images(args.img_dir)
    gt_cache = {ip.name: load_gt(ip, args.gt_dir) for ip in imgs}
    n_gt = sum(len(g) for g in gt_cache.values())
    det_cache = {}

    variants = args.variants or VARIANTS
    summary, detail = {}, {}
    for v in variants:
        print(f"\n===== 變體：{v} =====")
        rows, per_image = run_variant(v, args, imgs, gt_cache, det_cache)
        vrows = visible_rows(rows)
        errs = [r["err_px"] for r in vrows]
        st = error_stats(errs)
        st["all_points"] = error_stats([r["err_px"] for r in rows])  # 敏感度對照
        st["output_rate"] = (len(rows) / n_gt) if n_gt else None
        # 編號正確率：配對誤差 ≤ 門檻者視為編號正確
        st["cid_correct_rate"] = (float(np.mean(
            [r["err_px"] <= args.cid_thresh for r in vrows]))
            if vrows else None)
        summary[v] = st
        detail[v] = per_image
        print(f"  中位數 {fmt(st['median'])}  P90 {fmt(st['p90'])}  "
              f"輸出率 {fmt(st['output_rate'], pct=True)}  "
              f"編號正確率 {fmt(st['cid_correct_rate'], pct=True)}")

    save_result(args.out, args.name,
                {"args": vars(args), "summary": summary, "per_image": detail})

    print("\n========== 消融彙整（表 5.9）==========")
    print(f"{'變體':>12} {'中位數':>8} {'P90':>8} {'輸出率':>8} {'編號正確':>8}")
    for v in variants:
        st = summary[v]
        print(f"{v:>12} {fmt(st['median']):>8} {fmt(st['p90']):>8} "
              f"{fmt(st['output_rate'], pct=True):>8} "
              f"{fmt(st['cid_correct_rate'], pct=True):>8}")


def build_parser():
    ap = argparse.ArgumentParser(description="管線消融")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--device", default=None,
                    help="YOLO 推論裝置（0 / cuda:0 / cpu；預設自動）")
    ap.add_argument("--yolo_conf", type=float, default=0.4)
    ap.add_argument("--corner_conf", type=float, default=0.6)
    ap.add_argument("--variants", nargs="*", default=None,
                    help=f"預設全部：{VARIANTS}")
    ap.add_argument("--topo_radius", type=float, default=25.0)
    ap.add_argument("--cid_thresh", type=float, default=5.0)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="ablation")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
