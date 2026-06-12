# -*- coding: utf-8 -*-
"""
run_quality_discrim.py — 影像證據判別性與組成消融（對應論文表 5.6 / 5.7）
================================================================
依論文 5.5.1 / 5.5.2 設計：

正樣本：GT 角點位置。
負樣本三類：
  (a) bg      隨機背景點（距投影球場線與所有角點皆 ≥ d_min）
  (b) online  線上硬負樣本（沿 H 投影之球場線取樣、距任何角點 ≥ d_min）
  (c) shifted 偏移正樣本（GT 沿隨機方向偏移 shift_px）

證據方法兩種（同 ANMS、同評分公式，僅證據圖與峰過濾不同）：
  - gradgeo  本研究梯度幾何證據（vertex_quality.run_harris_steger_analysis）
  - legacy   原 Harris–Steger 差分（正規化 Harris-R − 正規化脊強度；
             峰過濾改用脊鄰近性）

組成消融（僅 gradgeo）：full / no_en_gate / no_nondeg / no_Rn / no_convfilter

H 來源：優先讀取 --main_result（run_main_eval 輸出，內含每張影像之 H），
找不到該影像之 H 時跳過該影像。

使用：
  python -m experiments.run_quality_discrim \\
      --img_dir data/test_imgs --gt_dir data/gt \\
      --main_result results/main_eval.json --out results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (find_images, load_gt, auc_score, roc_curve,
                                save_result, fmt, load_result)

from court_corner.vertex import vertex_quality as vq
from court_corner.shared.court_model import TEMPLATE_POINTS, GRID_CONNECTIONS

TAU_D = 5.0          # dist_score 的 τ（與 config.VQ_PEAK_RADIUS_PX 一致）
W_DIST, W_HEAT = 0.5, 0.5


# ================= 取樣 =================

def project(H, pts_m):
    p = np.asarray(pts_m, np.float64)
    ones = np.ones((p.shape[0], 1))
    q = (H @ np.hstack([p, ones]).T).T
    return q[:, :2] / q[:, 2:3]


def sample_points(H, gt, img_shape, n_each, d_min, shift_px, rng):
    """回傳 dict: {"pos": Nx2, "online": Nx2, "bg": Nx2, "shifted": Nx2}"""
    h, w = img_shape[:2]
    gt_pts = np.array([[g["x"], g["y"]] for g in gt.values()], np.float64) \
        if gt else np.zeros((0, 2))

    def far_from_gt(p):
        if gt_pts.size == 0:
            return True
        return np.min(np.linalg.norm(gt_pts - p, axis=1)) >= d_min

    def in_img(p, margin=20):
        return margin <= p[0] < w - margin and margin <= p[1] < h - margin

    # 線上硬負樣本：沿模板線段取樣後投影
    online = []
    for i1, i2 in GRID_CONNECTIONS:
        a, b = TEMPLATE_POINTS[i1], TEMPLATE_POINTS[i2]
        for t in np.linspace(0.18, 0.82, 9):
            online.append(a + t * (b - a))
    online = project(H, np.array(online))
    online = [p for p in online if in_img(p) and far_from_gt(p)]
    rng.shuffle(online)
    online = np.array(online[:n_each]) if online else np.zeros((0, 2))

    # 隨機背景：距線與角點皆遠
    line_pts = project(H, np.array(
        [TEMPLATE_POINTS[i1] + t * (TEMPLATE_POINTS[i2] - TEMPLATE_POINTS[i1])
         for i1, i2 in GRID_CONNECTIONS for t in np.linspace(0, 1, 12)]))
    bg = []
    tries = 0
    while len(bg) < n_each and tries < n_each * 60:
        tries += 1
        p = np.array([rng.uniform(20, w - 20), rng.uniform(20, h - 20)])
        if far_from_gt(p) and \
           np.min(np.linalg.norm(line_pts - p, axis=1)) >= d_min:
            bg.append(p)
    bg = np.array(bg) if bg else np.zeros((0, 2))

    pos = gt_pts.copy()
    ang = rng.uniform(0, 2 * np.pi, size=len(pos))
    shifted = pos + shift_px * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    shifted = np.array([p for p in shifted if in_img(p)]) \
        if len(shifted) else np.zeros((0, 2))
    pos = np.array([p for p in pos if in_img(p)]) \
        if len(pos) else np.zeros((0, 2))
    return {"pos": pos, "online": online, "bg": bg, "shifted": shifted}


# ================= 證據評分 =================

def composite_from(diff, peaks, vx, vy):
    """同論文評分公式：0.5·exp(−d/τ) + 0.5·clip(diff, 0, 1)。"""
    h, w = diff.shape
    vc = int(np.clip(round(vx), 0, w - 1))
    vr = int(np.clip(round(vy), 0, h - 1))
    heat = float(np.clip(diff[vr, vc], 0, 1))
    if len(peaks):
        pk = np.asarray(peaks, np.float64)
        d = float(np.min(np.hypot(pk[:, 0] - vy, pk[:, 1] - vx)))
        dist_s = float(np.exp(-d / TAU_D))
    else:
        dist_s = 0.0
    return W_DIST * dist_s + W_HEAT * heat


def score_gradgeo(roi):
    ana = vq.run_harris_steger_analysis(roi)
    return ana["diff"], ana["peaks"], ana


def score_legacy(roi, ana=None):
    """原 Harris–Steger 差分：norm(Harris R) − norm(脊強度)；峰過濾用脊鄰近。"""
    ana = ana or vq.run_harris_steger_analysis(roi)
    R = ana["harris_R"]
    S = ana["steger_strength"]
    Rn = R / (float(np.percentile(R, 99.5)) + 1e-9)
    Sn = S / (float(np.percentile(S, 99.5)) + 1e-9)
    diff = np.clip(Rn, 0, 1) - np.clip(Sn, 0, 1)
    ys, xs, vals = vq._topk_anms(diff.astype(np.float32), top_k=8, c=0.9,
                                 candidate_pool=200, loose_nms_radius=3)
    # 脊鄰近過濾（舊版邏輯）：峰須距 significant ridge ≤ 6px
    sig = ana["sig_ridge_mask"]
    if sig.any() and len(ys):
        dist_map = distance_transform_edt(~sig)
        keep = dist_map[ys, xs] <= 6.0
        ys, xs = ys[keep], xs[keep]
    peaks = list(zip(ys.tolist(), xs.tolist())) if len(ys) else []
    return diff.astype(np.float32), peaks


def score_gradgeo_variant(roi, variant):
    """組成消融：重算 diff 與峰（不動 ANMS/評分公式）。"""
    f32 = np.clip(roi, 0, 255).astype(np.float32)
    Ix, Iy = vq._gradients(f32, 1.5)
    Sxx, Syy, Sxy = vq._structure_tensor_field(Ix, Iy, 1.5)
    tr_S = Sxx + Syy
    En = (tr_S / (float(np.percentile(tr_S, 99)) + 1e-9)).astype(np.float32)
    disc = np.sqrt(np.maximum((Sxx - Syy) ** 2 + 4 * Sxy * Sxy, 0))
    coh = (disc / (tr_S + 1e-9)).astype(np.float32)
    Rn = vq._dense_conv_resid(Ix, Iy, int(round(7.0 * 1.5)))

    en_gate = np.clip(En / vq._EN_GATE, 0, 1)
    nondeg = np.clip(2.0 * (1.0 - coh), 0, 1)
    conv = (vq._RN_SPLIT - Rn) / vq._RN_SPLIT
    if variant == "no_en_gate":
        en_gate = np.ones_like(en_gate)
    elif variant == "no_nondeg":
        nondeg = np.ones_like(nondeg)
    elif variant == "no_Rn":
        conv = np.ones_like(conv)          # 僅能量×非退化構成證據
    diff = np.clip(en_gate * nondeg * conv, -1, 1).astype(np.float32)

    ys, xs, vals = vq._topk_anms(diff, top_k=8, c=0.9,
                                 candidate_pool=200, loose_nms_radius=3)
    if variant != "no_convfilter" and len(ys):
        win_r = max(3, int(round(3.0 * 1.5)))
        ys, xs, vals, _, _ = vq._convergence_filter(
            ys, xs, vals, Ix, Iy, max_dist=6.0, min_support=10, win_r=win_r)
    peaks = list(zip(np.asarray(ys).tolist(), np.asarray(xs).tolist())) \
        if len(ys) else []
    return diff, peaks


# ================= 主流程 =================

GRADGEO_VARIANTS = ["full", "no_en_gate", "no_nondeg", "no_Rn", "no_convfilter"]


def run(args):
    rng = np.random.default_rng(args.seed)
    imgs = find_images(args.img_dir)
    H_by_image = {}
    if args.main_result and Path(args.main_result).exists():
        mr = load_result(args.main_result)
        for p in mr.get("per_image", []):
            if p.get("H"):
                H_by_image[p["image"]] = np.asarray(p["H"], np.float64)

    half = args.roi_half
    scores = {"gradgeo": {}, "legacy": {}}
    var_scores = {v: {} for v in GRADGEO_VARIANTS}
    times = {"gradgeo": [], "legacy": []}
    n_img_used = 0

    for ip in imgs:
        gt = load_gt(ip, args.gt_dir)
        H = H_by_image.get(ip.name)
        if not gt or H is None:
            continue
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        n_img_used += 1
        samples = sample_points(H, gt, gray.shape, args.n_neg_per_img,
                                args.d_min, args.shift_px, rng)
        for cat, pts in samples.items():
            for p in pts:
                x0 = int(round(p[0])) - half
                y0 = int(round(p[1])) - half
                roi = gray[max(0, y0):y0 + 2 * half + 1,
                           max(0, x0):x0 + 2 * half + 1]
                if roi.shape[0] < 15 or roi.shape[1] < 15:
                    continue
                vx, vy = p[0] - max(0, x0), p[1] - max(0, y0)

                t0 = time.perf_counter()
                diff_g, peaks_g, ana = score_gradgeo(roi)
                times["gradgeo"].append(time.perf_counter() - t0)
                sc_g = composite_from(diff_g, peaks_g, vx, vy)
                scores["gradgeo"].setdefault(cat, []).append(sc_g)

                t0 = time.perf_counter()
                diff_l, peaks_l = score_legacy(roi, ana)
                times["legacy"].append(time.perf_counter() - t0)
                sc_l = composite_from(diff_l, peaks_l, vx, vy)
                scores["legacy"].setdefault(cat, []).append(sc_l)

                if args.with_component_ablation:
                    for v in GRADGEO_VARIANTS:
                        dv, pv = score_gradgeo_variant(roi, v)
                        var_scores[v].setdefault(cat, []).append(
                            composite_from(dv, pv, vx, vy))
        print(f"[done] {ip.name}")

    # ---- 彙整：表 5.6 ----
    def aucs(sc):
        pos = sc.get("pos", [])
        return {
            "auc_vs_bg": auc_score(pos, sc.get("bg", [])),
            "auc_vs_online": auc_score(pos, sc.get("online", [])),
            "auc_vs_shifted": auc_score(pos, sc.get("shifted", [])),
            "pos_median": float(np.median(pos)) if pos else None,
            "n_pos": len(pos),
        }

    table56 = {}
    for m in ("legacy", "gradgeo"):
        table56[m] = aucs(scores[m])
        table56[m]["time_ms_per_point"] = (
            1000 * float(np.mean(times[m])) if times[m] else None)
        table56[m]["roc_vs_online"] = roc_curve(
            scores[m].get("pos", []), scores[m].get("online", []))

    table57 = {v: aucs(var_scores[v]) for v in GRADGEO_VARIANTS} \
        if args.with_component_ablation else {}

    payload = {"args": vars(args), "n_images": n_img_used,
               "table_5_6": table56, "table_5_7": table57,
               "raw_scores": {m: {k: list(map(float, v))
                                  for k, v in sc.items()}
                              for m, sc in scores.items()}}
    save_result(args.out, args.name, payload)

    print(f"\n========== 判別性（表 5.6，{n_img_used} 張影像）==========")
    for m in ("legacy", "gradgeo"):
        t = table56[m]
        print(f"{m:>8}: AUC(線上) {fmt(t['auc_vs_online'])}  "
              f"AUC(偏移) {fmt(t['auc_vs_shifted'])}  "
              f"AUC(背景) {fmt(t['auc_vs_bg'])}  "
              f"pos中位數 {fmt(t['pos_median'])}  "
              f"耗時 {fmt(t['time_ms_per_point'], 2)} ms")
    if table57:
        print("\n========== 組成消融（表 5.7）==========")
        for v in GRADGEO_VARIANTS:
            t = table57[v]
            print(f"{v:>14}: AUC(背景) {fmt(t['auc_vs_bg'])}  "
                  f"AUC(線上) {fmt(t['auc_vs_online'])}")


def build_parser():
    ap = argparse.ArgumentParser(description="影像證據判別性實驗")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--main_result", default="results/main_eval.json")
    ap.add_argument("--roi_half", type=int, default=16)
    ap.add_argument("--n_neg_per_img", type=int, default=60)
    ap.add_argument("--d_min", type=float, default=12.0)
    ap.add_argument("--shift_px", type=float, default=3.0)
    ap.add_argument("--with_component_ablation", action="store_true",
                    help="同時跑表 5.7 組成消融（較慢）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="quality_discrim")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
