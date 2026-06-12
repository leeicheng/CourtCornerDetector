# -*- coding: utf-8 -*-
"""
run_main_eval.py — 主實驗（對應論文表 5.5「本方法」、表 5.8、表 5.11）
================================================================
對資料夾內所有影像執行完整四階段管線，與 GT 角點以 cid 配對後輸出：
  - 每角點紀錄（cid, x, y, conf, tier, type, source, err_px, visibility）
  - 角點定位誤差統計（整體 / 依交點類型 / 依 tier / 依可見性）
  - 信心分組之定位誤差（表 5.8）＋ Spearman 相關
  - 各階段執行時間（表 5.11）
  - 每張影像之 H 求解摘要（供 5.3 / 失敗案例分析）

使用：
  python -m experiments.run_main_eval \\
      --img_dir data/test_imgs --gt_dir data/gt --weights best.pt \\
      --yolo_conf 0.4 --corner_conf 0.0 --out results

備註：corner_conf 預設 0.0（輸出全部候選），由分析端再依信心切分，
      這樣同一次跑的結果可同時餵表 5.5（門檻後）與表 5.8（分組）。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (find_images, load_gt, match_by_cid,
                                match_by_cid_oriented, rot180_cid,
                                error_stats, spearman, save_result, fmt)

CONF_BINS = [(0.8, 1.01), (0.6, 0.8), (0.4, 0.6), (0.0, 0.4)]


def run(args):
    from court_corner.pipeline import CourtCornerPipeline

    pipe = CourtCornerPipeline(
        args.weights, yolo_conf=args.yolo_conf, corner_conf=args.corner_conf,
        dark=args.dark, min_line_support=args.min_line_support,
        verbose=not args.quiet)

    imgs = find_images(args.img_dir)
    if not imgs:
        print(f"找不到影像：{args.img_dir}")
        return

    viz_dir = None
    if args.save_viz:
        viz_dir = Path(args.out) / "viz_main_eval"
        viz_dir.mkdir(parents=True, exist_ok=True)

    per_image, all_rows = [], []
    n_gt_total = 0

    for ip in imgs:
        gt = load_gt(ip, args.gt_dir)
        n_gt_total += len(gt)
        rec = {"image": ip.name, "n_gt": len(gt)}
        try:
            res = pipe.run(str(ip))
        except Exception as e:
            rec.update(status="error", message=f"{type(e).__name__}: {e}")
            traceback.print_exc()
            per_image.append(rec)
            continue

        d = res.to_dict()
        rec.update(status=d["status"], message=d["message"],
                   n_detections=d["n_detections"],
                   homography=d.get("homography", {}),
                   stage_times=d.get("stage_times", {}),
                   elapsed_s=d.get("elapsed_s"),
                   H=d.get("H"))
        pred = d.get("corners", [])
        rec["n_pred"] = len(pred)
        rec["corners"] = pred

        if gt:
            rows, oflip = match_by_cid_oriented(pred, gt, canon=not args.no_orient_canon)
            rec["orientation_flipped"] = oflip
            hconf = (rec.get("homography") or {}).get("confidence", "")
            for r in rows:
                r["h_confidence"] = hconf
            for r in rows:
                r["image"] = ip.name
            rec["n_matched"] = len(rows)
            matched = []
            for r in rows:
                m = {k: r[k] for k in
                     ("cid", "x", "y", "conf", "tier", "err_px",
                      "gt_x", "gt_y", "visibility") if k in r}
                m["corner_type"] = r.get("corner_type", "")
                m["source"] = r.get("source", "")
                matched.append(m)
            rec["matched"] = matched
            all_rows.extend(rows)
            # 漏報：GT 有但輸出沒有
            pred_cids = {(rot180_cid(c["cid"]) if rec.get("orientation_flipped")
                          else int(c["cid"])) for c in pred}
            rec["missed_cids"] = sorted(set(gt) - pred_cids)
        per_image.append(rec)

        if viz_dir is not None and d["status"] == "ok":
            _save_viz(ip, pred, gt, viz_dir)

    summary = summarize(all_rows, per_image, n_gt_total, args)
    summary["orientation_flipped_images"] = sorted(
        r["image"] for r in per_image if r.get("orientation_flipped"))
    summary["n_orientation_flipped"] = len(summary["orientation_flipped_images"])
    payload = {"args": vars(args), "summary": summary, "per_image": per_image}
    save_result(args.out, args.name, payload)
    print_summary(summary)


def summarize(rows, per_image, n_gt_total, args):
    errs = [r["err_px"] for r in rows]
    s = {"overall": error_stats(errs)}

    # 輸出率 / 配對率
    n_pred = sum(p.get("n_pred", 0) for p in per_image)
    s["counts"] = {"n_images": len(per_image),
                   "n_ok": sum(1 for p in per_image if p.get("status") == "ok"),
                   "n_gt": n_gt_total, "n_pred": n_pred,
                   "n_matched": len(rows),
                   "output_rate": (len(rows) / n_gt_total) if n_gt_total else None}

    # 依類型 / tier / 可見性分層
    for key, name in (("corner_type", "by_type"), ("tier", "by_tier"),
                      ("visibility", "by_visibility")):
        groups = {}
        for r in rows:
            groups.setdefault(str(r.get(key, "")), []).append(r["err_px"])
        s[name] = {k: error_stats(v) for k, v in sorted(groups.items())}

    # 表 5.8：信心分組
    bins = []
    n_all = max(len(rows), 1)
    for lo, hi in CONF_BINS:
        sel = [r["err_px"] for r in rows if lo <= r.get("conf", 0) < hi]
        st = error_stats(sel)
        bins.append({"range": f"[{lo}, {hi})", "fraction": len(sel) / n_all, **st})
    s["conf_bins"] = bins
    s["spearman_conf_err"] = spearman([r.get("conf", 0) for r in rows],
                                      [-r["err_px"] for r in rows])

    # ── 兩段式指標（論文主表建議報法）──
    # (1) 影像級 H 信心閘控：線上可用（不需 GT）的接受策略
    gate = {}
    for lab, accept in (("high+medium", ("high", "medium")), ("high_only", ("high",))):
        sel = [r["err_px"] for r in rows if r.get("h_confidence") in accept]
        n_img = sum(1 for p in per_image
                    if (p.get("homography") or {}).get("confidence") in accept)
        gate[lab] = {"n_images_accepted": n_img, **error_stats(sel)}
    sel_low = [r["err_px"] for r in rows if r.get("h_confidence") == "low"]
    gate["rejected_low"] = {"n_images": sum(
        1 for p in per_image
        if (p.get("homography") or {}).get("confidence") == "low"),
        **error_stats(sel_low)}
    s["h_conf_gating"] = gate

    # (2) 角點級拆分：粗大錯誤（>gross_px，屬配對/標號層）vs 定位誤差（其餘）
    gp = float(getattr(args, "gross_px", 20.0))
    fine = [e for e in errs if e <= gp]
    s["association"] = {
        "gross_px_threshold": gp,
        "gross_rate": (sum(1 for e in errs if e > gp) / len(errs)) if errs else None,
        "localization_on_correct": error_stats(fine),
    }

    # 表 5.11：各階段時間
    stages = {}
    for p in per_image:
        for k, v in (p.get("stage_times") or {}).items():
            stages.setdefault(k, []).append(v)
    tot = sum(np.mean(v) for v in stages.values()) if stages else 0
    s["stage_times_ms"] = {
        k: {"mean_ms": 1000 * float(np.mean(v)),
            "share": float(np.mean(v)) / tot if tot else None}
        for k, v in stages.items()}

    # H 求解摘要（供 5.3）
    confs = [p.get("homography", {}).get("confidence") for p in per_image
             if p.get("status") == "ok"]
    s["h_confidence_counts"] = {c: confs.count(c) for c in ("high", "medium", "low")}
    s["mean_line_support"] = float(np.mean(
        [p["homography"].get("line_support", np.nan) for p in per_image
         if p.get("status") == "ok"])) if confs else None
    return s


def print_summary(s):
    o = s["overall"]
    print("\n========== 主實驗摘要 ==========")
    c = s["counts"]
    print(f"影像 {c['n_images']}（成功 {c['n_ok']}），GT {c['n_gt']}，"
          f"輸出 {c['n_pred']}，配對 {c['n_matched']}，"
          f"輸出率 {fmt(c['output_rate'], pct=True)}")
    print(f"誤差 px：中位數 {fmt(o['median'])}  平均 {fmt(o['mean'])}  "
          f"P90 {fmt(o['p90'])}  ≤1px {fmt(o.get('succ@1px'), pct=True)}  "
          f"≤2px {fmt(o.get('succ@2px'), pct=True)}")
    print(f"Spearman(conf, -err) = {fmt(s['spearman_conf_err'])}")
    print("信心分組（表 5.8）：")
    for b in s["conf_bins"]:
        print(f"  {b['range']:>12}  佔比 {fmt(b['fraction'], pct=True):>6}  "
              f"中位數 {fmt(b['median'])}  P90 {fmt(b['p90'])}")
    g = s.get("h_conf_gating", {})
    if g:
        hm = g.get("high+medium", {}); lo = g.get("rejected_low", {})
        print(f"H信心閘控（high+medium 接受）：影像 {hm.get('n_images_accepted')} 張  "
              f"誤差中位 {fmt(hm.get('median'))}  P90 {fmt(hm.get('p90'))}  "
              f"｜拒收(low) {lo.get('n_images')} 張")
    a = s.get("association", {})
    if a:
        loc = a.get("localization_on_correct", {})
        print(f"兩段式：粗大錯誤率(> {a.get('gross_px_threshold')}px) "
              f"{fmt(a.get('gross_rate'), pct=True)}；"
              f"正確子集定位 中位 {fmt(loc.get('median'))}  P90 {fmt(loc.get('p90'))}  "
              f"≤2px {fmt(loc.get('succ@2px'), pct=True)}")
    print("各階段耗時（表 5.11）：")
    for k, v in s["stage_times_ms"].items():
        print(f"  {k:>10}: {v['mean_ms']:8.1f} ms  ({fmt(v['share'], pct=True)})")


def _save_viz(img_path, pred, gt, viz_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for g in gt.values():
        cv2.drawMarker(img, (int(round(g["x"])), int(round(g["y"]))),
                       (255, 128, 0), cv2.MARKER_CROSS, 14, 2)
    for c in pred:
        col = (0, 200, 0) if c["conf"] >= 0.8 else \
              (0, 220, 220) if c["conf"] >= 0.65 else (0, 128, 255)
        cv2.circle(img, (int(round(c["x"])), int(round(c["y"]))), 4, col, 2)
    cv2.imwrite(str(viz_dir / f"{Path(img_path).stem}_eval.png"), img)


def build_parser():
    ap = argparse.ArgumentParser(description="主實驗：完整管線 vs GT")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.4)
    ap.add_argument("--corner_conf", type=float, default=0.0,
                    help="預設 0：輸出全部候選，分析端再切")
    ap.add_argument("--min_line_support", type=float, default=0.45)
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--gross_px", type=float, default=20.0,
                    help="角點級粗大錯誤門檻 (px)：超過視為配對/標號失敗")
    ap.add_argument("--no_orient_canon", action="store_true",
                    help="停用 180° 方向歸一化配對（嚴格方向）")
    ap.add_argument("--name", default="main_eval")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
