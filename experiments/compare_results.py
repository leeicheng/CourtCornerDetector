#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_results.py — 兩份 main_eval.json 的嚴格 A/B 對照

用途：驗證速度優化（或任何修改）前後，準度與行為是否一致。
比對內容：
  1. 摘要指標（配對數、誤差中位/平均/P90、成功率、信心分組）
  2. 逐影像：配對誤差中位數、配對數、H 是否一致（30 模板點投影差）、
     orientation_flipped 是否相同
  3. 各階段耗時（平均 + 排除首張暖機）

用法：
    python experiments/compare_results.py results/main_eval_old.json results/main_eval_new.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from experiments.common import fmt, md_table  # noqa: E402

GRID_M = [(x, y) for y in (0.0, 4.72, 8.68, 13.40) for x in (0.0, 3.05, 6.10)]


def _proj(H, X):
    v = np.asarray(H, float) @ np.array([X[0], X[1], 1.0])
    return (v[0] / v[2], v[1] / v[2]) if abs(v[2]) > 1e-12 else (np.nan, np.nan)


def per_image_stats(rec):
    errs = [m["err_px"] for m in rec.get("matched", [])]
    return {
        "med": float(np.median(errs)) if errs else float("nan"),
        "n": len(errs),
        "H": rec.get("H"),
        "flip": bool(rec.get("orientation_flipped", False)),
        "times": rec.get("stage_times", {}),
        "status": rec.get("status"),
    }


def h_diff_px(Ha, Hb):
    if Ha is None or Hb is None:
        return float("nan")
    d = [np.hypot(*(np.subtract(_proj(Ha, g), _proj(Hb, g)))) for g in GRID_M]
    return float(np.max(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--tol_px", type=float, default=0.05,
                    help="H 投影差視為一致的容忍 (px)")
    args = ap.parse_args()
    A = json.load(open(args.old, encoding="utf-8"))
    B = json.load(open(args.new, encoding="utf-8"))
    ra = {r["image"]: r for r in A.get("per_image", [])}
    rb = {r["image"]: r for r in B.get("per_image", [])}

    rows, n_same_H, n_flip_diff, n_med_diff = [], 0, 0, 0
    for name in sorted(set(ra) | set(rb)):
        a, b = ra.get(name), rb.get(name)
        if a is None or b is None:
            rows.append([name, "—", "—", "—", "缺一邊"]); continue
        sa, sb = per_image_stats(a), per_image_stats(b)
        hd = h_diff_px(sa["H"], sb["H"])
        same_h = np.isfinite(hd) and hd <= args.tol_px
        n_same_H += int(same_h)
        dmed = sb["med"] - sa["med"]
        if np.isfinite(dmed) and abs(dmed) > 0.05:
            n_med_diff += 1
        if sa["flip"] != sb["flip"]:
            n_flip_diff += 1
        note = []
        if not same_h:
            note.append(f"H差{fmt(hd, 2)}px")
        if sa["flip"] != sb["flip"]:
            note.append("flip不同")
        rows.append([name, fmt(sa["med"], 3), fmt(sb["med"], 3),
                     fmt(dmed, 3), " ".join(note) or "一致"])

    print(md_table(["影像", "舊·誤差中位", "新·誤差中位", "Δ", "備註"], rows))

    def summ(d, key, default=None):
        return d.get("summary", {}).get(key, default)

    print("\n摘要對照：")
    sa, sb = A.get("summary", {}), B.get("summary", {})
    ov_a, ov_b = sa.get("overall", {}), sb.get("overall", {})
    for k in ("median", "mean", "p90"):
        va, vb = ov_a.get(k), ov_b.get(k)
        if va is not None and vb is not None:
            print(f"  誤差 {k}: {fmt(va,3)} → {fmt(vb,3)}  (Δ {fmt(vb-va,3)})")

    # 耗時（平均 / 排除首張）
    def stage_means(data, skip_first):
        per = data.get("per_image", [])
        per = per[1:] if skip_first and len(per) > 1 else per
        acc = {}
        for r in per:
            for k, v in (r.get("stage_times") or {}).items():
                acc.setdefault(k, []).append(float(v))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    for skip in (False, True):
        ta, tb = stage_means(A, skip), stage_means(B, skip)
        tag = "（排除首張暖機）" if skip else "（含首張）"
        keys = sorted(set(ta) | set(tb))
        if not keys:
            continue
        print(f"\n各階段平均耗時 {tag}：")
        for k in keys:
            va, vb = ta.get(k, float("nan")), tb.get(k, float("nan"))
            sp = (va / vb) if vb and np.isfinite(va) and np.isfinite(vb) and vb > 0 else float("nan")
            print(f"  {k:>10}: {va*1000 if va<100 else va:8.1f} → "
                  f"{vb*1000 if vb<100 else vb:8.1f}  ({fmt(sp,2)}×)")

    n = len(rows)
    print(f"\n判定：H 一致 {n_same_H}/{n}、誤差中位變動>0.05px 的影像 {n_med_diff}、"
          f"方向旗標不同 {n_flip_diff}")
    if n_same_H == n and n_flip_diff == 0:
        print("→ 兩次結果逐影像一致，優化未影響準度。")
    else:
        print("→ 有差異的影像請用 results_viewer 逐圖檢視再決定是否採用。")


if __name__ == "__main__":
    main()
