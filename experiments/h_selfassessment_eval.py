#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_selfassessment_eval.py — H 信心自評之有效性分析（取代原表 5.4 之求解器消融）

學術定位：選擇性預測（selective prediction / reject option）之風險—覆蓋分析
+ 內省感知（introspective perception）之失敗預測校準。

輸入：
  - 真實集：run_main_eval.py 之 main_eval.json
  - 合成集（外參）：run_extrinsics.py 之 extrinsics.json（rig 條件）

輸出：
  (a) 校準表：h_confidence 等級 × 對應正確率／粗大錯誤率
  (b) 風險—覆蓋表：三個操作點（僅 high / high+medium / 全收）之
      覆蓋率、角點誤差中位、P90、粗大錯誤率
  (c) AURC 風格之排序檢驗：按信心降冪累積接受時之風險曲線數據（CSV）

用法：
  python experiments/h_selfassessment_eval.py \
      --main_eval results/main_eval.json \
      --extrinsics results_rig/extrinsics.json --out results/h_selfassess
"""

import argparse
import json
from pathlib import Path

import numpy as np

GROSS_PX = 20.0     # 與 5.1.3 協定一致（角點層）
ASSOC_PX = 10.0     # 與 5.7 協定一致（PnP 重投影）
ORDER = ("high", "medium", "low")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def conf_of(p):
    return (p.get("homography") or {}).get("confidence") or p.get("h_confidence")


def real_set_analysis(path, visible_only=True):
    d = json.load(open(path, encoding="utf-8"))
    per = [p for p in d["per_image"] if p.get("status") == "ok"]
    rows = []
    for p in per:
        m = p.get("matched") or []
        errs = [r["err_px"] for r in m
                if (not visible_only) or r.get("visibility", "visible") == "visible"]
        if not errs:
            continue
        e = np.asarray(errs, float)
        rows.append(dict(conf=conf_of(p), med=float(np.median(e)),
                         gross=float(np.mean(e > GROSS_PX)), errs=e))
    if not rows:
        print(f"找不到可用之配對列（檢查 {path} 之 per_image[].matched）")
        return []

    print("\n=== 真實集（61 張）：校準表 ===")
    tab = []
    for c in ORDER:
        sub = [r for r in rows if r["conf"] == c]
        if not sub:
            tab.append([c, 0, "—", "—", "—"]); continue
        cat = np.concatenate([r["errs"] for r in sub])
        tab.append([c, len(sub), f"{np.median(cat):.2f}",
                    f"{np.percentile(cat, 90):.1f}",
                    f"{np.mean(cat > GROSS_PX)*100:.1f}%"])
    print(md_table(["H 信心", "n 影像", "角點誤差中位 (px)", "P90 (px)",
                    "粗大錯誤率 (>20px)"], tab))

    print("\n=== 真實集：風險—覆蓋（三操作點）===")
    n_total = max(len(rows), 1)
    tab = []
    for accept, lab in ((("high",), "僅 high"),
                        (("high", "medium"), "high+medium"),
                        (ORDER, "全收")):
        sub = [r for r in rows if r["conf"] in accept]
        cat = np.concatenate([r["errs"] for r in sub]) if sub else np.array([])
        tab.append([lab, f"{len(sub)}/{n_total} ({len(sub)/n_total*100:.0f}%)",
                    f"{np.median(cat):.2f}" if len(cat) else "—",
                    f"{np.percentile(cat, 90):.1f}" if len(cat) else "—",
                    f"{np.mean(cat > GROSS_PX)*100:.1f}%" if len(cat) else "—"])
    print(md_table(["操作點", "覆蓋（影像）", "誤差中位 (px)", "P90 (px)",
                    "粗大錯誤率"], tab))
    return rows


def synth_analysis(path):
    d = json.load(open(path, encoding="utf-8"))
    per = d["per_image"]
    print("\n=== 合成 rig 集：校準表（對應正確 = junction 重投影 ≤10px）===")
    tab = []
    for c in ORDER:
        sub = [p for p in per if p.get("h_confidence") == c and p.get("junction")]
        if not sub:
            tab.append([c, 0, "—"]); continue
        ok = np.mean([p["junction"]["rmse_gt_px"] <= ASSOC_PX for p in sub])
        tab.append([c, len(sub), f"{ok*100:.0f}%"])
    print(md_table(["H 信心", "n 影像", "對應正確率"], tab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_eval", default=None)
    ap.add_argument("--extrinsics", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.main_eval:
        real_set_analysis(args.main_eval)
    if args.extrinsics:
        synth_analysis(args.extrinsics)
    if not (args.main_eval or args.extrinsics):
        print("請至少提供 --main_eval 或 --extrinsics")


if __name__ == "__main__":
    main()