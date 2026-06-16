# -*- coding: utf-8 -*-
"""
make_report.py — 彙整實驗結果 → report.md + figures/
================================================================
讀取 results/ 下各實驗 JSON，輸出：
  - report.md：依論文表號（5.3b / 5.5 / 5.6 / 5.7 / 5.8 / 5.9 / 5.11）
    排版好的 Markdown 表格，可直接貼回論文。
  - figures/：誤差 CDF、信心—誤差曲線、ROC、各階段耗時長條圖（PNG, 300dpi）。

使用：
  python -m experiments.make_report --results results --out results/report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.common import load_result, fmt, md_table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from matplotlib import font_manager
    for f in ("Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC",
              "Heiti TC", "Arial Unicode MS"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def maybe(results_dir, name):
    p = Path(results_dir) / f"{name}.json"
    return load_result(p) if p.exists() else None


def sec_main(main, fig_dir, lines):
    s = main["summary"]
    lines.append("## 表 5.5　角點定位誤差（本方法）\n")
    o = s["overall"]
    lines.append(md_table(
        ["方法", "中位數", "平均", "P90", "≤1px", "≤2px", "n"],
        [["本方法", fmt(o["median"]), fmt(o["mean"]), fmt(o["p90"]),
          fmt(o.get("succ@1px"), pct=True), fmt(o.get("succ@2px"), pct=True),
          o["n"]]]))
    lines.append("")

    for key, title in (("by_type", "依交點類型"), ("by_tier", "依 tier"),
                       ("by_visibility", "依可見性")):
        rows = [[k, fmt(v["median"]), fmt(v["mean"]), fmt(v["p90"]),
                 fmt(v.get("succ@2px"), pct=True), v["n"]]
                for k, v in s.get(key, {}).items()]
        if rows:
            lines.append(f"### {title}\n")
            lines.append(md_table(["分層", "中位數", "平均", "P90", "≤2px", "n"],
                                  rows))
            lines.append("")

    lines.append("## 表 5.8　信心分組之定位誤差\n")
    rows = [[b["range"], fmt(b["fraction"], pct=True), fmt(b["median"]),
             fmt(b["p90"]), b["n"]] for b in s["conf_bins"]]
    lines.append(md_table(["信心區間", "角點比例", "誤差中位數 (px)",
                           "誤差 P90 (px)", "n"], rows))
    lines.append(f"\nSpearman(conf, −err) = {fmt(s['spearman_conf_err'])}\n")

    lines.append("## 表 5.11　各階段執行時間\n")
    rows = [[k, f"{v['mean_ms']:.1f}", fmt(v["share"], pct=True)]
            for k, v in s["stage_times_ms"].items()]
    lines.append(md_table(["階段", "平均耗時 (ms)", "佔比"], rows))
    lines.append("")

    # ---- 圖：誤差 CDF / 信心—誤差 ----
    errs, confs = [], []
    for p in main.get("per_image", []):
        for r in p.get("matched", []):
            if r.get("visibility", "visible") != "visible":
                continue   # 主協定：圖與統計同採 visible 分母
            errs.append(r["err_px"])
            confs.append(r.get("conf", 0))
    if errs:
        errs = np.array(errs)
        confs = np.array(confs)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.sort(errs)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x))
        ax.set_xlabel("定位誤差 (px)")
        ax.set_ylabel("累積比例")
        ax.set_xlim(0, min(10, x.max()))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "error_cdf.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(confs, errs, s=8, alpha=0.35)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(confs, bins)
        bx, by = [], []
        for i in range(1, len(bins)):
            sel = errs[idx == i]
            if sel.size:
                bx.append(0.5 * (bins[i - 1] + bins[i]))
                by.append(np.median(sel))
        ax.plot(bx, by, "r-o", lw=2, ms=4, label="分箱中位數")
        ax.set_xlabel("信心值 Conf")
        ax.set_ylabel("定位誤差 (px)")
        ax.set_ylim(0, np.percentile(errs, 98))
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "conf_vs_error.png", dpi=300)
        plt.close(fig)
        lines.append("![誤差 CDF](figures/error_cdf.png)\n")
        lines.append("![信心—誤差](figures/conf_vs_error.png)\n")

    # 圖：耗時長條
    st = s["stage_times_ms"]
    if st:
        fig, ax = plt.subplots(figsize=(5, 3))
        ks = list(st)
        ax.barh(ks, [st[k]["mean_ms"] for k in ks])
        ax.set_xlabel("平均耗時 (ms)")
        fig.tight_layout()
        fig.savefig(fig_dir / "stage_times.png", dpi=300)
        plt.close(fig)


def sec_baselines(base, main, lines):
    lines.append("## 表 5.5　角點定位誤差（含基準）\n")
    rows = []
    name_map = {"harris": "Harris", "forstner": "Förstner",
                "extern": "YOLOPoint（外部）"}
    for m, st in base["summary"].items():
        if st.get("median") is None:
            continue
        rows.append([name_map.get(m, m), fmt(st["median"]), fmt(st["mean"]),
                     fmt(st["p90"]), fmt(st.get("succ@1px"), pct=True),
                     fmt(st.get("succ@2px"), pct=True), st["n"]])
    if main:
        o = main["summary"]["overall"]
        rows.append(["**本方法**", fmt(o["median"]), fmt(o["mean"]),
                     fmt(o["p90"]), fmt(o.get("succ@1px"), pct=True),
                     fmt(o.get("succ@2px"), pct=True), o["n"]])
    lines.append(md_table(["方法", "中位數", "平均", "P90", "≤1px 成功率",
                           "≤2px 成功率", "n"], rows))
    lines.append("")


def sec_sweep(sw, lines):
    lines.append("## 表 5.3b　信心門檻掃描\n")
    rows = [[r["threshold"], f"{r['candidates_per_img']:.1f}",
             fmt(r["solve_ok_rate"], pct=True),
             fmt(r["line_support_mean"], 2),
             fmt(r["mapping_correct_single"], pct=True),
             fmt(r["mapping_correct_multi"], pct=True),
             fmt(r["solve_time_mean_s"], 2)]
            for r in sw["table"]]
    lines.append(md_table(["門檻", "候選數/圖", "求解成功率", "線支持度",
                           "對應正確率（單球場）", "對應正確率（多球場）",
                           "求解耗時 (s)"], rows))
    lines.append("")


def sec_ablation(ab, lines):
    lines.append("## 表 5.9　方法消融\n")
    name_map = {"full": "完整方法", "no_topology": "無拓樸約束",
                "no_jacobian": "無 Jacobian 線寬",
                "no_steger": "無 Steger 精修", "no_quality": "無品質過濾"}
    rows = [[name_map.get(v, v), fmt(st["median"]), fmt(st["p90"]),
             fmt(st["output_rate"], pct=True),
             fmt(st["cid_correct_rate"], pct=True)]
            for v, st in ab["summary"].items()]
    lines.append(md_table(["變體", "誤差中位數", "誤差 P90", "輸出率",
                           "編號正確率"], rows))
    lines.append("")


def sec_discrim(qd, fig_dir, lines):
    lines.append("## 表 5.6　影像證據方法判別能力比較\n")
    name_map = {"legacy": "Harris–Steger 差分（原）",
                "gradgeo": "梯度幾何證據（本研究）"}
    rows = []
    for m in ("legacy", "gradgeo"):
        t = qd["table_5_6"][m]
        rows.append([name_map[m], fmt(t["auc_vs_online"]),
                     fmt(t["auc_vs_shifted"]), fmt(t["pos_median"]),
                     f"{fmt(t['time_ms_per_point'], 2)} ms"])
    lines.append(md_table(["證據方法", "AUC（vs 線上）", "AUC（vs 偏移）",
                           "角點 composite 中位數", "單點耗時"], rows))
    lines.append("")

    if qd.get("table_5_7"):
        lines.append("## 表 5.7　梯度幾何證據組成消融\n")
        name_map = {"full": "完整", "no_en_gate": "無能量門控",
                    "no_nondeg": "無 (1 − coh)", "no_Rn": "無 R_n",
                    "no_convfilter": "無收斂過濾"}
        rows = [[name_map.get(v, v), fmt(t["auc_vs_bg"]),
                 fmt(t["auc_vs_online"])]
                for v, t in qd["table_5_7"].items()]
        lines.append(md_table(["設定", "AUC（vs 背景）", "AUC（vs 線上）"],
                              rows))
        lines.append("")

    # ROC 圖
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for m in ("legacy", "gradgeo"):
        t = qd["table_5_6"][m]
        fpr, tpr = t.get("roc_vs_online", ([], []))
        if fpr:
            ax.plot(fpr, tpr,
                    label=f"{name_map.get(m, m) if m in name_map else m} "
                          f"(AUC={fmt(t['auc_vs_online'])})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_online.png", dpi=300)
    plt.close(fig)
    lines.append("![ROC（vs 線上硬負樣本）](figures/roc_online.png)\n")


def run(args):
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    main = maybe(args.results, "main_eval")
    base = maybe(args.results, "baselines")
    sweep = maybe(args.results, "conf_sweep")
    abla = maybe(args.results, "ablation")
    qd = maybe(args.results, "quality_discrim")

    lines = ["# 第五章實驗結果彙整", ""]
    if base:
        sec_baselines(base, main, lines)
    if main:
        sec_main(main, fig_dir, lines)
    if sweep:
        sec_sweep(sweep, lines)
    if abla:
        sec_ablation(abla, lines)
    if qd:
        sec_discrim(qd, fig_dir, lines)
    if len(lines) <= 2:
        print("results/ 內沒有任何結果 JSON。")
        return

    rp = out_dir / "report.md"
    rp.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {rp}")
    print(f"[save] {fig_dir}/*.png")


def build_parser():
    ap = argparse.ArgumentParser(description="實驗結果報表")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/report")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
