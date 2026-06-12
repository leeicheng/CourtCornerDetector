#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_one.py — 在單張影像上對完整管線做 cProfile，找出「你的資料上」的真熱點

用法：
    python experiments/profile_one.py <影像路徑> --weights ./weight/best.pt \
        [--yolo_conf 0.4] [--runs 2] [--top 25]

第 1 次 run 為暖機（模型載入/快取），profile 取後續 run，
輸出 cumulative 前 N 名 + 各階段耗時。把整段輸出貼回來即可分析。
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--yolo_conf", type=float, default=0.4)
    ap.add_argument("--runs", type=int, default=2, help="profile 取樣次數（暖機後）")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    from court_corner.pipeline import CourtCornerPipeline
    pipe = CourtCornerPipeline(weights=args.weights, yolo_conf=args.yolo_conf,
                               corner_conf=0.0, verbose=False)

    # 暖機（模型載入、首次 JIT/快取）
    t = time.perf_counter()
    res = pipe.run(args.image)
    print(f"暖機 run：{time.perf_counter()-t:.2f}s  status={res.to_dict().get('status')}")
    d = res.to_dict()
    print("stage_times:", {k: f"{float(v):.3f}s" for k, v in
                           (d.get("stage_times") or {}).items()})

    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    for _ in range(max(1, args.runs)):
        pipe.run(args.image)
    pr.disable()
    dt = (time.perf_counter() - t) / max(1, args.runs)
    print(f"\n穩態：{dt:.2f}s/張（{args.runs} 次平均）\n")

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("cumulative").print_stats(args.top)
    txt = s.getvalue()
    # 精簡路徑讓輸出好讀
    txt = txt.replace(str(_HERE.parent) + "/", "")
    print(txt)

    s2 = io.StringIO()
    st2 = pstats.Stats(pr, stream=s2)
    st2.sort_stats("tottime").print_stats(15)
    print("—— 自身耗時（tottime）前 15 ——")
    print(s2.getvalue().replace(str(_HERE.parent) + "/", ""))


if __name__ == "__main__":
    main()
