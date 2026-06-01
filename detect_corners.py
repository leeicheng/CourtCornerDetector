#!/usr/bin/env python3
"""
detect_corners.py — 羽球場角點定位指令工具
================================================================
四階段管線：交點偵測 → 拓樸求解 → 角點生成 → 品質評估與輸出。
輸出最終角點集合 (cid, x, y, conf)。

用法：
    python detect_corners.py --img_path court.jpg
    python detect_corners.py --img_path court.jpg --yolo.pt weights/best.pt \\
                             --yolo_conf 0.3 --corner_conf 0.6 --viz out.png

參數：
    --yolo.pt      YOLO 權重路徑（預設：與本程式同目錄下的 best.pt）
    --img_path     單張影像路徑（必填）
    --yolo_conf    YOLO 偵測信心門檻（預設 0.25）
    --corner_conf  角點輸出信心門檻（預設 0.6）
其他：
    --out          輸出 JSON 路徑（預設與影像同目錄 <影像名>_corners.json）
    --viz          輸出視覺化疊圖路徑（可選）
    --quiet        關閉逐階段訊息
"""

import os
import sys
import json
import argparse

# 確保可 import 同目錄下的 court_corner 套件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_parser():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="羽球場角點四階段定位工具（交點偵測→拓樸求解→角點生成→品質評估）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 依需求：權重旗標名為 yolo.pt，預設同目錄 best.pt
    p.add_argument("--yolo.pt", dest="yolo_pt",
                   default=os.path.join(here, "best.pt"),
                   help="YOLO 權重路徑（預設：與本程式同目錄下的 best.pt）")
    p.add_argument("--img_path", required=True, help="單張影像路徑")
    p.add_argument("--yolo_conf", type=float, default=0.25,
                   help="YOLO 偵測信心門檻")
    p.add_argument("--corner_conf", type=float, default=0.6,
                   help="角點輸出信心門檻")
    p.add_argument("--out", default=None,
                   help="輸出 JSON 路徑（預設 <影像名>_corners.json）")
    p.add_argument("--viz", default=None, help="輸出視覺化疊圖路徑（可選）")
    p.add_argument("--dark_lines", action="store_true",
                   help="若球場線為暗色（少見），加此旗標")
    p.add_argument("--quiet", action="store_true", help="關閉逐階段訊息")
    return p


def draw_viz(img_bgr, result, out_path):
    import cv2
    import numpy as np
    vis = img_bgr.copy()
    H = result.H
    # 畫 H 投影球場格線（淡藍）；依真實球場連線，場中央中線在發球線間不連
    if H is not None:
        from court_corner.shared.court_model import (
            _proj, _tpl_xy, N_COL, build_grid_connections)
        def P(idx):
            r, c = divmod(idx, N_COL)
            x, y = _proj(H, _tpl_xy(r, c))
            return (int(round(x)), int(round(y)))
        for a, b in build_grid_connections():
            cv2.line(vis, P(a), P(b), (255, 180, 60), 1, cv2.LINE_AA)
    # 畫偵測交點（黃）
    if result.detection is not None:
        for (x, y) in result.detection.node_pts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 220, 220), -1, cv2.LINE_AA)
    # 畫最終角點（綠），標 cid
    for c in result.corners:
        pt = (int(round(c.x)), int(round(c.y)))
        cv2.circle(vis, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, pt, 4, (0, 80, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, str(c.cid), (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    if not os.path.isfile(args.img_path):
        print(f"[錯誤] 找不到影像檔：{args.img_path}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.yolo_pt):
        print(f"[錯誤] 找不到 YOLO 權重：{args.yolo_pt}\n"
              f"       請以 --yolo.pt 指定，或將 best.pt 放在本程式同目錄。",
              file=sys.stderr)
        return 2

    from court_corner.pipeline import CourtCornerPipeline

    pipe = CourtCornerPipeline(
        yolo_weight=args.yolo_pt,
        yolo_conf=args.yolo_conf,
        corner_conf=args.corner_conf,
        dark=args.dark_lines,
        verbose=verbose,
    )
    result = pipe.run(args.img_path)

    # 主控台輸出最終角點
    print("\n=== 最終角點 (cid, x, y, conf) ===")
    if result.corners:
        for c in result.corners:
            print(f"  cid={c.cid:3d}  x={c.x:8.2f}  y={c.y:8.2f}  conf={c.conf:.3f}"
                  f"  [{c.corner_type}|{c.source}]")
    else:
        print(f"  （無角點通過門檻 conf≥{args.corner_conf}）")
    print(f"狀態：{result.message}")
    if result.status == "ok":
        st = result.stage_times or {}
        brk = "  ".join(f"{k}={v:.2f}s" for k, v in st.items())
        print(f"處理時間：{result.elapsed_s:.2f}s" + (f"（{brk}）" if brk else ""))
        hg = result._homography_dict()
        if "line_support" in hg:
            ok = "" if hg.get("line_support_ok", True) else "（⚠ 不足，H 可能不可靠）"
            print(f"H 信心：{result.confidence}　白線支持：{hg['line_support']:.2f}{ok}")

    # 寫 JSON
    out_path = args.out or os.path.splitext(args.img_path)[0] + "_corners.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"已輸出 JSON：{out_path}")

    # 視覺化
    if args.viz:
        import cv2
        img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
        draw_viz(img, result, args.viz)
        print(f"已輸出視覺化：{args.viz}")

    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
