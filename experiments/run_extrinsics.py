#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extrinsics.py — 表 5.10：以合成資料之已知相機位姿驗證角點層級輸出之下游價值

資料：gen_exp_corner_gt_dataset_clean.py 之輸出
  <data_dir>/images/sample_XXXXXX.png（或 --img_dir 指定）
  <data_dir>/corner_gt/sample_XXXXXX.json：K、cam_location、80 角點之
      world_xyz（Blender 世界座標，網中心原點、地板 z=0）與精確像素 GT

實驗：對每張影像跑管線，以三種對應來源解 PnP（平面 IPPE + LM 精修）：
  (a) 交點中心 — 求解器精修後之 junction 影像座標（≤30 點）
  (b) 全部角點 — 管線輸出角點（≤80 點，Steger 精修）
  (c) 高信心角點 — conf ≥ --conf_hi

方向規範（180° gauge）：管線之逐圖方向慣例與合成資料之固定世界座標可能相差
180°。本實驗對兩種標號表示（原始 / 180° 重映射）各解一次 PnP，以「對 GT 像素
之重投影誤差」選取正確表示——等價於每機位一位元方向先驗（5.1.3 節協定一）。

指標（逐影像，最後報中位數）：
  相機位置誤差 |C_est − C_gt|（cm）
  重投影 RMSE：以估計位姿投影全部可見 GT 角點之世界座標，對 GT 像素（px）

用法：
    python experiments/run_extrinsics.py --data_dir synth_pnp_200 \
        --weights ./weight/best.pt [--img_subdir images] [--conf_hi 0.8] --out results
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from experiments.common import fmt, md_table                     # noqa: E402
from experiments.diagnose_flip import WORLD_BY_CID               # noqa: E402  (template m)
from experiments.orientation import rot180_cid, rot180_junction  # noqa: E402
from court_corner.shared.court_model import TEMPLATE_POINTS      # noqa: E402

COURT_HALF_W, NET_Y = 3.05, 6.70


class _Timeout(Exception):
    pass


class time_limit:
    """單張影像逾時保護（Unix/macOS；不支援的平台自動跳過）。"""
    def __init__(self, seconds):
        self.seconds = int(seconds)

    def __enter__(self):
        import signal
        self._ok = hasattr(signal, "SIGALRM") and self.seconds > 0
        if self._ok:
            self._old = signal.signal(signal.SIGALRM,
                                      lambda *_: (_ for _ in ()).throw(_Timeout()))
            signal.alarm(self.seconds)

    def __exit__(self, *exc):
        if self._ok:
            import signal
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


def tpl_to_world(x, y):
    """pipeline template (m) → Blender 世界座標（網中心原點、z=0）。"""
    return (float(x) - COURT_HALF_W, float(y) - NET_Y, 0.0)


WORLD3D_BY_CID = {cid: tpl_to_world(*xy) for cid, xy in WORLD_BY_CID.items()}
WORLD3D_BY_JUNCTION = {j: tpl_to_world(*TEMPLATE_POINTS[j]) for j in range(30)}


# ---------------------------------------------------------------- PnP 核心
def estimate_pose(world_pts, img_pts, K):
    """平面 PnP（IPPE 候選 + 物理約束 + LM 精修）。
    回傳 (C, rvec, tvec, rmse_fit) 或 None。"""
    if len(world_pts) < 4:
        return None
    obj = np.asarray(world_pts, np.float64).reshape(-1, 1, 3)
    img = np.asarray(img_pts, np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, np.float64)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best = None
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        C = (-R.T @ tv).ravel()
        if C[2] <= 0:                      # 相機必在地板上方
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        rmse = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
        if best is None or rmse < best[3]:
            best = (C, rv, tv, rmse)
    if best is None:
        return None
    C, rv, tv, _ = best
    try:
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
    except cv2.error:
        pass
    R, _ = cv2.Rodrigues(rv)
    C = (-R.T @ tv).ravel()
    proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
    rmse = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
    return C, rv, tv, rmse


def reproj_rmse_on_gt(rv, tv, K, gt_corners):
    """以估計位姿投影可見 GT 角點之世界座標，對 GT 像素計 RMSE。
    （gauge 選取後位姿已表示於固定世界框，故一律以原 cid 取世界座標。）"""
    obj, img = [], []
    for c in gt_corners:
        if c.get("status") != "visible":
            continue
        w = WORLD3D_BY_CID.get(int(c["cid"]))
        if w is None:
            continue
        obj.append(w); img.append((float(c["x"]), float(c["y"])))
    if len(obj) < 4:
        return None, 0
    proj, _ = cv2.projectPoints(np.asarray(obj, np.float64).reshape(-1, 1, 3),
                                rv, tv, np.asarray(K, np.float64), None)
    d = proj.reshape(-1, 2) - np.asarray(img, np.float64)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))), len(obj)


def solve_with_gauge(corr_sets, K, gt_corners):
    """corr_sets: {source: (cids_or_jids, img_pts, kind)}，kind ∈ {'cid','junction'}。
    以最多點之來源決定 gauge（原始 / 180°），統一套用後解各來源 PnP。
    回傳 {source: dict(C, rmse_fit, rmse_gt, n)} 與 flip。"""
    ref = max(corr_sets, key=lambda s: len(corr_sets[s][0]))

    def build(src, flip):
        ids, pts, kind = corr_sets[src]
        world = []
        for i in ids:
            if kind == "cid":
                key = rot180_cid(int(i)) if flip else int(i)
                world.append(WORLD3D_BY_CID.get(key))
            else:
                key = rot180_junction(int(i)) if flip else int(i)
                world.append(WORLD3D_BY_JUNCTION.get(key))
        keep = [k for k, w in enumerate(world) if w is not None]
        return [world[k] for k in keep], [pts[k] for k in keep]

    # gauge 選取：參考來源兩種表示各解一次，取對 GT 像素重投影較小者
    cand = {}
    for flip in (False, True):
        w, p = build(ref, flip)
        est = estimate_pose(w, p, K)
        if est is None:
            continue
        r_gt, _ = reproj_rmse_on_gt(est[1], est[2], K, gt_corners)
        if r_gt is not None:
            cand[flip] = r_gt
    if not cand:
        return None, None
    flip = min(cand, key=cand.get)

    out = {}
    for src in corr_sets:
        w, p = build(src, flip)
        est = estimate_pose(w, p, K)
        if est is None:
            out[src] = None
            continue
        C, rv, tv, rmse_fit = est
        r_gt, n_gt = reproj_rmse_on_gt(rv, tv, K, gt_corners)
        out[src] = {"C": C.tolist(), "rmse_fit": rmse_fit,
                    "rmse_gt": r_gt, "n": len(w)}
    return out, flip


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="合成資料根目錄")
    ap.add_argument("--img_subdir", default="images")
    ap.add_argument("--gt_subdir", default="corner_gt")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--yolo_conf", type=float, default=0.4)
    ap.add_argument("--conf_hi", type=float, default=0.8,
                    help="（保留相容，已不使用）")
    ap.add_argument("--topk", type=int, nargs="+", default=[6, 10],
                    help="角點來源取信心前 k 點（可多個，如 --topk 6 10）")
    ap.add_argument("--spread", action="store_true",
                    help="top-k 改用空間分散選取（最高信心起步之最遠點貪婪），"
                         "避免高信心點空間聚集劣化 PnP 幾何")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout_s", type=int, default=30,
                    help="單張求解逾時（秒），逾時計為失敗並續跑；0=不限")
    ap.add_argument("--verbose", action="store_true", help="顯示管線各階段訊息")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from court_corner.pipeline import CourtCornerPipeline
    pipe = CourtCornerPipeline(args.weights, yolo_conf=args.yolo_conf,
                               corner_conf=0.0, verbose=args.verbose)

    root = Path(args.data_dir)
    # 容錯：--data_dir 可指資料根目錄（含 corner_gt/ 子目錄）或直接指 corner_gt/
    gt_dir = root / args.gt_subdir
    if not gt_dir.is_dir() and list(root.glob("*.json")):
        gt_dir = root
    gts = sorted(p for p in gt_dir.glob("*.json")
                 if not p.name.startswith("corner_gt_all"))
    if not gts:
        print(f"找不到 GT JSON：{gt_dir}")
        return
    img_roots = [root / args.img_subdir, root, gt_dir.parent / args.img_subdir,
                 gt_dir.parent]
    if args.limit:
        gts = gts[: args.limit]

    def find_image(name):
        for r in img_roots:
            p = r / name
            if p.exists():
                return p
        for r in (root, gt_dir.parent):
            hits = list(r.rglob(name))
            if hits:
                return hits[0]
        return None

    rows, per_image = [], []
    src_keys = ["box_center", "junction"] + [f"corners_top{k}" for k in args.topk]
    agg = {k: [] for k in src_keys}
    n_fail = 0
    for gp in gts:
        rec = json.loads(gp.read_text(encoding="utf-8"))
        ip = find_image(rec["image"])
        if ip is None:
            continue
        K = np.asarray(rec["camera"]["K"], np.float64)
        C_gt = np.asarray(rec["camera"]["cam_location"], np.float64)

        import time as _time
        t0 = _time.perf_counter()
        try:
            with time_limit(args.timeout_s):
                res = pipe.run(str(ip))
            d = res.to_dict()
        except _Timeout:
            n_fail += 1
            print(f"[{len(per_image)+n_fail}/{len(gts)}] {rec['image']}  逾時(>{args.timeout_s}s)，略過")
            continue
        except Exception as e:
            n_fail += 1
            print(f"[{len(per_image)+n_fail}/{len(gts)}] {rec['image']}  例外：{type(e).__name__}")
            continue
        if d.get("H") is None or not d.get("corners"):
            n_fail += 1
            print(f"[{len(per_image)+n_fail}/{len(gts)}] {rec['image']}  求解失敗  "
                  f"{_time.perf_counter()-t0:.1f}s")
            continue

        corners = d["corners"]
        srcs = {}
        # (a) 求解器精修 junction
        line = getattr(res, "line", None)
        if line is not None and getattr(line, "junctions", None):
            jid = [int(j) for j, _ in line.junctions]
            jpt = [(float(p[0]), float(p[1])) for _, p in line.junctions]
            srcs["junction"] = (jid, jpt, "junction")
            # (a0) YOLO 偵測框中心（粗對應）：精修交點最近之偵測中心 ≤15px
            det = getattr(res, "detection", None)
            cxy = None
            v = getattr(det, "node_pts", None) if det is not None else None
            if v is not None and len(v):
                cxy = np.asarray(v, np.float64)[:, :2]
            if cxy is not None and len(cxy):
                bj, bp = [], []
                for j, ptp in zip(jid, jpt):
                    dd = np.hypot(cxy[:, 0] - ptp[0], cxy[:, 1] - ptp[1])
                    k = int(np.argmin(dd))
                    if dd[k] <= 15.0:
                        bj.append(j); bp.append((float(cxy[k, 0]), float(cxy[k, 1])))
                if len(bj) >= 4:
                    srcs["box_center"] = (bj, bp, "junction")
        # (b)(c) 角點
        ranked = sorted(corners, key=lambda c: -float(c.get("conf", 0)))

        def pick_topk(k):
            if not args.spread or len(ranked) <= k:
                return ranked[:k]
            sel = [ranked[0]]
            pool = ranked[1:]
            while len(sel) < k and pool:
                def min_d(c):
                    return min((c["x"] - q["x"]) ** 2 + (c["y"] - q["y"]) ** 2
                               for q in sel)
                j = max(range(len(pool)), key=lambda i: min_d(pool[i]))
                sel.append(pool.pop(j))
            return sel

        for k in args.topk:
            sub = pick_topk(int(k))
            if len(sub) >= 4:
                srcs[f"corners_top{k}"] = ([c["cid"] for c in sub],
                                           [(c["x"], c["y"]) for c in sub],
                                           "cid")

        out, flip = solve_with_gauge(srcs, K, rec["corners"])
        if out is None:
            n_fail += 1
            continue
        row = {"image": rec["image"], "flip": flip,
               "h_confidence": d.get("homography", {}).get("confidence",
                                getattr(res, "confidence", None))}
        for s in src_keys:
            r = out.get(s)
            if r is None:
                row[s] = None
                continue
            pos_err_cm = float(np.linalg.norm(np.asarray(r["C"]) - C_gt)) * 100.0
            row[s] = {"pos_err_cm": pos_err_cm, "rmse_gt_px": r["rmse_gt"],
                      "n": r["n"]}
            agg[s].append((pos_err_cm, r["rmse_gt"], r["n"]))
        per_image.append(row)
        b = row.get(src_keys[-1])
        print(f"[{len(per_image)+n_fail}/{len(gts)}] {rec['image']}  "
              f"{_time.perf_counter()-t0:.1f}s  flip={'是' if flip else '否'}  "
              + (f"位置誤差 {b['pos_err_cm']:.1f}cm  重投影 {b['rmse_gt_px']:.2f}px"
                 if b else "全部角點來源失敗"))

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "extrinsics.json", "w", encoding="utf-8") as f:
        json.dump({"per_image": per_image, "n_fail": n_fail,
                   "conf_hi": args.conf_hi}, f, ensure_ascii=False, indent=1)

    sel_tag = "（分散選取）" if args.spread else "（信心排序）"
    label = {"box_center": "YOLO 框中心", "junction": "精修交點"}
    for k in args.topk:
        label[f"corners_top{k}"] = f"角點 top-{k}{sel_tag}"
    print(f"\n========== 外參估計（表 5.10）：{len(per_image)} 張成功 / "
          f"{n_fail} 張失敗 ==========")
    GROSS = 10.0   # 重投影 >10px 視為對應層失敗（與 5.1.3 兩段式協定一致）
    tab = []
    for s in src_keys:
        v = agg[s]
        if not v:
            tab.append([label[s], "0", "—", "—", "—", "—"]); continue
        pos = np.array([x[0] for x in v]); rp = np.array([x[1] for x in v])
        npts = [x[2] for x in v]
        ok = rp <= GROSS
        if ok.sum():
            tab.append([label[s], f"{len(v)}",
                        f"{ok.mean()*100:.0f}%",
                        fmt(float(np.median(rp[ok])), 2),
                        fmt(float(np.median(pos[ok])), 2),
                        fmt(float(np.percentile(pos[ok], 90)), 1),
                        ])
        else:
            tab.append([label[s], f"{len(v)}", "0%", "—", "—", "—"])
    print(md_table(["對應來源", "n", f"對應正確率(≤{GROSS:.0f}px)",
                    "重投影中位 (px)", "位置誤差中位 (cm)", "位置 P90 (cm)"], tab))
    # 成對比較（核心主張檢驗）
    def paired(a, b):
        rows = [(p[a]["pos_err_cm"], p[b]["pos_err_cm"]) for p in per_image
                if p.get(a) and p.get(b)
                and p[a]["rmse_gt_px"] <= GROSS and p[b]["rmse_gt_px"] <= GROSS]
        if not rows:
            return None
        x = np.array(rows)
        return len(rows), float(np.median(x[:, 0])), float(np.median(x[:, 1])),             float((x[:, 1] < x[:, 0]).mean())
    pairs = [("box_center", "junction")]
    for k in args.topk:
        pairs.append(("junction", f"corners_top{k}"))
    if len(args.topk) >= 2:
        pairs.append((f"corners_top{args.topk[0]}", f"corners_top{args.topk[-1]}"))
    for a, b in pairs:
        r = paired(a, b)
        if r:
            print(f"成對（皆正確, n={r[0]}）：{label[a]} {r[1]:.2f}cm vs "
                  f"{label[b]} {r[2]:.2f}cm（後者較佳 {r[3]*100:.0f}%）")
    print("（重投影誤差：以估計位姿投影可見 GT 角點世界座標，對精確像素 GT；"
          "方向 gauge 以對 GT 重投影自動選取，等價每機位一位元先驗。）")


if __name__ == "__main__":
    main()
