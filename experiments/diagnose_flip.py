#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_flip.py (v2) — 粗大誤差型態診斷：翻轉 / 鏡像 / 平移 / 鎖鄰場 / H 歪斜

原理：
  1. 以 GT 角點 (cid → 模板公尺座標 ↔ 影像像素) 擬合「真單應」H_gt
  2. D = H_gt⁻¹ ∘ H_pred 是管線 H 在「模板公尺空間」中等效施加的變換
  3. 把 D 以相似變換（允許鏡像）分解 → (縮放 s, 旋轉角 θ, 鏡像?, 平移 t, 殘差)
  4. 自動分類：
       OK        D ≈ 恆等
       ROT180    θ≈180°（180° 對稱解）
       MIRROR    鏡像解（左右/上下翻）
       SHIFT     形狀保持但平移大（|t|>1m；|tx| 接近場寬以上 → 疑似鎖到鄰場）
       WARP      無法以相似變換解釋（H 本身歪斜：對應點品質差 / 抽線錯）
       GT_SUSPECT  GT 自身擬合殘差過大（GT 可能標錯空間或標錯點）
  另列出管線自報的 H 信心（high/medium/low），檢查失敗是否已被自我察覺。

用法：
    python experiments/diagnose_flip.py --main_result results/main_eval.json \
        --img_dir <影像資料夾> [--gt_dir <GT資料夾>] [--tol 0.30]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from court_corner.shared.court_model import (              # noqa: E402
    TEMPLATE_POINTS, NODE_TABLE, LINE_WIDTH_M,
    encode_corner, nx_ny_to_junction_idx,
)
from experiments.common import load_gt, fmt, md_table      # noqa: E402

_HALF_W = LINE_WIDTH_M / 2.0  # 0.02 m
COURT_W, COURT_L = 6.10, 13.40


def enumerate_valid_corners():
    """80 個物理角點：[{cid, world(tpl m)}]，與管線 cid 定義一致。"""
    out = []
    for (nx, ny), info in sorted(NODE_TABLE.items()):
        j = nx_ny_to_junction_idx(nx, ny)
        p0 = TEMPLATE_POINTS[j].astype(np.float64)
        for lcid in range(4):
            if not (info["valid_corner_mask"] >> lcid) & 1:
                continue
            dy = +_HALF_W if (lcid >> 1) == 0 else -_HALF_W   # bit1: N(0)/S(1)
            dx = +_HALF_W if (lcid & 1) == 1 else -_HALF_W    # bit0: W(0)/E(1)
            out.append({
                "cid": encode_corner(nx, ny, lcid),
                "world": np.array([p0[0] + dx, p0[1] + dy], dtype=np.float64),
            })
    return out


ALL_CORNERS = enumerate_valid_corners()
WORLD_BY_CID = {c["cid"]: c["world"] for c in ALL_CORNERS}


def flip_cid(cid: int) -> int:
    """180° 旋轉對稱的 cid 映射（保留供舊流程使用）。"""
    ny = (cid >> 5) & 0b111
    nx = (cid >> 2) & 0b111
    lcid = cid & 0b11
    return encode_corner(4 - nx, 6 - ny, 3 - lcid)


def median_err(H: np.ndarray, gt: dict, use_flip: bool = False) -> tuple:
    errs = []
    for spec in ALL_CORNERS:
        cid = flip_cid(spec["cid"]) if use_flip else spec["cid"]
        g = gt.get(cid)
        if g is None:
            continue
        p = cv2.perspectiveTransform(
            spec["world"].astype(np.float32).reshape(1, 1, 2), H).reshape(2)
        if not np.all(np.isfinite(p)):
            continue
        errs.append(float(np.hypot(p[0] - g["x"], p[1] - g["y"])))
    return (float(np.median(errs)), len(errs)) if errs else (float("nan"), 0)


def chirality_sign(H) -> int:
    """模板 CCW 三角形投影後的有向面積符號；合法 H 恆為 -1，+1 即鏡像解。"""
    q = []
    for X, Y in ((0.0, 0.0), (COURT_W, 0.0), (0.0, COURT_L)):
        v = np.asarray(H, np.float64) @ np.array([X, Y, 1.0])
        if abs(v[2]) < 1e-12:
            return 0
        q.append((v[0] / v[2], v[1] / v[2]))
    cr = ((q[1][0] - q[0][0]) * (q[2][1] - q[0][1])
          - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]))
    return 1 if cr > 0 else (-1 if cr < 0 else 0)


def fit_h_gt(gt: dict):
    """GT (world m ↔ px) 擬合真 H。回傳 (H_gt, 殘差中位 px)。"""
    src, dst = [], []
    for cid, g in gt.items():
        w = WORLD_BY_CID.get(int(cid))
        if w is None:
            continue
        src.append(w); dst.append([g["x"], g["y"]])
    if len(src) < 6:
        return None, float("nan")
    src = np.asarray(src, np.float64).reshape(-1, 1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, float("nan")
    proj = cv2.perspectiveTransform(src.astype(np.float32), H).reshape(-1, 2)
    res = np.hypot(*(proj - dst.reshape(-1, 2)).T)
    return H, float(np.median(res))


def similarity_decompose(D: np.ndarray):
    """
    把模板空間變換 D 以「相似變換（允許鏡像）」近似。
    取場地外框四角經 D 映射後，與原四角做 Umeyama 擬合。
    回傳 dict(s, theta_deg, mirrored, t=(tx,ty), resid_m)。
    """
    quad = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]],
                    dtype=np.float64)
    mapped = cv2.perspectiveTransform(
        quad.astype(np.float32).reshape(-1, 1, 2), D).reshape(-1, 2).astype(np.float64)
    if not np.all(np.isfinite(mapped)):
        return None

    def umeyama(A, B, allow_reflection):
        ma, mb = A.mean(0), B.mean(0)
        Ac, Bc = A - ma, B - mb
        Sigma = Bc.T @ Ac / len(A)
        U, Dg, Vt = np.linalg.svd(Sigma)
        S = np.eye(2)
        if not allow_reflection and np.linalg.det(U @ Vt) < 0:
            S[1, 1] = -1
        if allow_reflection and np.linalg.det(U @ Vt) >= 0:
            return None  # 這個分支專出鏡像解
        R = U @ S @ Vt
        var = (Ac ** 2).sum() / len(A)
        s = np.trace(np.diag(Dg) @ S) / var
        t = mb - s * R @ ma
        res = np.sqrt(((B - (s * (R @ A.T).T + t)) ** 2).sum(1)).mean()
        return dict(s=float(s), R=R, t=t, resid_m=float(res),
                    mirrored=bool(np.linalg.det(R) < 0))

    cands = [u for u in (umeyama(quad, mapped, False),
                         umeyama(quad, mapped, True)) if u is not None]
    if not cands:
        return None
    best = min(cands, key=lambda u: u["resid_m"])
    theta = float(np.degrees(np.arctan2(best["R"][1, 0], best["R"][0, 0])))
    best["theta_deg"] = theta
    return best


def classify(dec, tol_m: float):
    if dec is None:
        return "WARP", "投影發散"
    if dec["resid_m"] > tol_m:
        return "WARP", f"相似殘差 {dec['resid_m']:.2f}m"
    if abs(dec["s"] - 1.0) > 0.12:
        return "SCALE", (f"縮放 {dec['s']:.2f}：檢查 GT 與推論影像空間"
                         "（解析度/裁切/去畸變）是否一致")
    # 以「場地中心」的位移判斷平移（純旋轉/鏡像對稱解的中心不動）
    ctr = np.array([COURT_W / 2.0, COURT_L / 2.0])
    ctr_mapped = dec["s"] * (dec["R"] @ ctr) + dec["t"]
    tx, ty = (ctr_mapped - ctr)
    tnorm = float(np.hypot(tx, ty))
    ang = abs(((dec["theta_deg"] + 180) % 360) - 180)  # 0..180
    parts, note = [], []
    if dec["mirrored"]:
        parts.append(f"MIRROR(軸{dec['theta_deg'] / 2:.0f}°)")
    elif ang > 90:
        parts.append("ROT180")
    elif ang > 15:
        parts.append(f"ROT{dec['theta_deg']:.0f}")
    if tnorm > 1.0:
        parts.append("SHIFT")
        note.append(f"中心位移 ({tx:+.2f},{ty:+.2f})m")
        if abs(tx) > COURT_W * 0.8 or abs(ty) > COURT_L * 0.8:
            note.append("疑似鎖到鄰場")
    if not parts:
        return "OK", ""
    return "+".join(parts), " ".join(note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_result", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--tol", type=float, default=0.30,
                    help="相似分解殘差容忍 (m)，超過判 WARP")
    ap.add_argument("--gt_resid_max", type=float, default=3.0,
                    help="GT 自身擬合殘差中位上限 (px)，超過判 GT_SUSPECT")
    args = ap.parse_args()

    with open(args.main_result, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data.get("per_image", data.get("images", []))

    rows, counts = [], {}

    def tally(v):
        counts[v] = counts.get(v, 0) + 1

    for rec in recs:
        name = rec.get("image", rec.get("name", "?"))
        hconf = (rec.get("homography") or {}).get("confidence", "—")
        H = rec.get("H")
        if H is None:
            tally("NO_H"); rows.append([name, "—", "—", "NO_H", "", hconf, "—"]); continue
        H = np.asarray(H, dtype=np.float64)
        chi = {1: "+1", -1: "-1", 0: "0"}[chirality_sign(H)]
        gt = load_gt(Path(args.img_dir) / name, args.gt_dir)
        if not gt:
            tally("NO_GT"); rows.append([name, "—", "—", "NO_GT", "", hconf, chi]); continue

        m_id, _ = median_err(H, gt)
        H_gt, gt_res = fit_h_gt(gt)
        if H_gt is None or not np.isfinite(gt_res) or gt_res > args.gt_resid_max:
            tally("GT_SUSPECT")
            rows.append([name, fmt(m_id, 2), fmt(gt_res, 2), "GT_SUSPECT",
                         "GT 擬合殘差過大，檢查標註", hconf, chi])
            continue

        D = np.linalg.inv(H_gt) @ H
        D /= D[2, 2]
        dec = similarity_decompose(D)
        verdict, note = ("OK", "") if (np.isfinite(m_id) and m_id <= 8.0) \
            else classify(dec, args.tol)
        tally(verdict)
        rows.append([name, fmt(m_id, 2), fmt(gt_res, 2), verdict, note, hconf, chi])

    print(md_table(["影像", "原cid中位誤差(px)", "GT擬合殘差(px)",
                    "判定", "說明", "管線H信心", "掌性"], rows))
    order = ["OK", "ROT180", "SHIFT", "SCALE", "WARP",
             "GT_SUSPECT", "NO_GT", "NO_H"]
    shown = {k: counts.pop(k) for k in order if k in counts}
    shown.update(counts)
    print("\n統計： " + "、".join(f"{k} {v}" for k, v in shown.items()))
    print("\n掌性檢查：物理相機模型推導合法 H 掌性 = -1（OK 列應同號、MIRROR 列必反號）。"
          "請以 OK 列實測值設定 config.S2_PROPER_CHIRALITY_SIGN，"
          "並啟用 S2_REJECT_IMPROPER_CHIRALITY 於候選層剔除鏡像解。")
    print("\n判讀指南：")
    print("  ROT180 / MIRROR  → 對稱消歧失敗：H 幾何貼合但方向/掌性鎖反，"
          "可加球網位置、攝影機先驗或 YOLO 類別投票消歧")
    print("  SHIFT            → 格線標號平移（cross-ratio 線標號錯位）或鎖到鄰場")
    print("  WARP             → H 本身歪斜：抽線品質差/退化配置，用 viewer 看疊圖")
    print("  GT_SUSPECT       → 先檢查 GT（是否標在不同影像空間、蓋章未微調）")
    print("  另比對「管線H信心」欄：若失敗影像多為 low，代表管線已自我察覺，"
          "可在論文以信心分級討論；若 high 卻失敗，是求解器的盲點案例")


if __name__ == "__main__":
    main()
