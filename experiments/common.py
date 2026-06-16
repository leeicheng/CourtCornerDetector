# -*- coding: utf-8 -*-
"""
common.py — 實驗共用工具
================================================================
提供第五章各實驗腳本共用的功能：
  - 資料集影像列舉與 GT 角點載入
  - 預測 ↔ GT 以 cid 配對與誤差統計（中位數 / 平均 / P90 / 成功率）
  - AUC、Spearman 相關係數
  - 統一的結果 JSON 存取（results/ 目錄）

GT 標註格式（與影像同名、放在 gt_dir 或影像同目錄）：
  <影像檔名去副檔名>.gt.json
  {
    "image": "court001.jpg",
    "corners": [
      {"cid": 77, "x": 619.50, "y": 271.86, "visibility": "visible"},
      {"cid": 78, "x": 640.12, "y": 270.40, "visibility": "occluded"}
    ]
  }
  也接受最外層直接是 corners 陣列。visibility 可省略（預設 visible）。
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ================= 資料集 =================

def find_images(img_dir: str) -> List[Path]:
    p = Path(img_dir)
    imgs = sorted(q for q in p.iterdir()
                  if q.suffix.lower() in IMG_EXTS and q.is_file())
    return imgs


def gt_path_for(img_path: Path, gt_dir: Optional[str] = None) -> Optional[Path]:
    stem = img_path.stem
    cands = []
    if gt_dir:
        cands += [Path(gt_dir) / f"{stem}.gt.json", Path(gt_dir) / f"{stem}.json"]
    cands += [img_path.with_suffix(".gt.json"),
              img_path.parent / f"{stem}.json"]
    for c in cands:
        if c.exists():
            return c
    return None


def load_gt(img_path: Path, gt_dir: Optional[str] = None) -> Dict[int, dict]:
    """回傳 {cid: {"x":..., "y":..., "visibility":...}}；無 GT 回傳 {}。"""
    p = gt_path_for(img_path, gt_dir)
    if p is None:
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    corners = data.get("corners", data) if isinstance(data, dict) else data
    out = {}
    for c in corners:
        try:
            cid = int(c["cid"])
            out[cid] = {
                "x": float(c["x"]), "y": float(c["y"]),
                "visibility": str(c.get("visibility", "visible")),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ================= 配對與誤差統計 =================

def rot180_cid(cid: int) -> int:
    """球場 180° 對稱的 cid 重映射：(nx,ny,lcid) → (4−nx, 6−ny, 3−lcid)。"""
    cid = int(cid)
    ny = (cid >> 5) & 0b111
    nx = (cid >> 2) & 0b111
    lcid = cid & 0b11
    return (((6 - ny) & 0b111) << 5) | (((4 - nx) & 0b111) << 2) | ((3 - lcid) & 0b11)


def match_by_cid_oriented(pred: List[dict], gt: Dict[int, dict],
                          canon: bool = True):
    """方向歸一化配對：球場 180° 旋轉對稱，單張影像無外部語意線索時
    「哪端是 row0」原則上不可觀測；多相機環場部署下，逐圖正典慣例
    （row0=近端）與固定世界座標 GT 必然有半數相機差 180°。
    故分別以原 cid 與 rot180 cid 配對，取誤差中位數較小者。
    回傳 (rows, flipped)；flipped=True 表示本圖以 180° 重映射配對。
    canon=False 時退回原行為（嚴格方向）。"""
    rows_id = match_by_cid(pred, gt)
    if not canon:
        return rows_id, False
    pred_rot = [dict(c, cid=rot180_cid(c["cid"])) for c in pred]
    rows_rot = match_by_cid(pred_rot, gt)

    def key(rows):
        vis = [r for r in rows if r.get("visibility") == "visible"]
        use = vis if len(vis) >= 4 else rows
        if len(use) < 4:
            return (0, float("inf"))
        return (1, float(np.median([r["err_px"] for r in use])))

    k_id, k_rot = key(rows_id), key(rows_rot)
    if k_rot[0] > k_id[0] or (k_rot[0] == k_id[0] and k_rot[1] < k_id[1] - 1e-9):
        return rows_rot, True
    return rows_id, False


def match_by_cid(pred: List[dict], gt: Dict[int, dict]) -> List[dict]:
    """
    以 cid 配對預測與 GT。
    pred 元素需含 cid/x/y（可另含 conf/tier/...，會原樣帶出）。
    回傳每筆配對 {"cid", "err_px", "gt_x", "gt_y", "visibility", **pred欄位}。
    """
    rows = []
    for c in pred:
        cid = int(c["cid"])
        g = gt.get(cid)
        if g is None:
            continue
        err = float(np.hypot(c["x"] - g["x"], c["y"] - g["y"]))
        row = dict(c)
        row.update(err_px=err, gt_x=g["x"], gt_y=g["y"],
                   visibility=g["visibility"])
        rows.append(row)
    return rows


def visible_rows(rows):
    """主協定分母：僅 visibility == "visible" 之配對列（與基準同分母）。"""
    return [r for r in rows if r.get("visibility") == "visible"]


def error_stats(errors, thresholds=(1.0, 2.0, 5.0)) -> dict:
    """中位數 / 平均 / P90 / 各門檻成功率。空集合回傳 None 統計。"""
    e = np.asarray([x for x in errors if np.isfinite(x)], dtype=np.float64)
    if e.size == 0:
        d = {"n": 0, "median": None, "mean": None, "p90": None}
        d.update({f"succ@{t:g}px": None for t in thresholds})
        return d
    d = {
        "n": int(e.size),
        "median": float(np.median(e)),
        "mean": float(np.mean(e)),
        "p90": float(np.percentile(e, 90)),
    }
    for t in thresholds:
        d[f"succ@{t:g}px"] = float(np.mean(e <= t))
    return d


def auc_score(pos_scores, neg_scores) -> Optional[float]:
    """以 Mann–Whitney U 計算 ROC AUC（含 ties 修正）。"""
    pos = np.asarray(pos_scores, np.float64)
    neg = np.asarray(neg_scores, np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # 平均 rank 處理 ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[:pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def roc_curve(pos_scores, neg_scores, n_thr: int = 200):
    """回傳 (fpr, tpr) 供繪 ROC 曲線。"""
    pos = np.asarray(pos_scores, np.float64)
    neg = np.asarray(neg_scores, np.float64)
    if pos.size == 0 or neg.size == 0:
        return [], []
    thr = np.unique(np.concatenate([pos, neg]))
    if thr.size > n_thr:
        thr = np.quantile(thr, np.linspace(0, 1, n_thr))
    fpr, tpr = [], []
    for t in thr[::-1]:
        tpr.append(float(np.mean(pos >= t)))
        fpr.append(float(np.mean(neg >= t)))
    return fpr, tpr


def spearman(a, b) -> Optional[float]:
    try:
        from scipy.stats import spearmanr
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        if a.size < 3:
            return None
        r, _ = spearmanr(a, b)
        return None if np.isnan(r) else float(r)
    except ImportError:
        return None


# ================= 結果存取 =================

def env_info() -> dict:
    info = {"python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu"] = "Apple MPS"
    except ImportError:
        pass
    return info


def save_result(results_dir: str, name: str, payload: dict) -> Path:
    """存成 results/<name>.json，附 meta（時間 / 環境）。"""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("meta", {})
    payload["meta"].update(
        experiment=name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        env=env_info(),
    )
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save] {path}")
    return path


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================= 雜項 =================

def fmt(v, n=3, pct=False):
    """表格輸出格式化。"""
    if v is None:
        return "—"
    if pct:
        return f"{100*v:.1f}%"
    return f"{v:.{n}f}"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
