#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_gt_rot180.py — 將既有 GT 由「row0=遠端」慣例轉為正典「row0=近端」

背景：舊版 gt_annotator 小地圖 row0 畫在上方，自然配對產生 row0=遠端的 GT；
正典慣例統一為 row0=近端後，既有 GT 只需做 180° 重映射（不必重標）：
  - corners:  cid → encode(4−nx, 6−ny, 3−lcid)，x/y/visibility 不變
  - annot.correspondences:  junction j → 29 − j
  - annot.H:  H → H · T180，T180 = [[-1,0,6.10],[0,-1,13.40],[0,0,1]]
180° 旋轉是剛體（掌性不變），轉換後 GT 仍為合法標號。

用法：
    python experiments/convert_gt_rot180.py <GT資料夾或影像資料夾> [--dry_run]
預設就地覆寫並備份為 <name>.gt.json.bak；--dry_run 只列出將轉換的檔案。
已轉換過的檔案（含 "convention":"row0_near"）會自動跳過，避免重複套用。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from court_corner.shared.court_model import encode_corner  # noqa: E402

T180 = np.array([[-1.0, 0.0, 6.10], [0.0, -1.0, 13.40], [0.0, 0.0, 1.0]])


def rot180_cid(cid: int) -> int:
    ny = (cid >> 5) & 0b111
    nx = (cid >> 2) & 0b111
    lcid = cid & 0b11
    return encode_corner(4 - nx, 6 - ny, 3 - lcid)


LCID_NAMES = {0: "NW", 1: "NE", 2: "SW", 3: "SE"}


def convert_file(p: Path, dry: bool) -> str:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("convention") == "row0_near":
        return "skip(已轉換)"
    if dry:
        return "would-convert"
    shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    for c in data.get("corners", []):
        new = rot180_cid(int(c["cid"]))
        c["cid"] = new
        ny = (new >> 5) & 0b111; nx = (new >> 2) & 0b111
        if "node" in c:
            c["node"] = [nx, ny]
        if "lcid" in c:
            c["lcid"] = LCID_NAMES[new & 0b11]
    annot = data.get("annot") or {}
    for cr in annot.get("correspondences", []):
        cr["junction"] = 29 - int(cr["junction"])
    if annot.get("H"):
        H = np.asarray(annot["H"], dtype=np.float64) @ T180
        annot["H"] = (H / H[2, 2]).tolist()
    data["convention"] = "row0_near"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return "converted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_dir")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    files = sorted(Path(args.gt_dir).glob("*.gt.json"))
    if not files:
        files = sorted(Path(args.gt_dir).glob("*.json"))
    if not files:
        print("找不到 GT 檔"); return
    n = {}
    for p in files:
        r = convert_file(p, args.dry_run)
        n[r] = n.get(r, 0) + 1
        print(f"{p.name}: {r}")
    print("\n統計:", ", ".join(f"{k} {v}" for k, v in n.items()))
    if not args.dry_run and n.get("converted"):
        print("原檔已備份為 *.gt.json.bak；轉換後請重跑 diagnose_flip.py 驗證應全為 OK/WARP。")


if __name__ == "__main__":
    main()
