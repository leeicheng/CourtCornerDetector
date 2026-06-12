#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orientation.py — 每相機方向先驗（180° 規範化後處理）

背景：羽球場對 180° 旋轉完全對稱，單張影像沒有外部語意線索時
「哪端是 row0」原則上不可觀測；管線以逐圖慣例（row0=近端）選邊。
固定多相機環場下，逐圖慣例與固定世界座標必然有約半數相機差 180°。
這是規範（gauge）自由度，不是求解誤差——解法是供應每台相機一個
一次性的 1-bit 先驗，於管線輸出後套用，零管線侵入。

用法：
    from experiments.orientation import rot180_result, load_hints, hint_for

    hints = load_hints("camera_orientation.json")
    d = pipeline.run(img_path).to_dict()
    if hint_for(img_path, hints) == "flip":
        d = rot180_result(d)

camera_orientation.json 範例（鍵為檔名子字串，先長後短比對）：
    {
      "CameraReader1": "flip",
      "CameraReader_0": "flip",
      "CameraReader_1": "flip",
      "CameraReader_4": "flip",
      "default": "keep"
    }
取得每台相機的值：對每台相機標 1 張 GT 跑 diagnose_flip.py，
判定 ROT180 者填 "flip"、OK 者填 "keep"（rig 不動就永久有效）。
"""

import fnmatch
import json
from pathlib import Path

import numpy as np

# 模板 180° 旋轉（公尺座標）
T180 = np.array([[-1.0, 0.0, 6.10], [0.0, -1.0, 13.40], [0.0, 0.0, 1.0]])


def rot180_cid(cid: int) -> int:
    cid = int(cid)
    ny = (cid >> 5) & 0b111
    nx = (cid >> 2) & 0b111
    lcid = cid & 0b11
    return (((6 - ny) & 0b111) << 5) | (((4 - nx) & 0b111) << 2) | ((3 - lcid) & 0b11)


def rot180_junction(j: int) -> int:
    """junction_idx 0..29 的 180° 重映射：(r,c)→(5−r,4−c) ⇔ j→29−j。"""
    return 29 - int(j)


_TYPE_FLIP = {"++": "--", "--": "++", "+-": "-+", "-+": "+-"}


def rot180_result(d: dict) -> dict:
    """對 PipelineResult.to_dict() 套用 180° 規範化（就地修改並回傳）。
    位置 (x, y) 為影像座標、不變；只重映射語意（cid / junction / H）。"""
    if d.get("H") is not None:
        H = np.asarray(d["H"], dtype=np.float64) @ T180
        d["H"] = (H / H[2, 2]).tolist()
    for key in ("corners", "corner_candidates"):
        for c in d.get(key) or []:
            if "cid" in c:
                c["cid"] = rot180_cid(c["cid"])
            if "junction_idx" in c and c["junction_idx"] is not None \
                    and int(c["junction_idx"]) >= 0:
                c["junction_idx"] = rot180_junction(c["junction_idx"])
            if c.get("corner_type") in _TYPE_FLIP:
                c["corner_type"] = _TYPE_FLIP[c["corner_type"]]
    d["orientation"] = "rot180_applied"
    return d


def load_hints(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def hint_for(img_path, hints: dict) -> str:
    """回傳 'flip' 或 'keep'。鍵以「子字串或 fnmatch 萬用字元」比對檔名，
    先比對較長（較特定）的鍵；無命中時回 hints['default']（預設 'keep'）。"""
    name = Path(str(img_path)).name
    keys = sorted((k for k in hints if k != "default"), key=len, reverse=True)
    for k in keys:
        if k in name or fnmatch.fnmatch(name, k):
            return str(hints[k])
    return str(hints.get("default", "keep"))
