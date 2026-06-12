#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_mirror_gt.py — 修復「鏡像標號」之 GT（掌性 +1 → −1）

背景：GT 慣例為 row0_near + 掌性 −1；實體相機任何視角皆不可能產生 +1，
故 audit_gt 報出掌性 +1 之影像必為鏡像標號（左右 cid 對調）。球場左右
對稱使鏡像標號之投影仍貼合白線、擬合殘差仍小，疊圖無法目視判別——
唯一徵兆即掌性，以及評估時的災難性配對誤差。

做法：對每張影像嘗試數種鏡像 cid 重映射（lcid 位元配置不同的候選），
重擬合 H 後選「掌性 = −1 且殘差中位不劣於原值」者；同步修 lcid 字串
（E↔W）與 node 欄。預設 dry-run，--apply 才寫檔（自動留 .bak）。

用法：
  python experiments/fix_mirror_gt.py <gt_dir> --images img_0308 img_0310 img_0314 img_0316
  python experiments/fix_mirror_gt.py <gt_dir> --images ... --apply
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.diagnose_flip import WORLD_BY_CID, fit_h_gt          # noqa: E402
from experiments.common import load_gt                                # noqa: E402
from experiments.audit_gt import chirality                            # noqa: E402

LCID_STR = {"NE": "NW", "NW": "NE", "SE": "SW", "SW": "SE"}  # x 鏡像：E↔W


def decode(cid):
    return (cid >> 5) & 0b111, (cid >> 2) & 0b111, cid & 0b11


def encode(ny, nx, l):
    return (ny << 5) | (nx << 2) | l


def mirror_cid(cid, lmap):
    ny, nx, l = decode(int(cid))
    return encode(ny, 4 - nx, lmap(l))


# lcid 2-bit 的鏡像候選：x 側在 bit0 / bit1 / 兩者皆翻 / 不翻
LMAPS = {"bit0": lambda l: l ^ 1, "bit1": lambda l: l ^ 2,
         "both": lambda l: 3 - l, "none": lambda l: l}


def fit_stats(corners):
    gt = {int(c["cid"]): {"x": c["x"], "y": c["y"],
                          "visibility": c.get("visibility", "visible")}
          for c in corners if int(c["cid"]) in WORLD_BY_CID}
    H, res = fit_h_gt(gt)
    return H, res, chirality(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_dir")
    ap.add_argument("--images", nargs="+", required=True,
                    help="檔名子字串（如 img_0308）")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.gt_dir).glob("*.gt.json"))
    for key in args.images:
        hits = [f for f in files if key in f.name]
        if len(hits) != 1:
            print(f"[{key}] 比對到 {len(hits)} 個檔案，跳過：{[h.name for h in hits]}")
            continue
        fp = hits[0]
        d = json.load(open(fp, encoding="utf-8"))
        corners = d["corners"] if isinstance(d, dict) else d
        H0, res0, chi0 = fit_stats(corners)
        if chi0 == "-1":
            print(f"[{fp.name}] 掌性已為 −1，無需修復")
            continue

        best = None
        for name, lmap in LMAPS.items():
            cand = [dict(c, cid=mirror_cid(c["cid"], lmap)) for c in corners]
            if any(int(c["cid"]) not in WORLD_BY_CID for c in cand):
                continue
            try:
                _, res, chi = fit_stats(cand)
            except Exception:
                continue
            if chi == "-1" and (best is None or res < best[1]):
                best = (name, res, cand)

        if best is None:
            print(f"[{fp.name}] 找不到使掌性=-1 之重映射，請改用標註工具人工處理")
            continue
        name, res, cand = best
        ok = res <= res0 + 0.5
        print(f"[{fp.name}] lcid 映射={name}  殘差 {res0:.2f}→{res:.2f}px  "
              f"掌性 +1→−1  {'OK' if ok else '⚠ 殘差變差，請人工確認'}")
        if not (args.apply and ok):
            continue
        # 同步修 lcid 字串與 node
        for c, c2 in zip(corners, cand):
            c["cid"] = c2["cid"]
            if isinstance(c.get("lcid"), str) and c["lcid"] in LCID_STR:
                c["lcid"] = LCID_STR[c["lcid"]]
            if isinstance(c.get("node"), (list, tuple)) and len(c["node"]) == 2:
                a, b = c["node"]
                ny, nx, _ = decode(int(c["cid"]))
                # 依新 cid 重建 node（自動偵測原順序）
                c["node"] = [nx, ny] if [a, b] != [ny, nx] else [ny, nx]
        if isinstance(d, dict):
            d["convention"] = "row0_near"
        shutil.copy(fp, str(fp) + ".bak")
        json.dump(d, open(fp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"          已寫入（備份 {fp.name}.bak）")


if __name__ == "__main__":
    main()
