#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_env_info.py — 收集實驗環境規格（論文 5.9 節用）

跨平台（Windows / macOS / Linux），僅用標準庫 + 已安裝套件的版本查詢。
輸出人類可讀清單與一段可直接貼入論文的文字。

用法：python experiments/get_env_info.py
"""

import json
import platform
import subprocess
import sys


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def get_cpu():
    s = platform.system()
    if s == "Darwin":
        return sh("sysctl -n machdep.cpu.brand_string") or platform.processor()
    if s == "Windows":
        out = sh('powershell -NoProfile "(Get-CimInstance Win32_Processor).Name"')
        return out.splitlines()[0].strip() if out else platform.processor()
    out = sh("grep -m1 'model name' /proc/cpuinfo")
    return out.split(":", 1)[1].strip() if ":" in out else platform.processor()


def get_ram_gb():
    s = platform.system()
    try:
        if s == "Darwin":
            return round(int(sh("sysctl -n hw.memsize")) / 2**30)
        if s == "Windows":
            out = sh('powershell -NoProfile '
                     '"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"')
            return round(int(out) / 2**30)
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2**20)
    except Exception:
        pass
    return None


def get_gpu():
    out = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if out:
        name, mem = out.splitlines()[0].split(",")
        return f"{name.strip()}（{mem.strip()}）"
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name}（{p.total_memory // 2**30} GB）"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple Silicon 整合 GPU（MPS）"
    except ImportError:
        pass
    if platform.system() == "Darwin" and "Apple" in get_cpu():
        return "Apple Silicon 整合 GPU"
    return "無獨立 GPU / 未偵測到"


def pkg_versions():
    vers = {}
    for name, mod in (("Python", None), ("NumPy", "numpy"), ("OpenCV", "cv2"),
                      ("PyTorch", "torch"), ("Ultralytics", "ultralytics"),
                      ("SciPy", "scipy"), ("PyQt6", "PyQt6.QtCore")):
        if mod is None:
            vers[name] = platform.python_version(); continue
        try:
            m = __import__(mod, fromlist=["_"])
            vers[name] = getattr(m, "__version__", None) or \
                getattr(m, "PYQT_VERSION_STR", "?")
        except ImportError:
            pass
    return vers


def main():
    cpu, ram, gpu = get_cpu(), get_ram_gb(), get_gpu()
    osname = f"{platform.system()} {platform.release()} ({platform.machine()})"
    if platform.system() == "Darwin":
        osname = f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    vers = pkg_versions()

    info = {"OS": osname, "CPU": cpu, "RAM_GB": ram, "GPU": gpu, **vers}
    print(json.dumps(info, ensure_ascii=False, indent=2))

    print("\n──── 論文 5.9 可貼段落 ────")
    pkgs = "、".join(f"{k} {v}" for k, v in vers.items() if k != "Python")
    print(f"實驗於 {osname} 平台執行，處理器為 {cpu}、記憶體 {ram} GB、"
          f"GPU 為 {gpu}；軟體環境為 Python {vers.get('Python')}，"
          f"主要套件版本：{pkgs}。除 YOLO 推論可選用 GPU 加速外，"
          f"幾何求解與角點精修管線皆以單執行緒 CPU 執行。")


if __name__ == "__main__":
    main()
