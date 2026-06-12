import blenderproc as bproc

"""
高精度羽球場交點 + 白線外緣角點 GT 合成資料生成腳本

v3 相對 v2 的主要改動:
  - 反光斑由 Blender 端產生 (rather than 2D 後處理):
    * 地板材質: 30% 機率走「半拋光」路徑 (低 roughness + 高 specular),
      會在低仰角 AREA light 下產生大片鏡面反光斑
    * 燈光: 70% AREA (有方向性,產生光斑) + 30% POINT (補環境光)
    * AREA light size 隨機 [0.5, 3.0] m → 控制光斑邊緣銳利度
    * 燈光仰角下限從 30° 降到 15° → 低角度才打得出明顯光斑
    * Render: noise_threshold 0.01 → 0.005, 避免反光斑變 firefly
  - 2D 影像增強 重寫為「成像鏈」式 pipeline (對齊預覽工具):
    光照 (gamma + brightness/contrast) → 鏡頭 (vignette + defocus/motion blur)
    → sensor (gaussian noise + snow) → 壓縮 (JPEG)
    所有強化都是 pure-pixel, 不改變幾何, 保留 sub-pixel keypoint 標註精度
  - 場地色 / 線色 強制對比度: 兩者灰階亮度差至少 60/255, 避免線消失在地板裡
  - Distractor pool 移除: 真實場景的遮擋來源是人 / 網柱, 飄浮幾何體沒幫助
  - 相機 pose 改為「look-at 場上一點」而非永遠對準場地中心:
    * Look-at 點從所有交點抽選 (L 角加權 ×1.5, 因為角落是最關鍵的定位特徵)
    * 加入 close / mid / far 三段距離,模擬手機 / 邊線監控 / 轉播機位

核心設計 (繼承自 v2):
  1. 標註精度: 透過 bproc.camera.project_points() 把 3D 交點直接投影到像素,
     避免 marker cube 大小造成的標註誤差 (pixel-perfect ground truth)。
  2. Domain Randomization (Tobin et al. 2017):
     - 地板紋理: 純色 / gradient / checker pattern 隨機抽選
     - 標線顏色: 隨機 (不再假設只有白線), 但保證與地板有足夠對比
     - 多光源 + 隨機強度/色溫/位置
     - 相機 intrinsics (focal length) 也加入小幅 jitter
  3. Occlusion handling: 對每個交點做 raycasting,
     visibility ∈ {0=超出畫面, 1=被遮擋, 2=可見} (符合 COCO keypoint 慣例)
  4. 輸出: COCO bbox/keypoints (for YOLO) + corner_gt/*.json (80 個白線外緣角點 GT)
  5. 角點 GT: cid 使用 court_corner pipeline 的 corner_code，可直接供實驗 runner 以 cid 比對。

用法:
  blenderproc run gen_dataset3.py path/to/court.blend --num_samples 1000 --output_dir ./output
"""

import bpy  # 注意: 必須在 bproc.init() 之後才能正常使用部分功能
import random
import numpy as np
import argparse
import os
import math
import csv
import json
import cv2
from scipy.spatial.transform import Rotation


# =============================================================================
# PART 1: 羽球場幾何定義 (純幾何,和 Blender 無關)
# =============================================================================
COURT_W = 6.10
COURT_L = 13.40
COURT_HALF_W = COURT_W / 2.0
NET_Y = COURT_L / 2.0
SHORT_SERVE_D = 1.98
DBL_BACK_D = 0.76
SINGLES_OFFSET = 0.46


def to_blender_coords(x, y):
    """場地座標 → Blender 世界座標 (網中心為原點)"""
    return (x - COURT_HALF_W, y - NET_Y, 0.0)


PTS = {
    # 雙打外框
    "D_LT": to_blender_coords(0.0, 0.0),
    "D_RT": to_blender_coords(COURT_W, 0.0),
    "D_RB": to_blender_coords(COURT_W, COURT_L),
    "D_LB": to_blender_coords(0.0, COURT_L),
    # 單打邊線
    "S_LT": to_blender_coords(SINGLES_OFFSET, 0.0),
    "S_RT": to_blender_coords(COURT_W - SINGLES_OFFSET, 0.0),
    "S_RB": to_blender_coords(COURT_W - SINGLES_OFFSET, COURT_L),
    "S_LB": to_blender_coords(SINGLES_OFFSET, COURT_L),
    # 短發球線
    "T_SRV_L": to_blender_coords(0.0, NET_Y - SHORT_SERVE_D),
    "T_SRV_R": to_blender_coords(COURT_W, NET_Y - SHORT_SERVE_D),
    "B_SRV_L": to_blender_coords(0.0, NET_Y + SHORT_SERVE_D),
    "B_SRV_R": to_blender_coords(COURT_W, NET_Y + SHORT_SERVE_D),
    # 雙打長發球線
    "T_DBL_L": to_blender_coords(0.0, DBL_BACK_D),
    "T_DBL_R": to_blender_coords(COURT_W, DBL_BACK_D),
    "B_DBL_L": to_blender_coords(0.0, COURT_L - DBL_BACK_D),
    "B_DBL_R": to_blender_coords(COURT_W, COURT_L - DBL_BACK_D),
    # 中線
    "T_MID_TOP": to_blender_coords(COURT_HALF_W, 0.0),
    "T_MID_DBL": to_blender_coords(COURT_HALF_W, DBL_BACK_D),
    "T_MID_SRV": to_blender_coords(COURT_HALF_W, NET_Y - SHORT_SERVE_D),
    "B_MID_SRV": to_blender_coords(COURT_HALF_W, NET_Y + SHORT_SERVE_D),
    "B_MID_DBL": to_blender_coords(COURT_HALF_W, COURT_L - DBL_BACK_D),
    "B_MID_BOT": to_blender_coords(COURT_HALF_W, COURT_L),
}

LINES_DICT = {
    "top_baseline_doubles": ("D_LT", "D_RT"),
    "bottom_baseline_doubles": ("D_LB", "D_RB"),
    "left_doubles_sideline": ("D_LT", "D_LB"),
    "right_doubles_sideline": ("D_RT", "D_RB"),
    "left_singles_sideline": ("S_LT", "S_LB"),
    "right_singles_sideline": ("S_RT", "S_RB"),
    "top_short_service": ("T_SRV_L", "T_SRV_R"),
    "bottom_short_service": ("B_SRV_L", "B_SRV_R"),
    "top_doubles_long_service": ("T_DBL_L", "T_DBL_R"),
    "bottom_doubles_long_service": ("B_DBL_L", "B_DBL_R"),
    "center_line_top": ("T_MID_TOP", "T_MID_SRV"),
    "center_line_bottom": ("B_MID_SRV", "B_MID_BOT"),
}


def line_segment_intersection_2d(p1, p2, p3, p4):
    """計算線段交點並分類 L / T / X 型"""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    tol = 0.001
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        is_end_1 = (abs(t) < tol or abs(t - 1) < tol)
        is_end_2 = (abs(u) < tol or abs(u - 1) < tol)
        if is_end_1 and is_end_2:
            return ('L', ix, iy)
        elif is_end_1 or is_end_2:
            return ('T', ix, iy)
        else:
            return ('X', ix, iy)
    return None


def calculate_all_junctions():
    """列出所有交點 (帶穩定順序),回傳 [{type, location, name}, ...]"""
    lines_list = list(LINES_DICT.items())
    all_junctions = []
    for i in range(len(lines_list)):
        for j in range(i + 1, len(lines_list)):
            line1_name, (p1n, p2n) = lines_list[i]
            line2_name, (p3n, p4n) = lines_list[j]
            res = line_segment_intersection_2d(
                PTS[p1n], PTS[p2n], PTS[p3n], PTS[p4n]
            )
            if res:
                j_type, ix, iy = res
                all_junctions.append({
                    'type': j_type,                              # 'L' / 'T' / 'X'
                    'location': [float(ix), float(iy), 0.0],     # 世界座標 (m)
                    'name': f"{line1_name}__{line2_name}",       # 穩定 ID,跨樣本順序固定
                })
    return all_junctions


# =============================================================================
# PART 1-B: 白線外緣角點 GT 定義（與 court_corner/shared/court_model.py 對齊）
# =============================================================================
# 目的：產生「最終角點」Ground Truth，而不是只有交點中心。
# 注意：這裡的 cid 採用 pipeline 內部使用的 corner_code，方便 runner 直接以 cid 對應。

LINE_WIDTH_M = 0.04  # 40 mm, 必須與 Blender 場地線寬與 pipeline 設定一致

# pipeline template layout：x ∈ [0, 6.10] 為短邊，y ∈ [0, 13.40] 為長邊
TEMPLATE_POINTS = np.array([
    [0.00, 13.40], [0.46, 13.40], [3.05, 13.40], [5.64, 13.40], [6.10, 13.40],
    [0.00, 12.64], [0.46, 12.64], [3.05, 12.64], [5.64, 12.64], [6.10, 12.64],
    [0.00, 8.68],  [0.46, 8.68],  [3.05, 8.68],  [5.64, 8.68],  [6.10, 8.68],
    [0.00, 4.72],  [0.46, 4.72],  [3.05, 4.72],  [5.64, 4.72],  [6.10, 4.72],
    [0.00, 0.76],  [0.46, 0.76],  [3.05, 0.76],  [5.64, 0.76],  [6.10, 0.76],
    [0.00, 0.00],  [0.46, 0.00],  [3.05, 0.00],  [5.64, 0.00],  [6.10, 0.00],
], dtype=np.float64)

TEMPLATE_TYPES = np.array([
    0, 1, 1, 1, 0,
    1, 2, 2, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 2, 2, 1,
    0, 1, 1, 1, 0,
], dtype=np.int32)  # 0=L, 1=T, 2=X

N_COL, N_ROW = 5, 6

ROW_TO_NY = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
NY_TO_ROW = {v: k for k, v in ROW_TO_NY.items()}

LCID_NW, LCID_NE, LCID_SW, LCID_SE = 0, 1, 2, 3
ARM_N, ARM_E, ARM_S, ARM_W = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_ADJACENT_ARMS = {
    LCID_NW: (ARM_N, ARM_W),
    LCID_NE: (ARM_N, ARM_E),
    LCID_SW: (ARM_S, ARM_W),
    LCID_SE: (ARM_S, ARM_E),
}

X_LINES_BY_NY = {0: -6.70, 1: -5.94, 2: -1.98, 4: 1.98, 5: 5.94, 6: 6.70}
Y_LINES_BY_NX = {0: -3.05, 1: -2.59, 2: 0.00, 3: 2.59, 4: 3.05}

LOCAL_CORNER_NAME = {
    LCID_NW: 'NW',
    LCID_NE: 'NE',
    LCID_SW: 'SW',
    LCID_SE: 'SE',
}


def _compute_arms_mask(nx: int, ny: int) -> int:
    """與 pipeline court_model.py 的 30-node 拓樸一致。"""
    if nx not in Y_LINES_BY_NX or ny not in X_LINES_BY_NY:
        return 0

    valid_ny = sorted(X_LINES_BY_NY.keys())
    valid_nx = sorted(Y_LINES_BY_NX.keys())

    has_north = ny != valid_ny[0]
    has_south = ny != valid_ny[-1]
    has_west = nx != valid_nx[0]
    has_east = nx != valid_nx[-1]

    # 中線於上下短發球線位置為 T，而非 X
    if nx == 2 and ny == 2:
        return ARM_N | ARM_E | ARM_W
    if nx == 2 and ny == 4:
        return ARM_S | ARM_E | ARM_W

    mask = 0
    if has_north: mask |= ARM_N
    if has_south: mask |= ARM_S
    if has_east:  mask |= ARM_E
    if has_west:  mask |= ARM_W
    return mask


def _compute_valid_corner_mask(arms_mask: int) -> int:
    """bit lcid=1 表示該 local corner 是合法白線外緣角點。"""
    mask = 0
    for lcid, (arm_a, arm_b) in _ADJACENT_ARMS.items():
        a = 1 if (arms_mask & arm_a) else 0
        b = 1 if (arms_mask & arm_b) else 0
        if a == b:
            mask |= (1 << lcid)
    return mask


def _build_node_table() -> dict:
    table = {}
    for ny in X_LINES_BY_NY.keys():
        for nx in Y_LINES_BY_NX.keys():
            arms = _compute_arms_mask(nx, ny)
            valid = _compute_valid_corner_mask(arms)
            degree = bin(arms).count('1')
            ntype = {2: 'L', 3: 'T', 4: 'X'}.get(degree, '?')
            table[(nx, ny)] = {
                'arms_mask': arms,
                'valid_corner_mask': valid,
                'node_type': ntype,
                'degree': degree,
            }
    return table


NODE_TABLE = _build_node_table()


def junction_idx_to_nx_ny(junction_idx: int) -> tuple:
    row = int(junction_idx) // N_COL
    col = int(junction_idx) % N_COL
    return col, ROW_TO_NY[row]


def encode_corner(nx: int, ny: int, local_corner_id: int) -> int:
    return ((int(ny) & 0b111) << 5) | ((int(nx) & 0b111) << 2) | (int(local_corner_id) & 0b11)


def decode_corner(corner_code: int) -> tuple:
    cc = int(corner_code)
    ny = (cc >> 5) & 0b111
    nx = (cc >> 2) & 0b111
    lcid = cc & 0b11
    return nx, ny, lcid


def is_valid_corner(corner_code: int) -> bool:
    nx, ny, lcid = decode_corner(corner_code)
    info = NODE_TABLE.get((nx, ny))
    return bool(info is not None and (info['valid_corner_mask'] & (1 << lcid)))


def corner_code_to_dense_id(corner_code: int) -> int:
    return CORNER_CODE_TO_DENSE.get(int(corner_code), -1)


def corner_code_to_template_xy(corner_code: int, line_width_m: float = LINE_WIDTH_M) -> tuple:
    """
    corner_code → pipeline template 座標 (x, y)，單位 m。

    注意：pipeline template 座標是 [0,6.10]×[0,13.40]，而 corner_code 規格
    使用球場中心為原點且長短軸方向不同，因此這裡做一次座標轉換：
      template_x = y_world + 3.05
      template_y = 6.70 - x_world
    """
    nx, ny, lcid = decode_corner(corner_code)
    if nx not in Y_LINES_BY_NX or ny not in X_LINES_BY_NY:
        raise ValueError(f"invalid corner_code={corner_code}")

    x_world = X_LINES_BY_NY[ny]   # long axis, center origin, top is negative
    y_world = Y_LINES_BY_NX[nx]   # short axis, center origin
    h = 0.5 * float(line_width_m)

    # local_corner_id: bit1=N/S affects long-axis; bit0=W/E affects short-axis
    dx_world = -h if (lcid & 0b10) == 0 else +h
    dy_world = -h if (lcid & 0b01) == 0 else +h

    xw = x_world + dx_world
    yw = y_world + dy_world

    template_x = yw + COURT_HALF_W
    template_y = NET_Y - xw
    return float(template_x), float(template_y)


def corner_template_xy_to_blender_xyz(template_xy) -> list:
    x, y = float(template_xy[0]), float(template_xy[1])
    return [x - COURT_HALF_W, y - NET_Y, 0.0]


def _build_dense_id_tables():
    dense_to_code = []
    for ny in sorted(X_LINES_BY_NY.keys()):
        for nx in sorted(Y_LINES_BY_NX.keys()):
            for lcid in range(4):
                cc = encode_corner(nx, ny, lcid)
                if is_valid_corner(cc):
                    dense_to_code.append(cc)
    code_to_dense = {c: d for d, c in enumerate(dense_to_code)}
    return dense_to_code, code_to_dense


DENSE_TO_CORNER_CODE, CORNER_CODE_TO_DENSE = _build_dense_id_tables()


def calculate_all_corners(line_width_m: float = LINE_WIDTH_M):
    """
    產生 80 個合法白線外緣角點 GT。

    回傳欄位：
      cid              : pipeline 使用的 corner_code
      dense_id         : 0..79 連續編號，方便統計或訓練 heatmap
      junction_idx     : 對應 30 個 junction 的 row-major template id
      junction_type    : L/T/X
      local_corner_id  : NW/NE/SW/SE 的 0..3 編碼
      template_xy      : pipeline template 座標 (m)
      world_xyz        : Blender 世界座標 (m)
    """
    corners = []
    for dense_id, cc in enumerate(DENSE_TO_CORNER_CODE):
        nx, ny, lcid = decode_corner(cc)
        row = NY_TO_ROW[ny]
        col = nx
        junction_idx = row * N_COL + col
        template_xy = corner_code_to_template_xy(cc, line_width_m=line_width_m)
        world_xyz = corner_template_xy_to_blender_xyz(template_xy)
        jt_id = int(TEMPLATE_TYPES[junction_idx])
        jt_name = {0: 'L', 1: 'T', 2: 'X'}[jt_id]
        corners.append({
            'cid': int(cc),
            'dense_id': int(dense_id),
            'junction_idx': int(junction_idx),
            'junction_type': jt_name,
            'nx': int(nx),
            'ny': int(ny),
            'local_corner_id': int(lcid),
            'local_corner_name': LOCAL_CORNER_NAME[int(lcid)],
            'template_xy': [float(template_xy[0]), float(template_xy[1])],
            'world_xyz': [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])],
        })
    return corners


def compute_gt_homography_from_junction_annots(annots):
    """用 30 個 junction 投影點估計 template→image 的 GT Homography。"""
    try:
        src = []
        dst = []
        # annots 是 calculate_all_junctions 的順序，不一定等於 row-major；因此不用它估 H。
        # 這個函式保留作相容，實際 main 會用 template_junction_annots 計算。
        return None
    except Exception:
        return None



# =============================================================================
# PART 2: 程式化材質 (Domain Randomization 的核心)
# =============================================================================

def _set_specular(bsdf, value):
    """跨 Blender 版本相容的 Specular 設定。
    Blender 4.x: 'Specular IOR Level'; Blender 3.x: 'Specular'。
    """
    for key in ('Specular IOR Level', 'Specular'):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = float(value)
            return


def _rgb_to_gray(rgb):
    """sRGB → luminance (0~1),用 BT.601 係數,跟 cv2.cvtColor BGR2GRAY 一致"""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _sample_rgb_with_gray_avoiding(avoid_gray, min_diff=60 / 255.0, max_tries=20):
    """隨機抽一個 RGB,確保其灰度與 avoid_gray 差距至少 min_diff (0~1 scale)。
    避免地板色與線色在灰階下太接近導致線消失。
    """
    for _ in range(max_tries):
        rgb = np.random.rand(3)
        g = _rgb_to_gray(rgb)
        if abs(g - avoid_gray) >= min_diff:
            return rgb, g
    # fallback: 強制取對比極端的灰度
    if avoid_gray < 0.5:
        target = np.random.uniform(avoid_gray + min_diff, 1.0)
    else:
        target = np.random.uniform(0.0, avoid_gray - min_diff)
    # 純灰色 (R=G=B=target),保證灰度精準
    return np.array([target, target, target]), target


# 全域常數: 地板色採樣策略
# - 30% 機率走「真實球館先驗」: 深色 (PU/木地板),三個 channel 都壓在低區
# - 70% 機率走「完全隨機」: 維持 Tobin 2017 廣域隨機化精神,
#   讓 model 不會學到「線=亮、地板=暗」的 shortcut
DARK_FLOOR_PROB = 0.3
DARK_FLOOR_RANGE = (0.05, 0.35)  # 三個 channel 的均勻分布範圍


def _sample_floor_rgb():
    """單次抽一個地板用的 RGB。
    30% 走深色先驗,70% 走完全隨機 (np.random.rand(3))。
    """
    if np.random.rand() < DARK_FLOOR_PROB:
        return np.random.uniform(*DARK_FLOOR_RANGE, size=3)
    return np.random.rand(3)


def _build_mosaic_checker_subtree(nt, mapping_vector_out):
    """
    建構「碎、2 色交替但部分格子被翻轉」的 checker 子圖。

    與單純 TexChecker 的差別:
      - 格子尺度很小 (mapping scale 30~80,比原本 2~15 密很多倍) → 馬賽克感
      - 用 WhiteNoise(以格子整數座標為輸入) 隨機決定每格是否「翻色」,
        亦即原本該 c1 的格子有 swap_prob 機率被換成 c2,反之亦然
        → 整體仍保持 2 色交替的視覺結構,但被「打亂」一部分

    Returns:
        (color_socket, gray_estimate)
        color_socket: 子圖最終顏色 output,要接到下游 (mosaic_color)
        gray_estimate: 整塊馬賽克的代表灰度 (兩色平均,因為翻轉是對稱的)
    """
    # --- 1. 用第二個 Mapping 把座標縮放到「馬賽克」密度 ---
    # 注意: 馬賽克本身的旋轉用另一個獨立的 mapping,
    # 這樣補丁紋理不會跟馬賽克紋理共享旋轉
    mosaic_map = nt.nodes.new('ShaderNodeMapping')
    mosaic_scale = float(np.random.uniform(50.0, 180.0))  # 馬賽克密度
    mosaic_map.inputs['Scale'].default_value = (mosaic_scale, mosaic_scale, mosaic_scale)
    mosaic_map.inputs['Rotation'].default_value[2] = float(np.random.uniform(0, 2 * math.pi))
    nt.links.new(mapping_vector_out, mosaic_map.inputs['Vector'])

    # --- 2. 標準 TexChecker (c1, c2 交替) ---
    checker = nt.nodes.new('ShaderNodeTexChecker')
    c1 = _sample_floor_rgb()
    c2 = _sample_floor_rgb()
    checker.inputs['Color1'].default_value = (c1[0], c1[1], c1[2], 1.0)
    checker.inputs['Color2'].default_value = (c2[0], c2[1], c2[2], 1.0)
    checker.inputs['Scale'].default_value = 1.0  # 用 mapping 控制
    nt.links.new(mosaic_map.outputs['Vector'], checker.inputs['Vector'])

    # --- 3. 把座標 floor 到整數,讓 WhiteNoise 每格給「同一個」隨機值 ---
    # 不 floor 的話 WhiteNoise 會在格子內部產生連續變化 → 變雜訊不是翻格
    floor_node = nt.nodes.new('ShaderNodeVectorMath')
    floor_node.operation = 'FLOOR'
    nt.links.new(mosaic_map.outputs['Vector'], floor_node.inputs[0])

    # --- 4. WhiteNoise: 整數格子座標 → [0, 1] deterministic 隨機 ---
    wn = nt.nodes.new('ShaderNodeTexWhiteNoise')
    wn.noise_dimensions = '3D'
    nt.links.new(floor_node.outputs['Vector'], wn.inputs['Vector'])

    # --- 5. 比較 WhiteNoise < swap_prob → 翻轉 mask ---
    # swap_prob 是「該格被翻轉的機率」,小一點視覺效果比較自然 (太大就變雜訊)
    swap_prob = float(np.random.uniform(0.15, 0.40))
    less_than = nt.nodes.new('ShaderNodeMath')
    less_than.operation = 'LESS_THAN'
    less_than.inputs[1].default_value = swap_prob
    nt.links.new(wn.outputs['Value'], less_than.inputs[0])

    # --- 6. MixRGB: mask=0 走 checker, mask=1 走「反向 checker」(c1/c2 互換) ---
    # 反向 checker: 直接拿 checker 的「Fac」當 mix factor 在 c2/c1 之間選
    # 但 TexChecker 的輸出已經是 Color,要拿到 Fac 比較麻煩;
    # 簡化做法: 再做一個 TexChecker_inv,c1/c2 互換,然後用 mask 在兩者之間混合
    checker_inv = nt.nodes.new('ShaderNodeTexChecker')
    checker_inv.inputs['Color1'].default_value = (c2[0], c2[1], c2[2], 1.0)  # 互換
    checker_inv.inputs['Color2'].default_value = (c1[0], c1[1], c1[2], 1.0)
    checker_inv.inputs['Scale'].default_value = 1.0
    nt.links.new(mosaic_map.outputs['Vector'], checker_inv.inputs['Vector'])

    mix_swap = nt.nodes.new('ShaderNodeMixRGB')
    mix_swap.blend_type = 'MIX'
    nt.links.new(less_than.outputs['Value'], mix_swap.inputs['Fac'])
    nt.links.new(checker.outputs['Color'], mix_swap.inputs['Color1'])
    nt.links.new(checker_inv.outputs['Color'], mix_swap.inputs['Color2'])

    # 灰度估計: 翻轉對稱 → 平均不變
    gray = 0.5 * (_rgb_to_gray(c1) + _rgb_to_gray(c2))
    return mix_swap.outputs['Color'], gray


def _build_patch_mask_subtree(nt, mapping_vector_out):
    """
    用低頻 noise 切出一個「補丁區」mask:
      mask = 1 → 該位置在補丁內 (用馬賽克 checker)
      mask = 0 → 該位置在補丁外 (用主色)

    技術細節:
      - 用 TexNoise 而非 TexVoronoi/Magic,因為 noise 邊緣可以用 ColorRamp
        把銳利度調出來,而 Voronoi 邊緣太硬不像「磨損補丁」
      - ColorRamp 的兩個 stop 位置決定邊緣銳利度與補丁覆蓋率

    Returns:
        mask_socket: ColorRamp 的 output (灰階,可以直接當 mix Fac)
        patch_fraction_estimate: 補丁估計覆蓋率 (用於 floor_gray 加權,但不精確)
    """
    patch_map = nt.nodes.new('ShaderNodeMapping')
    patch_scale = float(np.random.uniform(0.8, 2.0))  # 低頻 → 大塊
    patch_map.inputs['Scale'].default_value = (patch_scale, patch_scale, patch_scale)
    # 旋轉、位移都隨機,避免每張圖補丁都在同樣位置
    patch_map.inputs['Rotation'].default_value[2] = float(np.random.uniform(0, 2 * math.pi))
    patch_map.inputs['Location'].default_value[0] = float(np.random.uniform(-3, 3))
    patch_map.inputs['Location'].default_value[1] = float(np.random.uniform(-3, 3))
    nt.links.new(mapping_vector_out, patch_map.inputs['Vector'])

    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 1.0  # 用 patch_map 控制
    noise.inputs['Detail'].default_value = float(np.random.uniform(0.0, 2.0))
    nt.links.new(patch_map.outputs['Vector'], noise.inputs['Vector'])

    # ColorRamp: 把 noise [0,1] 轉成 sharp-ish mask
    # stop_low 和 stop_high 之間是邊緣過渡區; 兩者越接近邊緣越銳利
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    # patch 中心值約在 0.5,我們希望「補丁區佔約 patch_frac」
    # 簡化: 補丁覆蓋率約 (1 - stop_high) 那段,設 stop_high = 1 - patch_frac
    patch_frac = float(np.random.uniform(0.15, 0.45))  # 補丁佔地板的比例
    edge_softness = float(np.random.uniform(0.02, 0.10))
    stop_high = 1.0 - patch_frac
    stop_low = max(0.0, stop_high - edge_softness)
    ramp.color_ramp.elements[0].position = stop_low
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = stop_high
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])

    return ramp.outputs['Color'], patch_frac


def make_random_floor_material(mat):
    """
    在已存在的材質上重建 node tree,隨機選擇下列三種紋理之一:
      (a) 純色          → 對應 Tobin 2017 (a)
      (b) gradient      → 對應 Tobin 2017 (b)
      (c) checker       → 對應 Tobin 2017 (c)

    v3: 額外有 30% 機率走「半拋光」表面 (低 roughness + 高 specular),
    搭配 low-elevation AREA light 會在地板上產生大片鏡面反光斑,
    模擬真實 PU / 木地板的高光現象 (對應你提供的 Image 2 / Image 3)。

    Returns:
        float: 地板的「代表灰度」(0~1),供線材質決定對比色用。
               純色直接是該色灰度;gradient/checker 取兩色平均。
    """
    # 拿到 underlying bpy 材質,直接操作 node tree
    bpy_mat = mat.blender_obj
    bpy_mat.use_nodes = True
    nt = bpy_mat.node_tree
    # 清空現有 nodes
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    # --- (v3.3: roughness/specular 移到 pattern 抽完後決定,
    #     因為深色地板 + 半拋光 + 強光會被鏡面反射打亮到看不出深色,
    #     需要先知道 floor_gray 才能判斷是否強制走霧面) ---

    # [特效全關版] 移除 checker 馬賽克,只保留 solid / gradient 兩種地板紋理。
    pattern = random.choice(['solid', 'gradient'])

    if pattern == 'solid':
        c = _sample_floor_rgb()
        bsdf.inputs['Base Color'].default_value = (c[0], c[1], c[2], 1.0)
        floor_gray = _rgb_to_gray(c)

    else:  # gradient
        # gradient texture → color ramp(兩個隨機顏色內插)
        tex_coord = nt.nodes.new('ShaderNodeTexCoord')
        mapping = nt.nodes.new('ShaderNodeMapping')
        # 隨機旋轉 gradient 方向
        mapping.inputs['Rotation'].default_value[2] = float(np.random.uniform(0, 2 * math.pi))
        grad = nt.nodes.new('ShaderNodeTexGradient')
        grad.gradient_type = random.choice(['LINEAR', 'EASING', 'DIAGONAL'])
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        c1 = _sample_floor_rgb()
        c2 = _sample_floor_rgb()
        ramp.color_ramp.elements[0].color = (c1[0], c1[1], c1[2], 1.0)
        ramp.color_ramp.elements[1].color = (c2[0], c2[1], c2[2], 1.0)
        nt.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        nt.links.new(mapping.outputs['Vector'], grad.inputs['Vector'])
        nt.links.new(grad.outputs['Color'], ramp.inputs['Fac'])
        nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
        floor_gray = 0.5 * (_rgb_to_gray(c1) + _rgb_to_gray(c2))
    # --- v3.3: 表面物理 (roughness/specular) 在這裡決定 ---
    # 規則:
    #   - 深色地板 (floor_gray < DARK_FLOOR_GRAY_THRESHOLD) 強制走霧面,
    #     避免「深色 + 拋光 + 強光」鏡面反射主導,把深色完全打亮看不見。
    #     真實深色 PU/橡膠表面反射率本來就低,這個耦合也比較合理。
    #   - 其餘地板維持 30% 半拋光 / 70% 霧面。
    DARK_FLOOR_GRAY_THRESHOLD = 0.30
    can_be_polished = floor_gray >= DARK_FLOOR_GRAY_THRESHOLD
    if can_be_polished and np.random.rand() < 0.3:
        # 半拋光: 低 roughness 配高 specular
        bsdf.inputs['Roughness'].default_value = float(np.random.uniform(0.05, 0.25))
        _set_specular(bsdf, np.random.uniform(0.5, 1.0))
    else:
        # 霧面 (包含: 深色被強制 / 70% 自然走霧面)
        bsdf.inputs['Roughness'].default_value = float(np.random.uniform(0.4, 1.0))
        # specular 用 Blender 預設 (0.5),不主動設定

    return float(floor_gray)


def randomize_line_material(mat, floor_gray, min_gray_diff=60 / 255.0):
    """標線材質: 隨機顏色 + 隨機 roughness。
    v3: 強制線色灰度與地板灰度差距 ≥ min_gray_diff,
    避免灰階下線消失於地板。
    """
    c, _ = _sample_rgb_with_gray_avoiding(floor_gray, min_diff=min_gray_diff)
    mat.set_principled_shader_value("Base Color", [c[0], c[1], c[2], 1.0])
    mat.set_principled_shader_value("Roughness", float(np.random.uniform(0.0, 1.0)))


# =============================================================================
# PART 3: 相機 pose 與 intrinsics 隨機化
# =============================================================================

# v5: 前發球線「中央」T 點 (羽球場上每側 1 個,總共 2 個)。
# 不是泛指所有 T 點 —— 只有「短發球線 × 中線」這個特定 T。
# 名字由 calculate_all_junctions 用 LINES_DICT key 組合而成,順序由 LINES_DICT 字典順序決定。
FRONT_SERVICE_T_NAMES = {
    'top_short_service__center_line_top',
    'bottom_short_service__center_line_bottom',
}


def _sample_lookat_target(junctions, l_weight=1.5, front_t_weight=1.5):
    """從 junctions 抽一個作為相機 look-at 中心。
    L 角抽中機率 ×l_weight,因為角落是 vertex localization 最關鍵的特徵。
    v5: 前發球線中央 T 點抽中機率 ×front_t_weight。
        這兩個點是發球攻防焦點,真實轉播/教練錄影的 look-at 機率特別高。
        加權只套用在 FRONT_SERVICE_T_NAMES 列出的 2 個 T 點上,
        不會影響其他 T (如邊線端點 T)。
    """
    weights = []
    for j in junctions:
        if j['type'] == 'L':
            w = l_weight
        elif j['name'] in FRONT_SERVICE_T_NAMES:
            w = front_t_weight
        else:
            w = 1.0
        weights.append(w)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    idx = np.random.choice(len(junctions), p=weights)
    return np.array(junctions[idx]['location'], dtype=float), junctions[idx]['type']


# 三段距離設定: (radius_min, radius_max, elev_min, elev_max, height_min, height_max)
# - close : 教練手機 / 邊線監控 (1~3 m, 低視角)
# - mid   : 看台前排 / 邊角架設攝影機 (3~8 m)
# - far   : 轉播機位 (8~20 m, 高架)
DIST_BUCKETS = {
    'close': dict(r_min=1.5, r_max=3.5, elev_min=10, elev_max=45,
                  h_min=0.8, h_max=2.5),
    'mid':   dict(r_min=3.5, r_max=9.0, elev_min=15, elev_max=60,
                  h_min=1.5, h_max=5.0),
    'far':   dict(r_min=9.0, r_max=20.0, elev_min=20, elev_max=75,
                  h_min=3.0, h_max=10.0),
}

# 三段距離各自的抽樣機率
DIST_PROBS = {'close': 0.30, 'mid': 0.40, 'far': 0.30}


# ── 實驗室 rig 機位（自 12 台已標定相機之 cfg 萃取）──
# 參數化為「相機相對其注視點的偏移向量」(dx,dy,dz)，與各場座標原點無關；
# 幾何包絡：水平距 4.4–8.9 m、高 4.6–5.5 m、仰角 29–47°、FOV_x 57–65.5°。
RIG_VIEWS = [
    dict(tag='court1_0', dx=5.702,  dy=6.294,  dz=5.471, fov_x=60.2),
    dict(tag='court1_1', dx=-4.192, dy=6.826,  dz=4.665, fov_x=65.5),
    dict(tag='court1_2', dx=-4.778, dy=-6.227, dz=4.812, fov_x=65.4),
    dict(tag='court1_3', dx=3.652,  dy=-7.051, dz=4.618, fov_x=65.4),
    dict(tag='court2_0', dx=3.318,  dy=7.938,  dz=5.308, fov_x=59.4),
    dict(tag='court2_1', dx=-3.515, dy=7.972,  dz=5.255, fov_x=59.4),
    dict(tag='court2_2', dx=-2.625, dy=-6.857, dz=5.317, fov_x=59.4),
    dict(tag='court2_3', dx=2.785,  dy=-6.398, dz=5.243, fov_x=59.4),
    dict(tag='court3_0', dx=-2.772, dy=8.428,  dz=4.842, fov_x=63.5),
    dict(tag='court3_1', dx=-3.39,  dy=-6.689, dz=4.803, fov_x=65.4),
    dict(tag='court3_2', dx=3.368,  dy=-2.817, dz=4.759, fov_x=64.2),
    dict(tag='court3_3', dx=2.908,  dy=3.939,  dz=4.985, fov_x=57.0),
]


def add_rig_camera_pose(junctions, pos_jitter=0.5, aim_jitter=1.0,
                        roll_deg=5.0):
    """模擬實驗室 rig 機位（+擾動）。

    1. 注視點 A = 球場中心（junctions 質心）+ 橢圓擾動
       （x: ±1.5 m、y: ±3.0 m，乘 aim_jitter）——保持「看向場中央區」
       之真實 rig 行為，但避免 200 張全看同一點。
    2. 自 RIG_VIEWS 均勻抽一台，相機位置 = A + (dx,dy,dz)
       + 位置擾動（xy: ±pos_jitter、z: ±0.6·pos_jitter，均勻分佈）。
    3. roll ±roll_deg（吊裝相機近水平）。

    回傳 (location, rotation_matrix, view_type, lookat, fov_x)。
    """
    center = np.mean([np.array(j['location'], dtype=float)
                      for j in junctions], axis=0)
    aim = center + np.array([
        np.random.uniform(-1.5, 1.5) * aim_jitter,
        np.random.uniform(-3.0, 3.0) * aim_jitter,
        0.0,
    ])
    v = RIG_VIEWS[np.random.randint(len(RIG_VIEWS))]
    offset = np.array([v['dx'], v['dy'], v['dz']], dtype=float)
    offset[0] += np.random.uniform(-pos_jitter, pos_jitter)
    offset[1] += np.random.uniform(-pos_jitter, pos_jitter)
    offset[2] += np.random.uniform(-0.6 * pos_jitter, 0.6 * pos_jitter)
    location = aim + offset

    forward_vec = aim - location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        forward_vec,
        inplane_rot=float(np.random.uniform(-math.radians(roll_deg),
                                            math.radians(roll_deg)))
    )
    location = np.asarray(location, dtype=float).reshape(-1)[:3]
    aim = np.asarray(aim, dtype=float).reshape(-1)[:3]
    cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)
    return location, rotation_matrix, f"rig@{v['tag']}", aim, float(v['fov_x'])


def add_biased_camera_pose(junctions, l_corner_weight=1.5, front_t_weight=1.5):
    """
    v3: 相機 pose 改為「以場上某個交點為 look-at 中心」的隨機化:
      1. 從所有 junctions 抽一個作為 look-at 目標 (L 角加權 ×1.5)
      2. 從 close / mid / far 三段距離各取一段機率採樣
      3. 相機位置 = look-at 點 + 隨機方向偏移 (azimuth 完全隨機,elevation 受距離段限制)
      4. 高度單獨採樣 (覆蓋掉 elevation 算出的 z),保證每段距離的相機架設高度合理

    v5: 額外對「前發球線中央 T」加權 (×front_t_weight)。

    Returns:
        location: (3,) np.ndarray
        rotation_matrix: (3,3) np.ndarray
        view_type: str (e.g. "close@L", "mid@X", "far@T")
        lookat: (3,) np.ndarray
    """
    # --- A. 選 look-at 目標 ---
    lookat, lookat_type = _sample_lookat_target(
        junctions, l_weight=l_corner_weight, front_t_weight=front_t_weight
    )

    # --- B. 選距離段 ---
    buckets = list(DIST_PROBS.keys())
    probs = [DIST_PROBS[b] for b in buckets]
    bucket = np.random.choice(buckets, p=probs)
    cfg = DIST_BUCKETS[bucket]

    # --- C. 採樣相機相對 look-at 的偏移向量 ---
    # 用 shell 採樣方向 (elevation 控制仰角,azimuth 完全隨機)
    offset = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=cfg['r_min'], radius_max=cfg['r_max'],
        elevation_min=cfg['elev_min'], elevation_max=cfg['elev_max'],
        azimuth_min=-180, azimuth_max=180,
    )
    offset = np.array(offset, dtype=float)

    # 強制 z 重新採樣到合理高度區間 (覆蓋 elevation 算出的 z)
    # 這樣可以解耦「相機距離」與「相機架設高度」,更符合真實裝設情境
    offset[2] = float(np.random.uniform(cfg['h_min'], cfg['h_max']))

    location = lookat + offset

    # --- D. 計算朝向 (從相機看向 look-at 點),加入小 roll ---
    forward_vec = lookat - location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        forward_vec,
        inplane_rot=float(np.random.uniform(-0.26, 0.26))  # ±15°
    )
    cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)

    view_type = f"{bucket}@{lookat_type}"
    return location, rotation_matrix, view_type, lookat


def randomize_camera_intrinsics(image_width, image_height,
                                base_fov_deg_range=(35.0, 75.0)):
    """
    隨機抽選 FOV 並重新設定 intrinsics (focal length),
    讓模型對不同焦段都有 robust 表現。

    重要: principal point (cx, cy) 必須維持在影像正中央。
    過去版本曾對 cx/cy 加 ±2% jitter,但 Blender 內部用 shift_x/shift_y
    (sensor 寬度比例) 表達 principal point 偏移,
    在不同 BlenderProc 版本中渲染端和 project_points 對它的處理可能不一致,
    導致 ground truth 點偏離真實線條交點。
    保持 (cx, cy) = (W/2, H/2) 完全消除此風險。
    Tobin et al. 2017 也只做 FOV 抖動,沒對 principal point 動手。
    """
    fov_deg = float(np.random.uniform(*base_fov_deg_range))
    fov_rad = math.radians(fov_deg)
    # 用 horizontal FOV 反推 fx,假設 pixel 是方的 (fy = fx)
    fx = (image_width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = image_width / 2.0   # 嚴格置中,不加抖動
    cy = image_height / 2.0
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0]
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)
    return K, fov_deg


# =============================================================================
# PART 4: (Removed in v3) Distractor pool
# =============================================================================
# v2 中有 distractor 物件 (cube/sphere/cylinder/cone/monkey) 浮在場景中,
# 用意是訓練模型忽略無關物體。v3 移除這部分,理由:
#   - 真實場景的遮擋來源是「人 / 網柱 / 椅子」等高度結構化物件,
#     飄浮的幾何體並不能讓模型學到真正有用的不變性
#   - 反而可能讓模型學到奇怪的 prior (例如「球場上空可能有圓錐」)
#   - 真正的人形遮擋應該透過載入人體 mesh 或在 .blend 場景中放固定家具
#     來模擬,留待後續版本處理


# =============================================================================
# PART 5: 投影 + Raycasting 算 visibility (核心改動)
# =============================================================================

def is_inside_image(uv, w, h, margin=0):
    u, v = uv
    return (margin <= u < w - margin) and (margin <= v < h - margin)


def compute_keypoint_annotations(
    junctions,
    image_width,
    image_height,
    cam_location,
    frame=None,                 # 明確指定要投影到哪一個 keyframe (None = current frame)
    occlusion_eps=0.02,  # 2 cm:接受射線比目標近這麼多就算「未遮擋」
):
    """
    對每個 3D 交點:
      1. 用 bproc.camera.project_points() 算其像素座標 (u, v)
      2. 若超出畫面 → visibility = 0
      3. 否則從相機發射射線往該點,看是否擊中其他物件
         - 命中距離 ≈ 到點的距離   → visibility = 2 (可見)
         - 命中距離 < 到點的距離   → visibility = 1 (被遮擋)
    回傳: list of dict {name, type, world_xyz, u, v, visibility}
    """
    cam_loc = np.asarray(cam_location, dtype=float).reshape(-1)[:3]
    pts_3d = np.array([j['location'] for j in junctions], dtype=float)  # (N, 3)

    # ----- (1) 數學投影 -----
    # bproc.camera.project_points 回傳 (N, 2) 像素座標
    # 明確傳 frame 確保用的是「剛剛 add_camera_pose 設定」的那一格,
    # 不要依賴 BlenderProc 自己判斷 current frame
    pts_2d = bproc.camera.project_points(pts_3d, frame=frame)

    annots = []
    for j, p3d, uv in zip(junctions, pts_3d, pts_2d):
        u, v = float(uv[0]), float(uv[1])
        inside = is_inside_image((u, v), image_width, image_height, margin=0)

        if not inside or not np.isfinite(u) or not np.isfinite(v):
            visibility = 0
        else:
            # ----- (2) Raycasting 判遮擋 -----
            ray_dir = p3d - cam_loc
            target_dist = float(np.linalg.norm(ray_dir))
            if target_dist < 1e-6:
                visibility = 2
            else:
                ray_dir_norm = ray_dir / target_dist
                # 不同 BlenderProc 版本回傳元組長度不一 (5 或 6),
                # 我們只需要前兩個:hit (bool) 和 hit_loc (3D 座標)
                ray_result = bproc.object.scene_ray_cast(
                    origin=[float(x) for x in np.asarray(cam_loc).reshape(-1)[:3]],
                    direction=[float(x) for x in np.asarray(ray_dir_norm).reshape(-1)[:3]],
                )
                hit = ray_result[0]
                hit_loc = ray_result[1]
                if not hit:
                    # 沒打到任何東西 (理論上不該發生,因為地板會擋,
                    # 但有可能因為標線材質透明度等問題)
                    visibility = 2
                else:
                    hit_dist = float(np.linalg.norm(np.array(hit_loc) - cam_loc))
                    # 若射線比目標點更早被擋住 → 被遮擋
                    if hit_dist < target_dist - occlusion_eps:
                        visibility = 1
                    else:
                        visibility = 2

        annots.append({
            'name': j['name'],
            'type': j['type'],
            'world_xyz': [float(p3d[0]), float(p3d[1]), float(p3d[2])],
            'u': u,
            'v': v,
            'visibility': int(visibility),
        })

    return annots




def compute_corner_annotations(
    corners,
    image_width,
    image_height,
    cam_location,
    frame=None,
    occlusion_eps=0.02,
):
    """
    對每個 3D 白線外緣角點：
      1. project_points → sub-pixel 像素座標
      2. 畫面外 visibility=0
      3. raycasting 判斷是否被遮擋，visibility ∈ {1,2}

    回傳 list of dict，可直接作為 runner GT JSON 的 corners 欄位。
    """
    cam_loc = np.asarray(cam_location, dtype=float).reshape(-1)[:3]
    pts_3d = np.array([c['world_xyz'] for c in corners], dtype=float)
    pts_2d = bproc.camera.project_points(pts_3d, frame=frame)

    annots = []
    for c, p3d, uv in zip(corners, pts_3d, pts_2d):
        u, v = float(uv[0]), float(uv[1])
        inside = is_inside_image((u, v), image_width, image_height, margin=0)

        if not inside or not np.isfinite(u) or not np.isfinite(v):
            visibility = 0
        else:
            ray_dir = p3d - cam_loc
            target_dist = float(np.linalg.norm(ray_dir))
            if target_dist < 1e-6:
                visibility = 2
            else:
                ray_dir_norm = ray_dir / target_dist
                ray_result = bproc.object.scene_ray_cast(
                    origin=[float(x) for x in np.asarray(cam_loc).reshape(-1)[:3]],
                    direction=[float(x) for x in np.asarray(ray_dir_norm).reshape(-1)[:3]],
                )
                hit = ray_result[0]
                hit_loc = ray_result[1]
                if not hit:
                    visibility = 2
                else:
                    hit_dist = float(np.linalg.norm(np.array(hit_loc) - cam_loc))
                    visibility = 1 if hit_dist < target_dist - occlusion_eps else 2

        annots.append({
            **c,
            'x': u,                 # runner 直接讀 x/y
            'y': v,
            'u': u,
            'v': v,
            'visible': visibility == 2,
            'valid': visibility == 2,
            'visibility': int(visibility),
            'status': 'visible' if visibility == 2 else ('occluded' if visibility == 1 else 'outside'),
        })
    return annots


def compute_template_junction_annotations(
    image_width,
    image_height,
    cam_location,
    frame=None,
    occlusion_eps=0.02,
):
    """
    以 pipeline 的 30 個 row-major TEMPLATE_POINTS 產生 junction GT。
    這份資料與 Homography / corner_code 的 template id 對齊。
    """
    template_junctions = []
    for idx, p in enumerate(TEMPLATE_POINTS):
        world_xyz = corner_template_xy_to_blender_xyz(p)
        jt_id = int(TEMPLATE_TYPES[idx])
        template_junctions.append({
            'junction_idx': int(idx),
            'type_id': jt_id,
            'type': {0: 'L', 1: 'T', 2: 'X'}[jt_id],
            'template_xy': [float(p[0]), float(p[1])],
            'world_xyz': [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])],
        })

    cam_loc = np.asarray(cam_location, dtype=float).reshape(-1)[:3]
    pts_3d = np.array([j['world_xyz'] for j in template_junctions], dtype=float)
    pts_2d = bproc.camera.project_points(pts_3d, frame=frame)

    annots = []
    for j, p3d, uv in zip(template_junctions, pts_3d, pts_2d):
        u, v = float(uv[0]), float(uv[1])
        inside = is_inside_image((u, v), image_width, image_height, margin=0)
        if not inside or not np.isfinite(u) or not np.isfinite(v):
            visibility = 0
        else:
            ray_dir = p3d - cam_loc
            target_dist = float(np.linalg.norm(ray_dir))
            if target_dist < 1e-6:
                visibility = 2
            else:
                ray_dir_norm = ray_dir / target_dist
                ray_result = bproc.object.scene_ray_cast(origin=[float(x) for x in np.asarray(cam_loc).reshape(-1)[:3]], direction=[float(x) for x in np.asarray(ray_dir_norm).reshape(-1)[:3]])
                hit = ray_result[0]
                hit_loc = ray_result[1]
                if not hit:
                    visibility = 2
                else:
                    hit_dist = float(np.linalg.norm(np.array(hit_loc) - cam_loc))
                    visibility = 1 if hit_dist < target_dist - occlusion_eps else 2
        annots.append({
            **j,
            'x': u,
            'y': v,
            'u': u,
            'v': v,
            'visible': visibility == 2,
            'valid': visibility == 2,
            'visibility': int(visibility),
            'status': 'visible' if visibility == 2 else ('occluded' if visibility == 1 else 'outside'),
        })
    return annots


def estimate_h_template_to_image_from_template_junctions(template_junction_annots):
    """由 30 個 template junction 投影點估計 GT H，用於 oracle H 評估。"""
    src = []
    dst = []
    for a in template_junction_annots:
        if np.isfinite(a['u']) and np.isfinite(a['v']):
            src.append(a['template_xy'])
            dst.append([a['u'], a['v']])
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64), 0)
    if H is None:
        return None
    return H.astype(float).tolist()

def annots_to_coco_records(annots, image_width, image_height, bbox_half=8):
    """
    把每個交點轉成一筆 COCO annotation:
      - bbox: 以 (u,v) 為中心的固定大小方框 (2*bbox_half × 2*bbox_half)
              clip 到影像範圍內。bbox_half 預設 8 px ≈ YOLO 適合的小目標尺寸。
      - keypoints: [u, v, visibility]
    類別 ID: L=1, T=2, X=3
    僅輸出 visibility >= 1 的點 (在畫面內,可能可見或被遮擋)。
    """
    class_map = {'L': 1, 'T': 2, 'X': 3}
    records = []
    for a in annots:
        if a['visibility'] == 0:
            continue  # 不在畫面內就不寫入
        u, v = a['u'], a['v']
        x = max(0.0, u - bbox_half)
        y = max(0.0, v - bbox_half)
        w = min(2 * bbox_half, image_width - x)
        h = min(2 * bbox_half, image_height - y)
        if w <= 0 or h <= 0:
            continue
        records.append({
            'category_id': class_map[a['type']],
            'bbox': [float(x), float(y), float(w), float(h)],
            'area': float(w * h),
            'iscrowd': 0,
            'keypoints': [float(u), float(v), int(a['visibility'])],
            'num_keypoints': 1 if a['visibility'] > 0 else 0,
            'junction_name': a['name'],
            'junction_type': a['type'],
        })
    return records


# =============================================================================
# PART 6: 影像增強 (v3: 對齊 aug_preview.py 工具的成像鏈)
# =============================================================================
# 設計原則:
#   1. 所有 augmentation 都是 pure-pixel,不改變幾何,保留 sub-pixel 標註精度
#      (沒有 wave / perspective / elastic)
#   2. 按真實成像鏈順序套用: 光照 → 鏡頭 → sensor → 壓縮
#   3. 每個 augmentation 獨立機率啟用,參數從預覽工具確認過的範圍內隨機抽
#   4. 反光斑由 Blender 端的 specular material + 低仰角 AREA light 處理,
#      不在此處做後處理

def aug_gamma(img, gamma):
    """gamma > 1 → 變暗; gamma < 1 → 變亮"""
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def aug_brightness_contrast(img, brightness, contrast):
    """brightness: -100~100; contrast: 0.5~2.0"""
    out = img.astype(np.float32)
    out = (out - 127.5) * contrast + 127.5 + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_vignette(img, strength):
    """strength 0~1, 徑向 (r/r_max)^2 衰減"""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    r2 = ((X - cx) ** 2 + (Y - cy) ** 2) / (cx ** 2 + cy ** 2)
    mask = np.clip(1.0 - strength * r2, 0, 1).astype(np.float32)
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def aug_gaussian_blur(img, sigma):
    """isotropic defocus blur"""
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def aug_motion_blur(img, ksize, angle_deg):
    """方向可調的 motion blur"""
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    M = cv2.getRotationMatrix2D(
        (ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    s = kernel.sum()
    if s > 1e-6:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def aug_gaussian_noise(img, sigma):
    """高斯 sensor noise"""
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_snow_noise(img, density, size=1):
    """雪花 (salt) noise - 保留 v2 邏輯"""
    if density == 0:
        return img
    output = img.copy()
    h, w = output.shape[:2]
    num = int((density / 1000.0) * (h * w))
    ys = np.random.randint(0, h, num)
    xs = np.random.randint(0, w, num)
    for x, y in zip(xs, ys):
        bright = np.random.randint(200, 256)
        if len(img.shape) == 2:
            output[y, x] = bright
        else:
            cv2.circle(output, (x, y), size, (bright,), -1)
    return output


def apply_local_occluders_at_junctions(
    img, annots, n_min=1, n_max=3,
    radius_range=(5, 25),
    bright_prob=0.5,
    bright_range=(200, 255),
    dark_range=(0, 60),
    soft_edge_prob=0.5,
):
    """
    在 1~3 個「可見」交點周圍隨機畫一個圓盤,模擬局部反光斑 / 髒污 / 貼紙遮擋。

    與 apply_snow_noise 的差別:
      - snow: 空間均勻撒小亮點 → 模擬 sensor hot pixel
      - 本函式: 結構化地打在 keypoint GT 位置 → 模擬局部 occlusion,
               同時把該交點 visibility 從 2 (visible) 降為 1 (occluded),
               避免「明明被蓋住卻教 model 預測 visible」的標籤不一致。

    Args:
        img: H×W (gray) 或 H×W×C 的 uint8 影像
        annots: compute_keypoint_annotations() 的輸出,會被「就地修改」visibility
        n_min, n_max: 要打幾個交點 (從 visibility==2 的交點中隨機抽)
        radius_range: 圓盤半徑 (px) 的均勻抽樣範圍
        bright_prob: 亮斑機率 (其餘為暗斑)
        bright_range / dark_range: 圓盤填充亮度
        soft_edge_prob: 用 Gaussian blur 把邊緣糊掉的機率,看起來更像反光而非貼紙

    Returns:
        (output_img, annots)  annots 中被蓋到的點 visibility 已降為 1
    """
    # 只挑「目前 visible」的點來打,被遮擋或出畫面的就不挑
    visible_idx = [i for i, a in enumerate(annots) if a['visibility'] == 2]
    if not visible_idx:
        return img, annots

    n = random.randint(n_min, min(n_max, len(visible_idx)))
    chosen = random.sample(visible_idx, n)

    output = img.copy()
    h, w = output.shape[:2]

    for idx in chosen:
        a = annots[idx]
        cx, cy = int(round(a['u'])), int(round(a['v']))
        r = random.randint(*radius_range)

        # 亮 or 暗
        if random.random() < bright_prob:
            val = random.randint(*bright_range)
        else:
            val = random.randint(*dark_range)

        # 畫在獨立的 mask 上,方便做軟邊
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        if random.random() < soft_edge_prob:
            # Gaussian blur mask → 邊緣不再是硬邊,像反光暈開
            k = max(3, (r // 2) * 2 + 1)  # 奇數 kernel
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # alpha blend: alpha = mask/255
        alpha = mask.astype(np.float32) / 255.0
        if output.ndim == 2:
            output = (output.astype(np.float32) * (1 - alpha)
                      + val * alpha).clip(0, 255).astype(np.uint8)
        else:
            for c in range(output.shape[2]):
                output[..., c] = (output[..., c].astype(np.float32) * (1 - alpha)
                                  + val * alpha).clip(0, 255).astype(np.uint8)

        # 標籤同步: 蓋到了就標 occluded
        a['visibility'] = 1

    return output, annots


def aug_jpeg(img, quality):
    """JPEG 壓縮 artifact"""
    if quality >= 100:
        return img
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img
    if len(img.shape) == 2:
        return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# 每個 augmentation 在「啟用 augmentation 的樣本」中,個別啟用的機率與參數範圍。
# 機率乘 (1 - p_no_aug) 才是該 aug 在整個 dataset 的最終啟用率。
AUG_CFG = {
    'gamma':       {'p': 0.7, 'range': (0.6, 1.6)},
    'brightness':  {'p': 0.5, 'range': (-40, 40)},
    'contrast':    {'p': 0.5, 'range': (0.7, 1.4)},
    'vignette':    {'p': 0.4, 'range': (0.2, 0.6)},
    'gblur':       {'p': 0.4, 'range': (0.5, 2.5)},      # sigma
    'mblur':       {'p': 0.3, 'k_choices': [3, 5, 7, 9, 11], 'angle': (-30, 30)},
    'gnoise':      {'p': 0.6, 'range': (2.0, 12.0)},     # sigma
    'snow':        {'p': 0.15, 'range': (5, 80)},        # density (‰)
    'local_occ':   {'p': 0.25,                            # 局部交點遮擋
                    'n_range': (1, 3),
                    'radius_range': (5, 25),
                    'bright_prob': 0.5},
    'jpeg':        {'p': 0.4, 'range': (40, 90)},        # quality
    # v3.4: 弱光模式 (每張圖 25% 機率)。
    # 啟動時 main() 會抽是否進入弱光,然後同步調整光源 energy/盞數,
    # augment_image 在弱光模式下會強制觸發 gamma 與 brightness 往暗的方向走。
    'low_light':   {'p': 0.25,
                    'energy_range': (300, 800),     # 光源 energy 下限上限
                    'n_lights_range': (1, 3),       # 弱光時的盞數
                    'gamma_range': (1.3, 2.2),      # 強制 gamma > 1 (壓暗)
                    'brightness_range': (-50, -10)},  # 強制 brightness < 0
}


def augment_image(image, annots=None, p_no_aug=0.4, low_light=False):
    """[特效全關版] 僅做灰階轉換,不套用任何 2D 成像鏈強化。

    本版本刻意關閉所有 2D 後處理特效 (gamma / brightness / contrast /
    vignette / defocus / motion blur / gaussian noise / snow /
    local occluder / JPEG 壓縮) 與弱光模式。
    多樣性僅來自 3D 端:相機鏡位、場地光源、地板/線材質變化。

    Args / Returns 維持與原版相同簽名,確保 main() 不需改動:
        image: H×W×C BGR 或 H×W gray 的 uint8
        annots: 原樣回傳 (不再被 local_occ 修改,visibility 保持渲染端結果)
        p_no_aug / low_light: 保留參數但不再生效
    Returns:
        (out_img, annots)
    """
    # 只轉灰階,其餘特效全部關閉
    if len(image.shape) == 3:
        out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        out = image.copy()
    return out, annots


# =============================================================================
# PART 7: COCO writer (手寫,因為 bproc.writer 的 bbox 來源是 segmap)
# =============================================================================

class CocoWriter:
    """累積式 COCO writer:每張圖 add 一次,結束再 dump。"""

    CATEGORIES = [
        {'id': 1, 'name': 'L', 'supercategory': 'court_junction',
         'keypoints': ['junction'], 'skeleton': []},
        {'id': 2, 'name': 'T', 'supercategory': 'court_junction',
         'keypoints': ['junction'], 'skeleton': []},
        {'id': 3, 'name': 'X', 'supercategory': 'court_junction',
         'keypoints': ['junction'], 'skeleton': []},
    ]

    def __init__(self, coco_dir, image_dir_name='images'):
        self.coco_dir = coco_dir
        self.image_dir = os.path.join(coco_dir, image_dir_name)
        os.makedirs(self.image_dir, exist_ok=True)
        self.images = []
        self.annotations = []
        self._next_ann_id = 1

    def add_sample(self, image_id, file_name, width, height,
                   image_array, ann_records):
        # 寫影像
        path = os.path.join(self.image_dir, file_name)
        # OpenCV 預期 BGR;這裡若是灰階 (H,W) 直接寫即可
        cv2.imwrite(path, image_array)

        self.images.append({
            'id': image_id,
            'file_name': os.path.join(os.path.basename(self.image_dir), file_name),
            'width': int(width),
            'height': int(height),
        })

        for r in ann_records:
            ann = {
                'id': self._next_ann_id,
                'image_id': image_id,
                **r,
            }
            self.annotations.append(ann)
            self._next_ann_id += 1

    def dump(self, json_name='coco_annotations.json'):
        coco = {
            'info': {
                'description': 'Badminton court junctions (L/T/X) generated via BlenderProc with domain randomization.',
            },
            'licenses': [],
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.CATEGORIES,
        }
        out_path = os.path.join(self.coco_dir, json_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"  -> Wrote {out_path}  "
              f"({len(self.images)} images, {len(self.annotations)} annotations)")




class CornerGTWriter:
    """輸出每張影像的 corners GT JSON，以及一份 aggregate JSON。"""

    def __init__(self, output_dir, per_image_dir_name='corner_gt'):
        self.output_dir = output_dir
        self.gt_dir = os.path.join(output_dir, per_image_dir_name)
        os.makedirs(self.gt_dir, exist_ok=True)
        self.samples = []

    def add_sample(self, sample_id, file_name, image_width, image_height,
                   corners, junctions, template_junctions, H_gt,
                   camera_info=None):
        record = {
            'image_id': int(sample_id),
            'image': file_name,
            'width': int(image_width),
            'height': int(image_height),
            'line_width_m': float(LINE_WIDTH_M),
            'H_template_to_image': H_gt,
            'camera': camera_info or {},
            'corners': corners,
            'junctions': junctions,
            'template_junctions': template_junctions,
        }
        out_name = os.path.splitext(file_name)[0] + '.json'
        out_path = os.path.join(self.gt_dir, out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.samples.append(record)

    def dump(self, json_name='corner_gt_all.json'):
        summary = {
            'description': 'Synthetic badminton court outer-corner GT. cid is pipeline corner_code.',
            'line_width_m': float(LINE_WIDTH_M),
            'n_samples': len(self.samples),
            'samples': self.samples,
        }
        out_path = os.path.join(self.output_dir, json_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  -> Wrote {out_path} ({len(self.samples)} samples)")

# =============================================================================
# PART 8: 主邏輯
# =============================================================================

def main(args):
    bproc.init()

    # ----- 載入場景 -----
    objs = bproc.loader.load_blend(args.blend_file)
    print("=== Loaded objects ===")
    for o in objs:
        try:
            print(" -", o.get_name())
        except Exception:
            pass
    print("======================")

    # 移除原本所有 lights,改用我們自己加的
    existing_lights = bproc.filter.all_with_type(objs, bproc.types.Light)
    if len(existing_lights) > 0:
        bproc.object.delete_multiple(existing_lights)
        print(f"Deleted {len(existing_lights)} existing lights from scene.")

    court_floor = bproc.filter.one_by_attr(objs, "name", "Court_Floor")
    if court_floor is None:
        raise RuntimeError("找不到名為 'Court_Floor' 的物件,請檢查 .blend 中物件名稱。")
    line_objects = [o for o in objs if "Line" in o.get_name()]

    # 建立可重複使用的材質 (節點在每張圖會被重建)
    court_mat = bproc.material.create("CourtFloorDyn")
    line_mat = bproc.material.create("CourtLineDyn")
    court_floor.replace_materials(court_mat)
    for l_obj in line_objects:
        l_obj.replace_materials(line_mat)

    # 場景中其他物件預設不進 COCO
    for o in objs:
        o.set_cp("category_id", 0)

    # ----- (v3: distractor pool 已移除,見 PART 4 註解) -----

    # ----- 光源策略 (v3.1) -----
    # 改動: 不再 main 啟動時預建固定盞數,改為每張圖隨機 [n_min, n_max] 盞,
    #       每盞 type 也每張重抽 (AREA / POINT),
    #       色溫和能量分布拉開,模擬真實場館「主光 + 補光 + 點光」混合。
    # 上一輪每張圖內所有光源 color 都用 np.random.rand(3),導致色光偏濃;
    # 這版改為 70% 走「接近白光 + 微色溫」、30% 走「強色光」(維持隨機化精神)。
    dynamic_lights = []  # 用一個 list 累積每張圖建立的光源,渲染後刪除

    # ----- 渲染共用設定 -----
    bproc.renderer.set_output_format(enable_transparency=False)
    # v3: noise_threshold 0.01 → 0.005, 避免低 roughness 表面產生 firefly
    bproc.renderer.set_noise_threshold(0.005)

    img_w, img_h = args.image_width, args.image_height

    # ----- 計算所有交點與白線外緣角點 (整個 dataset 共用) -----
    junctions = calculate_all_junctions()
    gt_corners_template = calculate_all_corners(line_width_m=LINE_WIDTH_M)
    print(f"Total junctions defined: {len(junctions)}")
    print(f"Total valid outer-corners defined: {len(gt_corners_template)}")

    # ----- 輸出目錄 -----
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    coco_dir = os.path.join(output_dir, 'coco_data')
    coco = CocoWriter(coco_dir)
    corner_gt_writer = CornerGTWriter(output_dir)

    # 相機 pose CSV (v3: 多紀錄 lookat 點)
    csv_path = os.path.join(output_dir, 'camera_poses.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'sample_id', 'view_type', 'fov_deg',
        'pos_x', 'pos_y', 'pos_z',
        'euler_x_deg', 'euler_y_deg', 'euler_z_deg',
        'lookat_x', 'lookat_y', 'lookat_z',
        'n_visible', 'n_occluded', 'n_outside',
    ])

    print(f"Start generating {args.num_samples} samples...\n")

    for i in range(args.num_samples):
        sample_id = i + 1
        print(f"[Sample {sample_id}/{args.num_samples}]")

        bproc.utility.reset_keyframes()

        # --- A. 相機 pose (v3: look-at 場上一點 + L 角加權 + 三段距離) ---
        cam_loc, cam_rot, view_type, lookat = add_biased_camera_pose(
            junctions=junctions,
            l_corner_weight=args.l_corner_weight,
            front_t_weight=args.front_service_t_weight,
        ) if args.camera_mode == 'random' else (None, None, None, None)
        rig_fov = None
        if args.camera_mode == 'rig':
            cam_loc, cam_rot, view_type, lookat, rig_fov = \
                add_rig_camera_pose(junctions,
                                    pos_jitter=args.rig_pos_jitter,
                                    aim_jitter=args.rig_aim_jitter)

        # --- B. 相機 intrinsics (必須在 add_camera_pose 之後設定,
        #     否則 reset_keyframes 之後 intrinsics 可能被綁到錯誤的 frame,
        #     導致投影的 K 跟實際渲染用的 K 不一致 → 點偏移) ---
        if args.camera_mode == 'rig':
            K, fov_deg = randomize_camera_intrinsics(
                img_w, img_h,
                base_fov_deg_range=(rig_fov - 3.0, rig_fov + 3.0))
        else:
            K, fov_deg = randomize_camera_intrinsics(img_w, img_h)

        # --- C. 燈光隨機化 (v3.1: 每張圖重建,盞數隨機,type 隨機) ---
        # 為什麼要每張圖重建 (而非 set_location 重設):
        #   1. 原版盞數固定 → 多樣性不足,model 容易過擬合到「3 盞燈」的光照分布
        #   2. 原版每盞燈的 type (AREA/POINT) 在 main 啟動時抽一次就固定 →
        #      整個 dataset 每張圖的「主光方向性」分布幾乎一樣,失去隨機化目的
        # 反光斑、能量、色溫、仰角範圍等行為跟 v3 一致。
        #
        # v3.4: 每張圖 25% 機率走「弱光模式」,
        #       此時光源 energy 降低、盞數減少,
        #       並把 is_low_light 傳給 augment_image 強制 gamma+brightness 變暗。

        # [特效全關版] 弱光模式已停用 (弱光屬於影像特效,本版只保留正常場地光)。
        # 永遠走正常光源 energy 與使用者指定盞數範圍。
        is_low_light = False
        energy_range = (800, 3000)
        n_lights_lo, n_lights_hi = args.n_lights_min, args.n_lights_max

        # 先把上一張的光源刪掉 (第一張時 list 為空,跳過)
        if dynamic_lights:
            bproc.object.delete_multiple(dynamic_lights)
            dynamic_lights = []

        # 本張盞數
        n_lights = random.randint(n_lights_lo, n_lights_hi)

        for _ in range(n_lights):
            light = bproc.types.Light()
            # 70% AREA (方向性反光斑) + 30% POINT (補環境光) —— 每盞獨立抽
            is_area = np.random.rand() < 0.7
            light.set_type("AREA" if is_area else "POINT")

            loc = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=5, radius_max=25,
                elevation_min=15, elevation_max=80,
            )
            light.set_location(loc)
            # v3.4: energy 範圍依 is_low_light 決定
            light.set_energy(float(np.random.uniform(*energy_range)))

            # 色溫策略: 70% 接近白光 (微色溫漂移), 30% 強色光
            # 接近白光: 三個 channel 都在 0.75~1.0,模擬真實場館鹵素 / LED
            # 強色光: 維持原本 np.random.rand(3) 的廣域隨機化
            if np.random.rand() < 0.7:
                light.set_color(np.random.uniform(0.75, 1.0, size=3).tolist())
            else:
                light.set_color(np.random.rand(3).tolist())

            # AREA light 額外: 隨機 size + 朝下對準場地中心
            if is_area:
                light.blender_obj.data.size = float(np.random.uniform(0.5, 3.0))
                forward = -np.array(loc, dtype=float)
                rot = bproc.camera.rotation_from_forward_vec(forward)
                light.blender_obj.rotation_euler = (
                    Rotation.from_matrix(rot).as_euler('xyz').tolist()
                )

            dynamic_lights.append(light)

        # --- D. 材質隨機化 (v3: 線色根據地板色決定,確保灰階對比) ---
        floor_gray = make_random_floor_material(court_mat)
        randomize_line_material(line_mat, floor_gray=floor_gray,
                                min_gray_diff=args.min_gray_diff)

        # --- E. (v3: distractor 已移除) ---

        # --- F. 渲染 ---
        data = bproc.renderer.render()  # 只要 RGB 即可
        rgb = data["colors"][0]  # 一個 frame

        # --- G. 投影 + raycasting 算 keypoints / corners ---
        # frame=0 因為我們每張圖都 reset_keyframes,所以 add_camera_pose 寫進 frame 0
        annots = compute_keypoint_annotations(
            junctions, img_w, img_h,
            cam_location=cam_loc,
            frame=0,
        )
        template_junction_annots = compute_template_junction_annotations(
            img_w, img_h,
            cam_location=cam_loc,
            frame=0,
        )
        corner_annots = compute_corner_annotations(
            gt_corners_template, img_w, img_h,
            cam_location=cam_loc,
            frame=0,
        )
        H_gt = estimate_h_template_to_image_from_template_junctions(template_junction_annots)

        # --- H. 影像增強 (在 coco_records / corner_gt 之前! local_occ 會改 visibility) ---
        # local occluder 若啟用，必須同步作用於 junction 與 corner，否則角點 GT 會與影像不一致。
        combined_annots_for_occ = annots + corner_annots
        aug, _ = augment_image(rgb, annots=combined_annots_for_occ, low_light=is_low_light)  # 灰階 H×W

        # --- I. annots → coco records (用「augment 後」的 visibility) ---
        coco_records = annots_to_coco_records(
            annots, img_w, img_h, bbox_half=args.bbox_half
        )

        # 統計 (放在 augment 之後,反映實際標籤)
        n_vis = sum(1 for a in annots if a['visibility'] == 2)
        n_occ = sum(1 for a in annots if a['visibility'] == 1)
        n_out = sum(1 for a in annots if a['visibility'] == 0)
        light_tag = "LOW_LIGHT" if is_low_light else "normal"
        print(f"  junctions: visible={n_vis} occluded={n_occ} outside={n_out}  "
              f"fov={fov_deg:.1f}°  view={view_type}  lights={n_lights}({light_tag})")

        # --- J. 寫 COCO + 角點 GT ---
        file_name = f"sample_{sample_id:06d}.png"
        coco.add_sample(sample_id, file_name, img_w, img_h, aug, coco_records)

        camera_info = {
            'K': K.astype(float).tolist(),
            'fov_deg': float(fov_deg),
            'cam_location': [float(x) for x in cam_loc],
            'view_type': view_type,
            'lookat': [float(x) for x in lookat],
        }
        corner_gt_writer.add_sample(
            sample_id=sample_id,
            file_name=file_name,
            image_width=img_w, image_height=img_h,
            corners=corner_annots,
            junctions=annots,
            template_junctions=template_junction_annots,
            H_gt=H_gt,
            camera_info=camera_info,
        )

        # --- K. CSV ---
        euler_deg = Rotation.from_matrix(cam_rot).as_euler('xyz', degrees=True)
        csv_writer.writerow([
            sample_id, view_type, round(float(fov_deg), 4),
            round(float(cam_loc[0]), 6), round(float(cam_loc[1]), 6), round(float(cam_loc[2]), 6),
            round(float(euler_deg[0]), 4), round(float(euler_deg[1]), 4), round(float(euler_deg[2]), 4),
            round(float(lookat[0]), 6), round(float(lookat[1]), 6), round(float(lookat[2]), 6),
            n_vis, n_occ, n_out,
        ])

    csv_file.close()
    coco.dump()
    print(f"\nDone. Output directory: {output_dir}")
    print(f"  - Images + COCO json: {coco_dir}")
    print(f"  - Camera poses: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('blend_file', help='Path to the .blend file')
    parser.add_argument('--output_dir', default='./output', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of images to render')
    parser.add_argument('--image_width', type=int, default=640)
    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--bbox_half', type=int, default=8,
                        help='Half-size of bbox in pixels around each junction')
    parser.add_argument('--n_lights_min', type=int, default=3,
                        help='Minimum number of dynamic lights per sample. '
                             'Each sample randomly draws n_lights ~ U[n_lights_min, n_lights_max].')
    parser.add_argument('--n_lights_max', type=int, default=6,
                        help='Maximum number of dynamic lights per sample.')
    parser.add_argument('--num_lights', type=int, default=None,
                        help='[DEPRECATED] Use --n_lights_min/--n_lights_max instead. '
                             'If set, overrides both to this fixed value.')
    parser.add_argument('--camera_mode', choices=['random', 'rig'],
                        default='random',
                        help="random=原三段距離隨機取樣；"
                             "rig=模擬實驗室 12 台已標定機位（含擾動）")
    parser.add_argument('--rig_pos_jitter', type=float, default=0.5,
                        help='rig 模式：相機位置擾動幅度 (m)，xy ±j、z ±0.6j')
    parser.add_argument('--rig_aim_jitter', type=float, default=1.0,
                        help='rig 模式：注視點橢圓擾動倍率（x ±1.5m、y ±3.0m 之倍率）')
    parser.add_argument('--l_corner_weight', type=float, default=1.5,
                        help='Sampling weight multiplier for L-corner junctions '
                             'when picking camera look-at target (>1 favors corners)')
    parser.add_argument('--front_service_t_weight', type=float, default=1.5,
                        help='Sampling weight multiplier for the 2 front-service-line center '
                             'T-junctions (top_short_service x center_line, top/bottom). '
                             'Predefined by name in FRONT_SERVICE_T_NAMES, does not affect '
                             'other T-junctions. Default matches --l_corner_weight.')
    parser.add_argument('--min_gray_diff', type=float, default=60 / 255.0,
                        help='Minimum grayscale brightness difference between floor and line '
                             '(0~1 scale). Prevents lines from disappearing in grayscale.')
    args = parser.parse_args()

    # 處理 deprecated --num_lights
    if args.num_lights is not None:
        print(f"[WARN] --num_lights is deprecated, using fixed n_lights={args.num_lights}")
        args.n_lights_min = args.num_lights
        args.n_lights_max = args.num_lights

    if args.n_lights_min > args.n_lights_max:
        raise ValueError(
            f"--n_lights_min ({args.n_lights_min}) > --n_lights_max ({args.n_lights_max})"
        )
    if args.n_lights_min < 1:
        raise ValueError(f"--n_lights_min must be >= 1, got {args.n_lights_min}")

    main(args)
