"""
Court Model - 標準羽毛球場地定義與拓撲
==========================================
包含:
- 標準場地點位定義 (TEMPLATE_POINTS, TEMPLATE_TYPES)
- 網格連接 (GRID_CONNECTIONS)
- Junction 拓撲 (JUNCTION_INCIDENT)
- Junction 主方向計算
- 對稱性懲罰計算
"""

import numpy as np

# ================= 標準場地定義 =================

TEMPLATE_POINTS = np.array([
    [0.00, 13.40], [0.46, 13.40], [3.05, 13.40], [5.64, 13.40], [6.10, 13.40],
    [0.00, 12.64], [0.46, 12.64], [3.05, 12.64], [5.64, 12.64], [6.10, 12.64],
    [0.00, 8.68], [0.46, 8.68], [3.05, 8.68], [5.64, 8.68], [6.10, 8.68],
    [0.00, 4.72], [0.46, 4.72], [3.05, 4.72], [5.64, 4.72], [6.10, 4.72],
    [0.00, 0.76], [0.46, 0.76], [3.05, 0.76], [5.64, 0.76], [6.10, 0.76],
    [0.00, 0.00], [0.46, 0.00], [3.05, 0.00], [5.64, 0.00], [6.10, 0.00],
], dtype=np.float32)

TEMPLATE_TYPES = np.array([
    0, 1, 1, 1, 0,
    1, 2, 2, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 1, 2, 1,
    1, 2, 2, 2, 1,
    0, 1, 1, 1, 0,
])

LINE_WIDTH_M = 0.04  # 40mm

# ================= 格網尺寸與投影小工具 =================
# （供畫格線 / 投影使用；junction_idx = r * N_COL + c）
N_COL, N_ROW = 5, 6


def _tpl_xy(r, c):
    """範本格 (r, c) 的場地座標 (公尺)。"""
    p = TEMPLATE_POINTS[r * N_COL + c]
    return (float(p[0]), float(p[1]))


def _proj(H, X):
    """以單應矩陣 H 把場地點 X=(x,y) 投影到影像座標。"""
    H = np.asarray(H, dtype=np.float64)
    v = H @ np.array([X[0], X[1], 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-12:
        return (float("nan"), float("nan"))
    return (float(v[0] / v[2]), float(v[1] / v[2]))


# ================= 網格連接 =================

def build_grid_connections():
    """建立網格連接"""
    conns = []
    for r in range(6):
        for c in range(4):
            conns.append((r * 5 + c, r * 5 + c + 1))
    for c in range(5):
        for r in range(5):
            if not (c == 2 and r == 2):  # 中間斷開
                conns.append((r * 5 + c, (r + 1) * 5 + c))
    return conns


GRID_CONNECTIONS = build_grid_connections()


def build_junction_incident_edges():
    """
    建立每個 junction 的 incident edges
    回傳 dict: junction_idx -> list of neighbor_idx
    """
    incident = {i: [] for i in range(30)}
    for i1, i2 in GRID_CONNECTIONS:
        incident[i1].append(i2)
        incident[i2].append(i1)
    return incident


JUNCTION_INCIDENT = build_junction_incident_edges()


# ================= Junction 主方向計算 =================

def compute_junction_principal_directions(junction_idx: int) -> list:
    """
    計算 junction 的兩條主方向 (template 空間)

    做法：
    1. 取得所有 incident edges 的方向向量
    2. 將方向做 180° 等價分群（因為 ±v 是同一條線）
    3. 回傳 2 個主方向 unit vector
    """
    neighbors = JUNCTION_INCIDENT[junction_idx]
    p0 = TEMPLATE_POINTS[junction_idx]

    # 收集所有方向
    directions = []
    for n_idx in neighbors:
        p1 = TEMPLATE_POINTS[n_idx]
        v = p1 - p0
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            v = v / norm
            directions.append(v)

    if len(directions) == 0:
        return []

    # 將方向做 180° 等價分群
    def normalize_angle(v):
        angle = np.arctan2(v[1], v[0])
        if angle < 0:
            angle += np.pi
        if angle >= np.pi:
            angle -= np.pi
        return angle

    angles = [normalize_angle(d) for d in directions]

    # Clustering：把角度差 < 15° 的視為同一條線
    angle_threshold = np.deg2rad(15)
    clusters = []
    used = [False] * len(directions)

    for i, ang_i in enumerate(angles):
        if used[i]:
            continue
        cluster = [directions[i]]
        used[i] = True
        for j, ang_j in enumerate(angles):
            if used[j]:
                continue
            diff = abs(ang_i - ang_j)
            diff = min(diff, np.pi - diff)
            if diff < angle_threshold:
                cluster.append(directions[j])
                used[j] = True

        # 在同 cluster 裡先把向量「對齊到同半平面」再平均
        if len(cluster) > 1:
            ref = cluster[0]
            aligned_cluster = [ref]
            for k in range(1, len(cluster)):
                v = cluster[k]
                if np.dot(v, ref) < 0:
                    v = -v
                aligned_cluster.append(v)
            cluster = aligned_cluster

        # 取 cluster 的平均方向
        avg_dir = np.mean(cluster, axis=0)
        norm = np.linalg.norm(avg_dir)
        if norm > 1e-6:
            avg_dir = avg_dir / norm
            clusters.append(avg_dir)

    return clusters[:2] if len(clusters) >= 2 else clusters


# ================= 對稱性懲罰 =================

# ================= T Junction Stem 分析 =================

def get_t_junction_stem_info(junction_idx: int) -> dict:
    """
    分析 T junction 的 stem/bar 結構
    
    T junction 有 3 個 neighbor：2 個在 bar 軸上（兩側），1 個在 stem 軸上
    
    Returns:
        dict with:
        - 'stem_axis': 'A' or 'B' (stem 在哪個主方向)
        - 'stem_sign': +1 or -1 (stem 在該方向的正或負側)
        - 'bar_axis': 'A' or 'B' (bar 在哪個主方向)
        None if not a T junction
    """
    jt = TEMPLATE_TYPES[junction_idx]
    if jt != 1:  # 不是 T junction
        return None
    
    neighbors = JUNCTION_INCIDENT[junction_idx]
    if len(neighbors) != 3:
        return None
    
    p0 = TEMPLATE_POINTS[junction_idx]
    principal_dirs = compute_junction_principal_directions(junction_idx)
    
    if len(principal_dirs) < 2:
        return None
    
    tA, tB = principal_dirs[0], principal_dirs[1]
    
    # 分析每個 neighbor 在 A/B 軸上的投影
    neighbor_data = []
    for n_idx in neighbors:
        p1 = TEMPLATE_POINTS[n_idx]
        delta = p1 - p0
        proj_A = np.dot(delta, tA)
        proj_B = np.dot(delta, tB)
        # 判斷主要落在哪個軸
        if abs(proj_A) > abs(proj_B):
            axis = 'A'
            sign = +1 if proj_A > 0 else -1
        else:
            axis = 'B'
            sign = +1 if proj_B > 0 else -1
        neighbor_data.append({'idx': n_idx, 'axis': axis, 'sign': sign})
    
    # 統計：bar 軸應該有 2 個 neighbor（兩側），stem 軸只有 1 個
    axis_count = {'A': 0, 'B': 0}
    axis_signs = {'A': [], 'B': []}
    for nd in neighbor_data:
        axis_count[nd['axis']] += 1
        axis_signs[nd['axis']].append(nd['sign'])
    
    # 找出 stem_axis（只有 1 個 neighbor 的那個軸）
    if axis_count['A'] == 1 and axis_count['B'] == 2:
        stem_axis = 'A'
        bar_axis = 'B'
    elif axis_count['B'] == 1 and axis_count['A'] == 2:
        stem_axis = 'B'
        bar_axis = 'A'
    else:
        # 不標準的 T 結構
        return None
    
    stem_sign = axis_signs[stem_axis][0]  # 只有一個 neighbor
    
    return {
        'stem_axis': stem_axis,
        'stem_sign': stem_sign,
        'bar_axis': bar_axis
    }
# ============================================================================
#  Corner Code Encoding System (8-bit)
# ============================================================================
#
# 規格依《羽球場角點編碼》v1（規格文件 §1–§22）。每個外緣角點有一個全域
# 8-bit 編碼，落在 template 6×5 grid 上的哪一個物理角點是固定的（不會被
# template-id relabel/翻轉影響）。
#
# 編碼格式（§9.1）：
#   corner_code = (ny << 5) | (nx << 2) | local_corner_id
#       ny ∈ {0..6}  3-bit (其中 ny=3 是網線位置，保留索引但不生成角點)
#       nx ∈ {0..4}  3-bit
#       local_corner_id ∈ {NW=0, NE=1, SW=2, SE=3}  2-bit
#         bit1 = N(0)/S(1)，bit0 = W(0)/E(1)
#
# 規格 §6.1 世界座標系：球場正中心 = (0,0)，長邊∥x 軸，短邊∥y 軸，單位 m。
#   ny 對應世界 x 方向（長邊）：ny=0 → x=-6.70（規格 §17.4 範例），
#                              ny=6 → x=+6.70
#   nx 對應世界 y 方向（短邊）：nx=0 → y=-3.05，nx=4 → y=+3.05
#
# 對應到本檔案的 TEMPLATE_POINTS（左上角為原點的 layout）:
#   TEMPLATE_POINTS[:, 0] ∈ [0, 6.10]   → y_world = TEMPLATE_POINTS[:,0] - 3.05
#                                       → nx 變大方向 (即 E 方向)
#   TEMPLATE_POINTS[:, 1] ∈ [0, 13.40]  跟 ny 反向：
#       row 0 (TEMPLATE_POINTS[:,1]=13.40) → ny=0 = Top Baseline → x_world = -6.70
#       row 5 (TEMPLATE_POINTS[:,1]=0.00)  → ny=6 = Bottom Baseline → x_world = +6.70
#       → TEMPLATE_POINTS[:,1] 變大方向 = ny 變小方向 = N 方向
#
# 注意 topo id 與 corner_code 的對應（從規格反推 + 與本 codebase 整合）：
#   topo id 0..4  (row 0)   = ny=0 = TB    → codes 0..18      (含 4 個 L 8 codes)
#   topo id 5..9  (row 1)   = ny=1 = TDLS  → codes 32..50
#   topo id 10..14(row 2)   = ny=2 = TSS   → codes 64..82
#   topo id 15..19(row 3)   = ny=4 = BSS   → codes 128..146
#   topo id 20..24(row 4)   = ny=5 = BDLS  → codes 160..178
#   topo id 25..29(row 5)   = ny=6 = BB    → codes 192..211
# ============================================================================

# Junction grid row (0..5) ↔ ny mapping
#   依《羽球場角點編碼》§7.2:
#     ny=0 = Top Baseline,  ny=6 = Bottom Baseline (中間跳過 ny=3 = 網線)
#   TEMPLATE_POINTS 在本檔案的 layout（左上角原點）:
#     row 0 (TEMPLATE_POINTS[:,1]=13.40) → ny=0 (TB)
#     row 1 (12.64) → ny=1 (TDLS)
#     row 2 ( 8.68) → ny=2 (TSS)
#     row 3 ( 4.72) → ny=4 (BSS)            ← 跳過 ny=3 = 網線
#     row 4 ( 0.76) → ny=5 (BDLS)
#     row 5 ( 0.00) → ny=6 (BB)
ROW_TO_NY = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
NY_TO_ROW = {v: k for k, v in ROW_TO_NY.items()}

# local_corner_id 常數（§8）
LCID_NW = 0
LCID_NE = 1
LCID_SW = 2
LCID_SE = 3

# arms_mask bit 定義（§12.1）
ARM_N = 1 << 0  # bit0
ARM_E = 1 << 1  # bit1
ARM_S = 1 << 2  # bit2
ARM_W = 1 << 3  # bit3

# local_corner_id 對應的相鄰兩條臂（§13）
#   NW = (N, W), NE = (N, E), SW = (S, W), SE = (S, E)
_ADJACENT_ARMS = {
    LCID_NW: (ARM_N, ARM_W),
    LCID_NE: (ARM_N, ARM_E),
    LCID_SW: (ARM_S, ARM_W),
    LCID_SE: (ARM_S, ARM_E),
}


# ---------------------------------------------------------------------------
# 世界座標表（§6.2 + §7）— 中心原點，公尺
# ---------------------------------------------------------------------------
# x_world 由 ny 決定（長邊方向）
X_LINES_BY_NY = {
    0: -6.70,  # Top Baseline
    1: -5.94,  # Top Doubles Long Service Line
    2: -1.98,  # Top Short Service Line
    # 3: 0.00 網線位置，不生成 node（§6.3）
    4:  1.98,  # Bottom Short Service Line
    5:  5.94,  # Bottom Doubles Long Service Line
    6:  6.70,  # Bottom Baseline
}
# y_world 由 nx 決定（短邊方向）
Y_LINES_BY_NX = {
    0: -3.05,  # 左雙打邊線 LD
    1: -2.59,  # 左單打邊線 LS
    2:  0.00,  # 中線 C
    3:  2.59,  # 右單打邊線 RS
    4:  3.05,  # 右雙打邊線 RD
}


# ---------------------------------------------------------------------------
# Node table (§12, §14)：每個合法 (nx, ny) 的 arms_mask + valid_corner_mask
# ---------------------------------------------------------------------------
# 30 個 node 排列（§14.1）：
#   ny\nx |  0(LD) 1(LS)  2(C) 3(RS) 4(RD)
#   ------+---------------------------------
#   0(TB) |   L     T      T     T     L
#   1(TDLS)|  T     X      X     X     T
#   2(TSS)|   T     X      T     X     T
#   4(BSS)|   T     X      T     X     T
#   5(BDLS)|  T     X      X     X     T
#   6(BB) |   L     T      T     T     L
#
# arms_mask 規則（從鄰居存在性決定，§12.1）：
#   邊界角 L：例如 (0, 0) 在球場左上角，只有 E 跟 S 兩臂 → arms = E|S
#   T 型：缺一個方向；X 型：四個都有
#
# valid_corner_mask 規則（§13.1）：
#   bit lcid 為 1 ⇔ adjacent arms 兩個位元相同（11 或 00）
#
# 為了精準與可審查，下面直接列出每個 node 的 arms_mask；valid_corner_mask
# 在初始化時用 §13 規則自動推導。

def _compute_arms_mask(nx: int, ny: int) -> int:
    """
    根據 §14.1 的 30-node 結構，判斷某個 (nx, ny) 的 arms_mask。
    規則：node 的 N/S/E/W 方向有沒有「鄰居存在」，每個鄰居就是 grid 上
    沿該方向走，直到遇到另一個合法 (nx, ny) 為止。
    """
    if nx not in Y_LINES_BY_NX or ny not in X_LINES_BY_NY:
        return 0

    # 在 valid_ny / valid_nx 序列上的位置
    valid_ny = sorted(X_LINES_BY_NY.keys())  # [0,1,2,4,5,6]
    valid_nx = sorted(Y_LINES_BY_NX.keys())  # [0,1,2,3,4]

    has_north = ny != valid_ny[0]   # 還有更小的 ny
    has_south = ny != valid_ny[-1]  # 還有更大的 ny
    has_west  = nx != valid_nx[0]
    has_east  = nx != valid_nx[-1]

    # 規格 §14.1 中 (nx=2, ny=2) 與 (nx=2, ny=4) 都是 T 型而非 X 型——
    # 因為中線 (nx=2) 在球場中央發球線（TSS, BSS）位置不畫穿過：
    # 中線只從 TDLS 到 BDLS 之間是「斷開」的，但 §14.1 表已經告訴我們這兩個
    # 點是 T 型，stem 朝外（朝 baseline）。實際上中線在 TSS / BSS 處被「橫線
    # 加上中線終止」形成 T，stem 朝外的那一側才有臂。
    #
    # 用文件 §14.1 直接補正 X→T 的情況：
    if nx == 2 and ny == 2:
        # T at top short service line, stem 朝 N（中線在發球區那側沒有畫）
        # 邏輯：E/W 是短發球線，N 是中線朝底線方向（有），S 是中線朝網（沒有）
        # → arms = N | E | W
        return ARM_N | ARM_E | ARM_W
    if nx == 2 and ny == 4:
        # T at bottom short service line, stem 朝 S
        return ARM_S | ARM_E | ARM_W

    mask = 0
    if has_north: mask |= ARM_N
    if has_south: mask |= ARM_S
    if has_east:  mask |= ARM_E
    if has_west:  mask |= ARM_W
    return mask


def _compute_valid_corner_mask(arms_mask: int) -> int:
    """
    §13.1：對每個 local_corner_id (NW/NE/SW/SE)，取它的兩條相鄰臂
    (a, b)，當 a == b（都存在或都不存在）時，該角為有效角點。
    回傳 4-bit mask（bit lcid 為 1 ⇔ 該 lcid 有效）。
    """
    mask = 0
    for lcid, (arm_a, arm_b) in _ADJACENT_ARMS.items():
        a = 1 if (arms_mask & arm_a) else 0
        b = 1 if (arms_mask & arm_b) else 0
        if a == b:
            mask |= (1 << lcid)
    return mask


def _build_node_table() -> dict:
    """
    建立 NODE_TABLE: {(nx, ny): {'arms_mask', 'valid_corner_mask',
                                  'node_type', 'degree'}}
    包含全部 30 個合法 (nx, ny)。
    """
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


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def junction_idx_to_nx_ny(junction_idx: int) -> tuple:
    """junction_idx (0..29, row-major in TEMPLATE_POINTS) → (nx, ny)"""
    row = junction_idx // 5
    col = junction_idx % 5
    return col, ROW_TO_NY[row]


def nx_ny_to_junction_idx(nx: int, ny: int) -> int:
    """(nx, ny) → junction_idx (0..29)"""
    return NY_TO_ROW[ny] * 5 + nx


def encode_corner(nx: int, ny: int, local_corner_id: int) -> int:
    """§9.1: corner_code = (ny << 5) | (nx << 2) | local_corner_id"""
    return ((int(ny) & 0b111) << 5) | ((int(nx) & 0b111) << 2) | (int(local_corner_id) & 0b11)


def decode_corner(corner_code: int) -> tuple:
    """§9.2: corner_code → (nx, ny, local_corner_id)"""
    cc = int(corner_code)
    ny = (cc >> 5) & 0b111
    nx = (cc >> 2) & 0b111
    lcid = cc & 0b11
    return nx, ny, lcid


def get_node_code(corner_code: int) -> int:
    """§10: node_code = corner_code >> 2，去掉 local_corner_id。"""
    return int(corner_code) >> 2


def same_node(c1: int, c2: int) -> bool:
    """§10: 兩個 corner_code 是否屬於同一個 node。"""
    return get_node_code(c1) == get_node_code(c2)


def rotate180_corner(corner_code: int) -> int:
    """§11: 繞球場中心 180° 旋轉後的對應 corner_code。"""
    nx, ny, lcid = decode_corner(corner_code)
    nx2 = 4 - nx
    ny2 = 6 - ny
    lcid2 = lcid ^ 0b11
    return encode_corner(nx2, ny2, lcid2)


def is_valid_corner(corner_code: int) -> bool:
    """
    §13: 該 corner_code 是否在球場上實際存在的角點。
    依 NODE_TABLE 的 valid_corner_mask 判定。
    """
    nx, ny, lcid = decode_corner(corner_code)
    info = NODE_TABLE.get((nx, ny))
    if info is None:
        return False
    return bool(info['valid_corner_mask'] & (1 << lcid))


def corner_kind(corner_code: int) -> str:
    """
    §13 + §18 補充：區分 L 型內側 / 外側。
    return 'inner' (兩臂都存在 = 11), 'outer' (兩臂都不存在 = 00),
           或 'invalid' (a != b，角點不存在)。
    """
    nx, ny, lcid = decode_corner(corner_code)
    info = NODE_TABLE.get((nx, ny))
    if info is None:
        return 'invalid'
    arms = info['arms_mask']
    arm_a, arm_b = _ADJACENT_ARMS[lcid]
    a = 1 if (arms & arm_a) else 0
    b = 1 if (arms & arm_b) else 0
    if a == 1 and b == 1: return 'inner'
    if a == 0 and b == 0: return 'outer'
    return 'invalid'


def corner_code_to_world_xy(
    corner_code: int,
    line_width_m: float = LINE_WIDTH_M,
) -> tuple:
    """
    §16: corner_code → 世界座標 (x, y)，球場中心為原點，單位 m。

    流程：
      1. node 中心：x0 = X_LINES_BY_NY[ny],  y0 = Y_LINES_BY_NX[nx]
      2. 加 ±h 偏移：h = line_width_m / 2
         NW: (dx, dy) = (-h, -h)
         NE: (-h, +h)
         SW: (+h, -h)
         SE: (+h, +h)
      3. 回傳 (x0 + dx, y0 + dy)

    若 line_width_m = 0 則退化為純線中心。
    """
    nx, ny, lcid = decode_corner(corner_code)
    if nx not in Y_LINES_BY_NX or ny not in X_LINES_BY_NY:
        raise ValueError(f"corner_code {corner_code} has invalid (nx={nx}, ny={ny})")
    x0 = X_LINES_BY_NY[ny]
    y0 = Y_LINES_BY_NX[nx]
    h = 0.5 * float(line_width_m)
    # bit1 of lcid: 0=N (dx=-h), 1=S (dx=+h)
    # bit0 of lcid: 0=W (dy=-h), 1=E (dy=+h)
    dx = -h if (lcid & 0b10) == 0 else +h
    dy = -h if (lcid & 0b01) == 0 else +h
    return (x0 + dx, y0 + dy)


# ---------------------------------------------------------------------------
# Dense ID 映射（§20.3 / §21.2）— 80 個有效角點重新編號 0..79
# ---------------------------------------------------------------------------

def _build_dense_id_tables():
    """
    枚舉所有 corner_code（依 (ny, nx, lcid) 遞增順序），過濾出 valid 者，
    給出 dense_id (0..79)。固定順序方便當作訓練 label。
    """
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


def corner_code_to_dense_id(corner_code: int) -> int:
    """有效角點 → dense_id (0..79)；無效角點回傳 -1。"""
    return CORNER_CODE_TO_DENSE.get(int(corner_code), -1)


def dense_id_to_corner_code(dense_id: int) -> int:
    """dense_id → corner_code。out-of-range raise IndexError。"""
    return DENSE_TO_CORNER_CODE[int(dense_id)]


def get_valid_corner_codes() -> list:
    """所有 80 個有效 corner_code，順序與 dense_id 一致。"""
    return list(DENSE_TO_CORNER_CODE)


# ---------------------------------------------------------------------------
# 從 (junction_idx, corner_type) 換到 corner_code 的橋接 API
# （給 s3 vertex / h_refine 用，沿用既有的 ++/+-/-+/-- 系統）
# ---------------------------------------------------------------------------

def corner_type_to_local_corner_id(
    corner_type: str,
    tA_m: np.ndarray,
    tB_m: np.ndarray,
) -> int:
    """
    把 junction 局部的 corner_type ('++','+-','-+','--') 映射到全域的
    local_corner_id ∈ {NW=0, NE=1, SW=2, SE=3}。

    流程：
      1. corner_type → (sign_A, sign_B) ∈ {±1}
      2. offset_m = sign_A * tA_m + sign_B * tB_m  (template 空間方向)
      3. 用 offset_m 的 (x, y) 符號判斷象限：
           TEMPLATE_POINTS 的 layout：左上角原點，
             [:, 0] ∈ [0, 6.10]   = y_world + 3.05  (nx 0→4 是 y 由負到正)
             [:, 1] ∈ [0, 13.40]  跟 ny 反向：row 0 ([:,1]=13.40) 是 ny=0=TB
           → offset[1] > 0 (TEMPLATE_POINTS[:,1] 變大) ⇔ ny 變小 ⇔ N 方向
           → offset[0] > 0 (TEMPLATE_POINTS[:,0] 變大) ⇔ nx 變大 ⇔ E 方向

    （NW=0=00, NE=1=01, SW=2=10, SE=3=11；bit1=N(0)/S(1), bit0=W(0)/E(1)）
    """
    if not isinstance(corner_type, str) or len(corner_type) < 2:
        raise ValueError(f"invalid corner_type: {corner_type!r}")
    if corner_type[0] not in '+-' or corner_type[1] not in '+-':
        raise ValueError(f"corner_type must be '++/+-/-+/--', got {corner_type!r}")

    sign_A = +1.0 if corner_type[0] == '+' else -1.0
    sign_B = +1.0 if corner_type[1] == '+' else -1.0

    tA = np.asarray(tA_m, dtype=np.float64).reshape(2)
    tB = np.asarray(tB_m, dtype=np.float64).reshape(2)
    offset = sign_A * tA + sign_B * tB

    # offset[1] > 0 (TEMPLATE_POINTS[:,1] 大) ⇔ ny 小 ⇔ N
    # offset[0] > 0 (TEMPLATE_POINTS[:,0] 大) ⇔ nx 大 ⇔ E
    ns_bit = 0 if offset[1] > 0 else 1
    ew_bit = 1 if offset[0] > 0 else 0
    return (ns_bit << 1) | ew_bit


def corner_type_to_corner_code(
    junction_idx: int,
    corner_type: str,
    tA_m: np.ndarray,
    tB_m: np.ndarray,
) -> int:
    """
    (junction_idx, corner_type, tA_m, tB_m) → 8-bit corner_code。
    s3 vertex / h_refine 統一輸出層的主要入口。
    """
    nx, ny = junction_idx_to_nx_ny(int(junction_idx))
    lcid = corner_type_to_local_corner_id(corner_type, tA_m, tB_m)
    return encode_corner(nx, ny, lcid)