#!/usr/bin/env python3
"""
generate_cif.py — MotifBank ベンチマーク用 CIF ファイル生成

実行:
  python3 generate_cif.py

生成ファイル:
  examples/ice_Ih.cif     — 氷 Ih (H36O12, 2×2×1 スーパーセル)
  examples/ice_2x.cif     — 氷 Ih (H72O24, 4×4×1 スーパーセル)
  examples/quartz.cif     — α-quartz SiO2 (テスト用)
"""

import os
os.makedirs("examples", exist_ok=True)

try:
    import numpy as np
    from ase import Atoms
    from ase.io import write
except ImportError:
    print("pip install ase が必要です")
    raise

# ─────────────────────────────────────────────
# 1. 氷 Ih 構造: 手動で六方晶ユニットセルを構築
# ─────────────────────────────────────────────
# 参考: Kuhs & Lehmann (1983), JPC 87, 4312
# a = 4.5119 A, c = 7.3521 A, hexagonal P63/mmc

def make_ice_Ih(supercell=(2, 2, 1)):
    """
    氷 Ih ユニットセル (4H2O, hexagonal) を構築してスーパーセルを返す
    水素位置: 平均的な半占有位置を使用
    """
    # ユニットセルパラメータ
    a = 4.5119   # Å
    c = 7.3521   # Å

    # 格子ベクトル (hexagonal)
    import numpy as np
    lv = np.array([
        [a,             0.,             0.     ],
        [-a/2,  a*np.sqrt(3)/2,         0.     ],
        [0.,            0.,             c      ],
    ])

    # 原子位置 (分数座標)
    # O: Wyckoff 4f position (1/3, 2/3, z) and equiv.
    # H: average positions
    z_O  = 0.0620
    z_H1 = 0.1972   # H bonded along c-axis (bifurcated model, avg)
    z_H2 = 0.0000   # H in basal plane

    frac_O = np.array([
        [1/3,  2/3,        z_O],
        [2/3,  1/3,    0.5+z_O],
        [1/3,  2/3,    0.5-z_O],
        [2/3,  1/3,       -z_O],  # ≡ 1-z_O
    ])

    # H 位置 (簡略: O-H = 0.9572, 最近傍 O-O 方向)
    rOH = 0.9572   # Å
    # 各 O に 2 個の H を配置 (O-O 方向に沿った位置)
    # 接続: O[0]-O[2], O[1]-O[3], basal plane, c-axis 方向

    # 簡単のため、各 H2O を独立に O 周りに配置
    # 方向: (1) c 軸方向, (2) 面内 a1 方向
    def o_frac_to_cart(f):
        return lv[0]*f[0] + lv[1]*f[1] + lv[2]*f[2]

    atoms_list = []
    symbols_list = []

    for f_O in frac_O:
        O_pos = o_frac_to_cart(f_O)
        atoms_list.append(O_pos)
        symbols_list.append('O')

        # H1: along +c direction from O
        H1_cart = O_pos + np.array([0., 0., rOH])
        atoms_list.append(H1_cart)
        symbols_list.append('H')

        # H2: along O-O nearest neighbor direction (in-plane)
        # neighbor at a1 direction: a1/|a1| * rOH
        a1_dir = lv[0] / np.linalg.norm(lv[0])
        H2_cart = O_pos + a1_dir * rOH
        atoms_list.append(H2_cart)
        symbols_list.append('H')

    positions = np.array(atoms_list)
    cell = lv.copy()

    atoms = Atoms(symbols=symbols_list,
                  positions=positions,
                  cell=cell,
                  pbc=True)

    if supercell != (1, 1, 1):
        atoms = atoms * supercell

    return atoms


# ─────────────────────────────────────────────
# 2. より正確な氷 Ih: ASE の構造生成 or 手動 P1
# ─────────────────────────────────────────────
def make_ice_Ih_p1(n_mol_xy=3):
    """
    2D grid 状に並べた H2O 分子を直交セルに入れた P1 ice
    build_ice2d と同じジオメトリ → from_cif で正確に読み込める
    """
    import numpy as np

    a = 2.76    # O-O 距離
    rOH = 0.9572
    angle_HOH = 104.52 * np.pi / 180

    # 六方晶 2D ベクトル
    lv1 = np.array([a * np.sqrt(3), 0., 0.])
    lv2 = np.array([a * np.sqrt(3)/2, a * 3/2, 0.])

    positions = []
    symbols = []

    for i in range(n_mol_xy):
        for j in range(n_mol_xy):
            for kb, b in enumerate([np.zeros(3),
                                     np.array([a*np.sqrt(3)/2, a/2, 0.])]):
                o = i*lv1 + j*lv2 + b
                phi = kb * np.pi / 3
                Ox, Oy, Oz = o

                positions.append([Ox, Oy, Oz])
                symbols.append('O')

                positions.append([
                    Ox + rOH * np.cos(phi + angle_HOH/2),
                    Oy + rOH * np.sin(phi + angle_HOH/2),
                    Oz
                ])
                symbols.append('H')

                positions.append([
                    Ox + rOH * np.cos(phi - angle_HOH/2),
                    Oy + rOH * np.sin(phi - angle_HOH/2),
                    Oz
                ])
                symbols.append('H')

    positions = np.array(positions)

    # セルサイズ: 分子が全て入る最小直交ボックス + マージン
    margin = 5.0
    xmax = positions[:,0].max() + margin
    ymax = positions[:,1].max() + margin
    zmax = 10.0

    # 全座標を正にシフト
    positions[:,0] -= positions[:,0].min() - 1.0
    positions[:,1] -= positions[:,1].min() - 1.0

    cell = [[xmax, 0, 0], [0, ymax, 0], [0, 0, zmax]]

    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=cell, pbc=True)
    return atoms


# ─────────────────────────────────────────────
# 生成と保存
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np

    # --- 小セル: 3×3 グリッド (18 H2O) ---
    print("生成中: ice_Ih_3x3.cif (18 H2O) ...")
    ice_s = make_ice_Ih_p1(n_mol_xy=3)
    write("examples/ice_Ih_3x3.cif", ice_s, format="cif")
    print(f"  -> {len(ice_s)} atoms")

    # --- 大セル: 6×6 グリッド (72 H2O) ---
    print("生成中: ice_Ih_6x6.cif (72 H2O) ...")
    ice_l = make_ice_Ih_p1(n_mol_xy=6)
    write("examples/ice_Ih_6x6.cif", ice_l, format="cif")
    print(f"  -> {len(ice_l)} atoms")

    # --- 元の hexagonal ice (より正確) ---
    print("生成中: ice_Ih_hex.cif (2x2x1, 16 H2O) ...")
    try:
        ice_hex = make_ice_Ih(supercell=(2, 2, 1))
        write("examples/ice_Ih_hex.cif", ice_hex, format="cif")
        print(f"  -> {len(ice_hex)} atoms")
    except Exception as e:
        print(f"  警告: hexagonal ice 生成失敗 ({e})")

    print("\n完了:")
    for f in ["examples/ice_Ih_3x3.cif", "examples/ice_Ih_6x6.cif",
              "examples/ice_Ih_hex.cif"]:
        if os.path.exists(f):
            print(f"  {f}")
    print()
    print("使い方:")
    print('  python3 motifbank_cli.py benchmark examples/ice_Ih_3x3.cif --qc mock')
    print('  python3 motifbank_cli.py benchmark examples/ice_Ih_3x3.cif --qc pyscf')
