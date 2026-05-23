#!/usr/bin/env python3
"""失敗する Si(OH)4 断片の診断: Si-O 距離と角度を出力"""
import sys, numpy as np
sys.path.insert(0, '/home/yoiyoi')
from motifbank_cli import from_cif, geom_key

CIF = '/home/yoiyoi/examples/MFI_iza.cif'
mols, atypes, _ = from_cif(CIF, supercell=(1,1,1), mol_type='si_oh4', verbose=False)

# 失敗した断片: Si=(1.579, 11.272, 2.494)
TARGET = np.array([1.57907400, 11.27237180, 2.49435160])

seen = set()
print("Si(OH)4 断片の幾何情報 (最初の 10 ユニーク):")
print(f"  {'idx':>4}  {'Si_x':>7} {'Si_y':>7} {'Si_z':>7}  "
      f"{'Si-O[min]':>9} {'Si-O[max]':>9}  {'O-Si-O[min]':>11} {'問題?':>5}")

count = 0
problem_idx = None
for i, (m, at) in enumerate(zip(mols, atypes)):
    gk = geom_key([m])
    if gk in seen:
        continue
    seen.add(gk)

    si = m[0]
    o_coords = m[1:5]
    si_o = [np.linalg.norm(si - o) for o in o_coords]
    # O-Si-O 角度
    vecs = [o - si for o in o_coords]
    angles = []
    for a in range(4):
        for b in range(a+1, 4):
            cos_th = np.dot(vecs[a], vecs[b]) / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]))
            angles.append(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))

    is_target = np.linalg.norm(si - TARGET) < 0.01
    flag = " ← FAIL" if is_target else ""
    if is_target:
        problem_idx = i

    print(f"  {i:>4}  {si[0]:7.3f} {si[1]:7.3f} {si[2]:7.3f}  "
          f"{min(si_o):9.4f} {max(si_o):9.4f}  {min(angles):11.2f}{flag}")
    count += 1
    if count >= 15:
        break

if problem_idx is not None:
    print(f"\n  問題断片 idx={problem_idx}")
    m = mols[problem_idx]
    si = m[0]
    o_coords = m[1:5]
    h_coords = m[5:9]
    print(f"  Si-O 距離: {[round(np.linalg.norm(si-o),4) for o in o_coords]}")
    print(f"  O-H 距離: {[round(np.linalg.norm(o-h),4) for o,h in zip(o_coords,h_coords)]}")
    vecs = [o - si for o in o_coords]
    angles = []
    for a in range(4):
        for b in range(a+1,4):
            cos_th = np.dot(vecs[a], vecs[b]) / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]))
            angles.append(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))
    print(f"  O-Si-O 角度: {[round(a,1) for a in angles]}")
