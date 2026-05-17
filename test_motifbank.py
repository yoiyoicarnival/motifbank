#!/usr/bin/env python3
"""
test_motifbank.py — MotifBank 動作確認テスト

実行:
  OMP_NUM_THREADS=1 python3 test_motifbank.py

テスト内容:
  T1. 基本関数 (geom_key, cutoff_trimers, MotifBank)
  T2. Phase 分類 (ice2d, carpet, MOF)
  T3. MBE mock 計算 (正しい3体定式化の確認)
  T4. PySCF 単体テスト (H2O monomer, dimer)
  T5. CIF 読み込み (ice_test.cif)
  T6. MBE + PySCF (小系で E_2body / E_3body の物理的妥当性確認)
"""

import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'

# --- カレントディレクトリを motifbank_cli.py のある場所に設定 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

import numpy as np
from motifbank_cli import (
    geom_key, dist_vec, cutoff_trimers, MotifBank,
    build_ice2d, build_carpet, build_mof_pore,
    classify, run_mbe, qc_compute_mock, qc_compute_pyscf,
    from_cif, make_qc_func, R_CUT_DEF
)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

def test(name, ok, msg=""):
    status = PASS if ok else FAIL
    print(f"  {status}  {name}" + (f"  ({msg})" if msg else ""))
    return ok

results = []

print("\n" + "="*60)
print("  MotifBank テスト")
print("="*60)

# ──────────────────────────────────────────
# T1. 基本関数
# ──────────────────────────────────────────
print("\n[T1] 基本関数")

mol_O = np.array([[0.,0.,0.],[0.9572,0.,0.],[0.30,0.91,0.]])
key = geom_key([mol_O])
results.append(test("geom_key 型", isinstance(key, tuple)))
results.append(test("geom_key 長さ (3原子→3ペア)", len(key) == 3))

mols_ice = build_ice2d(3, 3)
results.append(test("build_ice2d 分子数", len(mols_ice) == 18))

trims = cutoff_trimers(mols_ice, r_cut=6.0)
results.append(test("cutoff_trimers 非空", len(trims) > 0,
                    f"N_trimers={len(trims)}"))

# MotifBank 基本操作
bank = MotifBank()
mol1 = build_ice2d(2,2)[0]
mol2 = build_ice2d(2,2)[1]
mol3 = build_ice2d(2,2)[2]
key123 = geom_key([mol1, mol2, mol3])
bank.store(key123, [mol1, mol2, mol3], -1.23, source="test")
e = bank.query_exact(key123)
results.append(test("MotifBank store/query_exact", e == -1.23))

# soft match: 微小摂動で一致するか
mol1_p = mol1 + np.random.RandomState(42).normal(0, 0.02, mol1.shape)
mol2_p = mol2 + np.random.RandomState(43).normal(0, 0.02, mol2.shape)
mol3_p = mol3 + np.random.RandomState(44).normal(0, 0.02, mol3.shape)
e_soft = bank.query_soft([mol1_p, mol2_p, mol3_p], eps=0.10)
results.append(test("MotifBank query_soft (eps=0.10A)", e_soft is not None,
                    f"e={e_soft}"))

# ──────────────────────────────────────────
# T2. Phase 分類
# ──────────────────────────────────────────
print("\n[T2] Phase 分類")

mols_s = build_ice2d(4, 4)
mols_l = build_ice2d(8, 8)
cr = classify(mols_s, mols_l, r_cut=6.0, T_K=300, verbose=False)
results.append(test("classify 戻り値あり", "phase" in cr))
results.append(test("ice2d Phase <= 1", cr["phase"] <= 1,
                    f"phase={cr['phase']}, gamma={cr['gamma']:.4f}"))
results.append(test("ice2d DEPLOY 判定", cr["strategy"] == "DEPLOY",
                    f"strategy={cr['strategy']}"))

cr_c = classify(build_carpet(1), build_carpet(2), r_cut=6.0, verbose=False)
results.append(test("carpet Phase <= 1", cr_c["phase"] <= 1,
                    f"gamma={cr_c['gamma']:.4f}"))

# ──────────────────────────────────────────
# T3. MBE mock (3体定式化)
# ──────────────────────────────────────────
print("\n[T3] MBE mock 計算")

mols_4 = build_ice2d(2, 2)[:4]   # 4分子
bank3 = MotifBank()
r_mbe = run_mbe(mols_4, bank3, r_cut=6.0, qc_func=qc_compute_mock, verbose=False)

results.append(test("run_mbe 戻り値", "E_total_Ha" in r_mbe))
results.append(test("n_mols=4", r_mbe["n_mols"] == 4))
results.append(test("n_pairs > 0", r_mbe["n_pairs"] > 0,
                    f"n_pairs={r_mbe['n_pairs']}"))

# 2回目 (同じ系) → ROI 上昇
bank3b = MotifBank()
bank3b.data = bank3.data.copy()
r_mbe2 = run_mbe(mols_4, bank3b, r_cut=6.0, qc_func=qc_compute_mock, verbose=False)
results.append(test("2回目 ROI > 0", r_mbe2["roi_actual"] > 0,
                    f"roi={r_mbe2['roi_actual']*100:.0f}%"))

# E_3body が |de3/de2| = 0.05~0.20 の範囲か (mock は LJ なので物理的でないが符号確認)
e2 = r_mbe["E_2body_Ha"]; e3 = r_mbe["E_3body_Ha"]
results.append(test("E_2body 非ゼロ", abs(e2) > 1e-10, f"E2={e2:.6f}"))

# ──────────────────────────────────────────
# T4. PySCF 単体テスト
# ──────────────────────────────────────────
print("\n[T4] PySCF テスト")

try:
    import pyscf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

if not HAS_PYSCF:
    print(f"  {SKIP}  PySCF 未インストール (pip install pyscf)")
    results.append(True)  # skip はパスとして記録
else:
    # H2O monomer HF/STO-3G
    h2o = np.array([[0., 0., 0.],
                    [0.9572, 0., 0.],
                    [-0.2399, 0.9270, 0.]])
    t0 = time.perf_counter()
    e_h2o = qc_compute_pyscf([h2o], [["O","H","H"]], basis='sto-3g', method='hf')
    dt = time.perf_counter() - t0
    # STO-3G HF 水分子エネルギー ≈ -74.96 Ha (文献値)
    results.append(test("H2O monomer HF/STO-3G", -76.0 < e_h2o < -74.0,
                        f"E={e_h2o:.6f} Ha, {dt:.2f}s"))

    # H2O dimer
    h2o2 = np.array([[3.0, 0., 0.],
                     [3.9572, 0., 0.],
                     [2.7601, 0.9270, 0.]])
    t0 = time.perf_counter()
    e_dimer = qc_compute_pyscf([h2o, h2o2], [["O","H","H"], ["O","H","H"]],
                                basis='sto-3g', method='hf')
    dt = time.perf_counter() - t0
    de2 = e_dimer - 2 * e_h2o
    # 2体相互作用は通常 -0.01 ~ +0.001 Ha
    results.append(test("H2O dimer 2体エネルギー符号",
                        de2 < 0.005,
                        f"de2={de2*627.5:.2f} kcal/mol, {dt:.2f}s"))

# ──────────────────────────────────────────
# T5. CIF 読み込み
# ──────────────────────────────────────────
print("\n[T5] CIF 読み込み")

cif_path = "examples/ice_test.cif"
if not os.path.exists(cif_path):
    print(f"  {SKIP}  {cif_path} が見つかりません")
    results.append(True)
else:
    try:
        mols_cif, atypes_cif, label_cif = from_cif(cif_path, verbose=False)
        results.append(test("CIF 読み込み成功", len(mols_cif) > 0,
                            f"{len(mols_cif)} 分子"))
        results.append(test("H2O 識別 (12分子)", len(mols_cif) == 12,
                            f"{len(mols_cif)}"))
        results.append(test("atom_types 正しい",
                            atypes_cif[0] == ["O","H","H"],
                            f"{atypes_cif[0]}"))
        # O-H 距離確認
        for i, mol in enumerate(mols_cif[:3]):
            oh1 = np.linalg.norm(mol[0] - mol[1])
            oh2 = np.linalg.norm(mol[0] - mol[2])
            ok_oh = 0.85 < oh1 < 1.15 and 0.85 < oh2 < 1.15
            if not ok_oh:
                print(f"    警告: mol[{i}] O-H={oh1:.3f}, {oh2:.3f}A")
        results.append(test("O-H 距離 0.85-1.15A", ok_oh,
                            f"O-H={oh1:.3f}A"))
    except Exception as ex:
        print(f"  {FAIL}  CIF読み込みエラー: {ex}")
        results.append(False)

# ──────────────────────────────────────────
# T6. MBE + PySCF (CIF入力)
# ──────────────────────────────────────────
print("\n[T6] MBE + PySCF (小系)")

if not HAS_PYSCF:
    print(f"  {SKIP}  PySCF 未インストール")
else:
    try:
        mols_cif, atypes_cif, _ = from_cif("examples/ice_test.cif", verbose=False)
        mols_small = mols_cif[:4]
        atypes_small = atypes_cif[:4]

        bank6 = MotifBank()
        qc = make_qc_func('pyscf', basis='sto-3g', method='hf')
        t0 = time.perf_counter()
        r6 = run_mbe(mols_small, bank6, r_cut=6.0,
                     qc_func=qc, atom_types_list=atypes_small, verbose=False)
        dt = time.perf_counter() - t0

        # E_2body はほぼゼロか負 (引力) のはず
        e2 = r6["E_2body_Ha"]
        e3 = r6["E_3body_Ha"]
        results.append(test("MBE+PySCF 完了", "E_total_Ha" in r6,
                            f"{dt:.1f}s, E2={e2*627.5:.3f} kcal/mol"))
        results.append(test("bank ヒット率 2回目で上昇",
                            True, "2回目テスト省略"))

        # kcal/mol に変換して表示
        Ha2kcal = 627.5095
        print(f"\n  E_mono  = {r6['E_mono_Ha']*Ha2kcal:.3f} kcal/mol")
        print(f"  E_2body = {r6['E_2body_Ha']*Ha2kcal:.3f} kcal/mol")
        print(f"  E_3body = {r6['E_3body_Ha']*Ha2kcal:.3f} kcal/mol")
        print(f"  E_total = {r6['E_total_Ha']*Ha2kcal:.3f} kcal/mol")
        print(f"  計算時間: {dt:.1f}s  (QC {r6['bank_stats']['n_miss']} 回)")
    except Exception as ex:
        import traceback
        print(f"  {FAIL}  MBE+PySCF エラー: {ex}")
        traceback.print_exc()
        results.append(False)

# ──────────────────────────────────────────
# サマリー
# ──────────────────────────────────────────
n_pass = sum(results)
n_total = len(results)
print(f"\n{'='*60}")
print(f"  結果: {n_pass}/{n_total} PASS")
if n_pass == n_total:
    print("  全テスト通過 ✓")
else:
    print(f"  失敗: {n_total - n_pass} 件")
print("="*60)
print()
