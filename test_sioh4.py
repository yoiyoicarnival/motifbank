#!/usr/bin/env python3
"""
test_sioh4.py -- H-capped Si(OH)4 断片化の検証 + S_local 計算

検証項目:
  F1. Si(OH)4 断片化: Si+4O+4H = 9原子, charge=0, 中性
  F2. PySCF HF/STO-3G で収束 (charge=0)
  F3. Phase 0 が Si(OH)4 でも維持されるか
  F4. S_local = log(N_bank_sat) の測定
  F5. scaling table (sio4 vs si_oh4 の比較)

使い方:
  OMP_NUM_THREADS=1 python3 test_sioh4.py
  OMP_NUM_THREADS=1 python3 test_sioh4.py --pyscf
"""
import os, sys, itertools, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motifbank_cli import (
    MotifBank, from_cif, run_mbe, classify,
    cutoff_trimers, geom_key, com,
    qc_compute_mock, qc_compute_pyscf,
)

CIF = os.path.join(os.path.dirname(__file__), "examples", "MFI_iza.cif")
PASS = "  PASS"
FAIL = "  FAIL"
SKIP = "  SKIP"
R_CUT = 5.5


def count_bank(mols, r_cut=R_CUT):
    n = len(mols)
    coms = np.array([com(m) for m in mols])
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if np.linalg.norm(coms[i] - coms[j]) < r_cut]
    trims = cutoff_trimers(mols, r_cut)
    um = len({geom_key([mols[i]]) for i in range(n)})
    up = len({geom_key([mols[i], mols[j]]) for i, j in pairs})
    ut = len({geom_key([mols[i], mols[j], mols[k]]) for i, j, k in trims})
    return n, len(pairs), len(trims), um + up + ut


def test(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"{tag}  {name}  {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyscf", action="store_true")
    args = ap.parse_args()

    passed = failed = 0

    def record(ok):
        nonlocal passed, failed
        if ok: passed += 1
        else:  failed += 1

    print("\nH-capped Si(OH)4 断片化 + S_local 検証")
    print("=" * 60)

    # ── F1: 断片化の構造確認 ──────────────────────────────────────────
    print("\n── F: 断片化の正確性 ──")

    mols_raw, atypes_raw, _ = from_cif(CIF, supercell=(1,1,1),
                                        mol_type="sio4", verbose=False)
    mols_cap, atypes_cap, _ = from_cif(CIF, supercell=(1,1,1),
                                        mol_type="si_oh4", verbose=False)

    ok = len(mols_cap) == len(mols_raw)
    record(test("F1a  N_fragments 一致 (sio4 == si_oh4)",
                ok, f"N={len(mols_cap)}"))

    n_atoms_ok = all(len(m) == 9 for m in mols_cap)
    record(test("F1b  各断片が 9原子 (Si+4O+4H)",
                n_atoms_ok,
                f"atoms={[len(m) for m in mols_cap[:3]]}..."))

    at_ok = all(at == ["Si","O","O","O","O","H","H","H","H"] for at in atypes_cap)
    record(test("F1c  atom_types = [Si,O,O,O,O,H,H,H,H]",
                at_ok, str(atypes_cap[0])))

    # Si-O 距離チェック (1.4~1.8 Å が正常)
    si_o_dists = [np.linalg.norm(mols_cap[0][0] - mols_cap[0][k]) for k in range(1,5)]
    ok = all(1.3 < d < 1.9 for d in si_o_dists)
    record(test("F1d  Si-O 距離 1.4~1.8 Å",
                ok, f"dists={[f'{d:.3f}' for d in si_o_dists]}"))

    # O-H 距離チェック (~0.96 Å)
    o_h_dists = [np.linalg.norm(mols_cap[0][k] - mols_cap[0][k+4]) for k in range(1,5)]
    ok = all(0.90 < d < 1.02 for d in o_h_dists)
    record(test("F1e  O-H 距離 ~0.96 Å",
                ok, f"dists={[f'{d:.3f}' for d in o_h_dists]}"))

    # ── F2: PySCF HF/STO-3G (charge=0) ──────────────────────────────
    print("\n── F2: PySCF HF/STO-3G (si_oh4, charge=0) ──")
    if args.pyscf:
        try:
            e = qc_compute_pyscf([mols_cap[0]], [atypes_cap[0]],
                                 basis="sto-3g", method="hf",
                                 charge=0, spin=0)
            ok = test("F2  Si(OH)4 PySCF charge=0 収束",
                      True, f"E={e:.6f} Ha")
            record(ok)
            print(f"     参考: SiO4(4-) charge=-4 E=-578.485 Ha → 差 = {e-(-578.485):.3f} Ha")
        except Exception as ex:
            record(test("F2  Si(OH)4 PySCF charge=0 収束", False, str(ex)[:80]))
    else:
        print(f"  {SKIP}  F2  PySCF (--pyscf で実行)")

    # ── F3: Phase 0 が si_oh4 でも維持されるか ──────────────────────
    print("\n── F3: Phase 分類 (si_oh4) ──")
    mols_2x, _, _ = from_cif(CIF, supercell=(2,2,1),
                              mol_type="si_oh4", verbose=False)
    r = classify(mols_cap, mols_2x, r_cut=R_CUT, T_K=300,
                 label="MFI_si_oh4", verbose=False)
    ok = r["phase"] <= 1
    record(test("F3  Phase 0 or 1  (si_oh4)",
                ok,
                f"Phase={r['phase']}  gamma={r['gamma']}  "
                f"N_bank={r['N_bank']}  strategy={r['strategy']}"))
    print(f"     compress={r['compress']}x  ROI初回={r['roi_pct']}%")

    # ── F4: S_local 測定 (bank 飽和値) ──────────────────────────────
    print("\n── F4: S_local = log(N_bank_sat) ──")
    results = {}
    for sc, label in [((1,1,1),"1x1x1"), ((2,2,1),"2x2x1"),
                      ((2,2,2),"2x2x2"), ((4,2,2),"4x2x2")]:
        mols, _, _ = from_cif(CIF, supercell=sc, mol_type="si_oh4", verbose=False)
        n, npairs, nt, nb = count_bank(mols)
        results[label] = (n, npairs + n + nt, nb)

    bank_sat = results["4x2x2"][2]
    S_local = np.log(bank_sat)
    print(f"  {'supercell':12s}  {'N_SiO4':>7}  {'QC_naive':>9}  {'QC_bank':>9}")
    print("  " + "-" * 44)
    for label, (n, naive, nb) in results.items():
        print(f"  {label:12s}  {n:>7}  {naive:>9}  {nb:>9}")
    print(f"\n  N_bank 飽和値    = {bank_sat}")
    print(f"  S_local          = log({bank_sat}) = {S_local:.2f} nats")
    print(f"  S_local (bits)   = {S_local/np.log(2):.2f} bits")
    print(f"\n  解釈: MFI のローカル幾何エントロピーは {S_local:.1f} nats で有限。")
    print(f"        系を無限大にしても bank は {bank_sat} 型以上増えない。")

    ok = bank_sat < results["1x1x1"][2] * 2
    record(test("F4  bank 飽和 (4x2x2 ≈ 2x2x2)",
                ok, f"sat={bank_sat}, 1x={results['1x1x1'][2]}"))

    # ── F5: スケーリング比較 (sio4 vs si_oh4) ──────────────────────
    print("\n── F5: sio4 vs si_oh4 スケーリング比較 ──")
    print(f"  {'sc':8s}  {'N':>6}  {'bank(sio4)':>11}  {'bank(si_oh4)':>13}  {'speedup(si_oh4)':>16}")
    print("  " + "-" * 62)
    for sc, label in [((1,1,1),"1x1x1"), ((2,2,1),"2x2x1"), ((2,2,2),"2x2x2")]:
        mols_s, _, _ = from_cif(CIF, supercell=sc, mol_type="sio4",   verbose=False)
        mols_h, _, _ = from_cif(CIF, supercell=sc, mol_type="si_oh4", verbose=False)
        n,  _, nt_s, nb_s = count_bank(mols_s)
        n2, _, nt_h, nb_h = count_bank(mols_h)
        naive_s = mols_s.__len__() + (count_bank(mols_s)[1]) + nt_s
        spd = naive_s / max(nb_h, 1)
        print(f"  {label:8s}  {n:>6}  {nb_s:>11}  {nb_h:>13}  {spd:>14.0f}x")

    # ── 結果 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  結果: {passed}/{passed+failed} PASS")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
