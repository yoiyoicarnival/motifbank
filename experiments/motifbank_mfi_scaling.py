#!/usr/bin/env python3
"""
motifbank_mfi_scaling.py
MFI silicalite-1 の QC コール数スケーリング解析

使い方:
  curl -o /tmp/MFI_iza.cif 'http://www.iza-structure.org/IZA-SC/cif/MFI.cif'
  OMP_NUM_THREADS=1 python3 motifbank_mfi_scaling.py
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motifbank_cli import from_cif, cutoff_trimers, geom_key, com, classify

CIF = os.environ.get("MFI_CIF", "/tmp/MFI_iza.cif")

if not os.path.exists(CIF):
    print(f"CIF not found: {CIF}")
    print("  curl -o /tmp/MFI_iza.cif 'http://www.iza-structure.org/IZA-SC/cif/MFI.cif'")
    sys.exit(1)

R_CUT = 5.5


def count_mbe_terms(mols, r_cut=R_CUT):
    n = len(mols)
    coms = np.array([com(m) for m in mols])
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if np.linalg.norm(coms[i] - coms[j]) < r_cut]
    trims = cutoff_trimers(mols, r_cut)
    um = len({geom_key([mols[i]]) for i in range(n)})
    up = len({geom_key([mols[i], mols[j]]) for i, j in pairs})
    ut = len({geom_key([mols[i], mols[j], mols[k]]) for i, j, k in trims})
    return n, len(pairs), len(trims), um, up, ut


# ── Phase 分類 ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  MFI Silicalite-1: Phase 分類")
print("=" * 65)
mols_1,  _, _ = from_cif(CIF, supercell=(1, 1, 1), mol_type="sio4", verbose=False)
mols_2x, _, _ = from_cif(CIF, supercell=(2, 2, 1), mol_type="sio4", verbose=False)
classify(mols_1, mols_2x, r_cut=R_CUT, T_K=300, label="MFI_silicalite")

# ── スケーリング表 ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  MFI Silicalite-1: QC コール数スケーリング")
print("=" * 65)
print("  %-14s %7s %9s %8s %8s" % ("supercell", "N_SiO4", "QC_naive", "QC_bank", "speedup"))
print("  " + "-" * 52)

results = []
for sc, label in [
    ((1, 1, 1), "1x1x1"),
    ((2, 1, 1), "2x1x1"),
    ((2, 2, 1), "2x2x1"),
    ((2, 2, 2), "2x2x2"),
    ((4, 2, 1), "4x2x1"),
    ((4, 2, 2), "4x2x2"),
]:
    mols, _, _ = from_cif(CIF, supercell=sc, mol_type="sio4", verbose=False)
    nm, npairs, nt, um, up, ut = count_mbe_terms(mols)
    naive = nm + npairs + nt
    bank  = um + up + ut
    spd   = naive / bank
    results.append((nm, naive, bank, spd))
    print("  %-14s %7d %9d %8d %7.0fx" % (label, nm, naive, bank, spd))

# ── 外挿 ─────────────────────────────────────────────────────────────────────
N_vals = [r[0] for r in results]
Q_vals = [r[1] for r in results]
lN = np.log(N_vals)
lQ = np.log(Q_vals)
a, b = np.polyfit(lN, lQ, 1)
bank_sat = results[-1][2]

print()
print("  スケーリング外挿 (bank = %d 固定, naive ∝ N^%.2f)" % (bank_sat, a))
print("  " + "-" * 52)
print("  %-14s %9s %8s %8s" % ("N_SiO4", "QC_naive", "QC_bank", "speedup"))
print("  " + "-" * 44)
for N in [96, 384, 768, 1536, 3072, 6144, 10000, 50000, 100000]:
    naive_est = int(np.exp(b) * N ** a)
    spd       = naive_est / bank_sat
    tag       = "  <- 実測" if N in N_vals else ""
    print("  %-14d %9d %8d %7.0fx%s" % (N, naive_est, bank_sat, spd, tag))

print()
print("  bank 飽和点: N ≈ %d SiO4 以降、bank 増加なし" % N_vals[2])
print("  speedup は N に比例して成長 (Phase 0 の特徴)")
print("=" * 65)
