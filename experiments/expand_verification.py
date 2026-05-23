#!/usr/bin/env python3
"""
expand_verification.py
L_geom の多系展開検証実験

3つの実験:
  [EXP-A] 複数系で L_geom を測定 (Si(OH)4, H2O, Al(OH)4)
  [EXP-B] L_geom(d_geom) 曲線: n=30 サンプルで L の d依存性を精密測定
  [EXP-C] Theorem 2' の統計的検証: 違反率・ratio 分布の定量化

実行:
  OMP_NUM_THREADS=1 python3 expand_verification.py --exp A [--exp B] [--exp C]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
from measure_lipschitz import ref_sioh4, geometry_distance

BASIS  = "def2-svp"
METHOD = "pbe"
L_GEOM = 0.705  # Ha/Å (Si(OH)4 実測値)
EPS    = 0.10   # Å


def qc_energy_and_key(coords, types, basis=BASIS, method=METHOD):
    from pyscf import gto, dft
    atom_str = "; ".join(
        f"{t} {x:.6f} {y:.6f} {z:.6f}"
        for t, (x, y, z) in zip(types, coords)
    )
    mol = gto.M(atom=atom_str, basis=basis, charge=0, spin=0,
                verbose=0, unit="Angstrom")
    mf = dft.RKS(mol)
    mf.xc = method
    mf.conv_tol = 1e-9
    mf.kernel()
    return float(mf.e_tot)


# ── 各系の参照ジオメトリ ──────────────────────────────────────────────────────

def ref_h2o():
    """H2O (PBE/def2-SVP 平衡, 気相)"""
    coords = np.array([
        [0.000,  0.000,  0.000],   # O
        [0.757,  0.586,  0.000],   # H
        [-0.757, 0.586,  0.000],   # H
    ])
    types = ["O", "H", "H"]
    return coords, types


def ref_aloh4():
    """Al(OH)4^- (正四面体, Al-O=1.77Å, O-H=0.96Å)"""
    al_o = 1.77
    o_h  = 0.96
    tet  = np.array([[1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1]], dtype=float)
    tet /= np.linalg.norm(tet, axis=1, keepdims=True)
    al   = np.array([0., 0., 0.])
    o_coords = [al + al_o * d for d in tet]
    h_coords = [oc + o_h * d for oc, d in zip(o_coords, tet)]
    coords = np.array([al] + o_coords + h_coords)
    types  = ["Al"] + ["O"]*4 + ["H"]*4
    return coords, types


SYSTEMS = {
    "Si(OH)4": (ref_sioh4,  0,  0),   # (ref_fn, charge, spin)
    "H2O":     (ref_h2o,    0,  0),
    "Al(OH)4": (ref_aloh4, -1,  0),
}


# ── EXP-A: 複数系 L_geom 測定 ──────────────────────────────────────────────

def exp_a(n_samples=12, sigma=0.03):
    print(f"\n{'='*65}")
    print(f"  [EXP-A] 複数系 L_geom 測定 (σ={sigma}Å, n={n_samples})")
    print(f"{'='*65}")
    rng = np.random.default_rng(42)
    results = {}

    for name, (ref_fn, charge, spin) in SYSTEMS.items():
        print(f"\n  [{name}] charge={charge}, spin={spin}")
        coords0, types = ref_fn()

        from pyscf import gto, dft
        atom_str0 = "; ".join(
            f"{t} {x:.6f} {y:.6f} {z:.6f}"
            for t, (x, y, z) in zip(types, coords0)
        )
        mol0 = gto.M(atom=atom_str0, basis=BASIS, charge=charge, spin=spin,
                     verbose=0, unit="Angstrom")
        mf0  = dft.RKS(mol0); mf0.xc = METHOD; mf0.conv_tol=1e-9
        mf0.kernel()
        E0 = float(mf0.e_tot)
        print(f"    E0 = {E0:.8f} Ha")

        L_vals = []
        for i in range(n_samples):
            delta  = rng.normal(0, sigma, coords0.shape)
            coords1 = coords0 + delta
            d_geom  = geometry_distance(coords0, coords1)
            if d_geom >= EPS or d_geom < 1e-10:
                continue

            atom_str1 = "; ".join(
                f"{t} {x:.6f} {y:.6f} {z:.6f}"
                for t, (x, y, z) in zip(types, coords1)
            )
            try:
                mol1 = gto.M(atom=atom_str1, basis=BASIS,
                             charge=charge, spin=spin,
                             verbose=0, unit="Angstrom")
                mf1  = dft.RKS(mol1); mf1.xc=METHOD; mf1.conv_tol=1e-9
                mf1.kernel()
                E1 = float(mf1.e_tot)
            except Exception as ex:
                print(f"    [{i}] SCF failed: {ex}")
                continue

            dE = abs(E1 - E0)
            L_emp = dE / d_geom
            L_vals.append({"d": d_geom, "dE": dE, "L": L_emp})
            print(f"    [{i:2d}] d={d_geom:.4f}  ΔE={dE*627.509:.3f} kcal  "
                  f"L={L_emp:.4f} Ha/Å")

        if L_vals:
            L_max = max(v["L"] for v in L_vals)
            L_mean = np.mean([v["L"] for v in L_vals])
            bound = L_max * EPS * 627.509
            print(f"    → L_max={L_max:.4f} Ha/Å  L_mean={L_mean:.4f}  "
                  f"Theorem2' bound={bound:.1f} kcal/mol")
            results[name] = {"L_max": L_max, "L_mean": L_mean,
                             "bound_kcal": bound, "samples": L_vals}

    print(f"\n  {'System':15s}  {'n_atoms':>8}  {'L_max (Ha/Å)':>14}  "
          f"{'Bound (kcal)':>14}")
    print("  " + "-"*55)
    for name, r in results.items():
        ref_fn, _, _ = SYSTEMS[name]
        coords, _ = ref_fn()
        print(f"  {name:15s}  {len(coords):>8}  {r['L_max']:>14.4f}  "
              f"{r['bound_kcal']:>14.1f}")

    return results


# ── EXP-B: L_geom(d_geom) 曲線 ──────────────────────────────────────────────

def exp_b(n_per_bin=8, d_bins=None):
    """d_geom を細かく区切って L_emp の d依存性を測定"""
    if d_bins is None:
        d_bins = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07]

    print(f"\n{'='*65}")
    print(f"  [EXP-B] L_geom(d_geom) 曲線 (Si(OH)4, {METHOD}/{BASIS})")
    print(f"{'='*65}")
    coords0, types = ref_sioh4()
    rng = np.random.default_rng(55)

    from pyscf import gto, dft
    atom_str0 = "; ".join(
        f"{t} {x:.6f} {y:.6f} {z:.6f}"
        for t, (x, y, z) in zip(types, coords0)
    )
    mol0 = gto.M(atom=atom_str0, basis=BASIS, charge=0, spin=0,
                 verbose=0, unit="Angstrom")
    mf0 = dft.RKS(mol0); mf0.xc = METHOD; mf0.conv_tol = 1e-9
    mf0.kernel()
    E0 = float(mf0.e_tot)

    curve = []
    print(f"  {'d_target':>10}  {'d_actual':>10}  {'L_max':>12}  {'L_mean':>12}")
    print("  " + "-"*48)

    for d_target in d_bins:
        # σ を調整して d_geom ≈ d_target になるようにする
        sigma = d_target * 0.7  # 経験的変換係数
        L_bin = []
        for _ in range(n_per_bin * 3):  # 余分にサンプルして d_target 近くのものを使う
            delta   = rng.normal(0, sigma, coords0.shape)
            coords1 = coords0 + delta
            d_geom  = geometry_distance(coords0, coords1)
            if abs(d_geom - d_target) > d_target * 0.5:
                continue
            if d_geom >= EPS:
                continue

            atom_str1 = "; ".join(
                f"{t} {x:.6f} {y:.6f} {z:.6f}"
                for t, (x, y, z) in zip(types, coords1)
            )
            try:
                mol1 = gto.M(atom=atom_str1, basis=BASIS, charge=0, spin=0,
                             verbose=0, unit="Angstrom")
                mf1 = dft.RKS(mol1); mf1.xc = METHOD; mf1.conv_tol = 1e-9
                mf1.kernel()
                E1 = float(mf1.e_tot)
            except Exception:
                continue

            dE = abs(E1 - E0)
            L_bin.append((d_geom, dE, dE/d_geom))
            if len(L_bin) >= n_per_bin:
                break

        if L_bin:
            L_max  = max(v[2] for v in L_bin)
            L_mean = np.mean([v[2] for v in L_bin])
            d_mean = np.mean([v[0] for v in L_bin])
            curve.append({"d_target": d_target, "d_mean": d_mean,
                          "L_max": L_max, "L_mean": L_mean, "n": len(L_bin)})
            print(f"  {d_target:>10.3f}  {d_mean:>10.4f}  {L_max:>12.4f}  {L_mean:>12.4f}")

    # L_geom の d依存性: 定数か増加か
    if len(curve) >= 3:
        xs = [c["d_mean"] for c in curve]
        ys = [c["L_max"] for c in curve]
        slope, intercept = np.polyfit(xs, ys, 1)
        print(f"\n  L_max vs d_geom の線形フィット: L = {slope:.2f}×d + {intercept:.4f}")
        if abs(slope) < 0.5 * intercept / max(xs):
            print("  → L はほぼ定数 (Lipschitz 定数が位置に依存しない)")
        else:
            print(f"  → L は d と {'正' if slope > 0 else '負'}の相関あり")

    return curve


# ── EXP-C: 統計的違反率分析 ──────────────────────────────────────────────────

def exp_c(n=30, sigma=0.04):
    """n=30 で ratio 分布と必要な L を統計的に定量化"""
    print(f"\n{'='*65}")
    print(f"  [EXP-C] 統計的違反率分析 (Si(OH)4, n={n}, σ={sigma}Å)")
    print(f"{'='*65}")
    coords0, types = ref_sioh4()
    rng = np.random.default_rng(77)

    from pyscf import gto, dft
    atom_str0 = "; ".join(
        f"{t} {x:.6f} {y:.6f} {z:.6f}"
        for t, (x, y, z) in zip(types, coords0)
    )
    mol0 = gto.M(atom=atom_str0, basis=BASIS, charge=0, spin=0,
                 verbose=0, unit="Angstrom")
    mf0 = dft.RKS(mol0); mf0.xc = METHOD; mf0.conv_tol = 1e-9
    mf0.kernel()
    E0 = float(mf0.e_tot)

    L_vals = []
    for i in range(n):
        delta  = rng.normal(0, sigma, coords0.shape)
        coords1 = coords0 + delta
        d_geom  = geometry_distance(coords0, coords1)
        if d_geom >= EPS or d_geom < 1e-10:
            continue

        atom_str1 = "; ".join(
            f"{t} {x:.6f} {y:.6f} {z:.6f}"
            for t, (x, y, z) in zip(types, coords1)
        )
        try:
            mol1 = gto.M(atom=atom_str1, basis=BASIS, charge=0, spin=0,
                         verbose=0, unit="Angstrom")
            mf1 = dft.RKS(mol1); mf1.xc = METHOD; mf1.conv_tol = 1e-9
            mf1.kernel()
            E1 = float(mf1.e_tot)
        except Exception as ex:
            print(f"  [{i}] failed: {ex}")
            continue

        dE = abs(E1 - E0)
        L_emp = dE / d_geom
        L_vals.append({"i": i, "d": d_geom, "dE": dE, "L": L_emp})
        print(f"  [{i:2d}] d={d_geom:.4f}  L={L_emp:.4f}")

    if L_vals:
        Ls = [v["L"] for v in L_vals]
        L_p50 = np.percentile(Ls, 50)
        L_p90 = np.percentile(Ls, 90)
        L_p99 = np.percentile(Ls, 99) if len(Ls) >= 10 else max(Ls)
        L_max = max(Ls)

        print(f"\n  L_geom 分布 (n={len(L_vals)}):")
        print(f"    p50  = {L_p50:.4f} Ha/Å")
        print(f"    p90  = {L_p90:.4f} Ha/Å")
        print(f"    p99  = {L_p99:.4f} Ha/Å")
        print(f"    max  = {L_max:.4f} Ha/Å")
        print(f"\n  Theorem 2' 上界 (各パーセンタイル × ε=0.10Å):")
        for p, L in [("p50", L_p50), ("p90", L_p90), ("max", L_max)]:
            print(f"    {p}: {L*EPS*627.509:.1f} kcal/mol")
        print(f"\n  信頼性の解釈:")
        print(f"    L_max (最悪ケース保証): {L_max*EPS*627.509:.1f} kcal/mol")
        print(f"    L_p90 (90%ケース保証): {L_p90*EPS*627.509:.1f} kcal/mol")
        print(f"    → 実際の誤差の90%は {L_p90*EPS*627.509:.1f} kcal/mol 以下")

    return L_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A","B","C"], action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="expand_verification.json")
    args = ap.parse_args()

    if args.all:
        args.exp = ["A", "B", "C"]
    if not args.exp:
        args.exp = ["A"]

    all_results = {}
    if "A" in args.exp:
        all_results["exp_a"] = exp_a(n_samples=8, sigma=0.03)
    if "B" in args.exp:
        all_results["exp_b"] = exp_b(n_per_bin=5)
    if "C" in args.exp:
        all_results["exp_c"] = exp_c(n=25, sigma=0.04)

    json.dump(all_results, open(args.out, "w"), indent=2, default=str)
    print(f"\n  結果: {args.out}")


if __name__ == "__main__":
    main()
