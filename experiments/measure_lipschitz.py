#!/usr/bin/env python3
"""
measure_lipschitz.py
Empirical measurement of Lipschitz constant L for QC energy surface.

定義:
  L = sup_{F ≠ F'} |E_QC(F) - E_QC(F')| / d_G(F, F')

d_G(F, F') = RMSD of distance vectors (MotifBank soft matching metric)

2通りの測定法:
  [A] 有限差分法: 変位前後のエネルギー差から直接 L を測定
  [B] 勾配法 (推奨): Hellmann-Feynman 定理より L ≤ max ||F||
      (力 = エネルギー勾配、その最大値が Lipschitz 上界)
      計算1回で済む。--grad オプションで有効化。

文献根拠:
  - Hirn et al. (2017, Wavelet scattering): PES の Lipschitz 連続性を
    ML汎化誤差の理論的根拠として明示的に仮定
  - Hellmann-Feynman: ∇_R E = −F(R)  →  L ≤ max_R ||F(R)||
  - 実験的目安: 平衡構造近傍で L ~ 1–10 eV/Å (Perplexity 調査)

実行:
  OMP_NUM_THREADS=1 python3 measure_lipschitz.py --grad          # 勾配法
  OMP_NUM_THREADS=1 python3 measure_lipschitz.py --n_samples 10  # 有限差分法
  OMP_NUM_THREADS=1 python3 measure_lipschitz.py --mock          # テスト
"""

import os, sys, argparse, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))

from motifbank_cli import dist_vec, qc_compute_pyscf, qc_compute_mock

# ── 参照ジオメトリ: Si(OH)4 最適化構造 (PBE/def2-SVP) ──────────────────────
# Si を原点、4本の O-H を正四面体方向に配置
# Si-O: 1.635 Å, O-H: 0.963 Å (PBE/def2-SVP 最適化値)

def ref_sioh4():
    """Si(OH)4 参照ジオメトリ (Å, Si 中心)"""
    si_o = 1.635
    o_h  = 0.963
    # 正四面体方向
    tet = np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
    ], dtype=float)
    tet /= np.linalg.norm(tet, axis=1, keepdims=True)

    si = np.array([0., 0., 0.])
    o_coords = [si + si_o * d for d in tet]
    h_coords = [oc + o_h * d for oc, d in zip(o_coords, tet)]

    coords = np.array([si] + o_coords + h_coords)  # shape (9, 3)
    types  = ["Si"] + ["O"]*4 + ["H"]*4
    return coords, types


def geometry_distance(coords1, coords2):
    """MotifBank の soft matching 距離 (RMSD of distance vectors)"""
    mol1 = [coords1]
    mol2 = [coords2]
    dv1 = dist_vec(mol1)
    dv2 = dist_vec(mol2)
    return float(np.sqrt(np.mean((dv1 - dv2)**2)))


def max_atom_disp(coords1, coords2):
    """最大原子変位 (Å)"""
    return float(np.max(np.linalg.norm(coords1 - coords2, axis=1)))


def measure_L(n_samples=20, sigmas=None, backend="pyscf",
              basis="def2-svp", method="pbe", verbose=True):
    """
    Lipschitz定数 L を実測する。

    Returns:
      results: list of dicts {sigma, sample_id, dE_Ha, d_geom, L_emp, max_disp}
    """
    if sigmas is None:
        sigmas = [0.01, 0.05, 0.10, 0.20, 0.30]

    coords0, types = ref_sioh4()
    mol0 = [coords0]
    atom_types0 = [types]

    # 参照エネルギー計算
    if verbose:
        print(f"[Lipschitz] 参照エネルギー計算中 ({method}/{basis})...")
    t0 = time.perf_counter()
    if backend == "pyscf":
        E0 = qc_compute_pyscf(mol0, atom_types0, basis=basis, method=method)
    else:
        E0 = qc_compute_mock(mol0)
    dt_ref = time.perf_counter() - t0
    if verbose:
        print(f"  E0 = {E0:.8f} Ha  ({dt_ref:.1f}s)")

    rng = np.random.default_rng(42)
    results = []

    for sigma in sigmas:
        if verbose:
            print(f"\n[σ={sigma:.2f}Å] {n_samples} サンプル計算中...")
        L_vals = []

        for i in range(n_samples):
            # ランダム変位を加えたジオメトリ
            delta = rng.normal(0, sigma, coords0.shape)
            coords1 = coords0 + delta
            mol1 = [coords1]

            # ジオメトリ距離
            d_geom = geometry_distance(coords0, coords1)
            d_max  = max_atom_disp(coords0, coords1)

            if d_geom < 1e-10:
                continue

            # エネルギー計算
            t1 = time.perf_counter()
            try:
                if backend == "pyscf":
                    E1 = qc_compute_pyscf(mol1, atom_types0,
                                          basis=basis, method=method)
                else:
                    E1 = qc_compute_mock(mol1)
            except Exception as ex:
                if verbose:
                    print(f"  sample {i}: SCF failed ({ex})")
                continue
            dt = time.perf_counter() - t1

            dE = abs(E1 - E0)
            L_emp = dE / d_geom

            results.append({
                "sigma":    sigma,
                "sample":   i,
                "dE_Ha":    dE,
                "dE_kcal":  dE * 627.509,
                "d_geom":   d_geom,
                "d_max_A":  d_max,
                "L_emp":    L_emp,
                "L_kcal":   L_emp * 627.509,
                "time_s":   dt,
            })
            L_vals.append(L_emp)

            if verbose:
                print(f"  [{i:2d}] σ={sigma:.2f}  d={d_geom:.4f}Å  "
                      f"ΔE={dE*627.509:.3f} kcal/mol  L={L_emp*627.509:.2f} kcal/mol/Å")

        if L_vals and verbose:
            print(f"  → max L = {max(L_vals)*627.509:.2f} kcal/mol/Å  "
                  f"mean L = {np.mean(L_vals)*627.509:.2f}")

    return results, E0


def summary(results, E0):
    """結果サマリーを表示し、Theorem 2 の誤差上界を計算"""
    if not results:
        print("結果なし")
        return

    import collections
    by_sigma = collections.defaultdict(list)
    for r in results:
        by_sigma[r["sigma"]].append(r["L_emp"])

    print("\n" + "="*65)
    print("  Lipschitz 定数 L の実測サマリー")
    print("  E_QC: PBE/def2-SVP,  fragment: Si(OH)4 (9 atoms)")
    print("  d_G: RMSD of distance vectors (MotifBank 距離)")
    print("="*65)
    print(f"  {'σ (Å)':>7}  {'N':>4}  {'L_max':>12}  {'L_mean':>12}  "
          f"{'ΔE_max(kcal)':>14}")
    print("  " + "-"*58)

    L_global_max = 0.0
    for sigma in sorted(by_sigma.keys()):
        Ls = by_sigma[sigma]
        L_max  = max(Ls)
        L_mean = np.mean(Ls)
        dE_max = max(r["dE_kcal"] for r in results if r["sigma"] == sigma)
        L_global_max = max(L_global_max, L_max)
        print(f"  {sigma:>7.2f}  {len(Ls):>4}  "
              f"{L_max*627.509:>10.2f} k/Å  "
              f"{L_mean*627.509:>10.2f} k/Å  "
              f"{dE_max:>12.3f}")

    # Theorem 2 誤差上界
    n_atoms = 9  # Si(OH)4
    eps_geom = 0.10  # Å (MotifBank soft matching threshold)
    m = n_atoms * (n_atoms - 1) // 2  # = 36 ペア

    # 理論的上界: L × ε × √(2(n-1))
    bound_Ha   = L_global_max * eps_geom * np.sqrt(2 * (n_atoms - 1))
    bound_kcal = bound_Ha * 627.509

    print()
    print("  Theorem 2 誤差上界 (ε=0.10Å, n=9, √2(n-1)=4.0):")
    print(f"  L_max = {L_global_max:.4f} Ha/Å = {L_global_max*627.509:.2f} kcal/mol/Å")
    print(f"  上界  = L × ε × √(2(n-1))")
    print(f"        = {L_global_max:.4f} × 0.10 × {np.sqrt(2*(n_atoms-1)):.2f}")
    print(f"        = {bound_Ha:.6f} Ha")
    print(f"        = {bound_kcal:.3f} kcal/mol  per fragment")
    print()
    if bound_kcal < 1.0:
        verdict = "✅ < 1 kcal/mol/fragment (化学精度内)"
    elif bound_kcal < 5.0:
        verdict = "⚠️  1〜5 kcal/mol/fragment (許容範囲)"
    else:
        verdict = "❌ > 5 kcal/mol/fragment (要注意)"
    print(f"  判定: {verdict}")
    print("="*65)
    return L_global_max


def measure_L_gradient(backend="pyscf", basis="def2-svp", method="pbe",
                        n_geoms=10, sigma=0.10, verbose=True):
    """
    Hellmann-Feynman 法: L ≤ max ||F(R)||
    複数ジオメトリで力を計算し、その最大ノルムを L の上界とする。
    有限差分法より約 (n_samples × n_sigmas) 倍 効率的。
    """
    try:
        from pyscf import gto, dft, grad
    except ImportError:
        print("PySCF が必要です (pip install pyscf)")
        return None

    coords0, types = ref_sioh4()
    rng = np.random.default_rng(42)
    F_norms = []

    geoms = [coords0] + [
        coords0 + rng.normal(0, sigma, coords0.shape)
        for _ in range(n_geoms - 1)
    ]

    for gi, coords in enumerate(geoms):
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

        g = mf.nuc_grad_method()
        forces = -g.kernel()          # shape (n_atoms, 3), Ha/Bohr
        forces_A = forces / 1.8897259  # Ha/Bohr → Ha/Å

        F_norm = float(np.linalg.norm(forces_A))  # Frobenius norm
        F_max  = float(np.max(np.linalg.norm(forces_A, axis=1)))
        F_norms.append(F_norm)

        if verbose:
            label = "ref" if gi == 0 else f"δ{gi}"
            print(f"  [{label}] ||F||={F_norm:.4f} Ha/Å  max_atom={F_max:.4f} Ha/Å")

    L_upper = max(F_norms)
    if verbose:
        print(f"\n  L ≤ max||F|| = {L_upper:.4f} Ha/Å = {L_upper*627.509:.2f} kcal/mol/Å")
    return L_upper, F_norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock",      action="store_true",
                    help="PySCF の代わりに mock を使う (テスト用)")
    ap.add_argument("--grad",      action="store_true",
                    help="勾配法 (Hellmann-Feynman) で L を測定 (高速・推奨)")
    ap.add_argument("--n_samples", type=int, default=10,
                    help="各σでのサンプル数 (デフォルト: 10)")
    ap.add_argument("--n_geoms",   type=int, default=10,
                    help="勾配法でのジオメトリ数 (デフォルト: 10)")
    ap.add_argument("--sigmas",    type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.20],
                    help="変位の標準偏差 σ (Å)")
    ap.add_argument("--basis",     default="def2-svp")
    ap.add_argument("--method",    default="pbe")
    ap.add_argument("--out",       default=None,
                    help="結果を JSON に保存するパス")
    args = ap.parse_args()

    backend = "mock" if args.mock else "pyscf"

    print(f"""
{'='*65}
  MotifBank Lipschitz 定数 実測スクリプト
  backend  : {backend}
  method   : {args.method}/{args.basis}
  mode     : {'勾配法 (Hellmann-Feynman)' if args.grad else '有限差分法'}
{'='*65}
""")

    if args.grad and not args.mock:
        # 勾配法: L ≤ max ||F||
        print("  [勾配法] 力ノルムから Lipschitz 上界を測定...")
        result = measure_L_gradient(
            backend=backend, basis=args.basis, method=args.method,
            n_geoms=args.n_geoms, sigma=0.10,
        )
        if result:
            L_max, F_norms = result
            n_atoms = 9
            eps_geom = 0.10
            bound_kcal = L_max * eps_geom * (2*(n_atoms-1))**0.5 * 627.509
            print(f"\n  Theorem 2 誤差上界 (ε=0.10Å, n=9):")
            print(f"  = {L_max:.4f} × 0.10 × {(2*(n_atoms-1))**0.5:.2f} = "
                  f"{bound_kcal:.3f} kcal/mol/fragment")
            if args.out:
                import json
                json.dump({"method": args.method, "basis": args.basis,
                           "mode": "gradient", "L_upper_Ha_per_A": L_max,
                           "L_upper_kcal_per_A": L_max*627.509,
                           "theorem2_bound_kcal": bound_kcal,
                           "F_norms": F_norms},
                          open(args.out, "w"), indent=2)
        return

    # 有限差分法
    results, E0 = measure_L(
        n_samples=args.n_samples,
        sigmas=args.sigmas,
        backend=backend,
        basis=args.basis,
        method=args.method,
    )

    L_max = summary(results, E0)

    if args.out and results:
        import json
        out = {
            "E0_Ha": E0,
            "method": args.method,
            "basis": args.basis,
            "mode": "finite_difference",
            "n_atoms": 9,
            "eps_geom_A": 0.10,
            "L_max_Ha_per_A": L_max,
            "L_max_kcal_per_A": L_max * 627.509,
            "theorem2_bound_kcal": L_max * 0.10 * (2*8)**0.5 * 627.509,
            "samples": results,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  結果を保存: {args.out}")


if __name__ == "__main__":
    main()
