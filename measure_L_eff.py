#!/usr/bin/env python3
"""
measure_L_eff.py
L_effective の実測: soft-match 対象配置 (sigma=0.03A) に限定した Lipschitz 定数

check_softmatch_range.py の結果より:
  sigma=0.03A → geom_RMSD 平均 0.042A < 0.10A (100% soft-match)
  sigma=0.10A → geom_RMSD 平均 0.137A > 0.10A (0%  soft-match)

よって L_eff = max||F|| at sigma=0.03A が Theorem 2 に使うべき値。
"""
import os, sys, numpy as np, json, time
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from measure_lipschitz import ref_sioh4, geometry_distance

SIGMA   = 0.03   # soft-match 確実な変位 (Å)
N_GEOMS = 8
BASIS   = "def2-svp"
METHOD  = "pbe"

def main():
    from pyscf import gto, dft

    coords0, types = ref_sioh4()
    rng = np.random.default_rng(99)

    print(f"\nL_eff 実測 (sigma={SIGMA}A, {METHOD}/{BASIS})")
    print(f"全配置が geom_RMSD < 0.10A を満たすことを確認しながら力を計算")
    print("="*60)

    geoms = [("ref", coords0)] + [
        (f"d{i}", coords0 + rng.normal(0, SIGMA, coords0.shape))
        for i in range(1, N_GEOMS)
    ]

    results = []
    for label, coords in geoms:
        # geom_key RMSD 確認
        d_geom = geometry_distance(coords0, coords)

        atom_str = "; ".join(
            f"{t} {x:.6f} {y:.6f} {z:.6f}"
            for t, (x, y, z) in zip(types, coords)
        )
        mol = gto.M(atom=atom_str, basis=BASIS, charge=0, spin=0,
                    verbose=0, unit="Angstrom")
        mf = dft.RKS(mol)
        mf.xc = METHOD
        mf.conv_tol = 1e-9
        t0 = time.perf_counter()
        mf.kernel()
        g = mf.nuc_grad_method()
        forces = -g.kernel()          # Ha/Bohr
        forces_A = forces / 1.8897259  # Ha/Å
        dt = time.perf_counter() - t0

        F_norm   = float(np.linalg.norm(forces_A))
        F_max_at = float(np.max(np.linalg.norm(forces_A, axis=1)))
        soft_ok  = d_geom < 0.10

        results.append({
            "label": label, "d_geom_A": d_geom, "soft_match": soft_ok,
            "F_norm_Ha_A": F_norm, "F_max_atom_Ha_A": F_max_at, "time_s": dt
        })
        mark = "OK" if soft_ok else "MISS"
        print(f"  [{label}] geom_RMSD={d_geom:.4f}A [{mark}]  "
              f"||F||={F_norm:.4f}  max_atom={F_max_at:.4f}  ({dt:.1f}s)")

    # soft-match 配置のみ集計
    hit = [r for r in results if r["soft_match"]]
    all_F = [r["F_norm_Ha_A"] for r in results]

    L_conservative = max(all_F)
    L_eff = max(r["F_norm_Ha_A"] for r in hit) if hit else float("nan")

    n = 9
    eps = 0.10
    factor = (2*(n-1))**0.5  # = 4.0

    print("\n" + "="*60)
    print("  Theorem 2 誤差上界 (Si(OH)4, n=9, eps=0.10A)")
    print("="*60)
    print(f"  L_conservative (全配置) = {L_conservative:.4f} Ha/A "
          f"= {L_conservative*627.509:.1f} kcal/mol/A")
    print(f"  L_eff (soft-match のみ) = {L_eff:.4f} Ha/A "
          f"= {L_eff*627.509:.1f} kcal/mol/A")
    print()
    print(f"  上界 (L_eff):   {L_eff:.4f} x {eps} x {factor:.1f} "
          f"= {L_eff*eps*factor:.4f} Ha "
          f"= {L_eff*eps*factor*627.509:.2f} kcal/mol/fragment")
    print()
    bound_eff = L_eff * eps * factor * 627.509
    if bound_eff < 1.0:
        verdict = "化学精度内 (< 1 kcal/mol/fragment)"
    elif bound_eff < 5.0:
        verdict = "工学精度 (< 5 kcal/mol/fragment)"
    else:
        verdict = f"保守的上界 ({bound_eff:.1f} kcal/mol) — 実際の誤差はこれより小"
    print(f"  判定: {verdict}")
    print("="*60)

    json.dump({
        "sigma_A": SIGMA, "method": METHOD, "basis": BASIS,
        "L_conservative_Ha_A": L_conservative,
        "L_eff_Ha_A": L_eff,
        "theorem2_bound_conservative_kcal": L_conservative*eps*factor*627.509,
        "theorem2_bound_eff_kcal": L_eff*eps*factor*627.509,
        "n_soft_match": len(hit), "n_total": len(results),
        "samples": results
    }, open("lipschitz_eff_result.json", "w"), indent=2)
    print("\n  -> lipschitz_eff_result.json に保存")

if __name__ == "__main__":
    main()
