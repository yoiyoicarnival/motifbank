#!/usr/bin/env python3
"""
compute_jacobian.py
J_d の解析的計算と条件数 κ = ||J_d^+||_2 = 1/σ_min の推定

Perplexity の提案に基づく:
- d(R) = sorted {||r_i - r_j||} の Jacobian は解析的に計算可能
- ソート操作は直交行列 P → singular values に影響しない
- SVD で σ_min → κ = 1/σ_min → L_geom ≤ L_atomic × κ
"""
import os, sys
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
from measure_lipschitz import ref_sioh4, geometry_distance


def compute_Jd_analytic(coords):
    """
    距離マップ d: R^{3N} -> R^{N(N-1)/2} の Jacobian を解析的に計算。
    各行 (i,j): ∂d_ij/∂r_i = (r_i - r_j)/||r_i-r_j||, ∂d_ij/∂r_j = -ditto
    返り値: J_d ∈ R^{M × 3N} (M = N(N-1)/2, 3N = 3*N)
    """
    N = len(coords)
    M = N * (N - 1) // 2
    dof = 3 * N
    J = np.zeros((M, dof))

    row = 0
    for i in range(N):
        for j in range(i+1, N):
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                row += 1
                continue
            unit = diff / dist   # shape (3,)
            # ∂d_ij/∂r_i = +unit_ij
            J[row, 3*i:3*i+3] =  unit
            # ∂d_ij/∂r_j = -unit_ij
            J[row, 3*j:3*j+3] = -unit
            row += 1
    return J


def analyze_jacobian(coords, label=""):
    N = len(coords)
    M = N * (N - 1) // 2
    dof = 3 * N
    print(f"\n{'='*55}")
    print(f"  {label}: N={N} atoms, M={M} pairs, 3N={dof} dof")
    print(f"  J_d shape: {M} × {dof}")

    J = compute_Jd_analytic(coords)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)

    # 3D剛体運動の null space (translation + rotation = 6 dof) を除外
    # → 最小の非ゼロ特異値を使う
    tol = 1e-6 * s[0]
    s_nonzero = s[s > tol]
    s_min = s_nonzero[-1]
    s_max = s_nonzero[0]

    kappa = 1.0 / s_min   # ||J_d^+||_2
    cond  = s_max / s_min  # 条件数

    print(f"  σ_max = {s_max:.6f}")
    print(f"  σ_min = {s_min:.6f}  (非ゼロ {len(s_nonzero)}/{len(s)})")
    print(f"  ||J_d^+||_2 = κ = 1/σ_min = {kappa:.4f}")
    print(f"  cond(J_d) = σ_max/σ_min = {cond:.4f}")
    L_atomic = 0.1140
    sqrt_M = float(np.sqrt(M))
    L_geom_theory = L_atomic * kappa * sqrt_M  # L_geom ≤ L_atomic × κ × √M

    print(f"\n  理論上界: L_geom ≤ L_atomic × κ × √M")
    print(f"  L_atomic = {L_atomic} Ha/Å,  √M = {sqrt_M:.4f},  κ = {kappa:.4f}")
    print(f"  L_geom_theory ≤ {L_atomic:.4f} × {kappa:.4f} × {sqrt_M:.4f} = {L_geom_theory:.4f} Ha/Å")

    print(f"\n  特異値 (上位10):")
    for i, sv in enumerate(s[:10]):
        marker = " ← σ_min (非ゼロ)" if abs(sv - s_min) < tol*10 else ""
        print(f"    σ_{i+1} = {sv:.6f}{marker}")

    return {"label": label, "N": N, "M": M, "sqrt_M": sqrt_M,
            "sigma_max": float(s_max), "sigma_min": float(s_min),
            "kappa": float(kappa), "cond": float(cond),
            "L_geom_theory_upper_Ha_A": float(L_geom_theory),
            "singular_values": s.tolist()}


def main():
    import json

    results = {}

    # Si(OH)4
    coords, types = ref_sioh4()
    r = analyze_jacobian(coords, label="Si(OH)4")
    results["Si(OH)4"] = r

    # H2O
    h2o = np.array([[0., 0., 0.], [0.757, 0.586, 0.], [-0.757, 0.586, 0.]])
    r2 = analyze_jacobian(h2o, label="H2O")
    results["H2O"] = r2

    # Al(OH)4^-
    al_o, o_h = 1.77, 0.96
    tet = np.array([[1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1]], dtype=float)
    tet /= np.linalg.norm(tet, axis=1, keepdims=True)
    al = np.array([0., 0., 0.])
    o_coords = [al + al_o * d for d in tet]
    h_coords = [oc + o_h * d for oc, d in zip(o_coords, tet)]
    aloh4 = np.array([al] + o_coords + h_coords)
    r3 = analyze_jacobian(aloh4, label="Al(OH)4^-")
    results["Al(OH)4"] = r3

    json.dump(results, open("jacobian_analysis.json", "w"), indent=2)
    print(f"\n  結果: jacobian_analysis.json")


if __name__ == "__main__":
    main()
