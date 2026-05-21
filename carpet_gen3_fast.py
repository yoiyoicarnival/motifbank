"""
carpet_gen3_fast.py — Gen3 Sierpinski Carpet γ∞ (高速版, scipy KDTree使用)

N³ の総当たりではなく、KDTree で r_cut 以内の近傍のみ探索する。
Gen3: N=512, O(N * k² * log N) ≈ manageable
"""
import numpy as np
import json, math
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

A_H3   = 0.75
D_CARP = 2.0 * A_H3   # 1.5 Å

OFFSETS_8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

def carpet_centers_2d(gen, d=D_CARP):
    if gen == 1:
        return np.array([(d*ox, d*oy) for (ox,oy) in OFFSETS_8])
    base = carpet_centers_2d(gen-1, d)
    D_n  = (3**(gen-1)) * d
    parts = []
    for (ox,oy) in OFFSETS_8:
        shifted = base + np.array([D_n*ox, D_n*oy])
        parts.append(shifted)
    return np.vstack(parts)

def geom_key_3pts(a, b, c):
    dists = sorted([
        round(np.linalg.norm(a-b), 5),
        round(np.linalg.norm(a-c), 5),
        round(np.linalg.norm(b-c), 5),
    ])
    return tuple(dists)

def count_bank_kdtree(centers_2d, r_cut, max_bank=100000):
    """Count unique trimer geometries using KDTree for neighbor lookup."""
    # Extend to 3D (z=0) for uniformity
    N = len(centers_2d)
    pts = np.column_stack([centers_2d, np.zeros(N)])
    tree = cKDTree(pts)
    bank = set()
    n_trim = 0
    for i in range(N):
        neighbors_i = tree.query_ball_point(pts[i], r_cut)
        neighbors_i = [j for j in neighbors_i if j > i]
        for ji, j in enumerate(neighbors_i):
            neighbors_j = tree.query_ball_point(pts[j], r_cut)
            for k in neighbors_j:
                if k <= j:
                    continue
                if np.linalg.norm(pts[i]-pts[k]) <= r_cut:
                    key = geom_key_3pts(pts[i], pts[j], pts[k])
                    bank.add(key)
                    n_trim += 1
                    if len(bank) >= max_bank:
                        return len(bank), n_trim
    return len(bank), n_trim

# Build geometries
print("Building geometries...")
centers = {}
for gen in range(1, 4):
    centers[gen] = carpet_centers_2d(gen)
    print(f"  Gen{gen}: N={len(centers[gen])}")

# Determine R_cut range
# Gen1 span ~3 Å, Gen2 span ~9 Å, Gen3 span ~27 Å
R_cuts = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 12.0, 15.0, 18.0, 22.0])

print("\nScanning R_cut for each generation...")
scan_results = {}
for gen in [1, 2, 3]:
    cx = centers[gen]
    N  = len(cx)
    span = np.max(np.linalg.norm(cx - cx.mean(axis=0), axis=1)) * 2
    print(f"\n  Gen{gen} (N={N}, span~{span:.1f} Å):")
    scan = []
    for R in R_cuts:
        if R > span * 0.6:
            # Skip R_cuts too large (all trimers merge into one)
            pass
        nb, nt = count_bank_kdtree(cx, R, max_bank=200000)
        if nt == 0:
            continue
        print(f"    R={R:.1f}: N_bank={nb}, N_trim={nt}")
        scan.append({'R': float(R), 'N_bank': nb, 'N_trim': nt})
    scan_results[gen] = scan

# Compute γ(R) for each transition
print("\n" + "="*55)
print("γ∞ ESTIMATION")
print("="*55)

def gamma_at_R(scan_lo, scan_hi, R, N_lo, N_hi):
    Nb_lo = next((s['N_bank'] for s in scan_lo if abs(s['R']-R) < 0.01), None)
    Nb_hi = next((s['N_bank'] for s in scan_hi if abs(s['R']-R) < 0.01), None)
    if Nb_lo is None or Nb_hi is None or Nb_lo < 1:
        return None
    return math.log(Nb_hi / Nb_lo) / math.log(N_hi / N_lo)

def model_A(R, gamma_inf, k, R_th):
    val = gamma_inf * (1.0 - np.exp(-k * np.maximum(R - R_th, 0)))
    return val

gamma_inf_results = {}
for (g_lo, g_hi) in [(1, 2), (2, 3)]:
    scan_lo = scan_results.get(g_lo, [])
    scan_hi = scan_results.get(g_hi, [])
    if not scan_lo or not scan_hi:
        print(f"  Gen{g_lo}→{g_hi}: insufficient data")
        continue

    N_lo = len(centers[g_lo])
    N_hi = len(centers[g_hi])
    Rs_common = [s['R'] for s in scan_lo]
    gammas = []
    Rs_used = []
    for R in Rs_common:
        g = gamma_at_R(scan_lo, scan_hi, R, N_lo, N_hi)
        if g is not None and 0 < g < 2.0:
            gammas.append(g)
            Rs_used.append(R)
            print(f"  Gen{g_lo}→{g_hi}  R={R:.1f}: γ={g:.4f}")

    if len(Rs_used) < 3:
        gamma_inf = max(gammas) if gammas else float('nan')
        R_th_val = Rs_used[-1] if Rs_used else 0.0
        k_val = 0.05
        gamma_inf_err = 0.1
        print(f"  Limited data: γ∞ lower bound = {gamma_inf:.3f}")
    else:
        Rs_arr = np.array(Rs_used)
        gs_arr = np.array(gammas)
        try:
            popt, pcov = curve_fit(
                model_A, Rs_arr, gs_arr,
                p0=[max(gs_arr)*1.1, 0.1, Rs_arr[0]],
                bounds=([0.3, 1e-4, 0.0], [2.0, 5.0, Rs_arr[-1]]),
                maxfev=5000
            )
            gamma_inf, k_val, R_th_val = popt
            gamma_inf_err = np.sqrt(np.diag(pcov))[0]
        except Exception as e:
            gamma_inf = max(gammas)
            k_val = 0.05
            R_th_val = Rs_used[0]
            gamma_inf_err = 0.05
            print(f"  Fit warning: {e}")

    gamma_inf_results[f'Gen{g_lo}→{g_hi}'] = {
        'gamma_inf': float(gamma_inf),
        'gamma_inf_err': float(gamma_inf_err),
        'D_info': float(2 * gamma_inf),
        'k': float(k_val),
        'R_th': float(R_th_val),
        'R_vals': Rs_used,
        'gamma_vals': gammas,
    }
    print(f"\n  >>> Gen{g_lo}→{g_hi}: γ∞ = {gamma_inf:.3f} ± {gamma_inf_err:.3f}")
    print(f"               2γ∞ = D_info = {2*gamma_inf:.3f}")

# Conjecture 1 assessment
print("\n" + "="*55)
print("CONJECTURE 1 — D_info → d_H = 1.893")
print("="*55)
d_H = 1.893
known = {'Gen1→2': 0.710, 'Gen2→3': 0.805}
all_gamma = {**known, **{k: v['gamma_inf'] for k, v in gamma_inf_results.items()}}

print(f"\n  {'Transition':<12} {'γ∞':>8} {'2γ∞':>8} {'err%':>8}")
for trans in ['Gen1→2', 'Gen2→3', 'Gen3→4']:
    if trans in all_gamma:
        g = all_gamma[trans]
        err = abs(2*g - d_H)/d_H * 100
        status = "✅" if err < 20 else ("→" if err < 14.9 else "→")
        print(f"  {trans:<12} {g:>8.3f} {2*g:>8.3f} {err:>7.1f}%  {status}")
    else:
        print(f"  {trans:<12} {'—':>8}")

# Check convergence
if 'Gen3→4' in all_gamma:
    g23 = all_gamma.get('Gen2→3', 0.805)
    g34 = all_gamma['Gen3→4']
    converging = g34 > g23
    in_range = 0.832 <= g34 <= 0.892
    print(f"\n  Converging (γ∞ increasing): {'✅' if converging else '❌'}")
    print(f"  In predicted range [0.832, 0.892]: {'✅' if in_range else '⚠️'}")
    verdict = "CONJECTURE 1 SUPPORTED" if (converging and in_range) else \
              "CONVERGING but outside range" if converging else "NOT CONVERGING"
    print(f"  => {verdict}")

out = {
    'scan_results': scan_results,
    'gamma_inf_results': gamma_inf_results,
    'known': known,
    'all_gamma_inf': all_gamma,
    'd_H': d_H,
}
with open('/home/yoiyoi/carpet_gen3_gamma_result.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved: carpet_gen3_gamma_result.json")
