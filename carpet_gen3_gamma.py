"""
carpet_gen3_gamma.py — Gen3 Sierpinski Carpet γ∞ 測定 (QC不要、幾何のみ)

Conjecture 1 検証: lim_{n→∞} 2γ∞(n→n+1) = d_H = 1.893

γ(R) = d(log N_bank)/d(log N) を R_cut の関数として測定し、
γ(R) = γ∞(1 - exp(-k(R - R_th))) にフィットして γ∞ を抽出する。
"""
import numpy as np
import json, itertools
from scipy.optimize import curve_fit

A_H3   = 0.75
D_CARP = 2.0 * A_H3   # = 1.5 Å

OFFSETS_8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

def carpet_centers(gen, d=D_CARP):
    if gen == 1:
        return [(d*ox, d*oy) for (ox,oy) in OFFSETS_8]
    base = carpet_centers(gen-1, d)
    D_n  = (3**(gen-1)) * d
    return [(D_n*ox + gx, D_n*oy + gy)
            for (ox,oy) in OFFSETS_8
            for (gx,gy) in base]

def geom_key_trimer(a, b, c):
    """Sort distances for 3-point motif."""
    pts = np.array([a, b, c])
    dists = tuple(sorted([
        round(np.linalg.norm(pts[i] - pts[j]), 5)
        for i in range(3) for j in range(i+1, 3)
    ]))
    return dists

def count_unique_trimers(centers, r_cut):
    centers = np.array(centers)
    N = len(centers)
    bank = set()
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(centers[i]-centers[j]) > r_cut:
                continue
            for k in range(j+1, N):
                if (np.linalg.norm(centers[i]-centers[k]) <= r_cut and
                    np.linalg.norm(centers[j]-centers[k]) <= r_cut):
                    key = geom_key_trimer(centers[i], centers[j], centers[k])
                    bank.add(key)
                    count += 1
                    if count > 5_000_000:
                        print(f"    [trimers capped at 5M, N_bank={len(bank)}]")
                        return len(bank), count
    return len(bank), count

def model_A(logR, gamma_inf, k, R_th):
    R = np.exp(logR)
    return gamma_inf * (1.0 - np.exp(-k * (R - R_th)))

# ----------------------------------------------------------------
# Build Gen3 geometry
# ----------------------------------------------------------------
print("Building Sierpinski Carpet geometries...")
centers = {
    1: carpet_centers(1),
    2: carpet_centers(2),
    3: carpet_centers(3),
}
for gen, c in centers.items():
    print(f"  Gen{gen}: N={len(c)} clusters")

# R_cut scan for each generation pair
R_cuts = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])

results = {}
for gen in [1, 2, 3]:
    cx = np.array(centers[gen])
    N  = len(cx)
    R_max = np.linalg.norm(cx.max(axis=0) - cx.min(axis=0)) * 0.5
    print(f"\n=== Gen{gen} (N={N}) R_max~{R_max:.1f} Å ===")

    scan = []
    for R in R_cuts:
        if R > R_max * 1.2:
            break
        n_bank, n_trim = count_unique_trimers(cx, R)
        if n_trim == 0:
            continue
        print(f"  R={R:.1f}: N_bank={n_bank}, N_trim={n_trim}")
        scan.append({'R': R, 'N_bank': n_bank, 'N_trim': n_trim, 'N': N})
    results[gen] = scan

# ----------------------------------------------------------------
# Compute γ(R) for each generation transition
# γ(R) = d log(N_bank) / d log(N)
# Use Gen1 and Gen2 to get N as function; compare bank at same R_cut
# ----------------------------------------------------------------
print("\n" + "="*60)
print("γ∞ ESTIMATION VIA BANK GROWTH RATE")
print("="*60)

def get_Nbank_at_R(gen_scan, R):
    """Interpolate N_bank at given R."""
    Rs = np.array([s['R'] for s in gen_scan])
    Ns = np.array([s['N_bank'] for s in gen_scan])
    if R < Rs.min() or R > Rs.max():
        return None
    return float(np.interp(R, Rs, Ns))

gamma_inf_results = {}
for (g_lo, g_hi) in [(1,2), (2,3)]:
    scan_lo = results[g_lo]
    scan_hi = results[g_hi]
    if not scan_lo or not scan_hi:
        continue

    N_lo = centers[g_lo].__len__()
    N_hi = centers[g_hi].__len__()
    log_N_ratio = np.log(N_hi / N_lo)

    common_Rs = []
    gamma_vals = []
    for s in scan_lo:
        R = s['R']
        nb_lo = s['N_bank']
        nb_hi = get_Nbank_at_R(scan_hi, R)
        if nb_hi is None or nb_lo < 1:
            continue
        g = np.log(nb_hi / nb_lo) / log_N_ratio
        common_Rs.append(R)
        gamma_vals.append(g)
        print(f"  Gen{g_lo}→{g_hi}  R={R:.1f}: γ={g:.4f}")

    if len(common_Rs) < 4:
        print(f"  Not enough points for Gen{g_lo}→{g_hi}")
        continue

    # Fit Model A: γ(R) = γ∞(1 - exp(-k(R - R_th)))
    log_Rs = np.log(common_Rs)
    try:
        popt, pcov = curve_fit(
            model_A, log_Rs, gamma_vals,
            p0=[0.85, 0.05, np.log(common_Rs[0])],
            bounds=([0.5, 1e-4, -5], [2.0, 5.0, 10.0]),
            maxfev=10000
        )
        gamma_inf, k, log_Rth = popt
        R_th = np.exp(log_Rth)
        perr = np.sqrt(np.diag(pcov))
        gamma_inf_err = perr[0]
    except Exception as e:
        # Fallback: use max observed gamma as lower bound
        gamma_inf = max(gamma_vals)
        R_th = common_Rs[-1]
        k = 0.05
        gamma_inf_err = 0.05
        print(f"  Fit failed ({e}), using max γ={gamma_inf:.3f}")

    gamma_inf_results[f'Gen{g_lo}→{g_hi}'] = {
        'gamma_inf': gamma_inf,
        'gamma_inf_err': gamma_inf_err,
        'D_info': 2 * gamma_inf,
        'k': k,
        'R_th': R_th,
        'R_vals': common_Rs,
        'gamma_vals': gamma_vals,
    }
    print(f"\n  Gen{g_lo}→{g_hi}: γ∞ = {gamma_inf:.3f} ± {gamma_inf_err:.3f}")
    print(f"           D_info = 2γ∞ = {2*gamma_inf:.3f}")
    print(f"           k={k:.4f}, R_th={R_th:.2f} Å")

# ----------------------------------------------------------------
# Conjecture 1 assessment
# ----------------------------------------------------------------
print("\n" + "="*60)
print("CONJECTURE 1 — D_info → d_H = 1.893")
print("="*60)
d_H = 1.893
known = {'Gen1→2': 0.710, 'Gen2→3': 0.805}

all_gamma_inf = {**known}
for k, v in gamma_inf_results.items():
    all_gamma_inf[k] = v['gamma_inf']

print(f"\n  {'Transition':<12} {'γ∞':>8} {'2γ∞':>8} {'err%':>8} {'status'}")
target_hit = {}
for trans in ['Gen1→2', 'Gen2→3', 'Gen3→4']:
    if trans in all_gamma_inf:
        g = all_gamma_inf[trans]
        err = abs(2*g - d_H)/d_H * 100
        conv = "↑ converging" if err < 25 else "–"
        print(f"  {trans:<12} {g:>8.3f} {2*g:>8.3f} {err:>7.1f}% {conv}")
        target_hit[trans] = {'gamma_inf': g, 'D_info': 2*g, 'err_pct': err}
    else:
        print(f"  {trans:<12} {'—':>8}")

# Check convergence
if 'Gen3→4' in all_gamma_inf:
    g3 = all_gamma_inf.get('Gen3→4', None)
    g2 = all_gamma_inf['Gen2→3']
    if g3 is not None and g3 > g2:
        print("\n  CONVERGENCE DIRECTION: ✅ γ∞ increasing toward d_H/2")
        if 0.832 <= g3 <= 0.892:
            print("  Gen3→4 γ∞ in predicted range [0.832, 0.892] ✅ CONJECTURE 1 SUPPORTED")
        else:
            print(f"  Gen3→4 γ∞ = {g3:.3f} (predicted 0.832-0.892)")
    else:
        print("  ⚠️ Gen3→4 γ∞ not clearly larger than Gen2→3")

# Save
out = {
    'gamma_inf_results': {k: {kk: (vv if not isinstance(vv, list) else vv)
                              for kk, vv in v.items()}
                          for k, v in gamma_inf_results.items()},
    'known': known,
    'all_gamma_inf': all_gamma_inf,
    'd_H': d_H,
    'conjecture1_assessment': target_hit,
}
with open('/home/yoiyoi/carpet_gen3_gamma_result.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved: carpet_gen3_gamma_result.json")
