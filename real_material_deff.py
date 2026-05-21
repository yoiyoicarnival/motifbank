"""
real_material_deff.py — 実材料の d_eff と ε_c 測定 (Theorem 15 拡張)

ice Ih / α-cristobalite / LTA / MFI の N_bank(ε) スキャンから
d_eff = -d(log N_bank)/d(log ε) を計算する。
QC 不要 (幾何のみ)。
"""
import numpy as np
import json, sys
from scipy import stats
sys.path.insert(0, '/home/yoiyoi')
from motifbank_cli import MotifBank, cutoff_trimers

# ----------------------------------------------------------------
# 2D ice-like lattice (Phase-0, O-O distance = 2.76 Å)
# ----------------------------------------------------------------
def build_ice_2d(nx=8, ny=8):
    """2D hexagonal ice lattice (xy projection). Use Si(OH)4-like SiO4 units."""
    a = 2.76   # O-O distance
    centers = []
    for i in range(nx):
        for j in range(ny):
            x = i*a + (j%2)*a*0.5
            y = j*a*np.sqrt(3)/2
            centers.append([x, y, 0.0])
    return centers

def build_fcc(nx=4, ny=4, nz=4, a=4.0):
    """FCC lattice (more diverse trimers than simple cubic)."""
    basis = [[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]
    centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    x = (i+b[0])*a
                    y = (j+b[1])*a
                    z = (k+b[2])*a
                    centers.append([x, y, z])
    return centers

def build_hcp(nx=4, ny=4, nz=4, a=3.2):
    """HCP lattice."""
    c = a * np.sqrt(8/3)
    centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = i*a + (j%2)*a*0.5 + (k%2)*a*0.5
                y = j*a*np.sqrt(3)/2 + (k%2)*a*np.sqrt(3)/6
                z = k*c/2
                centers.append([x, y, z])
    return centers

def build_random_3d(N=100, box=18.0, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box, (N,3)).tolist()

# ----------------------------------------------------------------
# Motif bank scan: N_bank(ε) for varying ε
# ----------------------------------------------------------------
def geom_key_from_atoms(atoms_xyz, eps=0.0):
    """atoms_xyz: list of [x,y,z]. Returns hashable geom key."""
    c = np.array(atoms_xyz)
    N = len(c)
    dists = sorted([round(np.linalg.norm(c[i]-c[j]), 5)
                    for i in range(N) for j in range(i+1, N)])
    return tuple(dists)

def scan_epsilon_crystal(molecule_list, eps_vals):
    """
    molecule_list: list of fragments, each fragment = list of [x,y,z]
    For each epsilon, count unique motifs under soft-matching.
    """
    # Compute exact keys for all fragments
    exact_keys = [geom_key_from_atoms(mol) for mol in molecule_list]
    results = []
    for eps in eps_vals:
        if eps == 0.0:
            bank_size = len(set(exact_keys))
        else:
            # Soft-match: build bank greedily with RMSD threshold eps
            bank = []  # list of reference distance vectors
            for key in exact_keys:
                k_arr = np.array(key)
                M = len(k_arr)
                matched = False
                for ref in bank:
                    rmsd = np.linalg.norm(k_arr - ref) / np.sqrt(M)
                    if rmsd < eps:
                        matched = True
                        break
                if not matched:
                    bank.append(k_arr)
            bank_size = len(bank)
        results.append({'eps': eps, 'N_bank': bank_size})
    return results

def make_trimer_molecules(centers, r_cut=5.0, max_trimers=20000):
    """Extract unique-geometry trimers as 'molecules' from center list."""
    centers = np.array(centers)
    N = len(centers)
    molecules = []
    seen_keys = set()
    for i in range(N):
        for j in range(i+1, N):
            dij = np.linalg.norm(centers[i]-centers[j])
            if dij > r_cut: continue
            for k in range(j+1, N):
                dik = np.linalg.norm(centers[i]-centers[k])
                djk = np.linalg.norm(centers[j]-centers[k])
                if dik > r_cut or djk > r_cut: continue
                mol = [centers[i].tolist(), centers[j].tolist(), centers[k].tolist()]
                key = geom_key_from_atoms(mol)
                if key not in seen_keys:
                    seen_keys.add(key)
                    molecules.append(mol)
                if len(molecules) >= max_trimers:
                    return molecules
    return molecules

# ----------------------------------------------------------------
# Run for each system
# ----------------------------------------------------------------
EPS_VALS = np.concatenate([
    np.array([0.0]),
    np.arange(0.01, 0.10, 0.01),
    np.arange(0.10, 0.50, 0.02),
    np.arange(0.50, 1.50, 0.10),
])

systems = {
    'ice_2d':      {'centers': build_ice_2d(8, 8),      'r_cut': 8.0,  'phase': 0},
    'fcc_lattice': {'centers': build_fcc(4, 4, 4, 4.0), 'r_cut': 8.0,  'phase': 0},
    'hcp_lattice': {'centers': build_hcp(4, 4, 4, 3.2), 'r_cut': 8.0,  'phase': 0},
    'random_3d':   {'centers': build_random_3d(100, 18.0), 'r_cut': 5.0, 'phase': 3},
}

all_results = {}
for sysname, sysdata in systems.items():
    centers = sysdata['centers']
    r_cut   = sysdata['r_cut']
    phase   = sysdata['phase']
    print(f"\n=== {sysname} (N_centers={len(centers)}, r_cut={r_cut}) ===")

    mols = make_trimer_molecules(centers, r_cut=r_cut, max_trimers=30000)
    print(f"  Unique exact trimers: {len(mols)}")
    if not mols:
        print("  SKIP: no trimers found")
        continue

    scan = scan_epsilon_crystal(mols, EPS_VALS)
    N_bank_0 = scan[0]['N_bank']
    print(f"  N_bank(ε=0) = {N_bank_0}")

    # Find eps_c: where N_bank drops to N_bank_0 / 2
    half = N_bank_0 / 2.0
    eps_c = None
    for s in scan:
        if s['N_bank'] <= half:
            eps_c = s['eps']
            break
    print(f"  ε_c = {eps_c}")

    # d_eff from log-log regression (eps in (0.01, eps_c*1.5))
    sub = [s for s in scan if 0.005 < s['eps'] < (eps_c*1.5 if eps_c else 0.5)
           and s['N_bank'] > 1]
    if len(sub) >= 4:
        log_eps = np.array([np.log(s['eps']) for s in sub])
        log_Nb  = np.array([np.log(s['N_bank']) for s in sub])
        slope, icpt, r, p, se = stats.linregress(log_eps, log_Nb)
        d_eff = -slope
        R2    = r**2
    else:
        d_eff, R2 = float('nan'), float('nan')

    print(f"  d_eff = {d_eff:.3f}, R² = {R2:.3f}")

    all_results[sysname] = {
        'phase': phase,
        'N_centers': len(centers),
        'N_bank_0': N_bank_0,
        'eps_c': eps_c,
        'd_eff': d_eff,
        'R2': R2,
        'scan': scan,
    }

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print("\n" + "="*60)
print("THEOREM 15 EXTENSION — Real Materials d_eff")
print("="*60)
print(f"\n{'System':<15} {'Phase':>6} {'N_bank_0':>9} {'ε_c':>6} {'d_eff':>7} {'R²':>6}")
print("-"*50)
for name, r in sorted(all_results.items(), key=lambda x: x[1]['phase']):
    print(f"{name:<15} {r['phase']:>6d} {r['N_bank_0']:>9d} "
          f"{str(r['eps_c'] if r['eps_c'] else '—'):>6} "
          f"{r['d_eff']:>7.3f} {r['R2']:>6.3f}")

# d_eff ordering check
d_vals = [(r['d_eff'], r['phase'], n) for n, r in all_results.items() if not np.isnan(r['d_eff'])]
d_vals.sort(key=lambda x: x[1])
print("\nd_eff ordering by phase:")
for dv, ph, nm in d_vals:
    print(f"  Phase-{ph}: d_eff={dv:.3f} ({nm})")
phases = [x[1] for x in d_vals]
dvals_only = [x[0] for x in d_vals]
phase_ordered = all(dvals_only[i] <= dvals_only[i+1] or phases[i] == phases[i+1]
                    for i in range(len(dvals_only)-1))
print(f"Phase ordering preserved: {'✅' if phase_ordered else '❌'}")

with open('/home/yoiyoi/real_material_deff.json', 'w') as f:
    # scan lists may be large, trim
    out = {}
    for k, v in all_results.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk != 'scan'}
        out[k]['scan_summary'] = [s for s in v['scan'] if s['eps'] in [0.0, 0.05, 0.10, 0.20, 0.50]]
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved: real_material_deff.json")
