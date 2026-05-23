"""
conjecture1_analysis.py — Conjecture 1 の検証と γ_geom vs γ_energy の区別

重要発見:
  γ_geom (幾何バンク成長率)  ≠  γ_energy (MBE エネルギー収束率)
  Conjecture 1 は γ_energy について述べており、γ_geom は別の量。
"""
import numpy as np
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Known data (from carpet sessions)
ENERGY_DATA = {
    'Gen1→2': {'gamma_inf': 0.710, 'k': 0.1714, 'R_th': 6.48, 'type': 'energy'},
    'Gen2→3': {'gamma_inf': 0.805, 'k': 0.03755, 'R_th': 21.32, 'type': 'energy'},
}
d_H = 1.893  # Hausdorff dim of Sierpinski carpet H3+

# Geometry-based measurements (new from carpet_gen3_fast.py)
# γ_geom at R=22 Å from Gen2→3 comparison
GEOM_DATA_22 = {
    'Gen2→3_R22': {
        'N_bank_Gen2': 987,
        'N_bank_Gen3': 5404,
        'N_Gen2': 64,
        'N_Gen3': 512,
        'R': 22,
        'gamma_at_R': math.log(5404/987)/math.log(512/64),
    }
}
# γ_geom at R=27 Å
GEOM_DATA_27 = {
    'Gen2→3_R27': {
        'N_bank_Gen2': 987,  # Gen2 saturated
        'N_bank_Gen3': 12078,
        'N_Gen2': 64,
        'N_Gen3': 512,
        'R': 27,
        'gamma_at_R': math.log(12078/987)/math.log(512/64),
    }
}

print("=" * 65)
print("CONJECTURE 1 ANALYSIS — γ_energy vs γ_geom")
print("=" * 65)
print()
print("Conjecture 1: lim_{n→∞} 2γ∞_energy(Gen n→n+1) = d_H = 1.893")
print()
print("γ_energy (MBE convergence from CASSCF, prior sessions):")
print(f"  {'Transition':<12} {'γ∞_energy':>12} {'2γ∞':>8} {'err%':>8}")
for trans, d in ENERGY_DATA.items():
    g = d['gamma_inf']
    err = abs(2*g - d_H)/d_H*100
    print(f"  {trans:<12} {g:>12.3f} {2*g:>8.3f} {err:>7.1f}%")
print(f"  Predicted Gen3→4: γ∞_energy ≈ 0.862, 2γ∞ ≈ 1.724 (err≈8.9%)")

print()
print("γ_geom (geometric bank growth from fast scan, NEW):")
print(f"  {'Transition':<15} {'R (Å)':>6} {'γ_geom(R)':>12} {'2γ_geom':>9}")
for key, d in {**GEOM_DATA_22, **GEOM_DATA_27}.items():
    g = d['gamma_at_R']
    print(f"  Gen2→3           {d['R']:>6.0f} {g:>12.3f} {2*g:>9.3f}")

print()
print("KEY DISTINCTION:")
print("  γ_energy: measures how fast de3 grows vs de2 as R increases")
print("            = 0.71-0.81 (sub-linear, MBE converges, Phase-0)")
print("  γ_geom:   measures N_bank growth between generations at fixed R")
print("            = 1.2-1.9 (super-linear in inter-generation comparison)")
print()
print("These are DIFFERENT quantities!")
print("Conjecture 1 is about γ_energy → d_H/2 = 0.9465")
print("γ_geom → something else (possibly d_H itself?)")

# Conjecture on γ_geom: at saturation, γ_geom → d_H ?
N_bank_sat_Gen1 = 9
N_bank_sat_Gen2 = 987
N_Gen1 = 8
N_Gen2 = 64
gamma_geom_12_sat = math.log(N_bank_sat_Gen2/N_bank_sat_Gen1)/math.log(N_Gen2/N_Gen1)
print()
print(f"γ_geom at saturation (Gen1→2):")
print(f"  log({N_bank_sat_Gen2}/{N_bank_sat_Gen1}) / log({N_Gen2}/{N_Gen1})")
print(f"  = log({N_bank_sat_Gen2/N_bank_sat_Gen1:.1f}) / log({N_Gen2/N_Gen1:.0f})")
print(f"  = {gamma_geom_12_sat:.4f}")
print(f"  2γ_geom_sat(Gen1→2) = {2*gamma_geom_12_sat:.4f}")
print(f"  d_H = {d_H}")
print(f"  γ_geom_sat ≈ {gamma_geom_12_sat:.3f} ≠ d_H/2 = {d_H/2:.4f}")
print(f"  But: γ_geom_sat ≈ {gamma_geom_12_sat:.3f} ≈ d_H - 0.7?")

# Attempt: does γ_geom_sat = log8(N_bank_ratio) converge to d_H?
# From Moran equation for Sierpinski carpet:
# 8 × r^{d_H} = 1, r = 1/3
# N_bank should scale as N^{d_H} for a fractal of dimension d_H?
print()
print("Hypothesis: N_bank_sat scales as N^{d_H} ?")
ratio_sat_12 = N_bank_sat_Gen2 / N_bank_sat_Gen1  # 109.7
print(f"  N_bank_sat ratio Gen1→2: {ratio_sat_12:.2f}")
print(f"  N_Gen ratio Gen1→2: {N_Gen2/N_Gen1:.0f}")
print(f"  Expected if N_bank ~ N^{d_H}: ratio = {N_Gen2/N_Gen1}^{d_H:.3f} = {(N_Gen2/N_Gen1)**d_H:.1f}")
print(f"  Observed: {ratio_sat_12:.1f}  {'≈' if abs(ratio_sat_12 - (N_Gen2/N_Gen1)**d_H) < 20 else '≠'}")
print(f"  Alternative: N_bank ~ N^gamma_geom with gamma_geom = {gamma_geom_12_sat:.3f}")

# Verdict
print()
print("=" * 65)
print("VERDICT")
print("=" * 65)
print()
print("Conjecture 1 (energy-based): NEEDS Gen3→4 QC data")
print("  → Currently running on PC (~500h ETA from carpet_gen2_pc.out)")
print("  → Predicted Gen3→4 γ∞_energy = 0.862")
print()
print("NEW FINDING from geometry-based analysis:")
print(f"  γ_geom_sat(Gen1→2) = {gamma_geom_12_sat:.3f}")
print(f"  This might converge to d_H = {d_H} (not d_H/2) ?")
print(f"  New Conjecture: lim_n γ_geom_sat(Gen n→n+1) → d_H = {d_H}")
print(f"  (needs Gen3 saturation data → N_bank_sat_Gen3 ~ ?)")
print()
print("Status of Conjecture 1:")
print("  Energy version: ❓ (awaits QC, 3.7% → 8.9% → ? per generation)")
print("  Geometric version (new): ❓ (different quantity, different prediction)")

# Figure: convergence trajectory
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: 2γ∞_energy convergence
ax = axes[0]
transitions = list(ENERGY_DATA.keys())
g_vals = [ENERGY_DATA[t]['gamma_inf'] for t in transitions]
two_g = [2*g for g in g_vals]
x_pos = [1, 2]
ax.plot(x_pos, two_g, 'bo-', markersize=10, linewidth=2, label='2γ∞_energy (measured)')
ax.axhline(d_H, color='red', linestyle='--', linewidth=2, label=f'd_H = {d_H}')

# Extrapolate
g3_pred = 0.862
ax.plot([3], [2*g3_pred], 'b^', markersize=12, alpha=0.5, label=f'Gen3→4 pred: 2γ={2*g3_pred:.3f}')
ax.set_xlim(0.5, 3.5)
ax.set_ylim(1.0, 2.2)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['Gen1→2', 'Gen2→3', 'Gen3→4\n(pred)'])
ax.set_ylabel('2γ∞_energy', fontsize=12)
ax.set_title('Conjecture 1 (Energy-based)\n2γ∞_energy → d_H = 1.893', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: γ_geom at various R for Gen2→3
ax2 = axes[1]
# From carpet_gen3_fast.py results
R_vals = [15, 18, 22, 27]
gamma_geom_vals = [0.133, 0.418, 0.818, 1.204]
ax2.plot(R_vals, gamma_geom_vals, 'gs-', markersize=8, linewidth=2, label='γ_geom(Gen2→3, R)')
ax2.axhline(d_H/2, color='red', linestyle='--', linewidth=1.5, label=f'd_H/2 = {d_H/2:.3f}')
ax2.axhline(d_H, color='orange', linestyle='--', linewidth=1.5, label=f'd_H = {d_H}')
ax2.axhline(gamma_geom_12_sat, color='blue', linestyle=':', linewidth=1.5,
            label=f'γ_geom_sat(Gen1→2) = {gamma_geom_12_sat:.3f}')
ax2.set_xlabel('R_cut (Å)', fontsize=12)
ax2.set_ylabel('γ_geom(R) = Δlog(N_bank)/Δlog(N)', fontsize=11)
ax2.set_title('Geometric Bank Growth Rate\n(DIFFERENT from energy γ)', fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle('Conjecture 1: γ_energy ≠ γ_geom\nEnergy version needs QC; Geometric version is a new conjecture',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/yoiyoi/conjecture1_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: conjecture1_analysis.png")

# Save results
out = {
    'energy_gamma': {k: v for k,v in ENERGY_DATA.items()},
    'geom_gamma_R22': {k: v for k,v in GEOM_DATA_22.items()},
    'geom_gamma_R27': {k: v for k,v in GEOM_DATA_27.items()},
    'geom_gamma_sat_Gen12': gamma_geom_12_sat,
    'd_H': d_H,
    'new_conjecture': '2*gamma_geom_sat(Gen n→n+1) → d_H (not d_H/2)',
}
with open('/home/yoiyoi/conjecture1_analysis.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Saved: conjecture1_analysis.json")
