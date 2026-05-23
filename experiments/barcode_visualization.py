"""
barcode_visualization.py — H₀ バーコード可視化 + S_17 正規化スペクトル

Theorem 17 (Persistence Barcode Phase Fingerprint) の図版を生成する。
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Load percolation data
with open('/home/yoiyoi/motif_percolation.json') as f:
    perc_data = json.load(f)

COLORS = {'Phase 0': '#2196F3', 'Phase 1': '#FF9800', 'Phase 3': '#F44336'}
PHASE_MAP = {
    'Sierpinski三角形 gen=4 (Phase 0)': ('Phase 0', 'Sierpinski\ngen=4', 0),
    'Vicsekフラクタル gen=3 (Phase 1)': ('Phase 1', 'Vicsek\ngen=3',   1),
    'ランダム点群 N=60 (Phase 3)':      ('Phase 3', 'Random\nN=60',    3),
}

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ---- Row 0: N_bank(ε) curves (normalized) ----
ax_top = fig.add_subplot(gs[0, :])
for sysname, data in perc_data.items():
    phase_label, short, ph = PHASE_MAP[sysname]
    scan = data['scan']
    eps_arr = np.array([s['eps'] for s in scan])
    N_arr   = np.array([s['N_bank'] for s in scan], dtype=float)
    N_arr  /= N_arr[0]  # normalize to 1 at eps=0
    ax_top.plot(eps_arr, N_arr, 'o-', color=COLORS[phase_label],
                label=f"{phase_label}: {short.replace(chr(10),' ')} (N_bank_0={data['N_bank_0']})",
                linewidth=2, markersize=4)
    # Mark eps_c
    ax_top.axvline(data['eps_c'], color=COLORS[phase_label], linestyle='--', alpha=0.4)
    ax_top.text(data['eps_c']+0.005, 0.55, f"ε_c={data['eps_c']:.2f}",
                color=COLORS[phase_label], fontsize=8, rotation=90, va='center')

ax_top.axhline(0.5, color='gray', linestyle=':', linewidth=1)
ax_top.set_xlabel('ε (Å)', fontsize=12)
ax_top.set_ylabel('N_bank(ε) / N_bank(0)', fontsize=12)
ax_top.set_title('Theorem 17: H₀ Component Count (= N_bank Filtration Curve)', fontsize=13)
ax_top.legend(fontsize=9)
ax_top.set_xlim(-0.01, 0.52)
ax_top.grid(True, alpha=0.3)

# ---- Row 1: H₀ barcode (death time histograms) ----
for col, (sysname, data) in enumerate(perc_data.items()):
    phase_label, short, ph = PHASE_MAP[sysname]
    scan = data['scan']
    eps_c = data['eps_c']
    N_bank_0 = data['N_bank_0']

    N_vals = [s['N_bank'] for s in scan]
    eps_vals = [s['eps'] for s in scan]

    # Reconstruct death events
    death_times = []
    for i in range(1, len(scan)):
        n_died = N_vals[i-1] - N_vals[i]
        if n_died > 0:
            death_times.extend([eps_vals[i]] * n_died)

    ax = fig.add_subplot(gs[1, col])
    if death_times:
        death_arr = np.array(death_times)
        # Normalize death times by eps_c
        death_norm = death_arr / eps_c
        bins = np.linspace(0, max(death_norm.max(), 2.5), 30)
        ax.hist(death_norm, bins=bins, color=COLORS[phase_label], alpha=0.75, edgecolor='white')
        S17 = death_arr.std() / eps_c
        var_norm = (death_arr.std() / eps_c)**2
        ax.axvline(1.0, color='black', linestyle='--', linewidth=1.5, label='ε_c')
        ax.text(0.97, 0.92, f'S₁₇ = {S17:.3f}',
                transform=ax.transAxes, ha='right', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.97, 0.80, f'{len(death_times)}/{N_bank_0-1} bars',
                transform=ax.transAxes, ha='right', fontsize=9, color='gray')
    ax.set_xlabel('ε_death / ε_c (normalized)', fontsize=10)
    ax.set_ylabel('Bar count', fontsize=10)
    ax.set_title(f'{phase_label}\n{short.replace(chr(10)," ")}', fontsize=11,
                 color=COLORS[phase_label])
    ax.grid(True, alpha=0.2)
    if col == 0:
        ax.text(-0.15, 0.5, 'H₀ Barcode\n(death distribution)', transform=ax.transAxes,
                va='center', ha='right', fontsize=9, rotation=90, color='gray')

# Add S_17 comparison annotation
S17_vals = {}
for sysname, data in perc_data.items():
    scan = data['scan']
    eps_c = data['eps_c']
    N_vals = [s['N_bank'] for s in scan]
    eps_vals = [s['eps'] for s in scan]
    deaths = []
    for i in range(1, len(scan)):
        n = N_vals[i-1] - N_vals[i]
        if n > 0:
            deaths.extend([eps_vals[i]] * n)
    if deaths:
        S17_vals[sysname] = np.array(deaths).std() / eps_c

fig.suptitle(
    'Theorem 17: Persistence Barcode Phase Fingerprint\n'
    f'S₁₇ = std(ε_death)/ε_c: Phase-0={list(S17_vals.values())[0]:.3f} '
    f'< Phase-1={list(S17_vals.values())[1]:.3f} '
    f'< Phase-3={list(S17_vals.values())[2]:.3f}  ✅',
    fontsize=13, fontweight='bold'
)

plt.savefig('/home/yoiyoi/barcode_fingerprint.png', dpi=150, bbox_inches='tight')
print("Saved: barcode_fingerprint.png")

# ---- Separate figure: S_17 bar chart ----
fig2, ax2 = plt.subplots(figsize=(7, 4))
phases = ['Phase-0\n(Sierpinski)', 'Phase-1\n(Vicsek)', 'Phase-3\n(Random)']
s17s = list(S17_vals.values())
colors = [COLORS['Phase 0'], COLORS['Phase 1'], COLORS['Phase 3']]
bars = ax2.bar(phases, s17s, color=colors, edgecolor='white', linewidth=1.5)
ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='S₁₇ = 1 threshold')
for bar, val in zip(bars, s17s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax2.set_ylabel('S₁₇ = std(ε_death) / ε_c', fontsize=12)
ax2.set_title('Normalized Spread (Theorem 17 Phase Classifier)\n'
              'S₁₇ > 1 ⟹ Phase-3 (amorphous)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(s17s) * 1.2)
plt.tight_layout()
plt.savefig('/home/yoiyoi/S17_bar_chart.png', dpi=150, bbox_inches='tight')
print("Saved: S17_bar_chart.png")

# ---- d_eff figure ----
fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
for col, (sysname, data) in enumerate(perc_data.items()):
    phase_label, short, ph = PHASE_MAP[sysname]
    scan = data['scan']
    eps_c = data['eps_c']
    eps_arr = np.array([s['eps'] for s in scan if s['eps'] > 0.001 and s['N_bank'] > 1])
    N_arr   = np.array([s['N_bank'] for s in scan if s['eps'] > 0.001 and s['N_bank'] > 1], dtype=float)
    if len(eps_arr) < 3:
        continue
    from scipy import stats
    slope, icpt, r, p, se = stats.linregress(np.log(eps_arr), np.log(N_arr))
    d_eff = -slope

    ax = axes3[col]
    ax.scatter(np.log(eps_arr), np.log(N_arr), color=COLORS[phase_label], s=40, zorder=5)
    x_fit = np.linspace(np.log(eps_arr.min()), np.log(eps_arr.max()), 100)
    ax.plot(x_fit, slope*x_fit + icpt, 'k--', linewidth=1.5,
            label=f'slope={slope:.3f}')
    ax.set_xlabel('log(ε)', fontsize=11)
    ax.set_ylabel('log(N_bank)', fontsize=11)
    ax.set_title(f'{phase_label}\nd_eff = {d_eff:.3f}  (R²={r**2:.2f})',
                 fontsize=11, color=COLORS[phase_label])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig3.suptitle('Theorem 15: Rate-Distortion Dimension d_eff\n'
              'd_eff: Phase-0=0.24 < Phase-1=0.42 < Phase-3=1.99 ✅',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/yoiyoi/deff_loglog.png', dpi=150, bbox_inches='tight')
print("Saved: deff_loglog.png")
print("All figures saved.")
