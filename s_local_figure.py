#!/usr/bin/env python3
"""
s_local_figure.py
S_local = log(N_bank_sat) 比較図 (材料別)

実測済み:
  ice Ih (ordered)    : N_bank=16,   S_local=2.77 nats
  alpha-cristobalite  : N_bank~400,  S_local~6.0  nats (収束中)
  MFI silicalite-1    : N_bank=644,  S_local=6.47 nats

使い方:
  python3 s_local_figure.py          (テキスト出力)
  python3 s_local_figure.py --plot   (PNG 出力)
"""
import os, sys, argparse
import numpy as np

DATA = [
    # (label,          N_bank_sat, measured, phase, note)
    ("ice Ih\n(ordered)",        16,   True,  0, "H2O, 3x3 CIF, um=1, sat at 8x"),
    ("α-cristobalite\n(SiO2)",   18,   True,  0, "si_oh4, um=1, sat at 4x (N=16)"),
    ("LTA\nzeolite",             66,   True,  0, "si_oh4, um=1, sat at 4x (N=96)"),
    ("MFI\nsilicalite-1",       282,   True,  0, "si_oh4, um=1, sat at 1x1x1 (N=96)"),
    ("defect MFI\n(M=12 types)",1800,  False, 1, "sparse 5% defect, γ=0.18"),
    ("amorphous\nSi(OH)4",      None,  False, 3, "N→∞, γ=1.56"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print("\nS_local = log(N_bank_sat): 材料別ローカル幾何エントロピー")
    print("=" * 62)
    print(f"  {'材料':25s}  {'N_bank':>8}  {'S_local':>8}  {'Phase':>5}  {'備考'}")
    print("  " + "-" * 56)
    for label, nb, measured, phase, note in DATA:
        lbl = label.replace("\n", " ")
        if nb is None:
            s = "→ ∞"
            nb_str = "→ ∞"
        else:
            s = f"{np.log(nb):.2f} nats"
            nb_str = str(nb)
        mark = "✅" if measured else "~"
        print(f"  {lbl:25s}  {nb_str:>8}  {s:>10}  {phase:>5}  {mark} {note}")

    print()
    print("  意味: S_local が小さいほど MotifBank の speedup が大きい")
    print("        S_local = log(N) → Phase 3 (amorphous, speedup → 1)")
    print("        S_local = 有限  → Phase 0 (crystal,   speedup → N/N_bank)")

    if not args.plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        labels  = [d[0] for d in DATA if d[1] is not None]
        vals    = [np.log(d[1]) for d in DATA if d[1] is not None]
        phases  = [d[3] for d in DATA if d[1] is not None]
        measured= [d[2] for d in DATA if d[1] is not None]

        colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
        bar_colors = [colors[p] for p in phases]
        hatches    = ['' if m else '///' for m in measured]

        bars = ax.bar(range(len(labels)), vals,
                      color=bar_colors, hatch=None, edgecolor='black', linewidth=0.8)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # 値ラベル
        for i, (v, m) in enumerate(zip(vals, measured)):
            suffix = "" if m else "~"
            ax.text(i, v + 0.05, f"{suffix}{v:.2f}", ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("$S_{\\rm local}$ = log($N_{\\rm bank,sat}$)  [nats]", fontsize=12)
        ax.set_title("MotifBank: Local Geometry Entropy $S_{\\rm local}$ by Material",
                     fontsize=11)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.axhline(np.log(644), color='#1f77b4', lw=0.8, ls='--', alpha=0.4)

        # 凡例 (Phase)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], label='Phase 0 (saturates)'),
            Patch(facecolor=colors[1], label='Phase 1 (sub-linear)'),
            Patch(facecolor=colors[3], label='Phase 3 (linear)'),
            Patch(facecolor='white', edgecolor='black', hatch='///',
                  label='estimated (~)'),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), "s_local_figure.png")
        plt.savefig(out, dpi=150)
        print(f"\n  図を保存: {out}")

    except ImportError:
        print("  matplotlib が必要 (pip install matplotlib)")


if __name__ == "__main__":
    main()
