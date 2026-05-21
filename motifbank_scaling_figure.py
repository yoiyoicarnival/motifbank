#!/usr/bin/env python3
"""
motifbank_scaling_figure.py
N_bank(N) スケーリング比較: crystal / defect / amorphous

3種類の系での bank サイズ成長を比較し、
Phase 0/1/3 の違いを実測で示す。

使い方:
  OMP_NUM_THREADS=1 python3 motifbank_scaling_figure.py
  OMP_NUM_THREADS=1 python3 motifbank_scaling_figure.py --plot  (PNG 出力)
"""
import os, sys, argparse, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motifbank_cli import from_cif, cutoff_trimers, geom_key, com

CIF   = os.path.join(os.path.dirname(__file__), "examples", "MFI_iza.cif")
R_CUT = 5.5
RNG   = np.random.default_rng(42)


# ── 共通ユーティリティ ────────────────────────────────────────────────────────

def count_unique_bank(mols, r_cut=R_CUT):
    n    = len(mols)
    coms = np.array([com(m) for m in mols])
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if np.linalg.norm(coms[i] - coms[j]) < r_cut]
    trims = cutoff_trimers(mols, r_cut)
    um = len({geom_key([mols[i]])              for i in range(n)})
    up = len({geom_key([mols[i], mols[j]])     for i, j in pairs})
    ut = len({geom_key([mols[i], mols[j], mols[k]]) for i, j, k in trims})
    return n, um + up + ut


def perturb_mols(mols, sigma_ang):
    """各原子に Gaussian ノイズ σ Å を加える (defect 模擬)"""
    return [m + RNG.normal(0, sigma_ang, m.shape) for m in mols]


def random_sio4_mols(n, box=50.0, sigma=0.03):
    """
    n 個のランダム配置 Si(OH)4 (amorphous 模擬)
    Si は box 内にランダム配置、O-H 結合方向はランダム
    sigma: O-H 長さのゆらぎ
    """
    mols = []
    for _ in range(n):
        si = RNG.uniform(0, box, 3)
        # 4 本の O-H 方向: 正四面体ベースに小さいランダム揺らぎ
        tet = np.array([[1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1]], dtype=float)
        tet += RNG.normal(0, 0.3, tet.shape)
        tet /= np.linalg.norm(tet, axis=1, keepdims=True)
        si_o = 1.61 * (1 + RNG.normal(0, sigma))
        o_coords = [si + si_o * d for d in tet]
        h_coords = [oc + 0.96 * d for oc, d in zip(o_coords, tet)]
        mol = np.array([si] + o_coords + h_coords)
        mols.append(mol)
    return mols


# ── 各系のデータ生成 ─────────────────────────────────────────────────────────

def gen_crystal(sizes):
    """Crystal: MFI supercells"""
    results = []
    for sc, label in sizes:
        mols, _, _ = from_cif(CIF, supercell=sc, mol_type="si_oh4", verbose=False)
        n, nb = count_unique_bank(mols)
        results.append((n, nb, label))
    return results


def gen_defect(sizes, defect_rate=0.05, n_defect_types=12, sigma_defect=0.40):
    """
    Defect crystal: crystal + sparse random defects
    defect_rate     : 5% のサイト が欠陥
    n_defect_types  : 欠陥の種類数 (有限の語彙, 例: Al 置換, Si-OH 空孔)
    sigma_defect    : 欠陥サイトの幾何ゆらぎ Å

    モデル: 有限の欠陥語彙 M から選択 → birthday problem → sub-linear 成長
    """
    # 欠陥語彙を生成 (M 種類の摂動プロトタイプ)
    mols_proto, _, _ = from_cif(CIF, supercell=(1,1,1),
                                 mol_type="si_oh4", verbose=False)
    defect_vocab = [perturb_mols([mols_proto[i % len(mols_proto)]], sigma_defect)[0]
                    for i in range(n_defect_types)]

    results = []
    for sc, label in sizes:
        mols_crys, _, _ = from_cif(CIF, supercell=sc, mol_type="si_oh4", verbose=False)
        n = len(mols_crys)
        mols = list(mols_crys)
        # defect_rate の割合のサイトに欠陥を割り当て
        defect_idx = RNG.choice(n, size=max(1, int(n * defect_rate)), replace=False)
        for idx in defect_idx:
            vocab_choice = int(RNG.integers(0, n_defect_types))
            mols[idx] = defect_vocab[vocab_choice]
        _, nb = count_unique_bank(mols)
        results.append((n, nb, label))
    return results


def gen_amorphous(n_vals):
    """Amorphous: fully random Si(OH)4 positions"""
    results = []
    for n in n_vals:
        mols = random_sio4_mols(n)
        _, nb = count_unique_bank(mols)
        results.append((n, nb, f"N={n}"))
    return results


# ── メイン ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="matplotlib PNG を出力")
    args = ap.parse_args()

    sizes_crys = [
        ((1,1,1), "1x1x1"), ((2,1,1), "2x1x1"), ((2,2,1), "2x2x1"),
        ((2,2,2), "2x2x2"), ((4,2,1), "4x2x1"), ((4,2,2), "4x2x2"),
    ]
    n_vals_amor = [24, 48, 96, 192, 384, 768]

    print("N_bank(N) スケーリング計算中...")
    print("  [1/3] crystal  (MFI supercells)...")
    data_crys = gen_crystal(sizes_crys)
    print("  [2/3] defect   (MFI + σ=0.25Å displacement)...")
    data_deft = gen_defect(sizes_crys)
    print("  [3/3] amorphous (random Si(OH)4)...")
    data_amor = gen_amorphous(n_vals_amor)

    # ── テーブル出力 ──────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  N_bank(N) スケーリング比較  (si_oh4, R_cut=5.5Å)")
    print("=" * 65)
    print(f"  {'N_SiO4':>8}  {'crystal':>10}  {'defect':>10}  {'amorphous':>10}")
    print("  " + "-" * 46)

    # N で揃える
    N_crys = [d[0] for d in data_crys]
    for i, n in enumerate(N_crys):
        nb_c = data_crys[i][1]
        nb_d = data_deft[i][1]
        # amorphous は最近傍点
        nb_a = min(data_amor, key=lambda x: abs(x[0] - n))[1]
        print(f"  {n:>8}  {nb_c:>10}  {nb_d:>10}  {nb_a:>10}")

    # ── スケーリング指数の推定 ────────────────────────────────────────────────
    def fit_exponent(data):
        if len(data) < 3:
            return float("nan")
        xs = np.log([d[0] for d in data[-4:]])
        ys = np.log([max(d[1], 1) for d in data[-4:]])
        a, _ = np.polyfit(xs, ys, 1)
        return a

    exp_c = fit_exponent(data_crys)
    exp_d = fit_exponent(data_deft)
    exp_a = fit_exponent(data_amor)

    print()
    print("  スケーリング指数 γ = d log(N_bank) / d log(N):")
    print(f"    crystal   γ = {exp_c:.3f}  ← Phase 0 (γ→0, 飽和)")
    print(f"    defect    γ = {exp_d:.3f}  ← Phase 1 (sub-linear)")
    print(f"    amorphous γ = {exp_a:.3f}  ← Phase 3 (linear)")

    # ── S_local ──────────────────────────────────────────────────────────────
    nb_sat_crys = data_crys[-1][1]
    print()
    print(f"  S_local (crystal, MFI)  = log({nb_sat_crys}) = {np.log(nb_sat_crys):.2f} nats")
    print(f"  wall-clock speedup 期待 (N=768, PBE/def2-SVP ~30s/call):")
    n768_naive = next(d[1] for d in [(d[0], d[0]+d[0]+d[0]) for d in data_crys] if d[0]==768)
    # 実際の QC call 数で計算
    nb_768 = next(d[1] for d in data_crys if d[0]==768)
    # naiveは motifbank_mfi_scaling.pyの実測値を使う
    naive_768 = 14690  # 実測
    t_qc_s = 30.0
    t_naive = naive_768 * t_qc_s / 3600
    t_bank  = nb_sat_crys  * t_qc_s / 3600
    print(f"    naive: {naive_768} calls × 30s = {t_naive:.1f} h")
    print(f"    bank:  {nb_sat_crys} calls × 30s = {t_bank:.1f} h")
    print(f"    speedup: {t_naive/t_bank:.0f}×  (wall-clock)")

    # ── PNG 出力 ──────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            plt.rcParams["font.family"] = "Noto Sans CJK JP"

            fig, ax = plt.subplots(figsize=(7, 5))

            Nc = [d[0] for d in data_crys]
            Nd = [d[0] for d in data_deft]
            Na = [d[0] for d in data_amor]

            ax.plot(Nc, [d[1] for d in data_crys],
                    "o-", color="#1f77b4", lw=2, ms=7, label=f"結晶（MFI, γ={exp_c:.2f}）")
            ax.plot(Nd, [d[1] for d in data_deft],
                    "s--", color="#ff7f0e", lw=2, ms=7, label=f"欠陥結晶（σ=0.25Å, γ={exp_d:.2f}）")
            ax.plot(Na, [d[1] for d in data_amor],
                    "^:", color="#d62728", lw=2, ms=7, label=f"非晶質（γ={exp_a:.2f}）")

            ax.axhline(nb_sat_crys, color="#1f77b4", lw=1, ls=":", alpha=0.5,
                       label=f"Phase 0 飽和 = {nb_sat_crys}")

            ax.set_xlabel("N（Si(OH)₄ フラグメント数）", fontsize=12)
            ax.set_ylabel("N_bank（バンク内ユニーク構造数）", fontsize=12)
            ax.set_title("MotifBank: N_bank(N) スケーリング — 結晶 vs 欠陥 vs 非晶質",
                         fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")
            ax.set_yscale("log")

            # S_local annotation
            ax.annotate(f"S_local = log({nb_sat_crys})\n= {np.log(nb_sat_crys):.1f} nats",
                        xy=(1536, nb_sat_crys), xytext=(200, nb_sat_crys * 1.3),
                        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
                        fontsize=9, color="#1f77b4")

            plt.tight_layout()
            out = os.path.join(os.path.dirname(__file__), "motifbank_scaling_figure.png")
            plt.savefig(out, dpi=150)
            print(f"\n  図を保存: {out}")
        except ImportError:
            print("\n  matplotlib が必要 (pip install matplotlib)")

    print("=" * 65)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
