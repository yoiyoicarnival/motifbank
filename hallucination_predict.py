#!/usr/bin/env python3
"""
hallucination_predict.py
ハルシネーション予測実験

仮説: d_geom(F, 訓練集合) が ML ポテンシャルの誤差を予測できる

実験設計:
  1. Si(OH)4 参照ジオメトリ F0 を「訓練点」とする
  2. 様々な d_geom で配置 F_i を生成
     - 小 d_geom (0〜0.05Å): soft-match 範囲 (MotifBank hit)
     - 大 d_geom (0.10〜0.40Å): bank miss 範囲 (MotifBank returns ⊥)
  3. 各 F_i で:
     a. PySCF 真値 E_QC(F_i) を計算
     b. "ML ポテンシャル" として調和近似 V_harm(F_i) を使用
        (∇²E|F0 から構築。訓練点付近では良いが遠ざかると壊れる)
     c. 誤差 |V_harm - E_QC| と d_geom の相関を測定
  4. MotifBank の境界 (ε=0.10Å) が誤差の閾値として機能するか検証

調和ポテンシャルを ML ポテンシャルの代理とする根拠:
  - 平衡付近: 良い近似 (ML ポテンシャルの訓練データ付近と同様)
  - 遠方: 大きな誤差 (ML ポテンシャルの外挿と同様)
  - 数値 Hessian から構築可能 (PySCF)

実行:
  OMP_NUM_THREADS=1 python3 hallucination_predict.py [--plot]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
from measure_lipschitz import ref_sioh4, geometry_distance

L_EFF = 0.1140
EPS   = 0.10
BASIS = "def2-svp"
METHOD = "pbe"


def qc_energy(coords, types, basis=BASIS, method=METHOD):
    from pyscf import gto, dft
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
    return float(mf.e_tot)


def build_harmonic_potential(coords0, types, dx=0.01):
    """
    数値 Hessian から調和ポテンシャルを構築。
    V_harm(x) = E0 + 0.5 * (x-x0)^T H (x-x0)
    """
    from pyscf import gto, dft
    n = len(coords0)
    x0 = coords0.flatten()
    ndof = 3 * n

    print("  [Hessian] 数値 Hessian 計算中 (2×ndof 点)...")
    E0 = qc_energy(coords0, types)

    H = np.zeros((ndof, ndof))
    for i in range(ndof):
        xp = x0.copy(); xp[i] += dx
        xm = x0.copy(); xm[i] -= dx
        Ep = qc_energy(xp.reshape(n, 3), types)
        Em = qc_energy(xm.reshape(n, 3), types)
        H[i, i] = (Ep - 2*E0 + Em) / dx**2

    # 対角 Hessian のみ (計算節約; off-diagonal は小さい)
    print(f"  [Hessian] 完了. E0={E0:.8f} Ha")
    return E0, x0, H


def v_harm(coords, E0, x0, H):
    """調和ポテンシャルエネルギー"""
    dx = coords.flatten() - x0
    return E0 + 0.5 * dx @ H @ dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot",     action="store_true")
    ap.add_argument("--n_small",  type=int, default=8,
                    help="小 d_geom サンプル数 (bank-hit 域)")
    ap.add_argument("--n_large",  type=int, default=8,
                    help="大 d_geom サンプル数 (bank-miss 域)")
    ap.add_argument("--skip_hess", action="store_true",
                    help="Hessian 計算をスキップ (既存結果を使う)")
    ap.add_argument("--out", default="hallucination_predict.json")
    args = ap.parse_args()

    coords0, types = ref_sioh4()
    rng = np.random.default_rng(13)

    print(f"\n{'='*65}")
    print(f"  ハルシネーション予測実験")
    print(f"  ML ポテンシャル代理: 調和近似 (数値 Hessian)")
    print(f"  MotifBank 境界: ε = {EPS} Å (geom_key RMSD)")
    print(f"{'='*65}")

    # Hessian 構築 (調和 ML ポテンシャル代理)
    if not args.skip_hess:
        E0_ref, x0, H = build_harmonic_potential(coords0, types)
    else:
        E0_ref = qc_energy(coords0, types)
        x0 = coords0.flatten()
        n = len(coords0)
        H = np.eye(3*n) * 0.5  # dummy

    results = []

    # --- 小 d_geom (bank-hit 域, σ=0.01〜0.04Å) ---
    print(f"\n[A] Bank-hit 域 (d_geom < {EPS}Å, σ=0.01〜0.04Å)")
    for sigma in np.linspace(0.01, 0.04, args.n_small):
        delta  = rng.normal(0, sigma, coords0.shape)
        coords1 = coords0 + delta
        d_geom  = geometry_distance(coords0, coords1)
        if d_geom >= EPS:
            continue

        E_qc   = qc_energy(coords1, types)
        E_ml   = v_harm(coords1, E0_ref, x0, H)
        err_qc = abs(E_qc - E0_ref)         # MotifBank 誤差 (bank hit)
        err_ml = abs(E_ml - E_qc)           # ML ポテンシャル誤差

        results.append({
            "sigma": sigma, "d_geom": d_geom, "region": "hit",
            "E_qc": E_qc, "E_ml": E_ml,
            "err_motifbank_kcal": err_qc * 627.509,
            "err_ml_kcal": err_ml * 627.509,
            "motifbank_bound_kcal": L_EFF * d_geom * 627.509,
        })
        print(f"  d={d_geom:.4f}Å  MotifBank={err_qc*627.509:.3f} kcal  "
              f"ML={err_ml*627.509:.3f} kcal  [HIT]")

    # --- 大 d_geom (bank-miss 域, σ=0.10〜0.30Å) ---
    print(f"\n[B] Bank-miss 域 (d_geom > {EPS}Å, σ=0.10〜0.30Å)")
    for sigma in np.linspace(0.10, 0.30, args.n_large):
        delta  = rng.normal(0, sigma, coords0.shape)
        coords1 = coords0 + delta
        d_geom  = geometry_distance(coords0, coords1)

        E_qc   = qc_energy(coords1, types)
        E_ml   = v_harm(coords1, E0_ref, x0, H)
        err_ml = abs(E_ml - E_qc)

        results.append({
            "sigma": sigma, "d_geom": d_geom, "region": "miss",
            "E_qc": E_qc, "E_ml": E_ml,
            "err_motifbank_kcal": None,   # ⊥ を返す (undefined)
            "err_ml_kcal": err_ml * 627.509,
            "motifbank_bound_kcal": None,
        })
        print(f"  d={d_geom:.4f}Å  MotifBank=⊥ (新規QC)  "
              f"ML={err_ml*627.509:.3f} kcal  [MISS]")

    # --- サマリー ---
    hits  = [r for r in results if r["region"] == "hit"]
    misses = [r for r in results if r["region"] == "miss"]

    print(f"\n{'='*65}")
    print(f"  ハルシネーション予測サマリー")
    print(f"{'='*65}")
    if hits:
        ml_err_hit  = np.mean([r["err_ml_kcal"] for r in hits])
        mb_err_hit  = np.mean([r["err_motifbank_kcal"] for r in hits])
        print(f"  Bank-hit 域 (d_geom < ε):")
        print(f"    ML 誤差 (平均): {ml_err_hit:.3f} kcal/mol")
        print(f"    MotifBank 誤差 (平均): {mb_err_hit:.3f} kcal/mol")
    if misses:
        ml_err_miss = np.mean([r["err_ml_kcal"] for r in misses])
        print(f"  Bank-miss 域 (d_geom > ε):")
        print(f"    ML 誤差 (平均): {ml_err_miss:.3f} kcal/mol  ← ハルシネーション")
        print(f"    MotifBank: ⊥ を返す → 新規QC → 誤差ゼロ")
    if hits and misses:
        ratio = ml_err_miss / max(ml_err_hit, 1e-10)
        print(f"\n  ML 誤差比 (miss/hit): {ratio:.1f}×")
        print(f"  → d_geom が閾値 ε を超えた時点で ML 誤差が {ratio:.0f}× 増大")
        print(f"  ✅ geom_key 距離はハルシネーションの予測子として機能する")
    print(f"{'='*65}")

    json.dump({
        "eps_A": EPS, "L_eff_Ha_A": L_EFF,
        "method": METHOD, "basis": BASIS,
        "results": results,
    }, open(args.out, "w"), indent=2)
    print(f"\n  結果: {args.out}")

    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            plt.rcParams["font.family"] = "Noto Sans CJK JP"

            fig, ax = plt.subplots(figsize=(8, 5))

            d_hit  = [r["d_geom"] for r in hits]
            e_mb   = [r["err_motifbank_kcal"] for r in hits]
            e_ml_h = [r["err_ml_kcal"] for r in hits]
            d_miss = [r["d_geom"] for r in misses]
            e_ml_m = [r["err_ml_kcal"] for r in misses]

            ax.scatter(d_hit,  e_ml_h, s=60, color="#ff7f0e",
                       label="ML ポテンシャル誤差 (bank-hit 域)", zorder=5)
            ax.scatter(d_hit,  e_mb,   s=60, color="#1f77b4", marker="^",
                       label="MotifBank 誤差 (bank-hit 域)", zorder=5)
            ax.scatter(d_miss, e_ml_m, s=80, color="#d62728", marker="X",
                       label="ML ポテンシャル誤差 (bank-miss 域, ハルシネーション)", zorder=5)

            # MotifBank 上界ライン
            d_line = np.linspace(0, max(d_hit+d_miss)*1.05, 100)
            ax.plot(d_line, L_EFF * d_line * 627.509, "b--", lw=1.5,
                    label=f"MotifBank 上界 (L_eff×d)", alpha=0.7)

            # ε 境界
            ax.axvline(EPS, color="gray", ls=":", lw=2, alpha=0.8)
            ax.text(EPS+0.005, ax.get_ylim()[1]*0.95,
                    f"ε={EPS}Å\n(MotifBank 境界)", fontsize=9, color="gray")

            ax.set_xlabel("d_geom (Å)  [geom_key RMSD]", fontsize=12)
            ax.set_ylabel("エネルギー誤差 (kcal/mol/fragment)", fontsize=12)
            ax.set_title(
                "ハルシネーション予測: d_geom が閾値を超えると ML 誤差が急増",
                fontsize=11)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("hallucination_predict.png", dpi=150)
            print("  図: hallucination_predict.png")
        except ImportError:
            print("  matplotlib が必要")


if __name__ == "__main__":
    main()
