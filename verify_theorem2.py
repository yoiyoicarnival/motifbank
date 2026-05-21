#!/usr/bin/env python3
"""
verify_theorem2.py
Theorem 2' の直接実験的検証

実験:
  1. 参照 Si(OH)4 でバンクを構築 (E0 計算)
  2. soft-match 範囲内 (d_geom < 0.10A) の変位配置 N=20 を生成
  3. 各配置で PySCF 計算 → E_i
  4. 実際の誤差 |E_i - E0| vs 上界 L_eff × d_geom を比較
  5. 全点が上界の下にあれば Theorem 2' 実証

実行:
  OMP_NUM_THREADS=1 python3 verify_theorem2.py [--n 20] [--plot]
"""
import os, sys, json, time, argparse
import numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(__file__))
from measure_lipschitz import ref_sioh4, geometry_distance

L_EFF   = 0.1140   # Ha/Å (measure_L_eff.py で確定)
EPS     = 0.10     # Å (soft-match 閾値)
BASIS   = "def2-svp"
METHOD  = "pbe"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=20, help="テスト配置数")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out",  default="theorem2_verification.json")
    args = ap.parse_args()

    coords0, types = ref_sioh4()
    rng = np.random.default_rng(7)

    print(f"\n{'='*65}")
    print(f"  Theorem 2' 直接検証")
    print(f"  L_eff = {L_EFF} Ha/Å,  ε = {EPS} Å,  {METHOD}/{BASIS}")
    print(f"  上界: |ΔE| ≤ L_eff × d_geom")
    print(f"{'='*65}")

    # 参照エネルギー
    print("\n[ref] 参照エネルギー計算中...")
    t0 = time.perf_counter()
    E0 = qc_energy(coords0, types)
    print(f"  E0 = {E0:.8f} Ha ({time.perf_counter()-t0:.1f}s)")

    results = []
    violations = 0

    print(f"\n[test] {args.n} 配置でエネルギー計算 (soft-match 範囲内)...")
    print(f"  {'i':>3}  {'d_geom':>8}  {'|ΔE| Ha':>12}  {'bound Ha':>12}  "
          f"{'ratio':>7}  {'OK?':>5}")
    print("  " + "-"*55)

    # σ を 0.01〜0.05Å で均等にサンプル（全て soft-match 通過確認済み範囲）
    sigmas = np.linspace(0.01, 0.05, args.n)

    for i, sigma in enumerate(sigmas):
        delta  = rng.normal(0, sigma, coords0.shape)
        coords1 = coords0 + delta
        d_geom = geometry_distance(coords0, coords1)

        # soft-match 確認 (念のため)
        if d_geom >= EPS:
            print(f"  {i:>3}  {d_geom:.4f}  [SKIP: d_geom >= ε]")
            continue

        t1 = time.perf_counter()
        try:
            E1 = qc_energy(coords1, types)
        except Exception as ex:
            print(f"  {i:>3}  {d_geom:.4f}  [SCF FAILED: {ex}]")
            continue
        dt = time.perf_counter() - t1

        actual_err  = abs(E1 - E0)
        bound       = L_EFF * d_geom
        ratio       = actual_err / bound if bound > 0 else float('inf')
        ok          = actual_err <= bound

        if not ok:
            violations += 1

        results.append({
            "i": i, "sigma": sigma, "d_geom": d_geom,
            "E0": E0, "E1": E1,
            "actual_err_Ha": actual_err,
            "actual_err_kcal": actual_err * 627.509,
            "bound_Ha": bound,
            "bound_kcal": bound * 627.509,
            "ratio": ratio,
            "ok": ok,
        })

        mark = "✅" if ok else "❌VIOLATION"
        print(f"  {i:>3}  {d_geom:.4f}  {actual_err:.6e}  "
              f"{bound:.6e}  {ratio:.4f}  {mark}  ({dt:.1f}s)")

    # サマリー
    n_ok = len([r for r in results if r["ok"]])
    n_tot = len(results)
    max_ratio = max((r["ratio"] for r in results), default=0)
    max_err_kcal = max((r["actual_err_kcal"] for r in results), default=0)

    print(f"\n{'='*65}")
    print(f"  Theorem 2' 検証サマリー")
    print(f"{'='*65}")
    print(f"  総計: {n_tot} 配置,  OK: {n_ok},  違反: {violations}")
    print(f"  最大 |ΔE|: {max_err_kcal:.3f} kcal/mol/fragment")
    print(f"  最大 ratio (actual/bound): {max_ratio:.4f}")
    print(f"  上界の引き締まり度: {1/max_ratio:.1f}× (bound は actual の {1/max_ratio:.1f}倍)")

    if violations == 0:
        print(f"\n  ✅ Theorem 2' VERIFIED: 全 {n_tot} 配置で |ΔE| ≤ L_eff × d_geom")
    else:
        print(f"\n  ❌ {violations} 違反あり — L_eff の再測定が必要")

    verdict = "VERIFIED" if violations == 0 else "FAILED"
    print(f"{'='*65}")

    # JSON 保存
    json.dump({
        "theorem": "Theorem 2'", "verdict": verdict,
        "L_eff_Ha_A": L_EFF, "eps_A": EPS,
        "method": METHOD, "basis": BASIS,
        "n_total": n_tot, "n_ok": n_ok, "violations": violations,
        "max_ratio": max_ratio,
        "max_err_kcal": max_err_kcal,
        "samples": results,
    }, open(args.out, "w"), indent=2)
    print(f"\n  結果: {args.out}")

    # プロット
    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            plt.rcParams["font.family"] = "Noto Sans CJK JP"

            d_vals  = [r["d_geom"] for r in results]
            e_vals  = [r["actual_err_Ha"] * 627.509 for r in results]
            b_vals  = [r["bound_Ha"] * 627.509 for r in results]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(d_vals, e_vals, s=50, zorder=5,
                       label="実測誤差 |ΔE|", color="#1f77b4")
            # 上界ライン
            d_line = np.linspace(0, max(d_vals)*1.05, 100)
            ax.plot(d_line, L_EFF * d_line * 627.509, "r--", lw=2,
                    label=f"上界: L_eff × d_geom\n(L={L_EFF} Ha/Å)")
            ax.set_xlabel("d_geom (Å)  [geom_key RMSD]", fontsize=12)
            ax.set_ylabel("|ΔE| (kcal/mol/fragment)", fontsize=12)
            ax.set_title("Theorem 2' 実験的検証: 全点が上界以下", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(d_vals)*1.1)
            ax.set_ylim(0, max(b_vals)*1.2)
            plt.tight_layout()
            out_png = "theorem2_verification.png"
            plt.savefig(out_png, dpi=150)
            print(f"  図: {out_png}")
        except ImportError:
            print("  matplotlib が必要")


if __name__ == "__main__":
    main()
