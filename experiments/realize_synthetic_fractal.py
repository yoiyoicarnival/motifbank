#!/usr/bin/env python3
"""
realize_synthetic_fractal.py
合成フラクタルの実現 — MotifBank γ スケーリング vs 理論フラクタル次元

実験:
  1. Sierpinski三角形・Vicsekフラクタル・Sierpinskiカーペットを世代別に生成
  2. 各世代で R_cut 付き N_bank(N) を計算 → γ_emp を実測
  3. 理論フラクタル次元 d_f と比較
  4. MotifBank Phase 0-3 図に配置

フラクタル次元 (理論):
  Sierpinski三角形: d_f = log3/log2 ≈ 1.585
  Vicsekフラクタル: d_f = log5/log3 ≈ 1.465
  Sierpinskiカーペット: d_f = log8/log3 ≈ 1.893

実行:
  python3 realize_synthetic_fractal.py [--plot]
"""
import os, sys, json, argparse, time
import numpy as np
from itertools import combinations
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'motifbank', 'core'))

# ── フラクタル生成関数 ────────────────────────────────────────────────────

def sierpinski_centers(gen, a=1.0):
    """Sierpinski三角形の中心座標 (gen世代目)"""
    if gen == 1:
        return np.array([[0,0],[a,0],[a/2, a*np.sqrt(3)/2]])
    prev = sierpinski_centers(gen-1, a)
    side = a * 2**(gen-2)
    offsets = np.array([[0,0],[side,0],[side/2, side*np.sqrt(3)/2]])
    pts = np.vstack([prev + off for off in offsets])
    # 重複除去
    rounded = np.round(pts, 8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(idx)]


def vicsek_centers(gen, a=1.0):
    """Vicsekフラクタルの中心座標"""
    if gen == 1:
        return np.array([[0,0],[a,0],[-a,0],[0,a],[0,-a]])
    prev = vicsek_centers(gen-1, a)
    D = 3 * a * (3**(gen-2))
    offsets = np.array([[0,0],[D,0],[-D,0],[0,D],[0,-D]])
    pts = np.vstack([prev + off for off in offsets])
    rounded = np.round(pts, 8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(idx)]


def carpet_centers(gen, a=1.0):
    """Sierpinskiカーペットの中心座標 (8近傍再帰)"""
    OFFS8 = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]], float)
    if gen == 1:
        return OFFS8 * a
    prev = carpet_centers(gen-1, a)
    D = 3 * a * (3**(gen-2))
    pts = np.vstack([prev + off * D for off in OFFS8])
    rounded = np.round(pts, 8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(idx)]


def cantor_dust_centers(gen, a=1.0):
    """カントール集合 (1D) → 2D ダスト (直積)"""
    def cantor1d(g, x0=0, x1=1):
        if g == 0:
            return np.array([(x0+x1)/2])
        mid1, mid2 = x0 + (x1-x0)/3, x0 + 2*(x1-x0)/3
        return np.concatenate([cantor1d(g-1, x0, mid1), cantor1d(g-1, mid2, x1)])
    xs = cantor1d(gen) * a * (3**gen)
    coords = np.array([[x, y] for x in xs for y in xs])
    return coords


FRACTALS = {
    'sierpinski': {
        'fn': sierpinski_centers,
        'df': np.log(3)/np.log(2),    # ≈ 1.585
        'label': 'Sierpinski三角形',
        'color': '#1f77b4',
        'max_gen': 6,
        'n_ref': 3,   # gen=1 の N
    },
    'vicsek': {
        'fn': vicsek_centers,
        'df': np.log(5)/np.log(3),    # ≈ 1.465
        'label': 'Vicsekフラクタル',
        'color': '#ff7f0e',
        'max_gen': 4,
        'n_ref': 5,
    },
    'carpet': {
        'fn': carpet_centers,
        'df': np.log(8)/np.log(3),    # ≈ 1.893
        'label': 'Sierpinskiカーペット',
        'color': '#2ca02c',
        'max_gen': 4,
        'n_ref': 8,
    },
    'cantor': {
        'fn': cantor_dust_centers,
        'df': 2*(np.log(2)/np.log(3)), # ≈ 1.261 (2D Cantor dust)
        'label': 'カントールダスト(2D)',
        'color': '#9467bd',
        'max_gen': 4,
        'n_ref': 4,
    },
}


# ── γ 計算 ────────────────────────────────────────────────────────────────

def compute_N_bank(centers, r_cut_factor=2.5, unit_a=1.0, eps=None):
    """
    R_cut = r_cut_factor × unit_a で trimer を列挙し、
    ユニーク geom_key 数 (N_bank) を返す。

    eps=None:  exact matching (ε=0)
    eps>0:     soft matching — RMSD(d3_i, d3_j) < eps を同一クラスとみなす
    """
    N = len(centers)
    r_cut = r_cut_factor * unit_a

    diff = centers[:, None, :] - centers[None, :, :]
    dmat = np.sqrt((diff**2).sum(axis=2))

    # 全有効 trimer の距離ベクトルを収集
    trimer_dvecs = []
    pair_dvecs   = []

    for i in range(N):
        for j in range(i+1, N):
            dij = dmat[i, j]
            if dij > r_cut:
                continue
            pair_dvecs.append(dij)
            for k in range(j+1, N):
                dik = dmat[i, k]
                dkj = dmat[k, j]
                if dik > r_cut or dkj > r_cut:
                    continue
                d3 = sorted([dij, dik, dkj])
                trimer_dvecs.append(np.array(d3))

    if not trimer_dvecs:
        return {'N': N, 'N_bank_pairs': len(set(round(d,6) for d in pair_dvecs)),
                'N_bank_trimers': 1, 'N_trimers_total': N*(N-1)*(N-2)//6}

    if eps is None or eps <= 0:
        # exact matching
        unique_t = set(tuple(round(d,6) for d in v) for v in trimer_dvecs)
        N_bank_t = len(unique_t)
    else:
        # soft matching: greedy クラスタリング (RMSD < eps)
        M = len(trimer_dvecs[0])  # = 3
        eps_scaled = eps * np.sqrt(M)   # RMSD(3次元) → L2 距離に変換
        centroids = []
        for dvec in trimer_dvecs:
            matched = False
            for c in centroids:
                if np.linalg.norm(dvec - c) / np.sqrt(M) < eps:
                    matched = True
                    break
            if not matched:
                centroids.append(dvec)
        N_bank_t = len(centroids)

    return {
        'N': N,
        'N_bank_pairs': len(set(round(d,6) for d in pair_dvecs)),
        'N_bank_trimers': max(N_bank_t, 1),
        'N_trimers_total': N*(N-1)*(N-2)//6,
    }


def analyze_scaling(fractal_name, max_gen=5, r_cut_factor=2.5, eps=None):
    """全世代 N_bank を計算し γ を実測する"""
    cfg = FRACTALS[fractal_name]
    fn = cfg['fn']
    df_theory = cfg['df']

    print(f"\n{'='*60}")
    print(f"  {cfg['label']} (理論次元 d_f = {df_theory:.4f})")
    print(f"{'='*60}")
    print(f"  {'Gen':>4}  {'N':>6}  {'N_bank':>8}  {'reuse':>7}  {'γ_emp':>7}")
    print("  " + "-"*40)

    data = []
    prev_N, prev_Nb = None, None

    for gen in range(1, min(max_gen+1, cfg['max_gen']+1)):
        centers = fn(gen, a=1.0)
        r = compute_N_bank(centers, r_cut_factor=r_cut_factor, unit_a=1.0, eps=eps)
        N   = r['N']
        Nb  = r['N_bank_trimers']
        reuse = r['N_trimers_total'] / Nb if Nb > 0 else 0

        # γ = Δlog(N_bank) / Δlog(N) — 世代間差分
        if prev_N is not None and prev_N > 0 and Nb != prev_Nb and N != prev_N:
            gamma_loc = np.log(Nb / prev_Nb) / np.log(N / prev_N)
        else:
            gamma_loc = float('nan')

        data.append({'gen': gen, 'N': N, 'N_bank': Nb, 'gamma_local': gamma_loc, 'reuse': reuse})
        marker = "" if np.isnan(gamma_loc) else f" γ={gamma_loc:.3f}"
        print(f"  {gen:>4}  {N:>6}  {Nb:>8}  {reuse:>7.1f}×{marker}")
        prev_N, prev_Nb = N, Nb

    # 全世代で log-log 線形フィット
    Ns  = [d['N']  for d in data if d['N'] > 1]
    Nbs = [d['N_bank'] for d in data if d['N'] > 1]
    if len(Ns) >= 2:
        slope, intercept = np.polyfit(np.log(Ns), np.log(Nbs), 1)
    else:
        slope, intercept = float('nan'), float('nan')

    gamma_global = slope
    print(f"\n  全世代 log-log フィット: γ_global = {gamma_global:.4f}")
    print(f"  理論フラクタル次元:       d_f      = {df_theory:.4f}")
    print(f"  γ / d_f = {gamma_global/df_theory:.4f}  (1に近ければ γ ≈ d_f)")

    # Phase 判定
    GAMMA_C = 0.48
    if gamma_global < 0.10:
        phase = "Phase 0 (結晶)"
    elif gamma_global < GAMMA_C:
        phase = "Phase 1 (準周期)"
    elif gamma_global < 0.80:
        phase = "Phase 2 (準非晶質)"
    else:
        phase = "Phase 3 (非晶質)"
    print(f"  MotifBank Phase: {phase}")

    return {
        'fractal': fractal_name,
        'label': cfg['label'],
        'df_theory': df_theory,
        'gamma_global': gamma_global,
        'gamma_intercept': intercept,
        'phase': phase,
        'data': data,
        'r_cut_factor': r_cut_factor,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot',  action='store_true')
    ap.add_argument('--out',   default='synthetic_fractal.json')
    ap.add_argument('--rcut',  type=float, default=2.5, help='R_cut / unit_a')
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  合成フラクタル実現実験")
    print(f"  R_cut = {args.rcut:.1f} × a")
    print(f"{'='*60}")

    results = {}
    for name in FRACTALS:
        r = analyze_scaling(name, max_gen=FRACTALS[name]['max_gen'],
                           r_cut_factor=args.rcut)
        results[name] = r

    # サマリー表
    print(f"\n\n{'='*60}")
    print(f"  合成フラクタル サマリー")
    print(f"{'='*60}")
    print(f"  {'フラクタル':20s}  {'d_f(理論)':>10}  {'γ_emp':>8}  {'γ/d_f':>7}  {'Phase'}")
    print("  " + "-"*65)
    for name, r in results.items():
        gf = r['gamma_global']
        df = r['df_theory']
        ratio = gf/df if df > 0 else float('nan')
        print(f"  {r['label']:20s}  {df:>10.4f}  {gf:>8.4f}  {ratio:>7.4f}  {r['phase']}")

    # 結晶 (γ=0) と非晶質 (γ=1) の基準を追加
    print(f"  {'結晶 (Phase 0)':20s}  {'---':>10}  {'0.000':>8}  {'---':>7}  Phase 0")
    print(f"  {'非晶質 (Phase 3)':20s}  {'---':>10}  {'1.000':>8}  {'---':>7}  Phase 3")

    json.dump(results, open(args.out, 'w'), indent=2, default=str)
    print(f"\n  結果: {args.out}")

    if args.plot:
        _make_plots(results)


def _make_plots(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        try:
            fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
        except Exception:
            pass

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: N_bank(N) log-log ──
        ax = axes[0]
        N_range = np.logspace(0, 3, 100)

        for name, r in results.items():
            cfg = FRACTALS[name]
            Ns  = [d['N']      for d in r['data'] if d['N'] > 1]
            Nbs = [d['N_bank'] for d in r['data'] if d['N'] > 1]
            if Ns:
                ax.scatter(Ns, Nbs, s=80, color=cfg['color'], zorder=5,
                           label=f"{cfg['label']} (γ={r['gamma_global']:.3f})")
                # フィット線
                gf = r['gamma_global']
                ic = r['gamma_intercept']
                ax.plot(N_range, np.exp(ic) * N_range**gf, '--',
                        color=cfg['color'], alpha=0.6, lw=1.5)

        # 参照線
        ax.plot(N_range, N_range**0,    'k:', lw=1, alpha=0.5, label='γ=0 (結晶)')
        ax.plot(N_range, N_range**0.48, 'gray', lw=1, alpha=0.5,
                linestyle='dashdot', label='γ=0.48 (Phase境界)')
        ax.plot(N_range, N_range**1,    'k--', lw=1, alpha=0.5, label='γ=1 (非晶質)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('N (フラグメント数)', fontsize=12)
        ax.set_ylabel('N_bank (ユニーク motif 数)', fontsize=12)
        ax.set_title('合成フラクタル: N_bank スケーリング', fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

        # ── Right: γ vs d_f (Phase 図) ──
        ax2 = axes[1]
        df_range = np.linspace(0, 2.2, 100)

        # γ = d_f の対角線（予想）
        ax2.plot(df_range, df_range, 'k--', lw=1.5, alpha=0.5, label='γ = d_f (仮説)')
        ax2.axhline(0.48, color='gray', ls=':', lw=1.5, alpha=0.7, label='γ=0.48 (Phase境界)')

        for name, r in results.items():
            cfg = FRACTALS[name]
            gf = r['gamma_global']
            df = r['df_theory']
            ax2.scatter([df], [gf], s=120, color=cfg['color'], zorder=5,
                        label=f"{cfg['label']}")
            ax2.annotate(cfg['label'].split('(')[0].strip(),
                        (df, gf), textcoords='offset points',
                        xytext=(8, 4), fontsize=8, color=cfg['color'])

        # 結晶・非晶質の基準点
        ax2.scatter([0], [0],    s=100, color='black', marker='s', zorder=5, label='結晶 (γ=0)')
        ax2.scatter([2], [1.0],  s=100, color='red',   marker='s', zorder=5, label='非晶質 (γ=1)')

        ax2.set_xlabel('理論フラクタル次元 d_f', fontsize=12)
        ax2.set_ylabel('実測 γ_emp (MotifBank)', fontsize=12)
        ax2.set_title('合成フラクタル: γ vs d_f', fontsize=12)
        ax2.legend(fontsize=8, loc='upper left')
        ax2.set_xlim(-0.1, 2.2)
        ax2.set_ylim(-0.1, 1.3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('synthetic_fractal.png', dpi=150)
        print(f"  図: synthetic_fractal.png")

        # ── Fractal visualizations ──
        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
        for ax3, (name, cfg) in zip(axes2, FRACTALS.items()):
            gen = min(4, cfg['max_gen'])
            centers = cfg['fn'](gen, a=1.0)
            ax3.scatter(centers[:,0], centers[:,1], s=8,
                       color=cfg['color'], alpha=0.7)
            ax3.set_title(f"{cfg['label']}\nGen{gen}, N={len(centers)}",
                         fontsize=9)
            ax3.set_aspect('equal')
            ax3.axis('off')
        plt.suptitle('合成フラクタル構造', fontsize=12)
        plt.tight_layout()
        plt.savefig('synthetic_fractal_vis.png', dpi=150)
        print(f"  図: synthetic_fractal_vis.png")

    except ImportError:
        print("  matplotlib が必要")


def analyze_tunable(sigma_noise=0.0, max_gen=5, base='sierpinski', r_cut_factor=2.5, eps=None):
    """
    フラクタルにノイズを加えて γ を連続チューニングする実験。
    sigma_noise=0: 純粋フラクタル (γ_min)
    sigma_noise→∞: 非晶質 (γ→1)
    """
    cfg = FRACTALS[base]
    fn  = cfg['fn']
    rng = np.random.default_rng(42)

    data = []
    prev_N, prev_Nb = None, None

    for gen in range(1, min(max_gen+1, cfg['max_gen']+1)):
        centers = fn(gen, a=1.0)
        if sigma_noise > 0:
            centers = centers + rng.normal(0, sigma_noise, centers.shape)
        r = compute_N_bank(centers, r_cut_factor=r_cut_factor, unit_a=1.0, eps=eps)
        N, Nb = r['N'], r['N_bank_trimers']
        if prev_N is not None and prev_N > 0 and Nb != prev_Nb and N != prev_N:
            gamma_loc = np.log(Nb / prev_Nb) / np.log(N / prev_N)
        else:
            gamma_loc = float('nan')
        data.append({'gen': gen, 'N': N, 'N_bank': Nb, 'gamma_local': gamma_loc})
        prev_N, prev_Nb = N, Nb

    Ns  = [d['N']      for d in data if d['N'] > 1]
    Nbs = [d['N_bank'] for d in data if d['N'] > 1]
    gamma_global = np.polyfit(np.log(Ns), np.log(Nbs), 1)[0] if len(Ns)>=2 else float('nan')
    return gamma_global, data


def tunable_phase_diagram(base='sierpinski', sigmas=None, max_gen=5, eps=None):
    """
    σ_noise を変化させて γ(σ) 曲線を描く。
    eps: soft-matching 閾値 (None=exact, 0.10=MotifBank標準)
    """
    if sigmas is None:
        sigmas = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    eps_label = f"ε={eps:.2f}" if eps else "exact(ε=0)"
    print(f"\n{'='*60}")
    print(f"  チューナブル合成フラクタル: {FRACTALS[base]['label']}")
    print(f"  soft-match: {eps_label}")
    print(f"{'='*60}")
    print(f"  {'σ_noise':>10}  {'γ_emp':>8}  Phase")
    print("  " + "-"*35)

    results = []
    GAMMA_C = 0.48
    for sigma in sigmas:
        gamma, _ = analyze_tunable(sigma_noise=sigma, max_gen=max_gen,
                                   base=base, eps=eps)
        if gamma < 0.10:      ph = "Phase 0"
        elif gamma < GAMMA_C: ph = "Phase 1"
        elif gamma < 0.80:    ph = "Phase 2"
        else:                 ph = "Phase 3"
        print(f"  {sigma:>10.3f}  {gamma:>8.4f}  {ph}")
        results.append({'sigma': sigma, 'gamma': gamma, 'phase': ph, 'eps': eps})

    return results


if __name__ == '__main__':
    main()

    # チューナブルフラクタル実験
    print(f"\n\n{'='*60}")
    print(f"  チューナブル合成フラクタル実験")
    print(f"{'='*60}")
    tunable_results = {}
    sigmas = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
    for base in ['sierpinski']:
        # exact matching vs soft matching (ε=0.10)
        tr_exact = tunable_phase_diagram(base=base, sigmas=sigmas, eps=None, max_gen=5)
        tr_soft  = tunable_phase_diagram(base=base, sigmas=sigmas, eps=0.10, max_gen=5)
        tunable_results[base] = {'exact': tr_exact, 'soft': tr_soft}

    # チューニング図
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        try:
            fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(8, 5))
        for base, trd in tunable_results.items():
            cfg = FRACTALS[base]
            for mode, tr in [('exact (ε=0)', trd['exact']),
                             ('soft  (ε=0.10)', trd['soft'])]:
                sigmas_pl = [t['sigma'] for t in tr]
                gammas_pl = [t['gamma'] for t in tr]
                ls = '-' if 'soft' in mode else '--'
                ax.plot(sigmas_pl, gammas_pl, ls, color=cfg['color'],
                        lw=2, ms=7, marker='o' if 'soft' in mode else 's',
                        label=f"{cfg['label']} {mode}")

        ax.axhline(0.48, color='gray', ls=':', lw=1.5, alpha=0.7,
                   label='γ=0.48 (Phase 1/2 境界)')
        ax.axhline(0,    color='blue',  ls='--', lw=1, alpha=0.4,
                   label='γ=0 (純フラクタル)')
        ax.axhline(1.0,  color='red',   ls='--', lw=1, alpha=0.4,
                   label='γ=1 (完全非晶質)')
        ax.fill_between([0,1.1], [0,0], [0.48,0.48], alpha=0.05, color='blue',
                        label='Phase 0-1 (MotifBank有効域)')
        ax.set_xlabel('σ_noise (位置ノイズ / 格子定数)', fontsize=12)
        ax.set_ylabel('γ_emp (MotifBank スケーリング指数)', fontsize=12)
        ax.set_title('チューナブル合成フラクタル: ノイズによる γ 制御', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tunable_fractal.png', dpi=150)
        print(f"\n  図: tunable_fractal.png")
    except ImportError:
        print("  matplotlib 不要")
