#!/usr/bin/env python3
"""
fractal3d_synthesis_decomposition.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
フラクタル合成 → MotifBank分解

合成 (synthesis):
  Menger Sponge, Sierpinski Tetrahedron, 3D Cantor Dust,
  3D Cellular Automaton (Class 2 / Class 3), Random 3D

分解 (decomposition):
  各フラクタルを r_cut 内の局所フラグメントに分解
  → geom_key でバンクに登録 → N_bank(N) スケーリングを計測

測定量:
  γ   = d log(N_bank) / d log(N)   (バンク成長指数 = Phase 指標)
  d_eff = -d log(N_bank) / d log(ε) (Theorem 15 量子化次元)
  Phase 0-3 分類

仮説: d_eff ≈ d_H (Hausdorff 次元)  [Conjecture 5 の 3D 拡張]

Usage:
  OMP_NUM_THREADS=1 python3 fractal3d_synthesis_decomposition.py [--gen N] [--plot]
"""

import sys, os, json, argparse, itertools, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. フラクタル合成 (Synthesis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def menger_sponge(gen):
    """
    Menger Sponge — 3D Sierpinski Carpet 類似
    d_H = log(20)/log(3) ≈ 2.727
    gen=1:20, gen=2:400, gen=3:8000点
    """
    # 3x3x3 の 27 格子点から 中心+6面中心 (各軸で2つが中点) を除いた 20点
    UNIT = np.array([(x, y, z)
                     for x in range(3) for y in range(3) for z in range(3)
                     if (x == 1) + (y == 1) + (z == 1) <= 1], dtype=float)

    pts = np.array([[0., 0., 0.]])
    for g in range(gen):
        scale = 3.0 ** (-(g + 1))
        new_pts = (pts[:, np.newaxis, :] + UNIT[np.newaxis, :, :] * scale).reshape(-1, 3)
        pts = np.unique(np.round(new_pts, 10), axis=0)
    return pts


def sierpinski_tetrahedron(gen):
    """
    3D Sierpinski Tetrahedron
    d_H = log(4)/log(2) = 2.000
    gen=1:4, gen=2:16, gen=3:64, gen=4:256点
    """
    # 正四面体の4頂点
    V = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float) * 0.5

    pts = V.copy()
    for g in range(1, gen):
        scale = 0.5 ** g
        new_pts = (pts[:, np.newaxis, :] + V[np.newaxis, :, :] * scale).reshape(-1, 3)
        pts = np.unique(np.round(new_pts, 10), axis=0)
    return pts


def cantor_dust_3d(gen):
    """
    3D Cantor Dust — Cantor集合の3D直積
    d_H = 3 × log(2)/log(3) ≈ 1.893
    gen=1:8, gen=2:64, gen=3:512, gen=4:4096点
    """
    def cantor1d(g):
        pts = np.array([0.0, 1.0])
        for _ in range(g):
            pts = np.concatenate([pts / 3, pts / 3 + 2 / 3])
        return np.unique(pts)

    xs = cantor1d(gen)
    grid = np.array([[x, y, z] for x in xs for y in xs for z in xs])
    return grid


def random_3d(n_pts, seed=42):
    """
    ランダム 3D 点群 — Phase 3 基準
    d_H ≈ 3
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n_pts, 3))


# ── 3D セルオートマトン ────────────────────────────────────────────────

def cellular_automaton_3d(size=16, steps=8, born={4, 5, 6}, survive={3, 4, 5},
                           density=0.25, seed=42):
    """
    3D Moore-26近傍 CA
    born    : 誕生に必要な生存近傍数
    survive : 生存に必要な生存近傍数
    density : 初期密度
    Returns: final state の生存セル座標 (N×3)
    """
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size, size)) < density).astype(np.int8)

    def count_neighbors(g):
        c = np.zeros_like(g)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    c += np.roll(np.roll(np.roll(g, dx, 0), dy, 1), dz, 2)
        return c

    for _ in range(steps):
        nb = count_neighbors(grid)
        new_grid = np.zeros_like(grid)
        new_grid[(grid == 1) & np.isin(nb, list(survive))] = 1
        new_grid[(grid == 0) & np.isin(nb, list(born))]   = 1
        if np.array_equal(new_grid, grid):
            break
        grid = new_grid

    coords = np.argwhere(grid == 1).astype(float)
    return coords


def ca_stable(size=20, steps=12):
    """安定構造 CA — Phase 0/1 期待 (B5/S4-6)"""
    return cellular_automaton_3d(size=size, steps=steps,
                                  born={5}, survive={4, 5, 6}, density=0.30)


def ca_chaotic(size=20, steps=4):
    """カオス CA — Phase 3 期待 (B4/S3456, 疎→不規則クラスター)"""
    return cellular_automaton_3d(size=size, steps=steps,
                                  born={4}, survive={3, 4, 5, 6}, density=0.15)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. MotifBank 分解 (Decomposition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def nn_distance_local(pts, sample=300):
    """点群の典型的最近傍距離 (中央値) を推定"""
    if len(pts) > sample:
        idx = np.random.choice(len(pts), sample, replace=False)
        sub = pts[idx]
    else:
        sub = pts
    dists = []
    for i in range(len(sub)):
        d = np.linalg.norm(sub - sub[i], axis=1)
        d[i] = np.inf
        dists.append(d.min())
    return float(np.median(dists))


def geom_key_3d(pts, decimal=2):
    """3D点群の距離タプル (geom_key: 元素・回転・並進不変, 固定decimal)"""
    n = len(pts)
    dists = [round(float(np.linalg.norm(pts[i] - pts[j])), decimal)
             for i in range(n) for j in range(i + 1, n)]
    return tuple(sorted(dists))


def extract_fragments_knn(pts, k=12, max_centers=400):
    """
    k最近傍ベースのフラグメント抽出。
    ・center は max_centers 件にサブサンプル
    ・近傍はフル点群から検索 → 真の局所幾何を取得
    ・global decimal (= structure の nn_dist から決定)

    Returns: frags (list of arrays), decimal (int)
    """
    n = len(pts)
    nn = nn_distance_local(pts)
    if nn < 1e-12:
        nn = 1.0
    # decimal: nn が 2桁有効数字になるように
    decimal = max(1, int(np.ceil(-np.log10(nn))) + 1)

    # center サブサンプル (近傍は全点から取る)
    if n > max_centers:
        center_idx = np.random.choice(n, max_centers, replace=False)
    else:
        center_idx = np.arange(n)

    k_use = min(k, n - 1)
    if k_use < 3:
        return [], decimal

    frags = []
    for ci in center_idx:
        d = np.linalg.norm(pts - pts[ci], axis=1)
        nn_idx = np.argsort(d)[1:k_use + 1]
        neighbors = pts[nn_idx] - pts[ci]  # 中心を原点に
        frags.append(neighbors)
    return frags, decimal


def bank_growth_curve(frags, decimal=2, shuffle=True, seed=42):
    """
    フラグメントをシャッフルしてバンクに順次登録。
    γ = bank growth index  ≈  1 - motif_reuse_rate
    Returns: n_vals, n_bank_vals, reuse_rate
    """
    if shuffle:
        rng = np.random.default_rng(seed)
        frags = [frags[i] for i in rng.permutation(len(frags))]

    bank = set()
    n_vals, n_bank_vals = [], []
    hits = 0
    for i, frag in enumerate(frags):
        gk = geom_key_3d(frag, decimal=decimal)
        if gk in bank:
            hits += 1
        bank.add(gk)
        n_vals.append(i + 1)
        n_bank_vals.append(len(bank))
    reuse_rate = hits / len(frags) if frags else 0.0
    return np.array(n_vals), np.array(n_bank_vals), reuse_rate


def fit_gamma(n_vals, n_bank_vals, fit_frac=0.5):
    """
    N_bank(N) ∝ N^γ の γ を後半 fit_frac で log-log 線形フィット
    """
    n = len(n_vals)
    start = int(n * (1 - fit_frac))
    if start < 2:
        start = 0
    log_n = np.log(n_vals[start:] + 1e-9)
    log_nb = np.log(n_bank_vals[start:] + 1e-9)
    if len(log_n) < 2:
        return 0.0
    slope = np.polyfit(log_n, log_nb, 1)[0]
    return float(np.clip(slope, 0, 3))


def classify_phase(gamma):
    if gamma < 0.05:  return 'Phase 0'
    if gamma < 0.48:  return 'Phase 1'
    if gamma < 0.80:  return 'Phase 2'
    return 'Phase 3'


# ── 相関次元 (Grassberger-Procaccia) ───────────────────────────────
# d_corr = d(log C(r)) / d(log r),  C(r) = fraction of pairs with dist < r

def correlation_dimension(pts, n_r=20, sample=500, seed=42):
    """
    Grassberger-Procaccia 相関次元。
    C(r) = (2/N²) × #{pairs (i,j): |xi-xj| < r}
    d_corr = slope of log C vs log r in the scaling region.

    これが本来の d_eff (Theorem 15 量子化次元の正確な代理量)。
    """
    rng = np.random.default_rng(seed)
    N = len(pts)
    if N < 10:
        return 0.0

    # ペアワイズ距離 (sample 点でサブサンプル)
    sub = pts[rng.choice(N, min(N, sample), replace=False)]
    n_sub = len(sub)
    dists = []
    for i in range(n_sub):
        d = np.linalg.norm(sub[i + 1:] - sub[i], axis=1)
        dists.extend(d.tolist())
    dists = np.array(dists)
    if len(dists) < 10:
        return 0.0

    # r の範囲: 1〜50%ile で log-equal に
    r_lo = np.percentile(dists, 1)
    r_hi = np.percentile(dists, 50)
    if r_lo <= 0 or r_hi <= r_lo:
        return 0.0
    r_vals = np.logspace(np.log10(r_lo), np.log10(r_hi), n_r)

    C_vals = np.array([(dists < r).mean() for r in r_vals])

    # スケーリング領域を推定 (C > 0.02 かつ C < 0.95)
    mask = (C_vals > 0.02) & (C_vals < 0.95)
    if mask.sum() < 4:
        mask = C_vals > 0
    log_r  = np.log(r_vals[mask])
    log_C  = np.log(C_vals[mask] + 1e-15)

    if len(log_r) < 3:
        return 0.0
    slope = np.polyfit(log_r, log_C, 1)[0]
    return float(np.clip(slope, 0, 4))


# ── γ(r) スケール依存バンク成長指数 ────────────────────────────────

def gamma_vs_scale(pts, k_vals=None, max_centers=200, seed=42, decimal=None):
    """
    γ(r_eff) — k-NN ベーススケール依存測定。

    k を増やす → r_eff (k番目近傍距離の平均) が増える → γ が変化。
    ε-ball と違い k が大きくてもフラグメントサイズは k で固定 → スケール問題なし。

    小 k → 局所モチーフ → reuse 高 → γ 低
    大 k → 大域構造   → reuse 崩壊 → γ 高

    Hypothesis: γ(r_eff) = 1 - exp(-k_fit(r_eff - r_th))
    """
    rng = np.random.default_rng(seed)
    N = len(pts)
    if N < 10:
        return np.array([]), np.array([])

    if k_vals is None:
        max_k = min(N - 1, 50)
        k_vals = list(set(int(v) for v in
                         np.geomspace(3, max_k, 10).tolist()))
        k_vals = sorted(k_vals)

    nn = nn_distance_local(pts)
    if decimal is None:
        decimal = max(1, int(np.ceil(-np.log10(nn + 1e-12))) + 1)

    center_idx = rng.choice(N, min(N, max_centers), replace=False)

    gammas = []
    r_effs = []
    for k in k_vals:
        k_use = min(k, N - 1)
        if k_use < 3:
            continue

        frags = []
        rs = []
        for ci in center_idx:
            d = np.linalg.norm(pts - pts[ci], axis=1)
            nn_idx = np.argsort(d)[1:k_use + 1]
            nb = pts[nn_idx] - pts[ci]
            frags.append(nb)
            rs.append(float(d[nn_idx[-1]]))

        if len(frags) < 10:
            continue

        n_vals, n_bank_vals, _ = bank_growth_curve(frags, decimal=decimal)
        g = fit_gamma(n_vals, n_bank_vals)
        gammas.append(g)
        r_effs.append(float(np.mean(rs)))

    return np.array(r_effs), np.array(gammas)


def fit_gamma_r(r_vals, gammas):
    """
    γ(r) = 1 × (1 - exp(-k(r - r_th)))  [γ∞=1 固定]

    物理的制約: γ ∈ [0,1] かつ reuse ∈ [0,1]
    → γ∞ ≤ 1 を保証するために γ∞=1 に固定し、k と r_th のみフィット

    Returns: (gamma_inf=1.0, k, r_th) or None
    """
    from scipy.optimize import curve_fit
    if len(r_vals) < 4:
        return None

    def model(r, k, r_th):
        return 1.0 - np.exp(-k * np.maximum(r - r_th, 0))

    try:
        p0 = [1.0 / float(np.median(r_vals) + 1e-12), float(r_vals[0])]
        bounds = ([0.0, 0.0], [200.0, float(r_vals[-1])])
        popt, _ = curve_fit(model, r_vals, gammas, p0=p0, bounds=bounds, maxfev=5000)
        return (1.0, float(popt[0]), float(popt[1]))  # (γ∞=1, k, r_th)
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. 実験実行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_llm_embeddings_pca(bank_path='/home/yoiyoi/radar_bank_cache.json', n_components=3):
    """
    ReasonBank の GPT-2 layer-11 embeddings を PCA で 3D に圧縮。
    embedding 空間上での γ(r) を測定するための 3D 点群を返す。
    """
    try:
        import json
        from sklearn.decomposition import PCA
        with open(bank_path) as f:
            data = json.load(f)
        embs = np.array([d['emb'] for d in data], dtype=float)
        pca = PCA(n_components=n_components, random_state=42)
        pts = pca.fit_transform(embs).astype(float)
        return pts
    except Exception as e:
        print(f'  [LLM load failed: {e}]')
        return np.random.randn(50, 3)  # fallback


EXPERIMENTS = {
    # name          : (generator_fn, d_H_theory)
    'Menger Sponge'      : (lambda g: menger_sponge(g),            2.727),
    'Sierpinski Tet'     : (lambda g: sierpinski_tetrahedron(g),   2.000),
    'Cantor Dust 3D'     : (lambda g: cantor_dust_3d(g),           1.893),
    'CA Stable (B5/S456)': (lambda g: ca_stable(size=16+g*4),       None),
    'CA Chaotic (B4/S3456)': (lambda g: ca_chaotic(size=16+g*4),    None),
    'Random 3D'          : (lambda g: random_3d(20 * 4**g),         3.0),
    'LLM Embeddings (PCA3D)': (lambda g: load_llm_embeddings_pca(), None),
}

def run_all(max_gen=3, k_nn=12, verbose=True):
    """
    k-NN ベース + per-fragment adaptive decimal で γ を測定。
    γ = 1 - motif_reuse_rate  (motif再利用率の相補量)
    """
    results = {}
    for name, (gen_fn, d_H) in EXPERIMENTS.items():
        results[name] = {'d_H': d_H, 'gens': []}
        if verbose:
            print(f'\n═══ {name}  (d_H={d_H}) ═══')

        for g in range(1, max_gen + 1):
            t0 = time.time()
            pts = gen_fn(g)
            if len(pts) < 5:
                continue

            frags, decimal = extract_fragments_knn(pts, k=k_nn, max_centers=400)
            if len(frags) < 5:
                if verbose:
                    print(f'  gen={g}: N={len(pts)}, fragments<5, skip')
                continue

            n_vals, n_bank_vals, reuse_rate = bank_growth_curve(frags, decimal=decimal)
            gamma  = fit_gamma(n_vals, n_bank_vals)
            phase  = classify_phase(gamma)
            n_sat  = int(n_bank_vals[-1])
            s_local = float(np.log(n_sat + 1))

            elapsed = time.time() - t0
            row = {
                'gen': g, 'N_pts': len(pts), 'N_frags': len(frags),
                'gamma': gamma, 'phase': phase,
                'N_bank_sat': n_sat, 'S_local': s_local,
                'reuse_rate': reuse_rate,
                'n_vals': n_vals.tolist(),
                'n_bank_vals': n_bank_vals.tolist(),
                'elapsed': elapsed,
            }
            results[name]['gens'].append(row)

            if verbose:
                print(f'  gen={g}: N={len(pts):5d}  frags={len(frags):4d}  '
                      f'N_bank={n_sat:4d}  γ={gamma:.3f}  reuse={reuse_rate:.2f}  '
                      f'{phase}  S={s_local:.2f}  ({elapsed:.1f}s)')

    return results


def run_corr_dim(max_gen=2, verbose=True):
    """相関次元 (Grassberger-Procaccia) 測定"""
    results = {}
    if verbose:
        print('\n═══ 相関次元 d_corr (Grassberger-Procaccia) ═══')
    for name, (gen_fn, d_H) in EXPERIMENTS.items():
        pts = gen_fn(max_gen)
        if len(pts) < 10:
            continue
        d_corr = correlation_dimension(pts)
        results[name] = {'d_H': d_H, 'd_corr': d_corr, 'N': len(pts)}
        if verbose:
            dh_str = f'{d_H:.3f}' if d_H else '—'
            diff = f'  Δ={d_corr - d_H:+.3f}' if d_H else ''
            print(f'  {name:28s}: d_corr={d_corr:.3f}  d_H={dh_str}{diff}  N={len(pts)}')
    return results


def run_gamma_r(max_gen=2, menger_gen4=False, verbose=True):
    """
    γ(r) スケール依存測定 + γ(r) = 1 - exp(-k(r-r_th)) フィット [γ∞=1 固定]
    menger_gen4=True のとき Menger gen=4 (N=160k) で r レンジを拡張
    """
    results = {}
    if verbose:
        print('\n═══ γ(r) スケール依存バンク成長指数 ═══')

    experiments = dict(EXPERIMENTS)
    if menger_gen4:
        experiments['Menger (gen=4)'] = (lambda g: menger_sponge(4), 2.727)
        COLORS['Menger (gen=4)'] = '#a5b4fc'

    for name, (gen_fn, d_H) in experiments.items():
        pts = gen_fn(max_gen)
        if len(pts) < 10:
            continue
        r_vals, gammas = gamma_vs_scale(pts)
        if len(r_vals) < 3:
            results[name] = {'d_H': d_H, 'r': [], 'gamma': [], 'fit': None}
            continue
        fit = fit_gamma_r(r_vals, gammas)
        results[name] = {
            'd_H': d_H, 'N': len(pts),
            'r': r_vals.tolist(), 'gamma': gammas.tolist(),
            'fit': list(fit) if fit is not None else None,
        }
        if verbose:
            if fit is not None:
                fit_str = f'γ∞=1(fixed)  k={fit[1]:.3f}  r_th={fit[2]:.4f}'
            else:
                fit_str = 'fit failed'
            print(f'  {name:30s}: γ(r_min)={gammas[0]:.3f}  '
                  f'γ(r_max)={gammas[-1]:.3f}  [{fit_str}]')
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. 可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COLORS = {
    'Menger Sponge':          '#6366f1',
    'Sierpinski Tet':         '#10b981',
    'Cantor Dust 3D':         '#f59e0b',
    'CA Stable (B5/S456)':    '#3b82f6',
    'CA Chaotic (B4/S3456)':  '#ef4444',
    'Random 3D':              '#94a3b8',
    'LLM Embeddings (PCA3D)': '#f0abfc',  # magenta — LLM
}

def plot_3d_structures(max_gen=2, out='fractal3d_structures.png'):
    fig = plt.figure(figsize=(18, 6), facecolor='#0f172a')
    names = list(EXPERIMENTS.keys())

    for idx, name in enumerate(names):
        gen_fn, d_H = EXPERIMENTS[name]
        pts = gen_fn(max_gen)
        if len(pts) > 3000:
            pts = pts[np.random.choice(len(pts), 3000, replace=False)]

        ax = fig.add_subplot(1, len(names), idx + 1, projection='3d',
                             facecolor='#1e293b')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=COLORS[name], s=1.5, alpha=0.6)
        dh_str = f'd_H={d_H:.3f}' if d_H else 'CA'
        ax.set_title(f'{name}\n({dh_str}, N={len(pts)})',
                     color='#e2e8f0', fontsize=7, pad=2)
        ax.tick_params(colors='#475569', labelsize=5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor('#334155')

    plt.suptitle('Fractal Synthesis — 3D Structures (gen=2)',
                 color='#f1f5f9', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_bank_growth(results, out='fractal3d_bank_growth.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0f172a')

    phase_colors = {'Phase 0': '#10b981', 'Phase 1': '#3b82f6',
                    'Phase 2': '#f59e0b', 'Phase 3': '#ef4444'}

    for name, data in results.items():
        color = COLORS[name]
        for row in data['gens']:
            n  = np.array(row['n_vals'])
            nb = np.array(row['n_bank_vals'])
            g  = row['gen']
            axes[0].plot(n, nb, color=color, alpha=0.6 + g * 0.1,
                         linewidth=1.2, label=f'{name} g{g}' if g == len(data["gens"]) else '_')
            if len(n) > 2:
                axes[1].loglog(n, nb, color=color, alpha=0.5 + g * 0.1,
                               linewidth=1.2, label=f'{name} g{g}' if g == len(data["gens"]) else '_')

    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, color='#334155', alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

    axes[0].set_xlabel('N (fragments seen)', color='#94a3b8')
    axes[0].set_ylabel('N_bank (unique motifs)', color='#94a3b8')
    axes[0].set_title('Bank Growth: N_bank(N)', color='#e2e8f0')
    axes[0].legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
                   labelcolor='#94a3b8', ncol=2)

    axes[1].set_xlabel('log N', color='#94a3b8')
    axes[1].set_ylabel('log N_bank', color='#94a3b8')
    axes[1].set_title('Log-Log Scaling  (slope = γ)', color='#e2e8f0')
    axes[1].legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
                   labelcolor='#94a3b8', ncol=2)

    plt.suptitle('Fractal Decomposition — MotifBank Growth Curves',
                 color='#f1f5f9', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_gamma_vs_dH(results, d_eff_results, out='fractal3d_gamma_dH.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0f172a')

    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, color='#334155', alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

    # γ vs generation
    ax = axes[0]
    for name, data in results.items():
        gens  = [r['gen']  for r in data['gens']]
        gammas = [r['gamma'] for r in data['gens']]
        if gens:
            ax.plot(gens, gammas, 'o-', color=COLORS[name],
                    label=name, linewidth=1.5, markersize=5)
    ax.axhline(0.48, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.7,
               label='γ_c = 0.48')
    ax.axhline(0.80, color='#ef4444', linestyle='--', linewidth=1, alpha=0.7,
               label='γ_c = 0.80')
    ax.set_xlabel('Generation', color='#94a3b8')
    ax.set_ylabel('γ (bank growth index)', color='#94a3b8')
    ax.set_title('γ per Generation\n(below 0.48 = Phase 0/1 = compressible)',
                 color='#e2e8f0', fontsize=9)
    ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

    # d_corr vs d_H (GP correlation dimension)
    ax = axes[1]
    xs, ys = [], []
    for name, d in d_eff_results.items():
        dc = d.get('d_corr')
        dh = d.get('d_H')
        if dc is not None and dh is not None:
            xs.append(dh); ys.append(dc)
            ax.scatter(dh, dc, c=COLORS.get(name, '#999'),
                       s=80, zorder=5, edgecolors='white', linewidths=0.8)
            ax.annotate(name.split('(')[0].strip(), (dh, dc),
                        textcoords='offset points', xytext=(5, 3),
                        fontsize=6.5, color='#cbd5e1')

    if xs:
        xr = np.linspace(min(xs) * 0.85, max(xs) * 1.1, 50)
        ax.plot(xr, xr, '--', color='#94a3b8', linewidth=1, alpha=0.6,
                label='d_corr = d_H (ideal)')
    ax.set_xlabel('d_H (Hausdorff dimension, theory)', color='#94a3b8')
    ax.set_ylabel('d_corr (GP correlation dimension)', color='#94a3b8')
    ax.set_title('GP Correlation Dimension vs d_H\n(Conjecture 5 — 3D extension)',
                 color='#e2e8f0', fontsize=9)
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

    plt.suptitle('Fractal3D Synthesis → Decomposition: Phase & Dimension Analysis',
                 color='#f1f5f9', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_summary_table(results, corr_results, out='fractal3d_summary.png'):
    """サマリーテーブル画像"""
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.axis('off')

    headers = ['System', 'd_H', 'd_corr', 'γ', 'reuse', 'Phase', 'N_bank', 'compress?']
    rows = []
    for name, data in results.items():
        if not data['gens']:
            continue
        last = data['gens'][-1]
        dh = data['d_H']
        dc = corr_results.get(name, {}).get('d_corr')
        rr = last.get('reuse_rate', 0.0)
        rows.append([
            name,
            f'{dh:.3f}' if dh else '—',
            f'{dc:.3f}' if dc is not None else '—',
            f'{last["gamma"]:.3f}',
            f'{rr:.2f}',
            last['phase'],
            str(last['N_bank_sat']),
            'yes' if last['gamma'] < 0.48 else 'no',
        ])

    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color='#e2e8f0', fontweight='bold')
        else:
            cell.set_facecolor('#1e293b' if r % 2 == 0 else '#263347')
            cell.set_text_props(color='#cbd5e1')
        cell.set_edgecolor('#334155')

    ax.set_title('Fractal 3D Synthesis → MotifBank Decomposition — Summary',
                 color='#f1f5f9', fontsize=11, fontweight='bold', pad=15)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_corr_dim(corr_results, out='fractal3d_corr_dim.png'):
    """相関次元 d_corr vs d_H のプロット"""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#334155', alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')

    xs, ys = [], []
    for name, d in corr_results.items():
        dc = d.get('d_corr', 0)
        dh = d.get('d_H')
        ax.scatter(dh if dh else -0.1, dc,
                   c=COLORS.get(name, '#999'), s=100,
                   zorder=5, edgecolors='white', linewidths=0.8,
                   label=f'{name} ({dc:.2f})')
        if dh:
            xs.append(dh); ys.append(dc)

    if xs:
        xr = np.linspace(min(xs) * 0.85, max(xs) * 1.1, 50)
        ax.plot(xr, xr, '--', color='#94a3b8', linewidth=1.2, alpha=0.7,
                label='d_corr = d_H (ideal)')
    ax.set_xlabel('d_H (Hausdorff theory)', color='#94a3b8')
    ax.set_ylabel('d_corr (GP correlation dim)', color='#94a3b8')
    ax.set_title('Correlation Dimension vs Hausdorff Dimension\n'
                 '(Grassberger-Procaccia method)',
                 color='#e2e8f0', fontsize=9)
    ax.legend(fontsize=6.5, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8', loc='upper left')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out}')


def plot_gamma_r(gamma_r_results, out='fractal3d_gamma_r.png'):
    """
    γ(r) スケール依存プロット — 論文レベルの核心図。
    自己相似フラクタル: 小スケール γ低、大スケール γ高 (転移あり)
    ランダム: 全スケールで γ≈1
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#334155', alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')

    for name, d in gamma_r_results.items():
        r_vals = np.array(d.get('r', []))
        gammas = np.array(d.get('gamma', []))
        if len(r_vals) < 2:
            continue
        ax.plot(r_vals, gammas, 'o-', color=COLORS.get(name, '#999'),
                linewidth=2, markersize=5, label=name, alpha=0.9)

        # フィット曲線
        fit = d.get('fit')
        if fit is not None:
            g_inf, k_fit, r_th = fit
            r_plot = np.linspace(r_vals[0] * 0.8, r_vals[-1] * 1.2, 100)
            y_plot = g_inf * (1 - np.exp(-k_fit * np.maximum(r_plot - r_th, 0)))
            ax.plot(r_plot, y_plot, '--', color=COLORS.get(name, '#999'),
                    linewidth=1.2, alpha=0.5)

    ax.axhline(0.48, color='#f59e0b', linestyle=':', linewidth=1.5, alpha=0.8,
               label='γ_c (Phase 1/2 boundary)')
    ax.axhline(0.80, color='#ef4444', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Fragment scale r (log)', color='#94a3b8', fontsize=10)
    ax.set_ylabel('γ(r)  =  bank growth index', color='#94a3b8', fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('γ(r): Scale-Dependent Motif Reuse\n'
                 r'Hypothesis: $\gamma(r) = \gamma_\infty(1 - e^{-k(r-r_{th})})$',
                 color='#e2e8f0', fontsize=10)
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. メイン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen',  type=int, default=3, help='max generation (default 3)')
    ap.add_argument('--plot', action='store_true', help='save figures')
    ap.add_argument('--gen4', action='store_true', help='include Menger gen=4 in γ(r)')
    args = ap.parse_args()

    os.chdir('/home/yoiyoi')
    np.random.seed(42)

    print('=' * 60)
    print('  Fractal 3D: Synthesis → MotifBank Decomposition')
    print('=' * 60)

    # § 合成 + 分解
    results = run_all(max_gen=args.gen, verbose=True)

    # § 相関次元 (Grassberger-Procaccia)
    print()
    corr_results = run_corr_dim(max_gen=min(args.gen, 2), verbose=True)

    # § γ(r) スケール依存 [γ∞=1 固定フィット]
    print()
    gamma_r_results = run_gamma_r(max_gen=min(args.gen, 2),
                                   menger_gen4=args.gen4, verbose=True)

    # § 最終サマリー
    print('\n' + '═' * 80)
    print(f'  {"System":28s} {"d_H":>6} {"d_corr":>7} {"γ":>6} {"reuse":>6} '
          f'{"Phase":>8} {"N_sat":>6}')
    print('  ' + '─' * 76)
    for name, data in results.items():
        if not data['gens']:
            continue
        last = data['gens'][-1]
        dh    = data['d_H']
        dc    = corr_results.get(name, {}).get('d_corr', float('nan'))
        flag  = 'compress' if last['gamma'] < 0.48 else 'no-compress'
        rr    = last.get('reuse_rate', 0.0)
        dh_s  = f'{dh:.3f}' if dh else '—'
        dc_s  = f'{dc:.3f}' if not np.isnan(dc) else '—'
        print(f'  {name:28s} {dh_s:>6}  {dc_s:>6}  '
              f'{last["gamma"]:>5.3f}  {rr:>5.2f}  {last["phase"]:>8}  '
              f'{last["N_bank_sat"]:>5}  {flag}')
    print('═' * 80)
    print('  γ ≈ 1 - reuse  |  d_corr = GP correlation dimension')

    # § 保存
    with open('/home/yoiyoi/fractal3d_results.json', 'w') as f:
        slim = {}
        for name, data in results.items():
            slim[name] = {'d_H': data['d_H'], 'gens': [
                {k: v for k, v in row.items()
                 if k not in ('n_vals', 'n_bank_vals')}
                for row in data['gens']
            ]}
        json.dump({
            'results': slim,
            'corr_dim': corr_results,
            'gamma_r': gamma_r_results,
        }, f, indent=2, default=lambda x: None)
    print('\nResults saved: fractal3d_results.json')

    if args.plot:
        print('\nGenerating figures...')
        plot_3d_structures(max_gen=min(args.gen, 2))
        plot_bank_growth(results)
        plot_gamma_vs_dH(results, corr_results)
        plot_summary_table(results, corr_results)
        plot_corr_dim(corr_results)
        plot_gamma_r(gamma_r_results)
        print('All figures saved.')


if __name__ == '__main__':
    main()
