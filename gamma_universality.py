#!/usr/bin/env python3
"""
gamma_universality.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
統一γ(r, d) 崩壊プロット

γ(r, d) = {  0                             (r + d ≤ r_th)
           {  1 - exp(-k(r + d - r_th))    (r + d > r_th)

r    = 観測スケール
d    = 知識クラスタからの距離 (純粋構造では d=0; LLM では d=d_min)
r_th = coherence radius (記憶の限界スケール)
k    = 崩壊率

普遍性仮説:
  s = k * max(r + d - r_th, 0) で正規化すると
  すべての系は γ(s) = 1 - exp(-s) に崩壊する

Usage:
  OMP_NUM_THREADS=1 python3 gamma_universality.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit as sigmoid

os.chdir('/home/yoiyoi')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. 統一γ(r, d) モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gamma_unified(r, d, k, r_th):
    """
    γ(r, d) = 1 - exp(-k * max(r + d - r_th, 0))

    piecewise: γ=0 for r+d ≤ r_th, increases exponentially beyond.
    γ ∈ [0, 1] が保証される (k > 0, r_th > 0 のとき)。
    """
    return 1.0 - np.exp(-k * np.maximum(r + d - r_th, 0.0))


def fit_gamma_r_unified(r_vals, gammas):
    """
    γ(r, d=0) = 1 - exp(-k * max(r - r_th, 0)) を k, r_th でフィット [γ∞=1 固定]
    """
    if len(r_vals) < 4:
        return None
    def model(r, k, r_th):
        return 1.0 - np.exp(-k * np.maximum(r - r_th, 0.0))
    try:
        r_arr = np.asarray(r_vals, float)
        g_arr = np.asarray(gammas, float)
        p0 = [2.0 / (r_arr.max() - r_arr.min() + 1e-12), float(r_arr.mean())]
        bounds = ([1e-6, 0.0], [500.0, float(r_arr.max())])
        popt, _ = curve_fit(model, r_arr, g_arr, p0=p0, bounds=bounds, maxfev=10000)
        return float(popt[0]), float(popt[1])  # k, r_th
    except Exception:
        return None


def normalized_s(r, d, k, r_th):
    """正規化座標 s = k * max(r + d - r_th, 0)  →  γ(s) = 1 - exp(-s)"""
    return k * np.maximum(r + d - r_th, 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. フラクタルデータ読み込みとフィット
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open('/home/yoiyoi/fractal3d_results.json') as f:
    frac_data = json.load(f)

COLORS = {
    'Menger Sponge':          '#6366f1',
    'Sierpinski Tet':         '#10b981',
    'Cantor Dust 3D':         '#f59e0b',
    'CA Stable (B5/S456)':    '#3b82f6',
    'CA Chaotic (B4/S3456)':  '#ef4444',
    'Random 3D':              '#94a3b8',
    'LLM Embeddings (PCA3D)': '#f0abfc',
    'Menger (gen=4)':         '#a5b4fc',
}

print('=' * 65)
print('  γ(r, d) Universal Collapse Analysis')
print('=' * 65)

fits = {}   # name → {'k': ..., 'r_th': ..., 'r': [...], 'gamma': [...]}
gamma_r = frac_data.get('gamma_r', {})

print('\n── フィット結果 (γ∞=1 固定) ──')
for name, d in gamma_r.items():
    r_arr = np.array(d.get('r', []))
    g_arr = np.array(d.get('gamma', []))
    if len(r_arr) < 4:
        print(f'  {name:30s}: skip (N={len(r_arr)})')
        continue
    res = fit_gamma_r_unified(r_arr, g_arr)
    if res is None:
        print(f'  {name:30s}: fit failed')
        continue
    k, r_th = res
    fits[name] = {'k': k, 'r_th': r_th, 'r': r_arr, 'gamma': g_arr}
    print(f'  {name:30s}: k={k:.3f}  r_th={r_th:.4f}  '
          f'(r_th * k = {k*r_th:.3f})')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. LLM d_min データ (hallucination_radar から)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# hallucination_radar の LLM 埋め込みを使って d_min を測定
# bank_cache を直接読み込んで d_min を計算 (PCA3D 空間での距離)
def get_llm_dmin_pca():
    """
    ReasonBank embeddings を PCA3D に変換し、
    既知クラスタからの d_min を各埋め込みに対して測定。
    (実際の hallucination_radar の d_min を 3D PCA 空間にマップ)
    """
    try:
        from sklearn.decomposition import PCA
        with open('/home/yoiyoi/radar_bank_cache.json') as f:
            data = json.load(f)
        embs = np.array([d['emb'] for d in data], dtype=float)
        pca = PCA(n_components=3, random_state=42)
        pts = pca.fit_transform(embs)

        # 各点の bank 内最近傍距離 (leave-one-out d_min)
        dmins = []
        for i in range(len(pts)):
            others = np.delete(pts, i, axis=0)
            dists = np.linalg.norm(others - pts[i], axis=1)
            dmins.append(float(dists.min()))
        return pts, np.array(dmins)
    except Exception as e:
        print(f'  [LLM dmin failed: {e}]')
        return None, None

llm_pts, llm_dmins = get_llm_dmin_pca()

if llm_dmins is not None:
    print(f'\n── LLM d_min (PCA3D, leave-one-out) ──')
    print(f'  d_min: min={llm_dmins.min():.3f}  med={np.median(llm_dmins):.3f}  '
          f'max={llm_dmins.max():.3f}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. γ(r, d) — LLM 固有パラメータで d_min を解釈
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LLM 固有の (k, r_th) を使用 (PCA3D 空間のスケールに対応)
# γ(r=0, d) = 1 - exp(-k_LLM * max(d - r_th_LLM, 0))
ref_name = 'Menger Sponge'  # 図の参照構造
if ref_name not in fits:
    ref_name = list(fits.keys())[0]
k_ref   = fits[ref_name]['k']
rth_ref = fits[ref_name]['r_th']

llm_key = 'LLM Embeddings (PCA3D)'
if llm_key in fits:
    k_llm   = fits[llm_key]['k']
    rth_llm = fits[llm_key]['r_th']
else:
    k_llm, rth_llm = 0.405, 0.283  # fallback

print(f'\n参照構造: {ref_name}  k={k_ref:.3f}  r_th={rth_ref:.4f}')
print(f'LLM 固有 :                    k={k_llm:.3f}  r_th={rth_llm:.4f}')

# LLM 固有パラメータで γ(r=0, d_min) を計算
if llm_dmins is not None:
    gamma_at_r0 = 1.0 - np.exp(-k_llm * np.maximum(llm_dmins - rth_llm, 0))
    print(f'\n── γ(r=0, d_min) — LLM 各埋め込み [k_LLM={k_llm:.3f}, r_th={rth_llm:.3f}] ──')
    print(f'  γ(d_min): min={gamma_at_r0.min():.3f}  med={np.median(gamma_at_r0):.3f}  '
          f'max={gamma_at_r0.max():.3f}')
    print(f'  d_min < r_th ({rth_llm:.3f}): {(llm_dmins < rth_llm).sum()} 点  '
          f'→ γ=0 (完全記憶)')
    print(f'  d_min > r_th ({rth_llm:.3f}): {(llm_dmins >= rth_llm).sum()} 点  '
          f'→ γ>0 (情報生成)')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. 普遍崩壊プロット
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def master_curve(s):
    """普遍マスター曲線 γ = 1 - exp(-s)  (s ≥ 0)"""
    return 1.0 - np.exp(-np.maximum(s, 0.0))


fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0f172a')
ax_raw, ax_norm, ax_d = axes

for ax in axes:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#334155', alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')

# ── Plot 1: γ(r) 生データ ─────────────────────────────────────────────
ax = ax_raw
r_master = np.linspace(0, 1.5, 300)
for name, fd in fits.items():
    r_arr = fd['r']
    g_arr = fd['gamma']
    c = COLORS.get(name, '#999')
    ax.plot(r_arr, g_arr, 'o', color=c, markersize=4, alpha=0.8)
    k, r_th = fd['k'], fd['r_th']
    r_fit = np.linspace(0, max(r_arr) * 1.3, 200)
    ax.plot(r_fit, gamma_unified(r_fit, 0, k, r_th), '-',
            color=c, linewidth=1.8, alpha=0.8, label=name.split('(')[0].strip())
ax.set_xlabel('r (scale)', color='#94a3b8')
ax.set_ylabel('γ(r)', color='#94a3b8')
ax.set_title('γ(r) raw — each system\n'
             r'$\gamma(r)=1-e^{-k\,\max(r-r_{th},0)}$',
             color='#e2e8f0', fontsize=9)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=6.5, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8', loc='upper left')

# ── Plot 2: 正規化崩壊 γ(s), s = k*(r - r_th) ──────────────────────────
ax = ax_norm
s_master = np.linspace(0, 4, 300)
ax.plot(s_master, master_curve(s_master), '--',
        color='white', linewidth=2.5, alpha=0.9, zorder=10,
        label=r'Master: $\gamma=1-e^{-s}$')

for name, fd in fits.items():
    r_arr = np.asarray(fd['r'])
    g_arr = np.asarray(fd['gamma'])
    k, r_th = fd['k'], fd['r_th']
    s_arr = normalized_s(r_arr, 0.0, k, r_th)
    c = COLORS.get(name, '#999')
    ax.scatter(s_arr, g_arr, color=c, s=25, alpha=0.85, zorder=5)
    # smooth fit curve in s space
    s_fit = np.linspace(0, s_arr.max() * 1.2 + 0.2, 100)
    ax.plot(s_fit, master_curve(s_fit), '-', color=c, linewidth=1.0, alpha=0.4)

ax.set_xlabel(r's = k · max(r − r_th, 0)  [normalized]', color='#94a3b8')
ax.set_ylabel('γ', color='#94a3b8')
ax.set_title('Universal Collapse\n'
             r'All systems → $\gamma(s)=1-e^{-s}$',
             color='#e2e8f0', fontsize=9)
ax.set_xlim(-0.1, 4.5)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=7.5, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8')

# ── Plot 3: d_min シフト — γ(d_min) vs r_th との比較 ────────────────────
ax = ax_d
r_d = np.linspace(0, 1.5, 300)

# 参照曲線 (d=0)
ax.plot(r_d, gamma_unified(r_d, 0, k_ref, rth_ref), '-',
        color='white', linewidth=2.5, alpha=0.9, label=f'{ref_name} (d=0)')

# d_min でシフトした LLM 曲線
if llm_dmins is not None:
    # d_min の分位点ごとにプロット
    for q, label, alpha in [(10, 'd_min p10 (銀行内)', 0.9),
                             (50, 'd_min p50 (中央値)', 0.75),
                             (90, 'd_min p90 (遠方)', 0.6)]:
        d_val = float(np.percentile(llm_dmins, q))
        y = gamma_unified(r_d, d_val, k_ref, rth_ref)
        ax.plot(r_d, y, '--', color='#f0abfc',
                linewidth=1.8, alpha=alpha,
                label=f'LLM {label} d={d_val:.3f}')
    # 各埋め込みの r=0 での γ(d_min) をスキャッタ
    ax.scatter(llm_dmins, gamma_at_r0,
               c='#f0abfc', s=18, alpha=0.6, zorder=5,
               label='LLM bank γ(r=0, d_min)')

ax.axvline(rth_ref, color='#f59e0b', linestyle=':', linewidth=1.5,
           alpha=0.8, label=f'r_th={rth_ref:.3f}')
ax.set_xlabel('r + d  (effective scale)', color='#94a3b8')
ax.set_ylabel('γ(r, d)', color='#94a3b8')
ax.set_title(r'Unified: $\gamma(r,d)=1-e^{-k\,\max(r+d-r_{th},0)}$' + '\n'
             'LLM d_min as knowledge distance',
             color='#e2e8f0', fontsize=9)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8', loc='lower right')

plt.suptitle(r'Universal $\gamma(r,d)$ Law: Structure × Knowledge × Scale',
             color='#f1f5f9', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig('/home/yoiyoi/gamma_universality.png', dpi=150,
            bbox_inches='tight', facecolor='#0f172a')
plt.close(fig)
print('\nSaved: gamma_universality.png')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §6. 崩壊の定量評価 (各系のマスター曲線からの残差)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print('\n── 崩壊品質: RMSE(γ vs master_curve) ──')
print(f'  {"System":30s}  {"k":>8}  {"r_th":>8}  {"RMSE":>8}')
print('  ' + '─' * 60)
for name, fd in fits.items():
    r_arr = np.asarray(fd['r'])
    g_arr = np.asarray(fd['gamma'])
    k, r_th = fd['k'], fd['r_th']
    s_arr = normalized_s(r_arr, 0.0, k, r_th)
    pred  = master_curve(s_arr)
    rmse  = float(np.sqrt(np.mean((g_arr - pred) ** 2)))
    print(f'  {name:30s}  {k:8.3f}  {r_th:8.4f}  {rmse:8.4f}')

# 解釈
print(f'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
統一式の意味:

  γ(r, d) = 1 - exp(-k·max(r + d - r_th, 0))

  r    = 観測スケール (k-NN の有効半径)
  d    = 知識からの距離 (LLM: d_min、フラクタル: d=0)
  r_th = coherence radius — 記憶の限界スケール
  k    = 崩壊率 (急峻 ↔ 緩やか)

普遍性: s = k·max(r+d-r_th, 0) で正規化 → γ(s) = 1-exp(-s)

LLM への適用:
  d_min < r_th → γ=0 (完全記憶、幻覚なし)
  d_min > r_th → γ>0 (記憶崩壊、幻覚リスク)

  これは hallucination_radar の d_min ベースリスクスコアと
  同じ相転移を異なる言語で記述している。

  統一スコア: H(d_min) = 1 - exp(-k·max(d_min - r_th, 0))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
''')
