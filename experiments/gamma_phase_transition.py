#!/usr/bin/env python3
"""
gamma_phase_transition.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase Transition in Embedding Space

安全プロンプトを段階的に「幻覚化」して γ の Phase 転移を測定。

手法:
  1. セーフプロンプト s_0 のエンベディング e_0 を取得
  2. ランダムな方向ベクトル v を PCA3D 空間で生成
  3. 補間: e(α) = e_0 + α * v  (α: 0 → α_max)
  4. 各 α での A(r) と γ_H を計算
  5. γ_H > 0.5 になる α* = Phase 転移点

また「セマンティック変形」も実施:
  - 実際に異なるプロンプトに変化させて d_min の変化を追跡

Usage:
  OMP_NUM_THREADS=1 python3 gamma_phase_transition.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

os.chdir('/home/yoiyoi')

K_GAMMA   = 0.405
RTH_GAMMA = 0.283
CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'

# セマンティック変形シーケンス: 段階的に "幻覚化"
MORPH_SEQUENCES = [
    {
        'title': 'Capital knowledge gradient',
        'prompts': [
            "What is the capital of France?",         # 完全にバンク内
            "What is the capital of Franconia?",      # 実在するが辺境
            "What is the capital of New Franconia?",  # 架空
            "What is the capital of Zorblaxia?",      # 完全に架空
        ],
        'truth': [True, True, False, False],
    },
    {
        'title': 'Scientist knowledge gradient',
        'prompts': [
            "Who discovered penicillin?",                      # バンク内
            "Who discovered lysozyme?",                        # 実在 (Fleming)
            "Who discovered the quantum biology of memory?",   # 曖昧
            "Who discovered the Kalderon field?",              # 架空
        ],
        'truth': [True, True, False, False],
    },
    {
        'title': 'Historical event gradient',
        'prompts': [
            "In what year did World War II end?",             # バンク内
            "In what year did the Korean War end?",           # 実在
            "In what year did the Martian War end?",          # 架空
            "In what year did the Zorblax Conflict end?",     # 完全架空
        ],
        'truth': [True, True, False, False],
    },
    {
        'title': 'Science fact gradient',
        'prompts': [
            "What is the boiling point of water in Celsius?",   # バンク内
            "What is the boiling point of ethanol in Celsius?", # 実在
            "What is the boiling point of xenon in Celsius?",   # 実在だが難
            "What is the boiling point of unobtainium?",        # 架空
        ],
        'truth': [True, True, True, False],
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. データ準備
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=' * 65)
print('  γ Phase Transition Analysis')
print('=' * 65)

_model = None
_tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    print("Loading GPT-2...", flush=True)
    _tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    _model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    _model.eval()
    print("  GPT-2 loaded", flush=True)


def get_embedding(text, layer=11):
    import torch
    _load_model()
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        out = _model(**inputs)
    return out.hidden_states[layer][0, -1, :].numpy()


print('\n§1. Loading bank...')
with open(CACHE_FILE) as f:
    bank_data = json.load(f)
bank_embs = np.array([d['emb'] for d in bank_data])
bank_qs   = [d['q'] for d in bank_data]

from sklearn.decomposition import PCA
pca3d = PCA(n_components=3, random_state=42)
bank_3d = pca3d.fit_transform(bank_embs)

# バンクの代表スケール
all_intra = []
for i in range(len(bank_3d)):
    d = np.linalg.norm(bank_3d - bank_3d[i], axis=1)
    d[i] = np.inf
    all_intra.append(float(d.min()))
bank_intra = np.array(all_intra)
r_nn_med = float(np.median(bank_intra))

print(f'  Bank: {len(bank_3d)} facts')
print(f'  Bank NN: min={bank_intra.min():.3f}  med={r_nn_med:.3f}  '
      f'max={bank_intra.max():.3f}')


def gamma_score(q3d):
    """γ_H for a point in PCA3D space"""
    dists = np.linalg.norm(bank_3d - q3d, axis=1)
    d_min = float(dists.min())
    return float(1.0 - np.exp(-K_GAMMA * max(d_min - RTH_GAMMA, 0.0))), d_min


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. セマンティック変形シーケンス
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§2. Semantic morphing sequences...')
sequences = []
for seq in MORPH_SEQUENCES:
    print(f'\n  [{seq["title"]}]')
    items = []
    for i, prompt in enumerate(seq['prompts']):
        emb = get_embedding(prompt)
        q3d = pca3d.transform(emb[np.newaxis, :])[0]
        gam, d_min = gamma_score(q3d)
        items.append({
            'prompt': prompt,
            'q3d':    q3d,
            'd_min':  d_min,
            'gamma':  gam,
            'truth':  seq['truth'][i],
        })
        mark = '✓' if seq['truth'][i] else '✗'
        print(f'    {mark} d={d_min:6.3f} γ={gam:.3f}  "{prompt[:50]}"')
    sequences.append({'title': seq['title'], 'items': items})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. エンベディング空間でのリニア補間 (Phase 転移可視化)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§3. Linear interpolation in PCA3D...')

interpolations = []
for seq in sequences:
    items = seq['items']
    if len(items) < 2:
        continue
    e_start = items[0]['q3d']   # safe
    e_end   = items[-1]['q3d']  # risky

    # 線形補間
    alphas = np.linspace(0, 1, 50)
    traj_gammas = []
    traj_dmins  = []
    for alpha in alphas:
        e_interp = (1 - alpha) * e_start + alpha * e_end
        gam, d_min = gamma_score(e_interp)
        traj_gammas.append(gam)
        traj_dmins.append(d_min)

    # Phase 転移点: γ > 0.5 になる最初の alpha
    arr = np.array(traj_gammas)
    cross_idx = np.where(arr > 0.5)[0]
    alpha_star = float(alphas[cross_idx[0]]) if len(cross_idx) > 0 else 1.0

    interpolations.append({
        'title':    seq['title'],
        'alphas':   alphas,
        'gammas':   np.array(traj_gammas),
        'dmins':    np.array(traj_dmins),
        'alpha_star': alpha_star,
        'items':    items,
    })
    print(f'  [{seq["title"]}]')
    print(f'    α* (γ>0.5 threshold) = {alpha_star:.2f}')
    print(f'    d_min(α=0)={traj_dmins[0]:.3f}  d_min(α=1)={traj_dmins[-1]:.3f}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. 可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§4. Generating figures...')

n_seq = len(sequences)
fig, axes = plt.subplots(3, n_seq, figsize=(5 * n_seq, 13), facecolor='#0f172a')
if n_seq == 1:
    axes = axes.reshape(3, 1)

for ax in axes.flatten():
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=7)
    ax.grid(True, color='#334155', alpha=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')

TIER_CMAP = plt.cm.RdYlGn_r

# master curve
d_master = np.linspace(0, 25, 300)
g_master = 1.0 - np.exp(-K_GAMMA * np.maximum(d_master - RTH_GAMMA, 0.0))

for col, (seq_data, interp) in enumerate(zip(sequences, interpolations)):
    items = seq_data['items']

    # ── Row 0: γ vs step (semantic morphing) ─────────────────────
    ax = axes[0, col]
    xs = list(range(len(items)))
    gammas = [it['gamma'] for it in items]
    dmins  = [it['d_min'] for it in items]

    colors = [TIER_CMAP(g) for g in gammas]
    for j in range(len(xs) - 1):
        ax.plot(xs[j:j+2], gammas[j:j+2], '-', color='#6366f1',
                linewidth=2.0, alpha=0.8)
    ax.scatter(xs, gammas, c=gammas, cmap='RdYlGn_r',
               vmin=0, vmax=1, s=100, edgecolors='white', linewidths=0.8, zorder=5)

    ax.axhline(0.5, color='#f59e0b', linestyle='--', linewidth=1.2, alpha=0.7,
               label='γ=0.5 threshold')
    ax.set_xticks(xs)
    ax.set_xticklabels([it['prompt'][:20] + '…' for it in items],
                       rotation=20, ha='right', fontsize=5.5, color='#94a3b8')
    ax.set_ylabel('γ_H', color='#94a3b8')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(seq_data['title'] + '\nγ per prompt step',
                 color='#e2e8f0', fontsize=8)
    ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

    # ── Row 1: γ(α) linear interpolation ─────────────────────────
    ax = axes[1, col]
    alphas = interp['alphas']
    g_arr  = interp['gammas']
    ax.plot(alphas, g_arr, '-', color='#6366f1', linewidth=2.5, label='γ(α)')
    ax.axvline(interp['alpha_star'], color='#f59e0b', linestyle='--',
               linewidth=1.5, alpha=0.8,
               label=f'α*={interp["alpha_star"]:.2f}  (γ=0.5 cross)')
    ax.axhline(0.5, color='#94a3b8', linestyle=':', linewidth=1.0, alpha=0.6)
    ax.fill_between(alphas, g_arr, 0.5, where=(g_arr > 0.5),
                    color='#ef4444', alpha=0.15, label='High risk zone')
    ax.fill_between(alphas, g_arr, 0.5, where=(g_arr <= 0.5),
                    color='#10b981', alpha=0.15, label='Safe zone')
    ax.set_xlabel('α  (0=safe, 1=risky)', color='#94a3b8')
    ax.set_ylabel('γ(α)', color='#94a3b8')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Phase transition (linear interp)\n'
                 f'α*={interp["alpha_star"]:.2f}',
                 color='#e2e8f0', fontsize=8)
    ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8', loc='upper left')

    # ── Row 2: d_min(α) + master curve ───────────────────────────
    ax = axes[2, col]
    d_arr = interp['dmins']
    ax.plot(d_master, g_master, '-', color='white', linewidth=2.0,
            alpha=0.8, label=f'H(d)=1-exp(-k·max(d-r_th,0))')
    ax.axvline(RTH_GAMMA, color='#f59e0b', linestyle=':', linewidth=1.5,
               alpha=0.8, label=f'r_th={RTH_GAMMA}')
    # trajectory on master curve
    ax.scatter(d_arr, g_arr, c=alphas, cmap='plasma',
               s=40, alpha=0.8, zorder=5)
    ax.plot(d_arr, g_arr, '-', color='#a5b4fc', linewidth=1.2, alpha=0.6)

    # 意味的変形点
    for it in items:
        ax.scatter(it['d_min'], it['gamma'], c=['#10b981' if it['truth'] else '#ef4444'],
                   s=100, edgecolors='white', linewidths=1.2, zorder=7)
        ax.annotate(it['prompt'][:15] + '…', (it['d_min'], it['gamma']),
                    textcoords='offset points', xytext=(4, 3),
                    fontsize=4.5, color='#cbd5e1')

    ax.set_xlabel('d_min_3d', color='#94a3b8')
    ax.set_ylabel('γ_H', color='#94a3b8')
    ax.set_title('Trajectory on master curve\n'
                 '(plasma=α, ●=semantic morph)',
                 color='#e2e8f0', fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=5.5, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

plt.suptitle(r'Phase Transition: Safe → Risky  ·  '
             r'$\gamma(d)=1-e^{-k\,\max(d-r_{th},0)}$  ·  '
             r'$r_{th}=0.283,\,k=0.405$',
             color='#f1f5f9', fontsize=11, fontweight='bold')
plt.tight_layout()
out = '/home/yoiyoi/gamma_phase_transition.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close(fig)
print(f'Saved: {out}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. 位相転移サマリー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print('═' * 70)
print('  Phase Transition Summary')
print('═' * 70)
print(f'  統一γ式: H(d) = 1-exp(-{K_GAMMA}·max(d-{RTH_GAMMA},0))')
print(f'  転移条件: H(d) = 0.5 → d* = {RTH_GAMMA} - ln(0.5)/{K_GAMMA:.3f} = '
      f'{RTH_GAMMA + np.log(2)/K_GAMMA:.3f}')
print()
print(f'  {"Sequence":35s} {"α*":>5} {"d*(theory)":>10} {"d*(measured)":>12}')
print('  ' + '─' * 65)
d_star_theory = RTH_GAMMA + np.log(2) / K_GAMMA
for interp in interpolations:
    d_at_alpha_star = float(interp['dmins'][
        np.argmin(np.abs(interp['alphas'] - interp['alpha_star']))])
    print(f'  {interp["title"]:35s} {interp["alpha_star"]:>5.2f} '
          f'{d_star_theory:>10.3f} {d_at_alpha_star:>12.3f}')

print()
print(f'  理論値 d*(H=0.5) = r_th - ln(1-0.5)/k = {d_star_theory:.3f}')
print(f'  これが「知識境界」— この距離を超えると幻覚リスク > 50%')
print()
print(f'  LLM: k={K_GAMMA}, r_th={RTH_GAMMA}')
print(f'    → d* = {d_star_theory:.3f} (PCA3D 単位)')
print(f'    → 銀行内NN平均距離との比: d*/r_nn = {d_star_theory/r_nn_med:.3f}')
print('═' * 70)
