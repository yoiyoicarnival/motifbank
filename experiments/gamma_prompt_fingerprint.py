#!/usr/bin/env python3
"""
gamma_prompt_fingerprint.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Radial Coverage Fingerprint — プロンプトカテゴリ別 Phase 分類

LLMへのγ理論の正しい適用:

  A(r) = fraction of ReasonBank within radius r from query (PCA3D)
  ← クエリを中心とした「知識カバレッジ曲線」

  Phase 0 (安全): A(0)>0, A(r_small)=1  ← クエリがバンク内
  Phase 1 (注意): A(r_small)=0, A(r_large)→1  ← バンクの端
  Phase 2 (リスク): A(r) ゆっくり上昇
  Phase 3 (危険): A(r) ≈ 0 for all r  ← バンクから遠い

  γ_A = 1 - A(r_th)  ← r_th での非カバレッジ率 ≈ Phase度

統一γスコアとの接続:
  H(d_min) = 1 - exp(-k·max(d_min-r_th, 0))
  大域: γ(r,d) = 1 - exp(-k·max(r+d-r_th, 0))

Usage:
  OMP_NUM_THREADS=1 python3 gamma_prompt_fingerprint.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/home/yoiyoi')

K_GAMMA   = 0.405
RTH_GAMMA = 0.283
CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'

PROMPTS = {
    'safe': [
        "What is the capital of France?",
        "Who discovered penicillin?",
        "In what year did World War II end?",
        "What is the chemical symbol for gold?",
        "What planet is closest to the Sun?",
    ],
    'borderline': [
        "What was Einstein's 1931 quantum biology paper?",
        "What is the cuisine of the lost city of Atlantis?",
        "What is Newton's theory of emotional gravity?",
        "Describe Marie Curie's discovery of element 120.",
        "What year did Shakespeare invent the internet?",
    ],
    'risky': [
        "Explain the Voynich manuscript's linguistic structure.",
        "Describe the grammar of the Elvish language Sindarin.",
        "Describe a Shakespeare play about Mars colonization.",
        "What are the laws of Klingon quantum mechanics?",
        "Explain the Zorblax paradox in astrophysics.",
    ],
}

TIER_COLOR = {'safe': '#10b981', 'borderline': '#f59e0b', 'risky': '#ef4444'}
TIER_LABEL = {
    'safe':       'Safe (Phase 0: in bank)',
    'borderline': 'Borderline (Phase 1-2)',
    'risky':      'Risky (Phase 3: far from bank)',
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. データ準備
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=' * 65)
print('  γ Radial Coverage Fingerprint')
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
    print(f"  GPT-2 loaded", flush=True)


def get_embedding(text, layer=11):
    import torch
    _load_model()
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        out = _model(**inputs)
    return out.hidden_states[layer][0, -1, :].numpy()


print('\n§1. Loading bank and building PCA3D...')
with open(CACHE_FILE) as f:
    bank_data = json.load(f)
bank_embs = np.array([d['emb'] for d in bank_data])

from sklearn.decomposition import PCA
pca3d = PCA(n_components=3, random_state=42)
bank_3d = pca3d.fit_transform(bank_embs)
print(f'  Bank: {len(bank_3d)} facts  explained={pca3d.explained_variance_ratio_.sum():.3f}')

# バンク内の典型スケール
all_intra = []
for i in range(len(bank_3d)):
    d = np.linalg.norm(bank_3d - bank_3d[i], axis=1)
    d[i] = np.inf
    all_intra.append(float(d.min()))
bank_intra = np.array(all_intra)
r_nn_med = float(np.median(bank_intra))  # 典型的な最近傍距離
r_max_plot = float(np.percentile(bank_intra, 90)) * 3

print(f'  Bank intra-NN: min={bank_intra.min():.3f}  '
      f'med={r_nn_med:.3f}  p90={np.percentile(bank_intra,90):.3f}')
print(f'  r_th={RTH_GAMMA}  r_plot_max={r_max_plot:.1f}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. ラジアルカバレッジ A(r) の計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§2. Computing radial coverage A(r)...')

r_vals = np.geomspace(0.01, r_max_plot, 60)


def coverage_curve(query_3d, bank_3d, r_vals):
    """A(r) = fraction of bank within radius r of query"""
    dists = np.linalg.norm(bank_3d - query_3d, axis=1)
    return np.array([float(np.mean(dists < r)) for r in r_vals])


def phase_from_coverage(A_arr, r_vals, r_th=RTH_GAMMA):
    """A(r) 曲線から Phase を分類"""
    # r_th でのカバレッジ
    r_th_idx = np.searchsorted(r_vals, r_th)
    r_th_idx = min(r_th_idx, len(A_arr) - 1)
    A_rth = A_arr[r_th_idx]

    # r_max でのカバレッジ
    A_max = A_arr[-1]

    if A_arr[0] > 0:           # 最小 r でバンクヒット
        return 0, 'Phase 0 (in-bank)'
    elif A_rth > 0.5:
        return 1, 'Phase 1 (near bank)'
    elif A_max > 0.3:
        return 2, 'Phase 2 (moderate OOD)'
    else:
        return 3, 'Phase 3 (far OOD)'


# バンク自身の平均 A(r) (leave-one-out)
print('  Computing bank self-coverage (leave-one-out)...')
bank_A = []
for i in range(0, min(len(bank_3d), 50), 5):  # サンプル 10 点
    others = np.delete(bank_3d, i, axis=0)
    A = coverage_curve(bank_3d[i], others, r_vals)
    bank_A.append(A)
bank_A_mean = np.mean(bank_A, axis=0)

# 各プロンプトの A(r)
print('  Computing per-prompt coverage curves...', flush=True)
fingerprints = {}
for tier, prompt_list in PROMPTS.items():
    fingerprints[tier] = []
    for prompt in prompt_list:
        emb = get_embedding(prompt)
        q3d = pca3d.transform(emb[np.newaxis, :])[0]
        dists = np.linalg.norm(bank_3d - q3d, axis=1)
        d_min = float(dists.min())
        A = coverage_curve(q3d, bank_3d, r_vals)
        ph, ph_str = phase_from_coverage(A, r_vals)
        gamma_h = float(1.0 - np.exp(-K_GAMMA * max(d_min - RTH_GAMMA, 0.0)))

        fingerprints[tier].append({
            'prompt':  prompt,
            'd_min':   d_min,
            'A':       A,
            'phase':   ph,
            'ph_str':  ph_str,
            'gamma_h': gamma_h,
        })
        print(f'    [{tier:12s}] d={d_min:6.3f} γ={gamma_h:.3f} {ph_str:25s} '
              f'A(r_th)={float(A[np.searchsorted(r_vals, RTH_GAMMA)]):.3f}  '
              f'"{prompt[:40]}"')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. γ(r) = 1 - A(r) (非カバレッジ率) の確認
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n── Phase 分類サマリー ──')
print(f'  {"Tier":12s} {"Phase":25s} {"d_min":>7} {"γ_H":>7}  Prompt')
print('  ' + '─' * 80)
for tier, fps in fingerprints.items():
    for fp in fps:
        print(f'  {tier:12s} {fp["ph_str"]:25s} {fp["d_min"]:7.3f} {fp["gamma_h"]:7.3f}'
              f'  "{fp["prompt"][:40]}"')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. 可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§4. Generating figures...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0f172a')
axes_flat = axes.flatten()
for ax in axes_flat:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#334155', alpha=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')


# ── Plot 1: A(r) tier 平均 ────────────────────────────────────────
ax = axes_flat[0]
ax.plot(r_vals, bank_A_mean, '--', color='white', linewidth=2.0,
        alpha=0.8, label='Bank self A(r) (leave-one-out)')
ax.axvline(RTH_GAMMA, color='#f59e0b', linestyle=':', linewidth=1.5,
           alpha=0.8, label=f'r_th={RTH_GAMMA}')
ax.axvline(r_nn_med, color='#94a3b8', linestyle=':', linewidth=1.0,
           alpha=0.6, label=f'bank NN med={r_nn_med:.2f}')

for tier, fps in fingerprints.items():
    c = TIER_COLOR[tier]
    all_A = np.array([fp['A'] for fp in fps])
    mean_A = all_A.mean(axis=0)
    std_A  = all_A.std(axis=0)
    ax.plot(r_vals, mean_A, '-', color=c, linewidth=2.5, label=TIER_LABEL[tier])
    ax.fill_between(r_vals, mean_A - std_A, mean_A + std_A, color=c, alpha=0.15)
    for fp in fps:
        ax.plot(r_vals, fp['A'], '-', color=c, linewidth=0.6, alpha=0.35)

ax.set_xscale('log')
ax.set_xlabel('r  (PCA3D radius, log scale)', color='#94a3b8')
ax.set_ylabel('A(r) = fraction of bank within r', color='#94a3b8')
ax.set_title('Radial Coverage A(r) per Tier\n'
             'Safe=in bank A(0)>0, Risky=A(r)≈0',
             color='#e2e8f0', fontsize=9)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=6.5, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8')


# ── Plot 2: γ(r) = 1 - A(r) ─────────────────────────────────────
ax = axes_flat[1]
for tier, fps in fingerprints.items():
    c = TIER_COLOR[tier]
    all_g = np.array([1.0 - fp['A'] for fp in fps])
    mean_g = all_g.mean(axis=0)
    std_g  = all_g.std(axis=0)
    ax.plot(r_vals, mean_g, '-', color=c, linewidth=2.5, label=TIER_LABEL[tier])
    ax.fill_between(r_vals, mean_g - std_g, mean_g + std_g, color=c, alpha=0.15)

# master curve 1-exp(-k*(r+d_min-r_th)) at three d_min values
r_smooth = np.geomspace(0.01, r_max_plot, 300)
for d_val, style, lab in [(0.0, '-', 'd=0 (safe)'),
                           (r_nn_med, '--', f'd={r_nn_med:.1f} (borderline)'),
                           (r_nn_med*3, ':', f'd={r_nn_med*3:.1f} (risky)')]:
    g_model = 1.0 - np.exp(-K_GAMMA * np.maximum(r_smooth + d_val - RTH_GAMMA, 0.0))
    ax.plot(r_smooth, g_model, style, color='white', linewidth=1.5,
            alpha=0.7, label=f'γ model {lab}')

ax.axvline(RTH_GAMMA, color='#f59e0b', linestyle=':', linewidth=1.5, alpha=0.8)
ax.set_xscale('log')
ax.set_xlabel('r  (radius)', color='#94a3b8')
ax.set_ylabel('γ_A(r) = 1 - A(r)', color='#94a3b8')
ax.set_title(r'γ_A(r) = 1-A(r)  vs  $\gamma(r,d)=1-e^{-k\max(r+d-r_{th},0)}$',
             color='#e2e8f0', fontsize=9)
ax.set_ylim(-0.02, 1.1)
ax.legend(fontsize=5.5, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8')


# ── Plot 3: d_min histogram per tier ─────────────────────────────
ax = axes_flat[2]
for tier, fps in fingerprints.items():
    dmins = [fp['d_min'] for fp in fps]
    c = TIER_COLOR[tier]
    ax.scatter(dmins, [tier] * len(dmins), c=[c], s=120,
               edgecolors='white', linewidths=1.0, zorder=5)
    for i, fp in enumerate(fps):
        ax.annotate(fp['prompt'][:14] + '…', (fp['d_min'], tier),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=4.5, color='#94a3b8')

ax.axvline(RTH_GAMMA, color='#f59e0b', linestyle=':', linewidth=1.5,
           alpha=0.9, label=f'r_th={RTH_GAMMA}')
ax.axvline(r_nn_med, color='#94a3b8', linestyle='--', linewidth=1.0,
           alpha=0.7, label=f'bank NN med={r_nn_med:.2f}')

ax.set_xlabel('d_min_3d  (distance to nearest bank in PCA3D)', color='#94a3b8')
ax.set_title('d_min_3d per Tier\n'
             'safe=0, borderline/risky>0',
             color='#e2e8f0', fontsize=9)
ax.tick_params(axis='y', labelcolor='#94a3b8')
ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8')


# ── Plots 4-6: per-tier individual A(r) ──────────────────────────
for i, (tier, fps) in enumerate(fingerprints.items()):
    ax = axes_flat[3 + i]
    c  = TIER_COLOR[tier]

    ax.plot(r_vals, bank_A_mean, '--', color='white', linewidth=1.5,
            alpha=0.6, label='Bank self (ref)')
    ax.axvline(RTH_GAMMA, color='#f59e0b', linestyle=':', linewidth=1.2,
               alpha=0.7, label=f'r_th={RTH_GAMMA}')

    for fp in fps:
        ax.plot(r_vals, fp['A'], '-', color=c, linewidth=1.8, alpha=0.8,
                label=f'[P{fp["phase"]}] ' + fp['prompt'][:25] + '…')

    phase_counts = {}
    for fp in fps:
        phase_counts[fp['phase']] = phase_counts.get(fp['phase'], 0) + 1
    phase_str = '  '.join(f'P{p}×{n}' for p, n in sorted(phase_counts.items()))

    ax.set_xscale('log')
    ax.set_xlabel('r', color='#94a3b8')
    ax.set_ylabel('A(r)', color='#94a3b8')
    ax.set_title(f'[{tier.upper()}] Coverage A(r)\nPhases: {phase_str}',
                 color='#e2e8f0', fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=5, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8', loc='lower right')


plt.suptitle(r'Radial Coverage Fingerprint  ·  $A(r) = $fraction of bank within $r$  '
             r'·  $\gamma_A(r)=1-A(r)$',
             color='#f1f5f9', fontsize=11, fontweight='bold')
plt.tight_layout()
out = '/home/yoiyoi/gamma_prompt_fingerprint.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close(fig)
print(f'Saved: {out}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. 統計サマリー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print('═' * 70)
print('  Radial Coverage Fingerprint Summary')
print('═' * 70)

# r_th でのカバレッジで tier 判別力を測る
rth_idx = int(np.searchsorted(r_vals, RTH_GAMMA))
rth_idx = min(rth_idx, len(r_vals) - 1)

print(f'  r_th = {RTH_GAMMA}  →  r_vals[{rth_idx}] = {r_vals[rth_idx]:.4f}')
print()
print(f'  {"Tier":12s} {"d_min mean":>11} {"A(r_th) mean":>13} '
      f'{"γ_H mean":>9} {"Phase mode":>11}')
print('  ' + '─' * 60)
for tier, fps in fingerprints.items():
    dmins   = [fp['d_min'] for fp in fps]
    a_rths  = [fp['A'][rth_idx] for fp in fps]
    gammas  = [fp['gamma_h'] for fp in fps]
    phases  = [fp['phase'] for fp in fps]
    mode_p  = max(set(phases), key=phases.count)
    print(f'  {tier:12s} {np.mean(dmins):11.3f} {np.mean(a_rths):13.4f} '
          f'{np.mean(gammas):9.3f}  Phase {mode_p}')

print()

# 判別力: safe vs (borderline+risky) のγ_H の差
safe_g   = [fp['gamma_h'] for fp in fingerprints['safe']]
risky_g  = ([fp['gamma_h'] for fp in fingerprints['risky']]
           + [fp['gamma_h'] for fp in fingerprints['borderline']])
print(f'  Discrimination (γ_H):')
print(f'    safe  : mean={np.mean(safe_g):.4f}  std={np.std(safe_g):.4f}')
print(f'    others: mean={np.mean(risky_g):.4f}  std={np.std(risky_g):.4f}')
sep = (np.mean(risky_g) - np.mean(safe_g)) / (
      np.std(risky_g) + np.std(safe_g) + 1e-9)
print(f'    separation (Cohen d proxy) = {sep:.3f}')
print()
print('  LLM Phase Analog:')
print('  γ_A(r) = 1 - A(r) = fraction of bank NOT within radius r of query')
print('  → Phase 0: A(r_small)=1  (query in bank)  γ_A≈0  Safe')
print('  → Phase 3: A(r)≈0        (query far OOD)  γ_A≈1  Risky')
print('═' * 70)
