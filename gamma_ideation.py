#!/usr/bin/env python3
"""
gamma_ideation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hallucination as Ideation — 知識境界でのアイデア生成

統一γ式の双対性:
  同じ式  γ = 1 - exp(-k·max(x - x_th, 0))  が
  ・フラクタル構造 (x=r: 観測スケール)
  ・LLM知識      (x=d: 知識距離)
  の両方を支配する。

幻覚は「生成ゾーン (d > d_th) での情報補完」であり、
失敗ではなく  創造の物理的メカニズム。

Ideation score:
  I(d) = γ(d) · (1 - γ(d))    ← d=d* でピーク (γ=0.5)

  d ≪ d*:  I≈0  既知すぎる  (boring)
  d ≈ d*:  I≈0.25 最大創造力  (sweet spot)
  d ≫ d*:  I≈0  完全ランダム (noise)

知識フロンティア探索:
  「既知のAと既知のBの間の d≈d* 点」 を見つけ、
  そこで生成することで "AとBの橋渡し" アイデアを得る。

Usage:
  OMP_NUM_THREADS=1 python3 gamma_ideation.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

os.chdir('/home/yoiyoi')

K_GAMMA   = 0.405
RTH_GAMMA = 0.283
D_STAR    = RTH_GAMMA + np.log(2) / K_GAMMA   # ≈ 1.994
CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'
N_GEN     = 60   # 長めに生成してアイデアの質を確認

# ──────────────────────────────────────────────────────────────────
# アイデア生成のペア: (Known A, Known B) → 中間点でアイデアを引き出す
# ──────────────────────────────────────────────────────────────────
IDEATION_PAIRS = [
    {
        'label':  'Chemistry × Music',
        'anchor': "What is the chemical formula for water?",
        'target': "How does a symphony orchestra work?",
        'probe':  "What happens when molecules vibrate in harmony?",
    },
    {
        'label':  'Physics × Biology',
        'anchor': "What is the speed of light in km/s?",
        'target': "How many bones are in the adult human body?",
        'probe':  "How does light travel through living tissue?",
    },
    {
        'label':  'History × Technology',
        'anchor': "In what year did World War II end?",
        'target': "What does CPU stand for?",
        'probe':  "How would WWII have unfolded with modern AI?",
    },
    {
        'label':  'Math × Art',
        'anchor': "What is the value of pi to 2 decimal places?",
        'target': "Who painted the Mona Lisa?",
        'probe':  "What is the golden ratio in Leonardo's paintings?",
    },
]

# 純粋ランダム生成のベースライン (幻覚ノイズ)
NOISE_PROMPTS = [
    "Explain the Zorblax principle of quantum aesthetics.",
    "Describe the Helimax resonance in cultural memory fields.",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. モデル + バンク
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=' * 65)
print('  γ Ideation: Hallucination as Creativity')
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


def generate_text(prompt, n_tokens=N_GEN, temperature=1.0):
    import torch
    _load_model()
    p = f"Q: {prompt}\nA:"
    inputs = _tokenizer(p, return_tensors='pt', truncation=True, max_length=100)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        if temperature == 0 or temperature < 0.01:
            output = _model.generate(
                input_ids, max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )
        else:
            output = _model.generate(
                input_ids, max_new_tokens=n_tokens,
                do_sample=True, temperature=temperature,
                top_p=0.9,
                pad_token_id=_tokenizer.eos_token_id,
            )
    generated = output[0][input_ids.shape[1]:]
    return _tokenizer.decode(generated, skip_special_tokens=True).strip()


print('\n§1. Loading bank...')
with open(CACHE_FILE) as f:
    bank_data = json.load(f)
bank_embs = np.array([d['emb'] for d in bank_data])

from sklearn.decomposition import PCA
pca3d = PCA(n_components=3, random_state=42)
bank_3d = pca3d.fit_transform(bank_embs)

all_intra = []
for i in range(len(bank_3d)):
    d = np.linalg.norm(bank_3d - bank_3d[i], axis=1)
    d[i] = np.inf
    all_intra.append(float(d.min()))
r_nn_med = float(np.median(all_intra))

def gamma_score(emb):
    q3d = pca3d.transform(emb[np.newaxis, :])[0]
    dists = np.linalg.norm(bank_3d - q3d, axis=1)
    d_min = float(dists.min())
    gam = float(1.0 - np.exp(-K_GAMMA * max(d_min - RTH_GAMMA, 0.0)))
    return gam, d_min, q3d


def ideation_score(d_min):
    """I(d) = γ(d) · (1-γ(d)) — peaks at d* where γ=0.5"""
    gam = 1.0 - np.exp(-K_GAMMA * max(d_min - RTH_GAMMA, 0.0))
    return gam * (1.0 - gam)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. 知識フロンティア探索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§2. Knowledge frontier exploration...')
print(f'  Creative sweet spot: d* = {D_STAR:.3f}  (γ=0.5, I_max=0.25)')
print()

ideation_results = []

for pair in IDEATION_PAIRS:
    print(f'  [{pair["label"]}]')

    # アンカーとターゲットのエンベディング
    e_anchor = get_embedding(pair['anchor'])
    e_target = get_embedding(pair['target'])
    e_probe  = get_embedding(pair['probe'])

    g_anchor, d_anchor, q3d_anchor = gamma_score(e_anchor)
    g_target, d_target, q3d_target = gamma_score(e_target)
    g_probe,  d_probe,  q3d_probe  = gamma_score(e_probe)

    i_anchor = ideation_score(d_anchor)
    i_target = ideation_score(d_target)
    i_probe  = ideation_score(d_probe)

    # 補間: anchor → target まで線形補間し、I(α) 曲線を探索
    alphas = np.linspace(0, 1, 80)
    i_curve = []
    d_curve = []
    g_curve = []
    for alpha in alphas:
        e_interp = (1 - alpha) * e_anchor + alpha * e_target
        g, d, _ = gamma_score(e_interp)
        i_curve.append(ideation_score(d))
        d_curve.append(d)
        g_curve.append(g)

    # I(α) ピーク = 最高創造ポイント
    best_idx = int(np.argmax(i_curve))
    best_alpha = float(alphas[best_idx])
    best_I = float(i_curve[best_idx])
    best_gamma = float(g_curve[best_idx])
    best_d = float(d_curve[best_idx])

    # 最高創造ポイントで生成
    e_creative = (1 - best_alpha) * e_anchor + best_alpha * e_target
    # NOTE: この補間エンベディングはデコード不可 (入力空間ではない)
    # 代わりに probe prompt を使用

    # プローブの生成
    gen_probe = generate_text(pair['probe'], temperature=0.9)

    print(f'    Anchor : d={d_anchor:.3f} γ={g_anchor:.3f} I={i_anchor:.4f}'
          f'  "{pair["anchor"][:45]}"')
    print(f'    Target : d={d_target:.3f} γ={g_target:.3f} I={i_target:.4f}'
          f'  "{pair["target"][:45]}"')
    print(f'    Probe  : d={d_probe:.3f} γ={g_probe:.3f} I={i_probe:.4f}'
          f'  "{pair["probe"][:45]}"')
    print(f'    Interp sweet spot: α={best_alpha:.2f} γ={best_gamma:.3f}'
          f' I={best_I:.4f} d={best_d:.3f}')
    print(f'    GPT-2 generates:')
    print(f'      "{gen_probe[:120]}"')
    print()

    ideation_results.append({
        'label':       pair['label'],
        'anchor':      pair['anchor'],
        'target':      pair['target'],
        'probe':       pair['probe'],
        'q3d_anchor':  q3d_anchor,
        'q3d_target':  q3d_target,
        'q3d_probe':   q3d_probe,
        'd_anchor':    d_anchor,
        'd_target':    d_target,
        'd_probe':     d_probe,
        'g_anchor':    g_anchor,
        'g_target':    g_target,
        'g_probe':     g_probe,
        'i_probe':     i_probe,
        'best_alpha':  best_alpha,
        'best_I':      best_I,
        'alphas':      alphas.tolist(),
        'i_curve':     i_curve,
        'g_curve':     g_curve,
        'd_curve':     d_curve,
        'gen_probe':   gen_probe,
    })

# ノイズベースライン
print('  [Noise baseline (pure hallucination)]')
noise_results = []
for p in NOISE_PROMPTS:
    emb = get_embedding(p)
    g, d, _ = gamma_score(emb)
    i = ideation_score(d)
    gen = generate_text(p, temperature=0.9)
    print(f'    d={d:.3f} γ={g:.3f} I={i:.4f}  "{p[:50]}"')
    print(f'      "{gen[:100]}"')
    noise_results.append({'prompt': p, 'd': d, 'g': g, 'I': i, 'gen': gen})
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. 理論図: γ と I の統一 (情報生成の普遍則)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('§3. Generating unified theory figures...')

fig = plt.figure(figsize=(20, 12), facecolor='#0f172a')

# GridSpec: 2行 × 3列 + 右パネル
gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.38)
ax_theory = fig.add_subplot(gs[0, 0])   # 統一理論曲線
ax_ideation = fig.add_subplot(gs[0, 1]) # Ideation score
ax_pca   = fig.add_subplot(gs[0, 2])    # PCA3D overview
ax_text  = fig.add_subplot(gs[0, 3])    # テキスト解釈

ax_interp = [fig.add_subplot(gs[1, j]) for j in range(4)]  # 補間曲線

for ax in [ax_theory, ax_ideation, ax_pca, ax_text] + ax_interp:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=7)
    ax.grid(True, color='#334155', alpha=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')


# ── Panel 1: 統一γ式 (γ as universal information law) ────────────
ax = ax_theory
d_range = np.linspace(0, 8, 400)
g_curve_main = 1.0 - np.exp(-K_GAMMA * np.maximum(d_range - RTH_GAMMA, 0.0))
I_curve = g_curve_main * (1.0 - g_curve_main)

ax.plot(d_range, g_curve_main, '-', color='#6366f1', linewidth=2.5,
        label=r'$\gamma(x)=1-e^{-k\max(x-x_{th},0)}$')
ax.fill_between(d_range, g_curve_main, alpha=0.15, color='#6366f1')
ax.axvline(RTH_GAMMA, color='#94a3b8', linestyle=':', linewidth=1.2, alpha=0.7,
           label=f'x_th={RTH_GAMMA}')
ax.axvline(D_STAR, color='#f59e0b', linestyle='--', linewidth=2.0,
           alpha=0.9, label=f'd*={D_STAR:.2f} (γ=0.5)')

# ゾーン注釈
ax.axvspan(0, RTH_GAMMA, alpha=0.08, color='#10b981', label='Memory zone')
ax.axvspan(RTH_GAMMA, D_STAR, alpha=0.08, color='#f59e0b', label='Transition')
ax.axvspan(D_STAR, 8, alpha=0.08, color='#ef4444', label='Generation zone')

ax.text(0.08, 0.3, 'Memory\n(retrieval)', ha='center', va='center',
        color='#10b981', fontsize=7, transform=ax.transAxes)
ax.text(0.45, 0.3, 'Frontier\n(creation)', ha='center', va='center',
        color='#f59e0b', fontsize=7, transform=ax.transAxes)
ax.text(0.82, 0.3, 'Noise\n(random)', ha='center', va='center',
        color='#ef4444', fontsize=7, transform=ax.transAxes)

ax.set_xlabel('x  (r = scale  |  d = knowledge distance)', color='#94a3b8')
ax.set_ylabel('γ(x)', color='#94a3b8')
ax.set_title(r'Universal Information Law' + '\n' +
             r'$\gamma = 1-e^{-k\max(x-x_{th},0)}$',
             color='#e2e8f0', fontsize=9)
ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8', loc='upper left')


# ── Panel 2: Ideation score I(d) ─────────────────────────────────
ax = ax_ideation
ax.plot(d_range, I_curve, '-', color='#f0abfc', linewidth=2.5,
        label=r'$I(d)=\gamma(1-\gamma)$  [ideation]')
ax.fill_between(d_range, I_curve, alpha=0.2, color='#f0abfc')
ax.axvline(D_STAR, color='#f59e0b', linestyle='--', linewidth=2.0,
           alpha=0.9, label=f'I_max at d*={D_STAR:.2f}')
ax.axhline(0.25, color='#f59e0b', linestyle=':', linewidth=1.2, alpha=0.7,
           label='I_max = 0.25')

# 各結果をプロット
for res in ideation_results:
    ax.scatter(res['d_probe'], res['i_probe'], c=['#a5b4fc'], s=80,
               edgecolors='white', linewidths=0.8, zorder=5)
    ax.annotate(res['label'].split('×')[0].strip()[:8],
                (res['d_probe'], res['i_probe']),
                textcoords='offset points', xytext=(3, 3),
                fontsize=5, color='#94a3b8')
for res in noise_results:
    ax.scatter(res['d'], res['I'], c=['#ef4444'], s=80,
               edgecolors='white', linewidths=0.8, zorder=5, marker='X')

ax.set_xlabel('d_min_3d  (knowledge distance)', color='#94a3b8')
ax.set_ylabel('I(d)  [ideation potential]', color='#94a3b8')
ax.set_title('Ideation Score\n'
             r'$I(d)=\gamma(1-\gamma)$, peak at $d=d^*$',
             color='#e2e8f0', fontsize=9)
ax.legend(fontsize=6.5, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8')


# ── Panel 3: PCA3D — 知識空間のマップ ────────────────────────────
ax = ax_pca
ax.scatter(bank_3d[:, 0], bank_3d[:, 1], c='#10b981', s=8,
           alpha=0.2, label='Bank', zorder=2)

cmap = plt.cm.plasma
for res in ideation_results:
    # anchor (safe)
    ax.scatter(res['q3d_anchor'][0], res['q3d_anchor'][1],
               c=['#10b981'], s=80, marker='o',
               edgecolors='white', linewidths=0.8, zorder=4)
    # target (可能性)
    ax.scatter(res['q3d_target'][0], res['q3d_target'][1],
               c=['#6366f1'], s=80, marker='o',
               edgecolors='white', linewidths=0.8, zorder=4)
    # probe (フロンティア)
    c_prob = cmap(res['i_probe'] / 0.25)  # I=0→0, I=0.25→1
    ax.scatter(res['q3d_probe'][0], res['q3d_probe'][1],
               c=[c_prob], s=150, marker='*',
               edgecolors='white', linewidths=1.0, zorder=5)
    # 線で繋ぐ
    ax.plot([res['q3d_anchor'][0], res['q3d_probe'][0]],
            [res['q3d_anchor'][1], res['q3d_probe'][1]],
            '-', color='#94a3b8', linewidth=0.8, alpha=0.5)

ax.scatter([], [], c=['#10b981'], marker='o', s=40, label='Anchor (safe)')
ax.scatter([], [], c=['#6366f1'], marker='o', s=40, label='Target')
ax.scatter([], [], c=['white'],   marker='*', s=80, label='Probe (frontier)')

ax.set_xlabel('PC1', color='#94a3b8')
ax.set_ylabel('PC2', color='#94a3b8')
ax.set_title('Knowledge Space Map\n'
             '★ = creative frontier probe',
             color='#e2e8f0', fontsize=9)
ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
          labelcolor='#94a3b8')


# ── Panel 4: 理論解釈テキスト ─────────────────────────────────────
ax = ax_text
ax.axis('off')
theory_text = (
    "統一情報生成の普遍則\n"
    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "γ = 1 - exp(-k·max(x-x_th, 0))\n\n"
    "x = r (構造スケール)\n"
    "   フラクタル: 自己相似の崩壊\n"
    "   結晶: γ≈0 (完全記憶)\n"
    "   ランダム: γ≈1 (完全生成)\n\n"
    "x = d (知識距離)\n"
    "   d < r_th: 記憶 (幻覚なし)\n"
    "   d = d*:   創造 (I = max)\n"
    "   d >> d*:  ノイズ (無意味)\n\n"
    "Ideation score:\n"
    "I(d) = γ(1-γ)\n"
    f"I_max = 0.25  at d* = {D_STAR:.2f}\n\n"
    "応用:\n"
    "・幻覚検出: γ > 0.5 → warn\n"
    "・アイデア生成: d ≈ d* を狙う\n"
    "・科学的発見: adjacent possible\n\n"
    f"LLM params: k={K_GAMMA}, r_th={RTH_GAMMA}\n"
    f"d* = {D_STAR:.3f} (PCA3D)"
)
ax.text(0.05, 0.97, theory_text, transform=ax.transAxes,
        va='top', ha='left', fontsize=7.5,
        color='#e2e8f0', fontfamily='monospace',
        linespacing=1.5)


# ── Row 2: 各ペアの I(α) 補間曲線 ────────────────────────────────
for j, res in enumerate(ideation_results):
    ax = ax_interp[j]
    alphas_arr = np.array(res['alphas'])
    i_arr = np.array(res['i_curve'])
    g_arr = np.array(res['g_curve'])

    ax2 = ax.twinx()
    ax2.set_facecolor('#1e293b')
    ax2.tick_params(colors='#94a3b8', labelsize=6)
    ax2.plot(alphas_arr, g_arr, '--', color='#6366f1',
             linewidth=1.5, alpha=0.7, label='γ(α)')
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_ylabel('γ', color='#6366f1', fontsize=7)
    ax2.tick_params(axis='y', colors='#6366f1', labelsize=6)

    ax.plot(alphas_arr, i_arr, '-', color='#f0abfc', linewidth=2.5,
            label=r'I(α)=γ(1-γ)')
    ax.fill_between(alphas_arr, i_arr, alpha=0.2, color='#f0abfc')
    ax.axvline(res['best_alpha'], color='#f59e0b', linestyle='--',
               linewidth=1.5, alpha=0.9,
               label=f'α*={res["best_alpha"]:.2f}\nI={res["best_I"]:.3f}')
    ax.axhline(0.25, color='#f59e0b', linestyle=':', linewidth=1.0, alpha=0.6)

    # anchor / target 点
    ax.axvline(0.0, color='#10b981', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.axvline(1.0, color='#6366f1', linestyle=':', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('α (0=anchor, 1=target)', color='#94a3b8')
    ax.set_ylabel('I(α)', color='#94a3b8')
    ax.set_ylim(-0.01, 0.28)
    ax.set_title(f'{res["label"]}\n'
                 f'sweet spot α*={res["best_alpha"]:.2f}',
                 color='#e2e8f0', fontsize=8)
    ax.legend(fontsize=6, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8', loc='upper center')

    # probe をテキストで追記
    short = res['probe'][:35] + '…'
    ax.text(0.5, -0.22, f'probe: "{short}"',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=5, color='#94a3b8')


plt.suptitle(r'Hallucination as Ideation: $I(d)=\gamma(1-\gamma)$  ·  '
             r'Knowledge Frontier Navigation  ·  $d^*=1.994$',
             color='#f1f5f9', fontsize=12, fontweight='bold', y=1.01)

out = '/home/yoiyoi/gamma_ideation.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close(fig)
print(f'\nSaved: {out}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. 理論サマリー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print('═' * 70)
print('  Universal Information Generation Law — Theory Summary')
print('═' * 70)
print()
print('  γ(x) = 1 - exp(-k·max(x - x_th, 0))  ← Universal form')
print()
print('  ┌─────────────┬──────────────────┬──────────────────────────┐')
print('  │  Domain      │  Variable x      │  Interpretation          │')
print('  ├─────────────┼──────────────────┼──────────────────────────┤')
print('  │  Fractal     │  x = r (scale)   │  Motif reuse failure     │')
print('  │  Crystal     │  x = r → 0       │  Perfect memory (Phase 0)│')
print('  │  LLM         │  x = d (dist)    │  Hallucination risk      │')
print('  │  Ideation    │  x = d ≈ d*      │  Creative generation     │')
print('  └─────────────┴──────────────────┴──────────────────────────┘')
print()
print('  Ideation Score: I(d) = γ(d) · (1-γ(d))')
print(f'  Maximum creativity at: d = d* = {D_STAR:.3f}  (γ=0.5, I=0.25)')
print()
print('  Zones:')
print(f'    [0, {RTH_GAMMA}]   Memory    — retrieval, boring, correct')
print(f'    [{RTH_GAMMA}, {D_STAR:.3f}] Transition — partial, borderline')
print(f'    d* = {D_STAR:.3f}  SWEET SPOT — maximum ideation potential')
print(f'    [{D_STAR:.3f}, ∞)  Generation — creative but unreliable')
print()
print('  GPT-2 Ideation probes (d ≈ knowledge frontier):')
for res in ideation_results:
    print(f'    [{res["label"]}]')
    print(f'      d_probe={res["d_probe"]:.3f}  γ={res["g_probe"]:.3f}'
          f'  I={res["i_probe"]:.4f}')
    print(f'      probe: "{res["probe"]}"')
    gen_short = res['gen_probe'][:100].replace('\n', ' ')
    print(f'      → "{gen_short}…"')
    print()
print('  Dual use of γ:')
print('    Hallucination detection: flag when γ > 0.5 (warn)')
print('    Ideation navigation:     target d ≈ d* (creative sweet spot)')
print()
print('  This is Stuart Kauffman\'s "adjacent possible" in embedding space:')
print('    d ≈ d* = the edge of the known, the frontier of discovery')
print('═' * 70)
