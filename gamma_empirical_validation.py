#!/usr/bin/env python3
"""
gamma_empirical_validation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Empirical Validation: γ Score vs Actual GPT-2 Hallucination

各プロンプトに対して:
  1. γ_H スコアを測定 (エンベディング距離ベース)
  2. GPT-2 の実際の答えを生成
  3. 答えの正確さを自動評価
  4. γ_H と実際の幻覚の相関を検証

評価指標:
  - AUC (ROC curve)
  - Brier score
  - 転移点 d* での精度

Usage:
  OMP_NUM_THREADS=1 python3 gamma_empirical_validation.py
"""

import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/home/yoiyoi')

K_GAMMA   = 0.405
RTH_GAMMA = 0.283
D_STAR    = RTH_GAMMA + np.log(2) / K_GAMMA  # ≈ 1.994
CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'
N_GEN = 30  # 生成トークン数

# テストセット: (prompt, expected_answer_keywords, truth_label)
# truth_label: True = GPT-2 が知っているはず, False = 幻覚しやすい
TEST_PROMPTS = [
    # ─── バンク内 (d_min=0) ───
    ("What is the capital of France?",            ["Paris"],                       True),
    ("Who discovered penicillin?",                ["Fleming", "Alexander"],        True),
    ("In what year did World War II end?",        ["1945"],                        True),
    ("What is the chemical symbol for gold?",     ["Au"],                          True),
    ("What planet is closest to the Sun?",        ["Mercury"],                     True),
    ("What is the boiling point of water in Celsius?", ["100"],                   True),
    ("What is the largest ocean on Earth?",       ["Pacific"],                     True),
    ("What is the atomic number of hydrogen?",    ["1"],                           True),
    # ─── 実在するが難しい (d_min > 0) ───
    ("In what year did the Korean War end?",       ["1953"],                       True),
    ("What is the capital of Bhutan?",             ["Thimphu"],                   True),
    ("Who invented the World Wide Web?",           ["Berners-Lee", "Tim"],        True),
    ("What is the speed of sound in air (m/s)?",   ["343", "340", "344"],         True),
    # ─── ボーダーライン (実在するが GPT-2 が知らない可能性) ───
    ("In what year did the Byzantine Empire fall?", ["1453"],                     True),
    ("What is the half-life of carbon-14 (years)?", ["5730", "5700", "5568"],    True),
    ("Who wrote Crime and Punishment?",             ["Dostoevsky", "Dostoyevsky"], True),
    # ─── 幻覚プロンプト (架空) ───
    ("What is the capital of New Franconia?",       [],                           False),
    ("Who discovered the Kalderon field?",          [],                           False),
    ("In what year did the Martian War end?",        [],                           False),
    ("What is the boiling point of unobtainium?",   [],                           False),
    ("What was Einstein's 1931 quantum biology paper?", [],                       False),
    ("What is Newton's theory of emotional gravity?",   [],                       False),
    ("Describe the grammar of the Elvish language Sindarin.", [],                 False),
    ("Explain the Zorblax paradox in astrophysics.", [],                          False),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. データ準備
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('=' * 65)
print('  γ Empirical Validation: γ Score vs Actual Hallucination')
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


def generate_answer(prompt, n_tokens=N_GEN):
    """GPT-2 で Q→A 形式で答えを生成"""
    import torch
    _load_model()
    p = f"Q: {prompt}\nA:"
    inputs = _tokenizer(p, return_tensors='pt', truncation=True, max_length=100)
    input_ids = inputs['input_ids']
    with torch.no_grad():
        output = _model.generate(
            input_ids,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    generated = output[0][input_ids.shape[1]:]
    text = _tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def check_answer(generated, expected_keywords):
    """答えが expected_keywords のいずれかを含むか"""
    if not expected_keywords:
        return None  # 幻覚プロンプトは評価不可
    gen_lower = generated.lower()
    return any(kw.lower() in gen_lower for kw in expected_keywords)


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
    return gam, d_min


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. 全テストプロンプトの評価
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§2. Evaluating all test prompts...')
print(f'  {"Prompt":45s} {"d":>6} {"γ":>6} {"gen":>6} {"hallu?"}')
print('  ' + '─' * 80)

results = []
for prompt, keywords, truth in TEST_PROMPTS:
    emb  = get_embedding(prompt)
    gam, d_min = gamma_score(emb)
    ans  = generate_answer(prompt)
    correct = check_answer(ans, keywords)

    # 幻覚ラベル: truth=True → hallucination=False (should be correct)
    #             truth=False → hallucination=True (will hallucinate)
    # 実測幻覚: correct=False かつ truth=True → 幻覚 (予期せず間違い)
    #           correct=True かつ truth=True → 正解
    #           truth=False → 幻覚とみなす (答え不能)
    if truth:
        is_hallu = (correct == False)  # 実在する事実を間違えた
        hallu_label = 1 if is_hallu else 0
    else:
        is_hallu = True  # 架空質問は必ず幻覚扱い
        hallu_label = 1

    # 答えをきれいに表示
    ans_short = ans[:20].replace('\n', ' ')
    if truth:
        check_str = f'{"✓" if correct else "✗"} ({ans_short})'
    else:
        check_str = f'[FICTIONAL] ({ans_short})'
    hallu_str = 'HALLU' if is_hallu else 'OK'

    print(f'  {prompt[:44]:44s} {d_min:6.3f} {gam:6.3f} {hallu_str:>6}  {check_str[:40]}')

    results.append({
        'prompt':    prompt,
        'd_min':     d_min,
        'gamma':     gam,
        'truth':     truth,
        'correct':   correct,
        'hallu':     hallu_label,
        'answer':    ans,
        'keywords':  keywords,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. AUC 計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§3. Computing AUC...')

gammas = np.array([r['gamma'] for r in results])
hallus = np.array([r['hallu'] for r in results])
dmins  = np.array([r['d_min'] for r in results])

# ROC curve (手動計算)
thresholds = np.linspace(0, 1, 200)
tpr_list, fpr_list = [], []
for thr in thresholds:
    pred = (gammas >= thr).astype(int)
    tp = np.sum((pred == 1) & (hallus == 1))
    fp = np.sum((pred == 1) & (hallus == 0))
    tn = np.sum((pred == 0) & (hallus == 0))
    fn = np.sum((pred == 0) & (hallus == 1))
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    tpr_list.append(tpr)
    fpr_list.append(fpr)

tpr_arr = np.array(tpr_list)
fpr_arr = np.array(fpr_list)
# AUC (台形公式)
auc = float(-np.trapz(tpr_arr, fpr_arr))
print(f'  γ_H AUC = {auc:.4f}')

# Brier score
brier = float(np.mean((gammas - hallus) ** 2))
print(f'  Brier score = {brier:.4f}')

# 最適閾値 (Youden's J)
j_scores = tpr_arr - fpr_arr
opt_idx = np.argmax(j_scores)
opt_thr = thresholds[opt_idx]
print(f'  Optimal threshold = {opt_thr:.3f}  (Youden J={j_scores[opt_idx]:.3f})')
print(f'  Theory threshold (H=0.5) = 0.500  '
      f'(d* = {D_STAR:.3f})')

# d* での性能
pred_dstar = (gammas >= 0.5).astype(int)
tp = np.sum((pred_dstar == 1) & (hallus == 1))
fp = np.sum((pred_dstar == 1) & (hallus == 0))
tn = np.sum((pred_dstar == 0) & (hallus == 0))
fn = np.sum((pred_dstar == 0) & (hallus == 1))
precision = tp / (tp + fp + 1e-9)
recall    = tp / (tp + fn + 1e-9)
f1        = 2 * precision * recall / (precision + recall + 1e-9)
acc       = (tp + tn) / len(results)
print(f'\n  γ≥0.5 threshold:')
print(f'    Accuracy  = {acc:.3f}')
print(f'    Precision = {precision:.3f}')
print(f'    Recall    = {recall:.3f}')
print(f'    F1        = {f1:.3f}')
print(f'    TP={tp} FP={fp} TN={tn} FN={fn}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. 可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n§4. Generating figures...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0f172a')
for ax in axes:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#334155', alpha=0.4)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')


# ── Plot 1: ROC curve ─────────────────────────────────────────────
ax = axes[0]
ax.plot(fpr_arr, tpr_arr, '-', color='#6366f1', linewidth=2.5,
        label=f'γ score (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='#475569', linewidth=1.2, alpha=0.7,
        label='Random (AUC=0.5)')
ax.scatter([fpr_arr[opt_idx]], [tpr_arr[opt_idx]],
           c=['#f59e0b'], s=150, edgecolors='white', linewidths=1.5, zorder=5,
           label=f'Optimal thr={opt_thr:.2f}')
ax.set_xlabel('False Positive Rate', color='#94a3b8')
ax.set_ylabel('True Positive Rate', color='#94a3b8')
ax.set_title(f'ROC Curve: γ Hallucination Score\nAUC={auc:.4f}',
             color='#e2e8f0', fontsize=9)
ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')


# ── Plot 2: γ score distribution ─────────────────────────────────
ax = axes[1]
ok_gammas  = gammas[hallus == 0]
hal_gammas = gammas[hallus == 1]
bins = np.linspace(0, 1, 25)
ax.hist(ok_gammas,  bins=bins, color='#10b981', alpha=0.7, label='No hallucination')
ax.hist(hal_gammas, bins=bins, color='#ef4444', alpha=0.7, label='Hallucination')
ax.axvline(0.5, color='#f59e0b', linestyle='--', linewidth=2.0, alpha=0.9,
           label=f'd* threshold (γ=0.5)')
ax.axvline(opt_thr, color='#a5b4fc', linestyle=':', linewidth=1.5, alpha=0.8,
           label=f'Optimal thr={opt_thr:.2f}')
ax.set_xlabel('γ_H score', color='#94a3b8')
ax.set_ylabel('Count', color='#94a3b8')
ax.set_title('γ Score Distribution\nHallucination vs No-Hallucination',
             color='#e2e8f0', fontsize=9)
ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')


# ── Plot 3: d_min vs hallucination ───────────────────────────────
ax = axes[2]
d_range = np.linspace(0, 25, 300)
g_model = 1.0 - np.exp(-K_GAMMA * np.maximum(d_range - RTH_GAMMA, 0.0))
ax.plot(d_range, g_model, '-', color='white', linewidth=2.5, alpha=0.9,
        label='H(d) master curve')
ax.axvline(D_STAR, color='#f59e0b', linestyle='--', linewidth=1.5,
           alpha=0.9, label=f'd*={D_STAR:.2f} (H=0.5)')
ax.axhline(0.5, color='#94a3b8', linestyle=':', linewidth=1.0, alpha=0.6)

jitter = np.random.RandomState(42).uniform(-0.03, 0.03, len(results))
for r, j in zip(results, jitter):
    c = '#ef4444' if r['hallu'] else '#10b981'
    marker = 'X' if r['hallu'] else 'o'
    ax.scatter(r['d_min'], r['gamma'] + j, c=[c], s=60,
               marker=marker, edgecolors='white', linewidths=0.5, zorder=5, alpha=0.85)

ax.scatter([], [], c=['#10b981'], marker='o', s=50, label='Correct')
ax.scatter([], [], c=['#ef4444'], marker='X', s=50, label='Hallucination')
ax.set_xlabel('d_min_3d  (PCA3D distance)', color='#94a3b8')
ax.set_ylabel('γ_H', color='#94a3b8')
ax.set_title(f'γ vs d_min: Empirical Results\n'
             f'Knowledge boundary at d*={D_STAR:.2f}',
             color='#e2e8f0', fontsize=9)
ax.set_ylim(-0.12, 1.15)
ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')


plt.suptitle(r'Empirical Validation: $\gamma_H$ Predicts GPT-2 Hallucination  ·  '
             f'AUC={auc:.3f}  ·  F1={f1:.3f}',
             color='#f1f5f9', fontsize=11, fontweight='bold')
plt.tight_layout()
out = '/home/yoiyoi/gamma_empirical_validation.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close(fig)
print(f'Saved: {out}')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. 詳細結果テーブル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print('═' * 75)
print('  Empirical Validation Results')
print('═' * 75)
print(f'  {"Prompt":42s} {"d":>6} {"γ":>6} {"Hallu":>6} {"Pred":>5} {"✓?":>3}')
print('  ' + '─' * 72)
for r in results:
    pred = 'H' if r['gamma'] >= 0.5 else 'OK'
    correct = (pred == 'H') == bool(r['hallu'])
    mark = '✓' if correct else '✗'
    print(f'  {r["prompt"][:41]:41s} {r["d_min"]:6.3f} {r["gamma"]:6.3f} '
          f'{"H" if r["hallu"] else "OK":>6} {pred:>5} {mark:>3}')

wrong = [r for r in results
         if (r['gamma'] >= 0.5) != bool(r['hallu'])]
print()
print(f'  Accuracy: {(1-len(wrong)/len(results))*100:.1f}%  '
      f'({len(results)-len(wrong)}/{len(results)} correct)')
print(f'  AUC={auc:.4f}  F1={f1:.4f}  Brier={brier:.4f}')
print()
if wrong:
    print('  誤分類ケース:')
    for r in wrong:
        pred = 'H' if r['gamma'] >= 0.5 else 'OK'
        print(f'    γ={r["gamma"]:.3f} → pred={pred}  true={"H" if r["hallu"] else "OK"}'
              f'  "{r["prompt"][:50]}"')
print('═' * 75)
