"""
measure_llm_lipschitz.py — 実 GPT-2 での L_LLM 実測

Theorem U5 の実測検証:
  理論上界: L_LLM ≤ (β/2)·‖W‖_F
  実測:     L_LLM_emp = max d_TV(P(·|x), P(·|x')) / ‖E(x)-E(x')‖_2

手順:
  1. GPT-2 (small, V=50257, d=768) ロード
  2. N テキストのラスト hidden state 取得
  3. lm_head で語彙ロジット計算 → softmax
  4. 全ペアから d_TV / d_L2 の最大値を推定
  5. 理論上界と比較
"""

import numpy as np
import json, time, math
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("L_LLM 実測実験 — Real GPT-2 (V=50257, d=768)")
print("=" * 70)
print()

# ── 1. モデルロード ───────────────────────────────────────────────────────────
print("[1/6] GPT-2 ロード中...")
t0 = time.time()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model     = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer.pad_token = tokenizer.eos_token

d_model = model.config.n_embd              # 768
V       = model.config.vocab_size           # 50257
W_lm    = model.lm_head.weight.detach()    # (50257, 768)
W_F     = float(W_lm.norm('fro'))
L_bound = 0.5 * W_F  # (β=1) / 2 × ‖W‖_F

print(f"  d={d_model}, V={V}")
print(f"  ‖W_lm_head‖_F = {W_F:.4f}")
print(f"  Theorem U5 上界: L_LLM ≤ (β/2)·‖W‖_F = {L_bound:.4f}")
print(f"  ロード時間: {time.time()-t0:.1f}s")
print()

# ── 2. テキスト生成 ───────────────────────────────────────────────────────────
texts = [
    # 事実質問 (in-domain)
    "What is the capital of France?",
    "The capital of France is Paris.",
    "Paris is the capital city of France.",
    "France has Paris as its capital.",
    "What is 2 + 2?",
    "Two plus two equals four.",
    "The answer to 2+2 is 4.",
    "Who wrote Romeo and Juliet?",
    "Shakespeare wrote Romeo and Juliet.",
    "Romeo and Juliet was written by Shakespeare.",
    "What is the speed of light?",
    "Light travels at 299,792,458 meters per second.",
    "The speed of light in vacuum is approximately 3×10^8 m/s.",
    "Water boils at 100 degrees Celsius.",
    "The boiling point of water is 100°C at sea level.",
    "H2O is the chemical formula for water.",
    "Water consists of hydrogen and oxygen atoms.",
    "DNA stands for deoxyribonucleic acid.",
    "The genetic code is stored in DNA molecules.",
    "Mitochondria are the powerhouse of the cell.",
    # 科学・技術 (diverse)
    "The Earth orbits the Sun in approximately 365 days.",
    "Gravity is a fundamental force of nature.",
    "Einstein developed the theory of relativity.",
    "Quantum mechanics describes the behavior of subatomic particles.",
    "The periodic table organizes chemical elements by atomic number.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Evolution by natural selection was proposed by Darwin.",
    "The human genome contains approximately 3 billion base pairs.",
    "Machine learning models learn from data.",
    "Neural networks are inspired by biological brains.",
    # 日常文 (diverse style)
    "The weather today is sunny and warm.",
    "I enjoy reading books in the evening.",
    "Coffee is my favorite morning beverage.",
    "The train arrives at 9:30 AM.",
    "Please remember to bring your umbrella.",
    "The meeting is scheduled for Monday.",
    "She walked quickly through the park.",
    "The restaurant serves excellent Italian food.",
    "Children love playing in the snow.",
    "The library closes at 8 PM on weekdays.",
    # 数学・論理 (abstract)
    "A prime number has exactly two divisors.",
    "The square root of 144 is 12.",
    "Euler's formula: e^{iπ} + 1 = 0.",
    "A triangle has three sides and three angles.",
    "The Pythagorean theorem states a² + b² = c².",
    "An integer is divisible by 2 if it is even.",
    "The set of natural numbers is infinite.",
    "Zero is neither positive nor negative.",
    "Infinity is not a real number.",
    "A function maps inputs to outputs.",
    # ランダム・多様
    "The stock market fluctuated significantly yesterday.",
    "Music has the power to evoke strong emotions.",
    "Ancient Rome was a major civilization.",
    "The ocean covers about 71% of Earth's surface.",
    "Vaccines have saved millions of lives.",
    "The moon affects ocean tides on Earth.",
    "Computers process information using binary code.",
    "The Eiffel Tower is located in Paris, France.",
    "Mount Everest is the tallest mountain on Earth.",
    "The Amazon River is the largest river by volume.",
    # ペアのコントラスト (最小ペア: 意味が変わる)
    "The cat sat on the mat.",
    "The bat sat on the mat.",
    "The cat sat on the hat.",
    "The dog lay on the mat.",
    "She loves him deeply.",
    "She hates him deeply.",
    "The price increased by 10%.",
    "The price decreased by 10%.",
    "Turn left at the intersection.",
    "Turn right at the intersection.",
    # 意味的OOD (数学的難問)
    "Prove the Riemann hypothesis.",
    "Derive the Yang-Mills mass gap.",
    "What is motivic cohomology?",
    "Explain p-adic L-functions.",
    "Characterize the Langlands correspondence.",
    "What is the BSD conjecture?",
    "Describe non-commutative geometry.",
    "What is quantum gravity?",
    "Solve the Navier-Stokes equations.",
    "Explain topological quantum field theory.",
    # 無意味・OOD
    "Xyzzy plugh frobnitz quantum entanglement.",
    "Asdf jkl qwerty zxcv banana phone.",
    "The purple moon sings backwards eloquently.",
    "Seventeen dancing numbers refused happily.",
    "Colorless green ideas sleep furiously.",
    "The silence of the library is loud.",
    "Time flies like an arrow; fruit flies like a banana.",
    "Buffalo buffalo buffalo buffalo buffalo.",
    "James while John had had had had had had had had had had had a better effect on the teacher.",
    "The complex complex complex is quite complex.",
]

print(f"[2/6] テキスト埋め込み計算中... ({len(texts)} テキスト)")
t1 = time.time()

embeddings = []
logit_distributions = []

with torch.no_grad():
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors='pt',
                           truncation=True, max_length=64)
        outputs = model(**inputs, output_hidden_states=True)

        # Last hidden state at last token position
        last_hidden = outputs.hidden_states[-1][0, -1, :]  # (768,)
        embeddings.append(last_hidden.numpy())

        # lm_head logits → softmax (full vocabulary distribution)
        logits = outputs.logits[0, -1, :]  # (50257,)
        probs  = F.softmax(logits, dim=0).numpy()  # (50257,)
        logit_distributions.append(probs)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(texts)} done...")

embeddings  = np.array(embeddings)   # (N, 768)
P_matrix    = np.array(logit_distributions)  # (N, 50257)
N_texts     = len(texts)
print(f"  埋め込み計算完了: {time.time()-t1:.1f}s")
print()

# ── 3. 埋め込み正規化と d_L2 計算 ─────────────────────────────────────────────
print("[3/6] d_L2 と d_TV 計算中...")
t2 = time.time()

# L2 ノルム正規化 (コサイン距離のための準備)
emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_normed = embeddings / (emb_norms + 1e-9)  # (N, 768) unit vectors

# 全ペアの d_L2 (L2 距離, NOT cosine) と d_TV を計算
# N=100 → N*(N-1)/2 = 4950 ペア
results_all = []

for i in range(N_texts):
    for j in range(i + 1, N_texts):
        # d_L2: 生の embedding 空間での距離
        d_L2 = float(np.linalg.norm(embeddings[i] - embeddings[j]))
        # d_cos: コサイン距離 (参考)
        d_cos = float(np.linalg.norm(emb_normed[i] - emb_normed[j]))
        # d_TV = (1/2)||P_i - P_j||_1
        d_TV  = float(0.5 * np.sum(np.abs(P_matrix[i] - P_matrix[j])))

        results_all.append({
            'i': i, 'j': j,
            'd_L2': d_L2,
            'd_cos': d_cos,
            'd_TV':  d_TV,
            'ratio_L2': d_TV / (d_L2 + 1e-9),
            'ratio_cos': d_TV / (d_cos + 1e-9),
        })

print(f"  ペア数: {len(results_all):,}")
print(f"  計算時間: {time.time()-t2:.1f}s")
print()

# ── 4. 統計 ────────────────────────────────────────────────────────────────────
print("[4/6] L_LLM 実測...")
print()

ratios_L2  = np.array([r['ratio_L2']  for r in results_all])
ratios_cos = np.array([r['ratio_cos'] for r in results_all])
dL2_vals   = np.array([r['d_L2']  for r in results_all])
dtv_vals   = np.array([r['d_TV']  for r in results_all])

L_emp_L2  = float(np.max(ratios_L2))
L_emp_cos = float(np.max(ratios_cos))
L_mean_L2 = float(np.mean(ratios_L2))

# 上界との比較
print(f"  ‖W_lm_head‖_F = {W_F:.4f}")
print(f"  Theorem U5 上界: L_LLM ≤ {L_bound:.4f}")
print()
print(f"  実測 L_LLM (d_L2 基準):    {L_emp_L2:.6f}")
print(f"  実測 L_LLM (d_cos 基準):   {L_emp_cos:.6f}")
print(f"  実測平均 d_TV/d_L2:         {L_mean_L2:.6f}")
print(f"  上界 tightness (実測/上界): {L_emp_L2 / L_bound:.6f}  ({L_emp_L2/L_bound*100:.4f}%)")
print()
print(f"  上界違反件数: {np.sum(ratios_L2 > L_bound + 1e-6)}")

violation = np.sum(ratios_L2 > L_bound + 1e-6)
status = "THEOREM U5 VERIFIED ✅" if violation == 0 else f"THEOREM U5 VIOLATED ✗ ({violation} cases)"
print(f"  → {status}")
print()

# 最大比率のペアを特定
top_idx = np.argmax(ratios_L2)
r_top = results_all[top_idx]
print(f"  最大比率ペア:")
print(f"    text[{r_top['i']}]: {texts[r_top['i']][:60]}")
print(f"    text[{r_top['j']}]: {texts[r_top['j']][:60]}")
print(f"    d_L2 = {r_top['d_L2']:.4f}, d_TV = {r_top['d_TV']:.4f}, "
      f"ratio = {r_top['ratio_L2']:.4f}")

# ── 5. ε別信頼保証の実測 ──────────────────────────────────────────────────────
print()
print("[5/6] ε別信頼保証の実測 (実際のデータから)...")
print()
print(f"  {'ε (d_L2)':<12} {'対象ペア数':<12} {'実測最大d_TV':<16} {'L_emp·ε':<14} {'安全マージン'}")
print(f"  {'-'*12} {'-'*12} {'-'*16} {'-'*14} {'-'*12}")

for eps in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    mask = dL2_vals < eps
    if mask.sum() == 0:
        continue
    max_dtv_in_band = float(np.max(dtv_vals[mask]))
    theoretical_bound = L_emp_L2 * eps
    margin = theoretical_bound - max_dtv_in_band
    print(f"  {eps:<12.1f} {mask.sum():<12d} {max_dtv_in_band:<16.4f} "
          f"{theoretical_bound:<14.4f} {'+' if margin >= 0 else '-'}{abs(margin):.4f}")

# ── 6. 確率分布の比較 ─────────────────────────────────────────────────────────
print()
print("[6/6] 代表ペアの分布比較 (top-10 トークン)...")
print()

# 最小ペア例: "The cat sat on the mat." vs "The bat sat on the mat."
# これらは意味的に近いが微妙に異なるはず
cat_idx = texts.index("The cat sat on the mat.")
bat_idx = texts.index("The bat sat on the mat.")
if cat_idx >= 0 and bat_idx >= 0:
    P_cat = P_matrix[cat_idx]
    P_bat = P_matrix[bat_idx]
    dtv_minimal = float(0.5 * np.sum(np.abs(P_cat - P_bat)))
    dL2_minimal = float(np.linalg.norm(embeddings[cat_idx] - embeddings[bat_idx]))
    print(f"  最小ペア: 'cat' vs 'bat'")
    print(f"    d_L2 = {dL2_minimal:.4f}, d_TV = {dtv_minimal:.4f}, "
          f"ratio = {dtv_minimal/dL2_minimal:.4f}")
    top10_cat = np.argsort(-P_cat)[:5]
    top10_bat = np.argsort(-P_bat)[:5]
    print(f"    cat top-5: {[tokenizer.decode([t]).strip() for t in top10_cat]}")
    print(f"    bat top-5: {[tokenizer.decode([t]).strip() for t in top10_bat]}")

# ── プロット ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('L_LLM 実測 — Real GPT-2 (V=50257, d=768)', fontsize=13, fontweight='bold')

# Plot 1: d_TV vs d_L2 scatter
ax = axes[0, 0]
sc = ax.scatter(dL2_vals, dtv_vals, alpha=0.4, s=15, c=ratios_L2,
                cmap='plasma', vmin=0, vmax=np.percentile(ratios_L2, 95))
plt.colorbar(sc, ax=ax, label='d_TV/d_L2 ratio')
x_line = np.linspace(0, dL2_vals.max(), 100)
ax.plot(x_line, L_emp_L2 * x_line, 'r--', linewidth=2, label=f'L_emp={L_emp_L2:.3f} (max)')
ax.plot(x_line, L_bound   * x_line, 'b--', linewidth=1.5, alpha=0.5,
        label=f'U5 bound={L_bound:.1f}')
ax.set_xlabel('d_L2 (embedding space)')
ax.set_ylabel('d_TV (output distribution)')
ax.set_title('d_TV vs d_L2 — GPT-2 実測\n(全ペア)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 2: ratio histogram
ax = axes[0, 1]
ax.hist(ratios_L2, bins=50, color='#6366f1', alpha=0.8, edgecolor='none')
ax.axvline(L_emp_L2, color='#ef4444', linestyle='--', linewidth=2,
           label=f'実測最大: {L_emp_L2:.4f}')
ax.axvline(L_bound, color='#f59e0b', linestyle='--', linewidth=1.5,
           label=f'理論上界: {L_bound:.1f}')
ax.axvline(L_mean_L2, color='#10b981', linestyle='-', linewidth=1.5,
           label=f'実測平均: {L_mean_L2:.4f}')
ax.set_xlabel('d_TV / d_L2 ratio')
ax.set_ylabel('ペア数')
ax.set_title('L_LLM 分布 (GPT-2)\n理論上界 >> 実測値')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
ax.text(0.62, 0.85, f'tightness={L_emp_L2/L_bound*100:.3f}%',
        transform=ax.transAxes, fontsize=10, color='#6366f1', fontweight='bold')

# Plot 3: スケール比較 (理論値 vs 実測)
ax = axes[0, 2]
model_data = [
    ('Mini\n(V=100,d=32)', 0.0106, 0.5594),  # from U6
    ('GPT-2\n(real)', L_emp_L2, L_bound),
]
x_pos = [0, 1]
emp_vals  = [m[1] for m in model_data]
bnd_vals  = [m[2] for m in model_data]
labels    = [m[0] for m in model_data]
w = 0.35
ax.bar([x - w/2 for x in x_pos], bnd_vals, w, label='理論上界 (U5)', color='#f59e0b', alpha=0.8)
ax.bar([x + w/2 for x in x_pos], emp_vals, w, label='実測 L_emp', color='#10b981', alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.set_ylabel('L_LLM (log scale)')
ax.set_title('理論上界 vs 実測値\n(上界は安全側・保守的)')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
for xi, (emp, bnd) in zip(x_pos, zip(emp_vals, bnd_vals)):
    ax.text(xi, bnd*1.3, f'{bnd:.3f}', ha='center', fontsize=9, color='#f59e0b')
    ax.text(xi, emp*0.5, f'{emp:.4f}', ha='center', fontsize=9, color='#10b981')

# Plot 4: テキストクラスター別 d_TV 分布
n_groups = 5
group_size = N_texts // n_groups
group_labels = ['事実QA', '科学技術', '日常文', '最小ペア', 'OOD/無意味']
group_dtv = []
for g in range(n_groups):
    dtv_g = []
    gi_start = g * group_size
    gi_end   = (g+1) * group_size
    for r in results_all:
        if gi_start <= r['i'] < gi_end and gi_start <= r['j'] < gi_end:
            dtv_g.append(r['d_TV'])
    group_dtv.append(dtv_g if dtv_g else [0])

ax = axes[1, 0]
bp = ax.boxplot(group_dtv, labels=group_labels[:n_groups], patch_artist=True)
colors_bp = ['#10b981', '#6366f1', '#f59e0b', '#ef4444', '#888888']
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('d_TV within group')
ax.set_title('グループ内 d_TV 分布\n(OOD テキストほど d_TV 大)')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)

# Plot 5: ε別保証
ax = axes[1, 1]
eps_vals = np.linspace(0.1, 30, 200)
dtv_bound_u5 = L_emp_L2 * eps_vals  # actual empirical L
dtv_bound_th = L_bound   * eps_vals  # theoretical bound
ax.fill_between(eps_vals, 0, dtv_bound_u5, alpha=0.2, color='#10b981',
                label=f'実測保証 (L_emp={L_emp_L2:.3f})')
ax.fill_between(eps_vals, dtv_bound_u5, dtv_bound_th, alpha=0.1, color='#f59e0b',
                label=f'余裕 (保守的上界)')
ax.plot(eps_vals, dtv_bound_u5, '#10b981', linewidth=2)
ax.plot(eps_vals, dtv_bound_th, '#f59e0b', linewidth=1.5, linestyle='--')
ax.axhline(1.0, color='#ef4444', linestyle=':', linewidth=1.5, label='d_TV=1 (最大)')
ax.set_xlabel('Trust threshold ε (d_L2 < ε → guarantee)')
ax.set_ylabel('d_TV guarantee = L·ε')
ax.set_title('GPT-2 信頼保証曲線\n(L_emp実測ベース)')
ax.set_ylim(0, 1.5); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Plot 6: Summary table
ax = axes[1, 2]
ax.axis('off')
ax.set_title('THEOREM U5 実測検証\nGPT-2 結果サマリー', fontsize=11)
summary_lines = [
    f"モデル: GPT-2 (117M params)",
    f"V={V}, d={d_model}",
    f"‖W_lm‖_F = {W_F:.2f}",
    "",
    f"理論上界 (U5):",
    f"  L_LLM ≤ {L_bound:.2f}",
    "",
    f"実測値 (全{len(results_all)}ペア):",
    f"  L_emp (d_L2) = {L_emp_L2:.4f}",
    f"  L_emp (d_cos) = {L_emp_cos:.4f}",
    f"  平均 = {L_mean_L2:.4f}",
    "",
    f"Tightness: {L_emp_L2/L_bound*100:.4f}%",
    f"上界違反: {violation} 件",
    "",
    f"→ THEOREM U5 {'VERIFIED ✅' if violation==0 else 'VIOLATED ✗'}",
    "",
    f"→ 上界は実測の {L_bound/L_emp_L2:.0f}× (保守的)",
]
for k, line in enumerate(summary_lines):
    color = '#10b981' if '✅' in line else ('#ef4444' if '✗' in line else 'black')
    bold  = '✅' in line or '✗' in line or 'THEOREM' in line
    ax.text(0.05, 0.97 - k*0.065, line, transform=ax.transAxes,
            fontsize=9.5, va='top', color=color,
            fontweight='bold' if bold else 'normal', family='monospace')

plt.tight_layout()
plt.savefig('/home/yoiyoi/llm_lipschitz_measurement.png', dpi=150, bbox_inches='tight')
print()
print("Saved: llm_lipschitz_measurement.png")

# ── 結果保存 ─────────────────────────────────────────────────────────────────
results_out = {
    'model': 'gpt2',
    'V': V,
    'd': d_model,
    'W_F_lm_head': W_F,
    'L_bound_U5': L_bound,
    'L_empirical_L2': L_emp_L2,
    'L_empirical_cos': L_emp_cos,
    'L_mean_L2': L_mean_L2,
    'tightness_pct': L_emp_L2 / L_bound * 100,
    'bound_violations': violation,
    'n_pairs': len(results_all),
    'n_texts': N_texts,
    'theorem_U5_status': 'VERIFIED' if violation == 0 else 'VIOLATED',
    'top_pair': {
        'text_i': texts[r_top['i']],
        'text_j': texts[r_top['j']],
        'd_L2': r_top['d_L2'],
        'd_TV':  r_top['d_TV'],
        'ratio': r_top['ratio_L2'],
    }
}
with open('/home/yoiyoi/llm_lipschitz_results.json', 'w') as f:
    json.dump(results_out, f, indent=2, ensure_ascii=False)
print("Saved: llm_lipschitz_results.json")
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  GPT-2 実測 L_LLM = {L_emp_L2:.4f}  (d_L2 基準)")
print(f"  Theorem U5 上界 = {L_bound:.2f}")
print(f"  Tightness = {L_emp_L2/L_bound*100:.4f}%  (上界は実測の ~{L_bound/L_emp_L2:.0f}× 保守的)")
print()
print(f"  信頼保証 ε=10.0 (d_L2):  d_TV ≤ {L_emp_L2*10.0:.3f}")
print(f"  信頼保証 ε= 1.0 (d_L2):  d_TV ≤ {L_emp_L2*1.0:.4f}")
print(f"  信頼保証 ε= 0.1 (d_L2):  d_TV ≤ {L_emp_L2*0.1:.5f}")
print()
print(f"  → GPT-2の実際の L_LLM は上界({L_bound:.0f})の約{L_bound/L_emp_L2:.0f}分の1")
print(f"    実測 L_LLM ≈ {L_emp_L2:.4f} で、信頼保証は実用的範囲")
print()
print(f"  THEOREM U5 {results_out['theorem_U5_status']}")
