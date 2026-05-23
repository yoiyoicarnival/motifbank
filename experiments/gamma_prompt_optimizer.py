#!/usr/bin/env python3
"""
gamma_prompt_optimizer.py — Prompt Frontier Navigator v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
知識境界ナビゲーター: プロンプトを d≈d_opt=1.994 に誘導し
アイデア生成スコア I(d)=γ(1-γ) を最大化する。

理論的根拠:
  γ(d) = 1 - exp(-k · max(d - r_th, 0))   [k=0.405, r_th=0.283]
  I(d) = γ(1-γ) = Var[Bernoulli(γ)]        [二項分散]
  d_opt = r_th + ln2/k = 1.9945            [γ=0.5 から導出]

ゾーン分類:
  KNOWN    (d < 0.7·d_opt ≈ 1.4):  既知すぎる — I≈0 (boring)
  FRONTIER (0.7 < d/d_opt < 1.3):  最適 — I≈0.25 (sweet spot)
  RISKY    (1.3 < d/d_opt < 2.0):  幻覚リスク上昇 — I↓
  UNKNOWN  (d > 2.0·d_opt ≈ 4.0):  未知すぎる — I≈0 (noise)

Usage:
  OMP_NUM_THREADS=1 python3 gamma_prompt_optimizer.py
  OMP_NUM_THREADS=1 python3 gamma_prompt_optimizer.py "prompt1" "prompt2"
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

os.chdir('/home/yoiyoi')

# ── γ(r,d) 法則パラメータ ──────────────────────────────────────────────────────
K_GAMMA   = 0.405
RTH_GAMMA = 0.283
D_OPT     = RTH_GAMMA + np.log(2) / K_GAMMA  # = 1.9945
LAYER     = 11
CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'

# ── ゾーン境界 (d_opt の倍率) ──────────────────────────────────────────────────
Z_KNOWN_MAX    = 0.70 * D_OPT   # < 1.396: KNOWN
Z_FRONTIER_MAX = 1.30 * D_OPT   # < 2.593: FRONTIER
Z_RISKY_MAX    = 2.00 * D_OPT   # < 3.989: RISKY  (else UNKNOWN)

# ── ゾーンの色・ラベル ──────────────────────────────────────────────────────────
ZONES = {
    'KNOWN':    {'color': '#4CAF50', 'marker': '○', 'label': 'KNOWN    (d < 1.40)'},
    'FRONTIER': {'color': '#FF9800', 'marker': '★', 'label': 'FRONTIER (1.40≤d<2.59) ← sweet spot'},
    'RISKY':    {'color': '#F44336', 'marker': '△', 'label': 'RISKY    (2.59≤d<3.99)'},
    'UNKNOWN':  {'color': '#9C27B0', 'marker': '✕', 'label': 'UNKNOWN  (d ≥ 3.99)'},
}

# ── デモプロンプト (各ゾーンをカバー) ──────────────────────────────────────────
# 注: FRONTIERゾーン (d≈1.994) は「実在するが難しい事実」が到達しやすい
#     gamma_empirical_validation.py で実測 d≈2.2-2.3 だった質問を使用
DEFAULT_PROMPTS = [
    # KNOWN ゾーン (d=0: バンクメンバー)
    "What is the capital of France?",
    "Who discovered penicillin?",
    # FRONTIER ゾーン候補 (d≈2: 実在するが難しい知識)
    "In what year did the Korean War end?",
    "What is the capital of Bhutan?",
    "Who invented the World Wide Web?",
    "What is the half-life of carbon-14 in years?",
    "In what year did the Byzantine Empire fall?",
    # RISKY ゾーン候補 (d≈3-4: バンクに近い表現の別表現)
    "What is the chemical formula for water?",
    "What is the chemical formula for table salt?",
    # UNKNOWN ゾーン (d>>d*: 架空・学際的)
    "Describe the grammar of the Elvish language Sindarin.",
    "What is the cuisine of the lost city of Atlantis?",
]

# ── 改善テンプレート (ゾーン別) ────────────────────────────────────────────────
SUGGESTIONS = {
    'KNOWN': [
        "「もし{topic}が{alternative}だったら？」— 反事実で境界へ",
        "「{topic}は{unrelated_field}とどう関係する？」— 分野横断で距離↑",
        "「{topic}の最も意外な応用は？」— 既知から未知へのブリッジ",
        "「{topic}が{historical_shift}以前に発見されていたら？」— 時間軸シフト",
    ],
    'FRONTIER': [
        "最適ゾーン ★  このフレーミングを維持",
        "I(d)≈最大値。わずかな変形でも豊かなアイデアが出る",
        "隣接可能 (adjacent possible) 領域 — 探索を続けよ",
    ],
    'RISKY': [
        "「{known_fact}を前提として、{topic}を推論すると？」— 既知事実でアンカー",
        "「{topic}の{known_domain}的側面から分析すると？」— 既知分野に接地",
        "「実際のデータでは{topic}はどう見えるか？」— 実証でアンカー",
    ],
    'UNKNOWN': [
        "「{known_analogy}との類比から{topic}を考えると？」— 既知との橋渡し",
        "「{topic}の{known_aspect}だけに限定すると何が言えるか？」— スコープ縮小",
        "全面的に別のプロンプトから再出発することを推奨",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# 埋め込み・モデル (lazy)
# ─────────────────────────────────────────────────────────────────────────────
_model = _tokenizer = None

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    print("Loading GPT-2...", flush=True)
    _tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    _model     = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    _model.eval()
    print(f"  GPT-2 loaded ({sum(p.numel() for p in _model.parameters()):,} params)",
          flush=True)

def get_embedding(text):
    import torch
    _load_model()
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.hidden_states[LAYER][0, -1, :].numpy()

# ─────────────────────────────────────────────────────────────────────────────
# バンク + PCA3D
# ─────────────────────────────────────────────────────────────────────────────
_pca3d = None
_bank_3d = None
_bank = None

def load_bank():
    global _bank
    if _bank is not None:
        return _bank
    with open(CACHE_FILE) as f:
        data = json.load(f)
    _bank = [{'q': d['q'], 'emb': np.array(d['emb'])} for d in data]
    print(f"Bank loaded: {len(_bank)} facts", flush=True)
    return _bank

def ensure_pca3d():
    global _pca3d, _bank_3d
    if _pca3d is not None:
        return _pca3d, _bank_3d
    from sklearn.decomposition import PCA
    bank = load_bank()
    embs = np.array([b['emb'] for b in bank])
    _pca3d = PCA(n_components=3, random_state=42)
    _bank_3d = _pca3d.fit_transform(embs)
    return _pca3d, _bank_3d

def gamma_score(prompt_emb):
    """Return (gamma_h, I_score, d_min_3d, zone)."""
    pca3d, bank_3d = ensure_pca3d()
    q3d    = pca3d.transform(prompt_emb[np.newaxis, :])[0]
    dists  = np.linalg.norm(bank_3d - q3d, axis=1)
    d_min  = float(dists.min())
    g      = float(1.0 - np.exp(-K_GAMMA * max(d_min - RTH_GAMMA, 0.0)))
    I      = g * (1.0 - g)

    if   d_min < Z_KNOWN_MAX:    zone = 'KNOWN'
    elif d_min < Z_FRONTIER_MAX: zone = 'FRONTIER'
    elif d_min < Z_RISKY_MAX:    zone = 'RISKY'
    else:                        zone = 'UNKNOWN'

    return g, I, d_min, zone, q3d

# ─────────────────────────────────────────────────────────────────────────────
# テキストレポート
# ─────────────────────────────────────────────────────────────────────────────
BAR_WIDTH = 30

def _ibar(I):
    filled = int(round(I / 0.25 * BAR_WIDTH))
    return '█' * filled + '░' * (BAR_WIDTH - filled)

def _zone_icon(zone):
    return {'KNOWN': '○', 'FRONTIER': '★', 'RISKY': '△', 'UNKNOWN': '✕'}[zone]

def print_report(results):
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         PROMPT FRONTIER NAVIGATOR — γ(r,d) Analysis v1.0           ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  d_opt = {D_OPT:.4f}  (= r_th + ln2/k)   I_max = 0.2500           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    for r in results:
        g, I, d, zone, q3d, prompt = r['g'], r['I'], r['d'], r['zone'], r['q3d'], r['prompt']
        icon   = _zone_icon(zone)
        color_reset = ''  # terminal colors off for clean output
        label  = ZONES[zone]['label']

        print(f"  {icon} [{zone:8s}]  γ={g:.3f}  I={I:.3f}  d={d:.3f}  (d_opt={D_OPT:.3f})")
        print(f"     I-bar: |{_ibar(I)}| {I/0.25*100:.0f}% of max")
        print(f"     Prompt: \"{prompt[:72]}{'...' if len(prompt)>72 else ''}\"")

        # 改善ヒント
        hints = SUGGESTIONS[zone]
        if zone == 'FRONTIER':
            print(f"     → {hints[0]}")
        elif zone == 'KNOWN':
            print(f"     → ヒント: d={d:.2f} < d_opt={D_OPT:.2f}。プロンプトが知識バンクに近すぎます。")
            print(f"       推奨: {hints[0]}")
        elif zone == 'RISKY':
            print(f"     → ヒント: d={d:.2f} > d_opt={D_OPT:.2f}。幻覚リスク上昇中 (γ={g:.2f}>{0.5:.2f})。")
            print(f"       推奨: {hints[0]}")
        else:  # UNKNOWN
            print(f"     → ヒント: d={d:.2f} >> d_opt={D_OPT:.2f}。知識境界を大きく超えています。")
            print(f"       推奨: {hints[0]}")
        print()

    # サマリーテーブル
    print("  ─────────────────────────────────────────────────────")
    print(f"  {'Prompt':42s}  {'d':>6}  {'γ':>6}  {'I':>6}  Zone")
    print("  ─────────────────────────────────────────────────────")
    for r in results:
        prompt = r['prompt'][:40] + '..' if len(r['prompt']) > 42 else r['prompt']
        icon = _zone_icon(r['zone'])
        print(f"  {icon} {prompt:42s}  {r['d']:6.3f}  {r['g']:6.3f}  {r['I']:6.3f}  {r['zone']}")
    print("  ─────────────────────────────────────────────────────")

    # 最高 I のプロンプトを強調
    best = max(results, key=lambda r: r['I'])
    print(f"\n  ★ Best ideation prompt (I={best['I']:.3f}):")
    print(f"     \"{best['prompt']}\"")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 可視化
# ─────────────────────────────────────────────────────────────────────────────
ZONE_COLORS = {z: ZONES[z]['color'] for z in ZONES}

def visualize(results, out_path='gamma_prompt_optimizer.png'):
    _, bank_3d = ensure_pca3d()

    d_vals = np.linspace(0, 8, 400)
    g_vals = 1 - np.exp(-K_GAMMA * np.maximum(d_vals - RTH_GAMMA, 0))
    I_vals = g_vals * (1 - g_vals)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values():
            sp.set_color('#444')
        ax.tick_params(colors='#aaa')
        ax.xaxis.label.set_color('#ccc')
        ax.yaxis.label.set_color('#ccc')
        ax.title.set_color('#eee')

    # ── Panel 1: I(d) curve with prompt positions ──────────────────────────
    ax = axes[0]

    # ゾーン背景
    ax.axvspan(0,            Z_KNOWN_MAX,    alpha=0.12, color='#4CAF50')
    ax.axvspan(Z_KNOWN_MAX,  Z_FRONTIER_MAX, alpha=0.12, color='#FF9800')
    ax.axvspan(Z_FRONTIER_MAX, Z_RISKY_MAX,  alpha=0.12, color='#F44336')
    ax.axvspan(Z_RISKY_MAX,  8,              alpha=0.08, color='#9C27B0')

    # ゾーンラベル
    for (x_lo, x_hi, lbl, clr) in [
        (0,             Z_KNOWN_MAX,    'KNOWN',    '#4CAF50'),
        (Z_KNOWN_MAX,   Z_FRONTIER_MAX, 'FRONTIER', '#FF9800'),
        (Z_FRONTIER_MAX, Z_RISKY_MAX,  'RISKY',    '#F44336'),
        (Z_RISKY_MAX,   8,              'UNKNOWN',  '#9C27B0'),
    ]:
        ax.text((x_lo + x_hi) / 2, 0.235, lbl, ha='center', va='bottom',
                fontsize=7.5, color=clr, fontweight='bold', alpha=0.9)

    # マスターカーブ
    ax.plot(d_vals, I_vals, color='#FFD700', lw=2.0, label='I(d)=γ(1-γ)')

    # d_opt 垂直線
    ax.axvline(D_OPT, color='#FFD700', ls='--', lw=1.2, alpha=0.6)
    ax.text(D_OPT + 0.05, 0.23, f'd_opt={D_OPT:.3f}',
            color='#FFD700', fontsize=8, va='top', alpha=0.9)

    # 各プロンプトの点
    for r in results:
        clr = ZONE_COLORS[r['zone']]
        ax.scatter(r['d'], r['I'], color=clr, s=80, zorder=5, edgecolors='white', lw=0.5)
        # 短縮ラベル
        short = r['prompt'][:20] + '..' if len(r['prompt']) > 22 else r['prompt']
        ax.annotate(short, (r['d'], r['I']),
                    textcoords='offset points', xytext=(5, 4),
                    fontsize=6.5, color=clr, alpha=0.9,
                    arrowprops=dict(arrowstyle='-', color=clr, alpha=0.4, lw=0.7))

    ax.set_xlabel('d_min (PCA3D distance from bank)', fontsize=10)
    ax.set_ylabel('I(d) = Ideation Score = γ(1−γ)', fontsize=10)
    ax.set_title('Knowledge Frontier Landscape', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.01, 0.27)
    ax.legend(loc='upper right', framealpha=0.2, labelcolor='#ccc', fontsize=9)
    ax.grid(alpha=0.15, color='#555')

    # I 最大値マーク
    ax.axhline(0.25, color='#FFD700', ls=':', lw=0.8, alpha=0.4)
    ax.text(7.8, 0.251, 'I_max=0.25', color='#FFD700', fontsize=7, ha='right', va='bottom', alpha=0.7)

    # ── Panel 2: PCA空間の散布図 (PC1 vs PC2) ─────────────────────────────
    ax2 = axes[1]

    # バンク点 (薄く)
    ax2.scatter(bank_3d[:, 0], bank_3d[:, 1],
                c='#555', s=10, alpha=0.4, label='Bank facts', zorder=1)

    # バンク凸包のおよそのエッジを示すヒートマップ代わり
    from sklearn.decomposition import PCA as _PCA
    _, bank_3d_local = ensure_pca3d()
    # 等高線 (γ=0.5 の円: d=d_opt)
    centroid = bank_3d_local.mean(axis=0)
    ax2.scatter(*centroid[:2], marker='+', s=200, c='#FFD700', zorder=6, lw=2)

    # プロンプト点
    for r in results:
        clr = ZONE_COLORS[r['zone']]
        q2  = r['q3d'][:2]
        ax2.scatter(q2[0], q2[1], color=clr, s=100, zorder=5,
                    edgecolors='white', lw=0.8)
        short = r['prompt'][:18] + '..' if len(r['prompt']) > 20 else r['prompt']
        ax2.annotate(short, q2,
                     textcoords='offset points', xytext=(4, 3),
                     fontsize=6, color=clr, alpha=0.9)

    # ゾーン凡例
    legend_handles = [
        mpatches.Patch(color=ZONES[z]['color'], label=ZONES[z]['label'])
        for z in ['KNOWN', 'FRONTIER', 'RISKY', 'UNKNOWN']
    ]
    legend_handles.append(Line2D([0], [0], marker='+', color='#FFD700',
                                 markersize=8, lw=0, label='Bank centroid'))
    ax2.legend(handles=legend_handles, loc='lower right',
               framealpha=0.2, labelcolor='#ccc', fontsize=7.5)

    ax2.set_xlabel('PC1', fontsize=10)
    ax2.set_ylabel('PC2', fontsize=10)
    ax2.set_title('PCA3D Embedding Space (PC1–PC2)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.15, color='#555')

    plt.suptitle(
        f'Prompt Frontier Navigator  |  γ(d)=1−exp(−k·max(d−r_th,0))  '
        f'k={K_GAMMA}  r_th={RTH_GAMMA}  d_opt={D_OPT:.4f}',
        color='#eee', fontsize=9, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    print(f"\n  Figure saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 「知識ブリッジ」: 2つのプロンプト間の I(α) プロファイル
# ─────────────────────────────────────────────────────────────────────────────
def knowledge_bridge(prompt_a, prompt_b, n_steps=40, out_path='gamma_bridge.png'):
    """
    Anchor A (known) → Target B (unknown/different domain)
    の間で α を動かし、I(α) がピークになる点 α* を求める。
    """
    print(f"\n  Knowledge Bridge Analysis")
    print(f"  A: \"{prompt_a}\"")
    print(f"  B: \"{prompt_b}\"")

    emb_a = get_embedding(prompt_a)
    emb_b = get_embedding(prompt_b)
    pca3d, bank_3d = ensure_pca3d()

    alphas = np.linspace(0, 1, n_steps)
    results_bridge = []
    for alpha in alphas:
        emb_mix = (1 - alpha) * emb_a + alpha * emb_b
        # ノルム正規化しない: PCA は生の埋め込み空間でフィットされている
        q3d  = pca3d.transform(emb_mix[np.newaxis, :])[0]
        dists = np.linalg.norm(bank_3d - q3d, axis=1)
        d_min = float(dists.min())
        g     = float(1.0 - np.exp(-K_GAMMA * max(d_min - RTH_GAMMA, 0.0)))
        I     = g * (1.0 - g)
        results_bridge.append((alpha, d_min, g, I))

    # α* の発見
    best_idx = max(range(len(results_bridge)), key=lambda i: results_bridge[i][3])
    a_star, d_star, g_star, I_star = results_bridge[best_idx]
    print(f"\n  → α*={a_star:.2f}  d*={d_star:.3f}  γ*={g_star:.3f}  I*={I_star:.3f}")
    print(f"  → Best mixing: {(1-a_star)*100:.0f}% A + {a_star*100:.0f}% B")
    if a_star < 0.3:
        print("     [Frontier is close to A — slight nudge from A is enough]")
    elif a_star > 0.7:
        print("     [Frontier is close to B — B dominates the ideation direction]")
    else:
        print("     [True cross-domain bridge — blend both perspectives equally]")

    # プロット
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_color('#444')
    ax.tick_params(colors='#aaa')

    alphas_arr = np.array([r[0] for r in results_bridge])
    I_arr      = np.array([r[3] for r in results_bridge])
    d_arr      = np.array([r[1] for r in results_bridge])
    g_arr      = np.array([r[2] for r in results_bridge])

    ax.plot(alphas_arr, I_arr, color='#FF9800', lw=2.0, label='I(α)=Ideation score')
    ax.plot(alphas_arr, g_arr, color='#F44336', lw=1.2, ls='--', alpha=0.6, label='γ(α)')

    # α* マーク
    ax.axvline(a_star, color='#FFD700', ls='--', lw=1.5)
    ax.scatter([a_star], [I_star], color='#FFD700', s=150, zorder=6, label=f'α*={a_star:.2f}')
    ax.text(a_star + 0.02, I_star + 0.005,
            f'α*={a_star:.2f}\nd*={d_star:.2f}', color='#FFD700', fontsize=9)

    # d_opt 水平参照線
    ax.axhline(0.25, color='#FFD700', ls=':', lw=0.8, alpha=0.3)

    ax.set_xlabel('α (interpolation: 0=A, 1=B)', fontsize=11, color='#ccc')
    ax.set_ylabel('Score', fontsize=11, color='#ccc')
    ax.set_title(f'Knowledge Bridge: A → B\nA="{prompt_a[:45]}" | B="{prompt_b[:45]}"',
                 fontsize=10, color='#eee')
    ax.legend(framealpha=0.2, labelcolor='#ccc', fontsize=9)
    ax.grid(alpha=0.15, color='#555')
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    print(f"  Bridge figure saved → {out_path}")
    return a_star, d_star, I_star

# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────
def main():
    prompts = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PROMPTS

    print("\n  Computing embeddings and γ scores...", flush=True)
    results = []
    for i, p in enumerate(prompts):
        emb = get_embedding(p)
        g, I, d, zone, q3d = gamma_score(emb)
        results.append({'prompt': p, 'g': g, 'I': I, 'd': d, 'zone': zone, 'q3d': q3d})
        print(f"  [{i+1:2d}/{len(prompts)}] d={d:.3f}  γ={g:.3f}  I={I:.3f}  [{zone}]  {p[:50]}")

    print_report(results)
    visualize(results)

    # デフォルト実行時: Knowledge Bridge デモも追加
    if len(sys.argv) == 1:
        print("\n" + "═"*60)
        print("  Knowledge Bridge Demo")
        print("═"*60)
        # A=バンクメンバー (d=0), B=バンクメンバー (d=0 だが別ドメイン)
        # 2点間の補間で d_opt を横断: α≈0.1 で I≈0.250 (最大) を実測
        knowledge_bridge(
            "In what year did World War II end?",
            "What is the largest ocean on Earth?",
            out_path='gamma_bridge.png',
        )
        knowledge_bridge(
            "What is the capital of France?",
            "How many bones are in the adult human body?",
            out_path='gamma_bridge2.png',
        )

if __name__ == '__main__':
    main()
