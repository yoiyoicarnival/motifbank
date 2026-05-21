"""
AI信用度検査機 — AI Credibility Checker
γ(r,d) Universal Law: hallucination detection via knowledge distance

Usage:
  pip install streamlit transformers torch scikit-learn
  streamlit run app.py
"""

import os, json
import numpy as np
import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── γ(r,d) 法則パラメータ ─────────────────────────────────────────
K_GAMMA   = 0.405
RTH_GAMMA = 0.283
D_OPT     = RTH_GAMMA + np.log(2) / K_GAMMA   # 1.9945
LAYER     = 11
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'radar_bank_cache.json')

# ── ゾーン境界 ────────────────────────────────────────────────────
Z_KNOWN_MAX    = 0.70 * D_OPT   # 1.396
Z_FRONTIER_MAX = 1.30 * D_OPT   # 2.593
Z_RISKY_MAX    = 2.00 * D_OPT   # 3.989

# ─────────────────────────────────────────────────────────────────
# キャッシュ: GPT-2 + bank + PCA3D (起動時1回だけ)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_resources():
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    from sklearn.decomposition import PCA

    tok = GPT2Tokenizer.from_pretrained('gpt2')
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    mdl.eval()

    with open(CACHE_FILE) as f:
        data = json.load(f)
    bank = [{'q': d['q'], 'emb': np.array(d['emb'])} for d in data]
    bank_arr = np.array([b['emb'] for b in bank])

    pca3d = PCA(n_components=3, random_state=42)
    bank_3d = pca3d.fit_transform(bank_arr)

    return tok, mdl, bank, bank_3d, pca3d

def get_score(text: str):
    import torch
    tok, mdl, bank, bank_3d, pca3d = load_resources()

    inp = tok(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        out = mdl(**inp)
    emb = out.hidden_states[LAYER][0, -1, :].numpy()

    q3d   = pca3d.transform(emb[np.newaxis, :])[0]
    dists = np.linalg.norm(bank_3d - q3d, axis=1)
    d     = float(dists.min())
    g     = float(1.0 - np.exp(-K_GAMMA * max(d - RTH_GAMMA, 0.0)))
    I     = g * (1.0 - g)

    if   d < Z_KNOWN_MAX:    zone = 'SAFE'
    elif d < Z_FRONTIER_MAX: zone = 'CREATIVE'
    elif d < Z_RISKY_MAX:    zone = 'RISKY'
    else:                    zone = 'DANGER'

    # 最も近いバンク質問
    nearest_q = bank[int(dists.argmin())]['q']

    return {'d': d, 'gamma': g, 'I': I, 'zone': zone, 'nearest': nearest_q}

def make_suggestion(zone: str, d: float) -> str:
    if zone == 'SAFE':
        return (
            f"**ヒント:** この質問はAIが確実に知っています (d={d:.2f} < {Z_KNOWN_MAX:.2f})。\n\n"
            "創造的なアイデアが欲しい場合は、少し「もし〜だったら？」を加えてみましょう。"
        )
    elif zone == 'CREATIVE':
        return (
            f"**🎯 最適ゾーン！** d={d:.2f} ≈ d*={D_OPT:.2f}\n\n"
            "AIの知識境界に近く、**信頼性と創造性のバランスが最高**です。\n"
            "このフレーミングを維持してください。"
        )
    elif zone == 'RISKY':
        return (
            f"**⚠️ リスクあり:** d={d:.2f} > d*={D_OPT:.2f} (γ={d:.2f})\n\n"
            "AIが**部分的に知識を持つ領域**です。答えの一部が不正確になる可能性があります。\n\n"
            "💡 改善策: 「〇〇という事実を前提として、〜について教えてください」と具体的な既知事実でアンカーを打ちましょう。"
        )
    else:
        return (
            f"**🚨 高リスク:** d={d:.2f} >> d*={D_OPT:.2f}\n\n"
            "AIの知識を大きく超えた領域です。**幻覚（hallucination）の可能性が高い**です。\n\n"
            "💡 改善策: 既知の事実や類比から出発してください。例:「〇〇と〜の共通点から考えると？」"
        )

# ─────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI信用度検査機",
    page_icon="🔬",
    layout="centered",
)

# カスタムCSS
st.markdown("""
<style>
  .main { background: #0d1117; }
  .result-card {
      padding: 1.2em 1.5em;
      border-radius: 12px;
      margin: 1em 0;
  }
  .safe    { background: #0d2e1a; border: 2px solid #22c55e; }
  .creative{ background: #2e1f0d; border: 2px solid #f59e0b; }
  .risky   { background: #2e150d; border: 2px solid #ef4444; }
  .danger  { background: #1a0d2e; border: 2px solid #a855f7; }
  .metric-row { display: flex; gap: 2em; margin-top: 0.5em; }
  .metric { text-align: center; }
  .metric-val { font-size: 1.8em; font-weight: bold; }
  .metric-lbl { font-size: 0.8em; color: #888; }
</style>
""", unsafe_allow_html=True)

# ヘッダー
st.title("🔬 AI信用度検査機")
st.caption("AI Credibility Checker — γ(r,d) Universal Law  |  知識境界の物理")

st.markdown("""
AIへの質問を入力すると、その質問がAIの**知識領域のどこに位置するか**を数式で診断します。

| ゾーン | 意味 |
|--------|------|
| 🟢 SAFE | AIが確実に知っている領域 |
| 🟡 CREATIVE | 知識境界 = 信頼性×創造性が最大 |
| 🔴 RISKY | 知識を超えつつある = 幻覚リスク上昇 |
| 🟣 DANGER | 完全に知識境界外 = 高幻覚リスク |
""")

st.divider()

# 入力
prompt = st.text_area(
    "質問またはプロンプトを入力",
    placeholder="例: What is the capital of France?",
    height=100,
)

col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("🔬 診断する", type="primary", use_container_width=True)

if run and prompt.strip():
    with st.spinner("分析中..."):
        result = get_score(prompt.strip())

    d, g, I, zone = result['d'], result['gamma'], result['I'], result['zone']

    # ゾーン表示
    ZONE_DISPLAY = {
        'SAFE':     ('🟢', 'SAFE',     'safe',     '#22c55e'),
        'CREATIVE': ('🟡', 'CREATIVE', 'creative', '#f59e0b'),
        'RISKY':    ('🔴', 'RISKY',    'risky',    '#ef4444'),
        'DANGER':   ('🟣', 'DANGER',   'danger',   '#a855f7'),
    }
    icon, label, css_class, color = ZONE_DISPLAY[zone]

    st.markdown(f"""
<div class="result-card {css_class}">
  <div style="font-size:2em; font-weight:bold; color:{color};">
    {icon} &nbsp; {label}
  </div>
  <div class="metric-row">
    <div class="metric">
      <div class="metric-val" style="color:{color};">{g:.3f}</div>
      <div class="metric-lbl">γ スコア<br>(幻覚確率)</div>
    </div>
    <div class="metric">
      <div class="metric-val" style="color:{color};">{d:.2f}</div>
      <div class="metric-lbl">d (知識距離)<br>d* = {D_OPT:.2f}</div>
    </div>
    <div class="metric">
      <div class="metric-val" style="color:{color};">{I:.3f}</div>
      <div class="metric-lbl">I (アイデア<br>スコア)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # I バー
    i_pct = int(I / 0.25 * 100)
    st.metric("アイデア生成ポテンシャル  I = γ(1−γ)", f"{I:.3f}  ({i_pct}% of max)")
    st.progress(i_pct / 100)

    # 提案
    st.markdown("---")
    st.markdown("### 💡 診断結果と改善提案")
    st.markdown(make_suggestion(zone, d))

    # 最近傍バンク質問
    with st.expander("📚 最も近い既知の知識"):
        st.info(f"**{result['nearest']}**\n\n"
                f"この質問との距離: d={d:.3f}")

    # 詳細数値
    with st.expander("🔢 詳細スコア"):
        st.code(f"""
γ(d)   = 1 - exp(-{K_GAMMA} × max(d - {RTH_GAMMA}, 0))
       = {g:.4f}   ← 幻覚確率の推定値

I(γ)   = γ(1-γ) = Var[Bernoulli(γ)]
       = {g:.4f} × {1-g:.4f}
       = {I:.4f}   ← d = d* = {D_OPT:.4f} で最大 (0.25)

d_min  = {d:.4f}   ← PCA3D 知識空間での最近傍距離
d*     = {D_OPT:.4f}   ← 最適アイデア生成点 (= r_th + ln2/k)
""")

elif run and not prompt.strip():
    st.warning("プロンプトを入力してください。")

# ─────────────────────────────────────────────────────────────────
# サイドバー: バッチ診断 + 理論説明
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 バッチ診断")
    batch_text = st.text_area(
        "複数のプロンプト (1行1件)",
        placeholder="What is the capital of France?\nIn what year did the Byzantine Empire fall?\nDescribe the Elvish language Sindarin.",
        height=150,
    )
    batch_run = st.button("一括診断", use_container_width=True)

    if batch_run and batch_text.strip():
        lines = [l.strip() for l in batch_text.split('\n') if l.strip()]
        with st.spinner(f"{len(lines)}件を分析中..."):
            rows = []
            for line in lines:
                r = get_score(line)
                rows.append({
                    'Prompt': line[:40] + '..' if len(line) > 42 else line,
                    'd': round(r['d'], 3),
                    'γ': round(r['gamma'], 3),
                    'I': round(r['I'], 3),
                    'Zone': r['zone'],
                })
        import pandas as pd
        df = pd.DataFrame(rows)

        def color_zone(val):
            colors = {'SAFE': '#0d2e1a', 'CREATIVE': '#2e1f0d',
                      'RISKY': '#2e150d', 'DANGER': '#1a0d2e'}
            return f'background-color: {colors.get(val, "")}'

        st.dataframe(df.style.applymap(color_zone, subset=['Zone']),
                     use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📐 理論")
    st.latex(r"\gamma(x) = 1 - e^{-k \cdot \max(x - x_{th},\, 0)}")
    st.latex(r"I = \gamma(1-\gamma) = \mathrm{Var}[\mathrm{Bern}(\gamma)]")
    st.latex(r"d^* = x_{th} + \frac{\ln 2}{k} = 1.994")
    st.caption(f"k = {K_GAMMA}  |  r_th = {RTH_GAMMA}  |  GPT-2 layer-11")

    st.divider()
    st.caption("Powered by γ(r,d) Universal Law\nFractal → LLM Knowledge Space")
