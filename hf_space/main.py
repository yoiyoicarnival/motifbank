"""
HallucinationInspectionMachine — API + Web UI
FastAPI on port 7860 (HF Spaces Docker)
"""
import os, json, secrets, smtplib, hmac, hashlib
from datetime import date
from contextlib import asynccontextmanager
from typing import Optional
from email.mime.text import MIMEText

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── γ(r,d) 定数 (sentence-transformers cosine distance 空間) ──────
K_GAMMA   = 5.0    # 崩壊率 (cosine distance [0,1] 空間)
RTH_GAMMA = 0.10   # coherence radius
D_OPT     = RTH_GAMMA + np.log(2) / K_GAMMA   # ≈ 0.239

Z_KNOWN_MAX    = 0.22   # SAFE: cosine distance < 0.22
Z_FRONTIER_MAX = 0.40   # CREATIVE: 0.22 ≤ d < 0.40
Z_RISKY_MAX    = 0.58   # RISKY: 0.40 ≤ d < 0.58   DANGER: d ≥ 0.58

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radar_bank_cache_st.json')

# ── 環境変数 ─────────────────────────────────────────────────────
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
GMAIL_USER            = os.environ.get('GMAIL_USER', '')
GMAIL_APP_PASSWORD    = os.environ.get('GMAIL_APP_PASSWORD', '')

# ── 有料 API キー (env + 起動後に追加されたもの) ─────────────────
PAID_KEYS: set = set(k.strip() for k in os.environ.get('API_KEYS', '').split(',') if k.strip())

KEYS_FILE = '/tmp/runtime_keys.json'

def _load_runtime_keys():
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            for k in json.load(f):
                PAID_KEYS.add(k)

def _save_runtime_keys():
    with open(KEYS_FILE, 'w') as f:
        json.dump(list(PAID_KEYS), f)

def add_paid_key(key: str):
    PAID_KEYS.add(key)
    _save_runtime_keys()

# ── 無料レート制限 ────────────────────────────────────────────────
FREE_LIMIT = 20
_free_counts: dict = {}

def check_free_limit(ip: str) -> tuple[bool, int]:
    key = f"{ip}-{date.today()}"
    count = _free_counts.get(key, 0)
    if count >= FREE_LIMIT:
        return False, count
    _free_counts[key] = count + 1
    return True, count + 1

# ── メール送信 ────────────────────────────────────────────────────
def send_api_key_email(to_email: str, api_key: str):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print(f"[EMAIL SKIP] key={api_key} to={to_email} (no SMTP config)")
        return
    body = f"""HallucinationInspectionMachine Pro へのご登録ありがとうございます！

あなたの API キー:
{api_key}

使い方:
curl -X POST \\
  https://yoiyoicarnival-hallucinationinspectionmachine.hf.space/api/score \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: {api_key}" \\
  -d '{{"text": "your prompt here"}}'

API仕様: https://yoiyoicarnival-hallucinationinspectionmachine.hf.space/docs

ご不明な点は yoiyoicarnival@gmail.com までどうぞ。
"""
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = '【HallucinationInspectionMachine】APIキーのご案内'
    msg['From']    = GMAIL_USER
    msg['To']      = to_email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.send_message(msg)
        print(f"[EMAIL OK] key sent to {to_email}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")

# ── モデル (起動時にロード) ───────────────────────────────────────
_resources = {}

def _load():
    from sentence_transformers import SentenceTransformer
    print("Loading all-MiniLM-L6-v2 ...", flush=True)
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(CACHE_FILE) as f:
        data = json.load(f)
    bank = [{'q': d['q'], 'emb': np.array(d['emb'])} for d in data]
    bank_arr = np.array([b['emb'] for b in bank])  # 正規化済み
    _resources.update({'st_model': st_model, 'bank': bank, 'bank_arr': bank_arr})
    print(f"Ready. Bank: {len(bank)} facts", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_runtime_keys()
    _load()
    yield

app = FastAPI(
    title="HallucinationInspectionMachine",
    description="AIの幻覚を γ(r,d) 普遍則で検出するAPI",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── スコア計算 ───────────────────────────────────────────────────
def _score(text: str) -> dict:
    st_model  = _resources['st_model']
    bank      = _resources['bank']
    bank_arr  = _resources['bank_arr']

    # コサイン類似度 (正規化済みベクトルのドット積)
    emb  = st_model.encode(text, normalize_embeddings=True)
    sims = bank_arr @ emb
    d    = float(1.0 - sims.max())   # コサイン距離 [0, 1]
    nearest = bank[int(sims.argmax())]['q']

    g = float(1.0 - np.exp(-K_GAMMA * max(d - RTH_GAMMA, 0.0)))
    I = g * (1.0 - g)

    if   d < Z_KNOWN_MAX:    zone = 'SAFE'
    elif d < Z_FRONTIER_MAX: zone = 'CREATIVE'
    elif d < Z_RISKY_MAX:    zone = 'RISKY'
    else:                    zone = 'DANGER'

    return {'text': text, 'd': round(d, 4), 'gamma': round(g, 4),
            'I': round(I, 4), 'zone': zone, 'nearest_fact': nearest,
            'd_opt': round(D_OPT, 4)}

# ─────────────────────────────────────────────────────────────────
# API エンドポイント
# ─────────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    text: str

class ScoreResponse(BaseModel):
    text: str
    d: float
    gamma: float
    I: float
    zone: str        # SAFE | CREATIVE | RISKY | DANGER
    nearest_fact: str
    d_opt: float

@app.post("/api/score", response_model=ScoreResponse, tags=["API"])
async def score(
    body: ScoreRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    """
    プロンプトの幻覚リスクを診断する。

    - **SAFE** (γ≈0): AIが確実に知っている領域
    - **CREATIVE** (γ≈0.5, I≈0.25): 知識境界 = 最大アイデア生成点
    - **RISKY** (γ>0.5): 幻覚リスク上昇中
    - **DANGER** (γ≈1): 完全に知識外 = 高幻覚リスク

    **Free tier**: 20回/日（APIキー不要）
    **Paid tier**: 無制限（X-API-Key ヘッダーに有料キーを指定）
    """
    paid = x_api_key and x_api_key in PAID_KEYS

    if not paid:
        ip = request.client.host or "unknown"
        ok, count = check_free_limit(ip)
        if not ok:
            raise HTTPException(
                status_code=429,
                detail=f"Free limit ({FREE_LIMIT}/day) reached. Get a paid key: /pricing",
            )

    return _score(body.text)

@app.get("/api/health", tags=["API"])
async def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2", "bank_size": len(_resources.get('bank', []))}

@app.post("/webhook/stripe", tags=["Webhook"])
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Stripe からの支払い完了通知を受け取り、APIキーを自動発行してメール送信。"""
    payload = await request.body()
    sig = request.headers.get('stripe-signature', '')

    # Stripe 署名を検証
    if STRIPE_WEBHOOK_SECRET:
        try:
            parts = {p.split('=')[0]: p.split('=')[1] for p in sig.split(',')}
            ts = parts.get('t', '')
            v1 = parts.get('v1', '')
            expected = hmac.new(
                STRIPE_WEBHOOK_SECRET.encode(),
                f"{ts}.{payload.decode()}".encode(),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, v1):
                raise HTTPException(status_code=400, detail="Invalid signature")
        except Exception:
            raise HTTPException(status_code=400, detail="Signature verification failed")

    event = json.loads(payload)
    etype = event.get('type', '')
    print(f"[WEBHOOK] {etype}")

    # サブスクリプション開始 or 支払い完了
    if etype in ('checkout.session.completed', 'customer.subscription.created',
                 'invoice.payment_succeeded'):
        obj = event['data']['object']
        # メールアドレスを取得
        email = (obj.get('customer_email')
                 or obj.get('customer_details', {}).get('email')
                 or obj.get('customer_email', ''))
        if email:
            api_key = secrets.token_urlsafe(32)
            add_paid_key(api_key)
            background_tasks.add_task(send_api_key_email, email, api_key)
            print(f"[KEY ISSUED] {api_key[:8]}... -> {email}")

    return {"received": True}

# ─────────────────────────────────────────────────────────────────
# Web UI (HTML)
# ─────────────────────────────────────────────────────────────────
HTML_INDEX = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔬 HallucinationInspectionMachine</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif;
         min-height: 100vh; padding: 2rem 1rem; }
  .container { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
  .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
          padding: 1.5rem; margin-bottom: 1.2rem; }
  textarea { width: 100%; background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
             color: #e6edf3; font-size: 1rem; padding: 0.8rem; resize: vertical;
             min-height: 90px; outline: none; }
  textarea:focus { border-color: #58a6ff; }
  button { background: #238636; color: white; border: none; border-radius: 8px;
           padding: 0.7rem 1.8rem; font-size: 1rem; cursor: pointer; font-weight: 600; }
  button:hover { background: #2ea043; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .result { display: none; }
  .zone-badge { display: inline-block; font-size: 1.4rem; font-weight: 700;
                padding: 0.5rem 1.2rem; border-radius: 8px; margin-bottom: 1rem; }
  .SAFE     { background: #0d2e1a; border: 2px solid #22c55e; color: #22c55e; }
  .CREATIVE { background: #2e1f0d; border: 2px solid #f59e0b; color: #f59e0b; }
  .RISKY    { background: #2e150d; border: 2px solid #ef4444; color: #ef4444; }
  .DANGER   { background: #1a0d2e; border: 2px solid #a855f7; color: #a855f7; }
  .metrics { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }
  .metric { text-align: center; min-width: 80px; }
  .metric-val { font-size: 1.6rem; font-weight: 700; }
  .metric-lbl { font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem; }
  .bar-bg { background: #21262d; border-radius: 4px; height: 8px; margin: 0.8rem 0; }
  .bar-fill { height: 8px; border-radius: 4px; background: #f59e0b; transition: width 0.4s; }
  .hint { color: #8b949e; font-size: 0.9rem; line-height: 1.6; }
  .nearest { background: #21262d; border-radius: 6px; padding: 0.6rem 0.9rem;
             font-size: 0.85rem; color: #8b949e; margin-top: 0.8rem; }
  .zone-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .zone-table td, .zone-table th { padding: 0.5rem 0.8rem; border-bottom: 1px solid #21262d; }
  .zone-table th { color: #8b949e; font-weight: 500; text-align: left; }
  .api-block { background: #21262d; border-radius: 8px; padding: 1rem;
               font-family: monospace; font-size: 0.85rem; overflow-x: auto; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .nav { display: flex; gap: 1.2rem; margin-bottom: 1.5rem; }
  .nav a { color: #8b949e; font-size: 0.9rem; }
  .nav a:hover { color: #e6edf3; }
  #loading { display: none; color: #8b949e; margin-top: 0.5rem; }
</style>
</head>
<body>
<div class="container">
  <div class="nav">
    <a href="/">🔬 Checker</a>
    <a href="/pricing">💳 Pricing & API</a>
    <a href="/docs">📖 API Docs</a>
  </div>

  <h1>🔬 HallucinationInspectionMachine</h1>
  <p class="subtitle">AIの幻覚リスクを γ(r,d) 普遍則でスコア化 — Free: 20回/日</p>

  <div class="card">
    <textarea id="prompt" placeholder="AIへの質問を入力してください&#10;例: What is the capital of France?&#10;例: In what year did the Byzantine Empire fall?&#10;例: Describe the grammar of Elvish language Sindarin."></textarea>
    <div style="margin-top:0.8rem; display:flex; align-items:center; gap:1rem;">
      <button onclick="runScore()" id="btn">🔬 診断する</button>
      <span id="loading">分析中...</span>
    </div>
  </div>

  <div class="card result" id="result">
    <div id="zone-badge" class="zone-badge"></div>
    <div class="metrics">
      <div class="metric"><div class="metric-val" id="m-gamma"></div><div class="metric-lbl">γ (幻覚確率)</div></div>
      <div class="metric"><div class="metric-val" id="m-d"></div><div class="metric-lbl">d (知識距離)</div></div>
      <div class="metric"><div class="metric-val" id="m-I"></div><div class="metric-lbl">I (アイデア力)</div></div>
      <div class="metric"><div class="metric-val" style="color:#8b949e;">0.239</div><div class="metric-lbl">d* (最適点)</div></div>
    </div>
    <div class="bar-bg"><div class="bar-fill" id="i-bar" style="width:0%"></div></div>
    <div style="font-size:0.75rem; color:#8b949e; margin-bottom:0.8rem;">アイデア生成ポテンシャル I = γ(1−γ) — d=d*(0.239) で最大 (0.25)</div>
    <div class="hint" id="hint"></div>
    <div class="nearest" id="nearest"></div>
  </div>

  <div class="card" style="margin-top:1.5rem;">
    <table class="zone-table">
      <tr><th>ゾーン</th><th>d の範囲</th><th>γ</th><th>意味</th></tr>
      <tr><td>🟢 SAFE</td><td>d &lt; 0.22</td><td>≈ 0</td><td>AIが確実に知っている</td></tr>
      <tr><td>🟡 CREATIVE</td><td>0.22 ≤ d &lt; 0.40</td><td>≈ 0.5</td><td>知識境界 = 最大創造力</td></tr>
      <tr><td>🔴 RISKY</td><td>0.40 ≤ d &lt; 0.58</td><td>0.5〜0.8</td><td>幻覚リスク上昇中</td></tr>
      <tr><td>🟣 DANGER</td><td>d ≥ 0.58</td><td>≈ 1</td><td>完全に知識外</td></tr>
    </table>
  </div>
</div>

<script>
const HINTS = {
  SAFE: "AIが確実に知っている領域です (d < 0.22)。<br>創造的なアイデアを引き出したい場合は、「もし〜だったら？」という反事実的フレーミングを加えてみましょう。",
  CREATIVE: "🎯 <strong>最適ゾーン！</strong> d ≈ d* = 0.239<br>AIの知識境界に位置し、信頼性と創造性のバランスが最高です。このフレーミングを維持してください。",
  RISKY: "⚠️ 幻覚リスクが上昇しています。<br>改善策: 「〇〇という事実を前提として、〜について教えてください」と既知事実でアンカーを打ちましょう。",
  DANGER: "🚨 AIの知識を大きく超えた領域です。幻覚の可能性が高いです。<br>改善策: 既知の事実や類比から出発してください。例:「〇〇と〜の共通点から考えると？」",
};
async function runScore() {
  const text = document.getElementById('prompt').value.trim();
  if (!text) return;
  document.getElementById('btn').disabled = true;
  document.getElementById('loading').style.display = 'inline';
  document.getElementById('result').style.display = 'none';
  try {
    const r = await fetch('/api/score', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text}),
    });
    if (r.status === 429) {
      const d = await r.json();
      alert(d.detail + '\\n\\n→ /pricing でAPIキーを取得してください');
      return;
    }
    const data = await r.json();
    const zone = data.zone;
    const colors = {SAFE:'#22c55e', CREATIVE:'#f59e0b', RISKY:'#ef4444', DANGER:'#a855f7'};
    const icons  = {SAFE:'🟢', CREATIVE:'🟡', RISKY:'🔴', DANGER:'🟣'};
    document.getElementById('zone-badge').textContent = icons[zone] + ' ' + zone;
    document.getElementById('zone-badge').className = 'zone-badge ' + zone;
    document.getElementById('m-gamma').textContent = data.gamma.toFixed(3);
    document.getElementById('m-gamma').style.color = colors[zone];
    document.getElementById('m-d').textContent = data.d.toFixed(3);
    document.getElementById('m-d').style.color = colors[zone];
    document.getElementById('m-I').textContent = data.I.toFixed(3);
    document.getElementById('m-I').style.color = colors[zone];
    document.getElementById('i-bar').style.width = (data.I / 0.25 * 100) + '%';
    document.getElementById('i-bar').style.background = colors[zone];
    document.getElementById('hint').innerHTML = HINTS[zone];
    document.getElementById('nearest').textContent = '📚 最も近い既知の知識: ' + data.nearest_fact;
    document.getElementById('result').style.display = 'block';
  } catch(e) {
    alert('エラー: ' + e.message);
  } finally {
    document.getElementById('btn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}
document.getElementById('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runScore();
});
</script>
</body>
</html>
"""

HTML_PRICING = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>💳 Pricing — HallucinationInspectionMachine</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif;
         min-height: 100vh; padding: 2rem 1rem; }
  .container { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem; }
  .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
          padding: 1.5rem; margin-bottom: 1.2rem; }
  .plan-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
  .plan { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem; }
  .plan.featured { border-color: #f59e0b; }
  .plan-name { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.3rem; }
  .plan-price { font-size: 2rem; font-weight: 700; color: #f59e0b; }
  .plan-price span { font-size: 1rem; color: #8b949e; }
  .plan-features { list-style: none; margin: 1rem 0; color: #8b949e; font-size: 0.9rem; line-height: 2; }
  .plan-features li::before { content: "✓ "; color: #22c55e; }
  .btn-pay { display: block; background: #f59e0b; color: #0d1117; border: none; border-radius: 8px;
             padding: 0.8rem; text-align: center; font-weight: 700; font-size: 1rem;
             cursor: pointer; text-decoration: none; margin-top: 1rem; }
  .btn-pay:hover { background: #d97706; }
  .api-block { background: #21262d; border-radius: 8px; padding: 1rem;
               font-family: monospace; font-size: 0.85rem; overflow-x: auto; white-space: pre; }
  a { color: #58a6ff; text-decoration: none; }
  .nav { display: flex; gap: 1.2rem; margin-bottom: 1.5rem; }
  .nav a { color: #8b949e; font-size: 0.9rem; }
  h2 { font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; color: #e6edf3; }
  p { color: #8b949e; line-height: 1.7; margin-bottom: 0.8rem; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="container">
  <div class="nav">
    <a href="/">🔬 Checker</a>
    <a href="/pricing">💳 Pricing & API</a>
    <a href="/docs">📖 API Docs</a>
  </div>

  <h1>💳 Pricing & API Access</h1>
  <p class="subtitle">AIの幻覚検出を自分のアプリに組み込む</p>

  <div class="plan-grid">
    <div class="plan">
      <div class="plan-name">Free</div>
      <div class="plan-price">¥0 <span>/月</span></div>
      <ul class="plan-features">
        <li>20回/日</li>
        <li>Webブラウザから利用</li>
        <li>APIキー不要</li>
      </ul>
      <a href="/" class="btn-pay" style="background:#238636; color:white;">今すぐ試す</a>
    </div>
    <div class="plan featured">
      <div class="plan-name">⭐ Pro</div>
      <div class="plan-price">¥980 <span>/月</span></div>
      <ul class="plan-features">
        <li>API アクセス (無制限)</li>
        <li>JSON レスポンス</li>
        <li>バッチ診断</li>
        <li>メールサポート</li>
      </ul>
      <a href="https://buy.stripe.com/14AbIU9LO5Sh6aT4if5Ne05" class="btn-pay" id="stripe-btn">
        💳 購入してAPIキーを取得
      </a>
    </div>
  </div>

  <div class="card">
    <h2>🔌 API の使い方</h2>
    <p>X-API-Key ヘッダーにAPIキーを付けてPOSTリクエストを送るだけ。</p>
    <div class="api-block">curl -X POST \\
  https://yoiyoicarnival-hallucinationinspectionmachine.hf.space/api/score \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: YOUR_API_KEY" \\
  -d '{"text": "In what year did the Byzantine Empire fall?"}'</div>

    <p style="margin-top:1rem;"><strong>レスポンス例:</strong></p>
    <div class="api-block">{
  "text": "In what year did the Byzantine Empire fall?",
  "d": 0.285,
  "gamma": 0.503,
  "I": 0.250,
  "zone": "CREATIVE",
  "nearest_fact": "What was the Byzantine Empire?",
  "d_opt": 0.239
}</div>

    <p style="margin-top:1rem;">APIキー不要で試す（無料枠 20回/日）:</p>
    <div class="api-block">curl -X POST \\
  https://yoiyoicarnival-hallucinationinspectionmachine.hf.space/api/score \\
  -H "Content-Type: application/json" \\
  -d '{"text": "What is the capital of France?"}'</div>
  </div>

  <div class="card">
    <h2>📮 購入後の流れ</h2>
    <p>1. 上の「購入」ボタンから Stripe で決済（¥980/月）</p>
    <p>2. 決済完了後、登録メールに <strong>APIキーを24時間以内に送付</strong> します</p>
    <p>3. X-API-Key ヘッダーにキーを付けてリクエストするだけ</p>
    <p>問い合わせ: <a href="mailto:yoiyoicarnival@gmail.com">yoiyoicarnival@gmail.com</a></p>
  </div>
</div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML_INDEX)

@app.get("/pricing", response_class=HTMLResponse)
async def pricing():
    return HTMLResponse(HTML_PRICING)
