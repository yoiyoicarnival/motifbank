"""
hallucination_radar.py — Hallucination Radar v0.4

Detects whether a language model will hallucinate on a given prompt
by measuring three complementary signals in GPT-2 hidden-state space:

  1. Geometric distance  d_min : distance from prompt to ReasonBank (768-dim)
  2. Token entropy       H     : uncertainty during generation
  3. Unified γ score    γ_H   : γ(r,d) law applied in PCA3D space

Composite risk score (Theorem U12 + entropy fusion):
  risk_d = σ(A  × (d_min − r_c))          # geometry (768-dim)
  risk_h = σ(B  × (H     − H_c))          # uncertainty
  risk   = 0.65 × risk_d + 0.35 × risk_h  # fusion (bank members: geometry only)

Unified γ score (fractal universality law):
  γ_H = 1 - exp(-k · max(d_min_3d − r_th, 0))
  k=0.405, r_th=0.283  (fit from LLM PCA3D bank, gamma_universality.py)
  d_min_3d = distance in PCA3D embedding space

Calibration (GPT-2 small, layer-11, N=133 prompts, 2026-05-19):
  A_CALIB  = 0.1044   r_c = 83.08   EPS_STAR = 92.7
  B_CALIB  = 1.5      H_c = 3.5     (entropy contribution)

Usage:
  python3 hallucination_radar.py                    # demo
  python3 hallucination_radar.py "prompt"           # single prompt
  python3 hallucination_radar.py "p1" "p2" "p3"    # multiple prompts
  python3 hallucination_radar.py --build            # rebuild bank cache
  python3 hallucination_radar.py --gamma            # demo + γ comparison plot
"""

import sys, json, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import expit as sigmoid

# ── Calibration constants ─────────────────────────────────────────────────────
A_CALIB  = 0.1044   # geometry: logistic sharpness
RC_CALIB = 83.08    # geometry: critical distance r_c
EPS_STAR = 92.7     # support density radius
B_CALIB  = 1.5      # entropy: logistic sharpness
HC_CALIB = 3.5      # entropy: critical entropy H_c (nats)
W_D      = 0.65     # weight for geometric component
W_H      = 0.35     # weight for entropy component
BANK_HIT_THRESHOLD = 20.0  # d_min < this → use geometry only (bank member)

LAYER      = 11
CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'
N_GEN_TOKENS = 24

# ── Unified γ(r,d) law — fit from LLM PCA3D (gamma_universality.py) ─────────
K_GAMMA   = 0.405   # decay rate k  (LLM bank, PCA3D space)
RTH_GAMMA = 0.283   # coherence radius r_th in PCA3D
_pca3d_model = None
_bank_3d     = None

# ── Demo prompts: three tiers — safe / risky / borderline ────────────────────
DEMO_PROMPTS = [
    # Safe (bank members, d_min = 0)
    "What is the capital of France?",
    "Who discovered penicillin?",
    "In what year did World War II end?",
    # Clearly risky (fictional / out-of-domain)
    "Explain the Voynich manuscript's linguistic structure.",
    "Describe the grammar of the Elvish language Sindarin.",
    "Describe a Shakespeare play about Mars colonization.",
    # Borderline (topic exists but fact is wrong or fictional)
    "What was Einstein's 1931 quantum biology paper?",
    "What is the cuisine of the lost city of Atlantis?",
]

# ── Factual QA bank (verified ground truth — used for embedding bank only) ────
FACTUAL_QA = [
    ("What is the capital of France?",       ["Paris"],          "easy"),
    ("What is the capital of Japan?",        ["Tokyo"],          "easy"),
    ("What is the capital of Germany?",      ["Berlin"],         "easy"),
    ("What is the capital of Italy?",        ["Rome"],           "easy"),
    ("What is the capital of Spain?",        ["Madrid"],         "easy"),
    ("What is the capital of China?",        ["Beijing"],        "easy"),
    ("What is the capital of Russia?",       ["Moscow"],         "easy"),
    ("What is the capital of Egypt?",        ["Cairo"],          "easy"),
    ("What is the capital of Mexico?",       ["Mexico City"],    "easy"),
    ("What is the capital of India?",        ["New Delhi", "Delhi"], "easy"),
    ("What is the capital of South Korea?",  ["Seoul"],          "easy"),
    ("What is the capital of Sweden?",       ["Stockholm"],      "easy"),
    ("What is the capital of Norway?",       ["Oslo"],           "easy"),
    ("What is the capital of Portugal?",     ["Lisbon"],         "easy"),
    ("What is the capital of Greece?",       ["Athens"],         "easy"),
    ("What is the capital of Poland?",       ["Warsaw"],         "easy"),
    ("What is the capital of Netherlands?",  ["Amsterdam"],      "easy"),
    ("What is the capital of Belgium?",      ["Brussels"],       "easy"),
    ("What is the capital of Austria?",      ["Vienna"],         "easy"),
    ("What is the capital of Thailand?",     ["Bangkok"],        "easy"),
    ("What is the capital of Australia?",    ["Canberra"],       "medium"),
    ("What is the capital of Canada?",       ["Ottawa"],         "medium"),
    ("What is the capital of Brazil?",       ["Brasilia"],       "medium"),
    ("What is the capital of Turkey?",       ["Ankara"],         "medium"),
    ("What is the capital of Switzerland?",  ["Bern"],           "medium"),
    ("What is the chemical symbol for gold?",       ["Au"],   "easy"),
    ("What is the chemical symbol for silver?",     ["Ag"],   "easy"),
    ("What is the chemical symbol for iron?",       ["Fe"],   "easy"),
    ("What is the chemical symbol for sodium?",     ["Na"],   "easy"),
    ("What is the chemical symbol for potassium?",  ["K"],    "easy"),
    ("What planet is closest to the Sun?",          ["Mercury"],  "easy"),
    ("What planet is largest in the solar system?", ["Jupiter"],  "easy"),
    ("What is the boiling point of water in Celsius?", ["100"],   "easy"),
    ("What is the freezing point of water in Celsius?", ["0"],    "easy"),
    ("What is the atomic number of hydrogen?",      ["1"],        "easy"),
    ("What is the formula for water?",              ["H2O"],      "easy"),
    ("What is the formula for carbon dioxide?",     ["CO2"],      "easy"),
    ("What element has the symbol O?",              ["oxygen","Oxygen"],   "easy"),
    ("What element has the symbol N?",              ["nitrogen","Nitrogen"],"easy"),
    ("What element has the symbol C?",              ["carbon","Carbon"],   "easy"),
    ("What element has the symbol H?",              ["hydrogen","Hydrogen"],"easy"),
    ("What is the hardest natural substance?",      ["diamond","Diamond"], "easy"),
    ("How many chromosomes do humans have?",        ["46"],   "medium"),
    ("In what year did World War II end?",          ["1945"], "easy"),
    ("In what year did World War I begin?",         ["1914"], "easy"),
    ("In what year did man first land on the Moon?",["1969"], "easy"),
    ("Who was the first President of the United States?", ["George Washington","Washington"], "easy"),
    ("Who discovered penicillin?",                 ["Alexander Fleming","Fleming"],  "easy"),
    ("Who invented the telephone?",                ["Alexander Graham Bell","Bell"], "easy"),
    ("Who painted the Mona Lisa?",                 ["Leonardo da Vinci","Leonardo","da Vinci"], "easy"),
    ("Who wrote Romeo and Juliet?",                ["William Shakespeare","Shakespeare"], "easy"),
    ("Who developed the theory of relativity?",    ["Albert Einstein","Einstein"],    "easy"),
    ("Who discovered gravity?",                    ["Isaac Newton","Newton"],         "easy"),
    ("Who invented the light bulb?",               ["Thomas Edison","Edison"],        "easy"),
    ("Who was the first woman to win a Nobel Prize?",["Marie Curie","Curie"],         "easy"),
    ("In what year was the Declaration of Independence signed?", ["1776"], "easy"),
    ("Who was the first human to travel to space?",["Yuri Gagarin","Gagarin"],  "easy"),
    ("What language is spoken in Brazil?",         ["Portuguese"], "easy"),
    ("What language is spoken in Egypt?",          ["Arabic"],     "easy"),
    ("How many letters are in the English alphabet?",["26"],       "easy"),
    ("What is the largest ocean on Earth?",        ["Pacific","Pacific Ocean"],  "easy"),
    ("What is the largest continent?",             ["Asia"],       "easy"),
    ("What is the longest river in the world?",    ["Nile","Nile River"],        "easy"),
    ("What is the tallest mountain in the world?", ["Mount Everest","Everest"],  "easy"),
    ("What is the largest country by area?",       ["Russia"],     "easy"),
    ("What is the smallest country in the world?", ["Vatican City","Vatican"],   "easy"),
    ("What is the largest animal on Earth?",       ["blue whale","Blue Whale"],  "easy"),
    ("What is the fastest land animal?",           ["cheetah","Cheetah"],        "easy"),
    ("How many legs does a spider have?",          ["8","eight"],  "easy"),
    ("What do bees produce?",                      ["honey","Honey"], "easy"),
    ("Who founded Microsoft?",                     ["Bill Gates","Gates","Paul Allen"], "easy"),
    ("What does CPU stand for?",                   ["Central Processing Unit"], "easy"),
    ("What does AI stand for?",                    ["Artificial Intelligence"], "easy"),
    ("How many days are in a leap year?",          ["366"], "easy"),
    ("How many days are in a year?",               ["365"], "easy"),
    ("How many hours are in a day?",               ["24"],  "easy"),
    ("How many minutes are in an hour?",           ["60"],  "easy"),
    ("How many months are in a year?",             ["12"],  "easy"),
    ("How many sides does a hexagon have?",        ["6","six"],   "easy"),
    ("How many sides does an octagon have?",       ["8","eight"], "easy"),
    ("How many planets are in the solar system?",  ["8","eight"], "easy"),
    ("What is the name of Earth's moon?",          ["Moon","Luna"],"easy"),
    ("What is the name of the galaxy we live in?", ["Milky Way"],  "easy"),
    ("What planet has rings?",                     ["Saturn"],     "easy"),
    ("What is the closest star to Earth?",         ["Sun","the Sun"], "easy"),
    ("What is the currency of Japan?",             ["yen","Yen"],    "easy"),
    ("What is the currency of the UK?",            ["pound","Pound","GBP"], "easy"),
    ("How many colors are in a rainbow?",          ["7","seven"],    "easy"),
    ("Who invented the printing press?",           ["Gutenberg","Johannes Gutenberg"], "medium"),
    ("What year did the Titanic sink?",            ["1912"],  "easy"),
    ("What is the largest organ in the human body?",["skin","Skin"], "medium"),
    ("What is the normal human body temperature in Celsius?",["37"],"easy"),
    ("What is the study of stars called?",         ["astronomy"],   "easy"),
    ("What does DNA stand for?",                   ["Deoxyribonucleic acid"], "medium"),
    ("What ocean did the Titanic sink in?",        ["Atlantic","Atlantic Ocean"], "easy"),
    ("How many strings does a standard guitar have?",["6","six"],   "easy"),
    ("Who sang Bohemian Rhapsody?",                ["Queen"],       "easy"),
    ("How many players are on a soccer team?",     ["11","eleven"], "easy"),
    ("How many players are on a basketball team?", ["5","five"],    "easy"),
    ("What organ filters blood in the body?",      ["kidney","kidneys","Kidney"], "easy"),
    ("What vitamin is produced by the body when exposed to sunlight?",["Vitamin D","D"],"medium"),
    ("What gas do plants absorb from the atmosphere?",["carbon dioxide","CO2"],"easy"),
    ("What is the most abundant gas in Earth's atmosphere?",["nitrogen","Nitrogen"],"medium"),
    ("What is the value of pi to 2 decimal places?",["3.14"],  "easy"),
    ("What is the square root of 144?",            ["12"],     "easy"),
    ("How many bones are in the adult human body?",["206"],    "medium"),
    ("What is the Roman numeral for 10?",          ["X"],      "easy"),
    ("What is the Roman numeral for 5?",           ["V"],      "easy"),
    ("What is the Roman numeral for 100?",         ["C"],      "easy"),
    ("What is the chemical formula for table salt?",["NaCl"],  "medium"),
    ("What year was the Eiffel Tower built?",      ["1889"],   "medium"),
    ("What is the longest bone in the human body?",["femur","Femur"], "medium"),
    ("How many teeth does an adult human have?",   ["32"],     "medium"),
    ("What is the speed of light in km/s?",        ["300000","299792"], "medium"),
    ("What is the atomic number of carbon?",       ["6"],      "medium"),
    ("What is the atomic number of oxygen?",       ["8"],      "medium"),
    ("What is 2 to the power of 10?",             ["1024"],   "medium"),
    ("What is the most spoken language in the world?",["Mandarin","Chinese","Mandarin Chinese"],"medium"),
    ("How many moons does Mars have?",             ["2","two"],"medium"),
    ("What year did the Soviet Union dissolve?",   ["1991"],   "medium"),
    ("In what year did the Berlin Wall fall?",     ["1989"],   "medium"),
    ("What year did Christopher Columbus reach the Americas?",["1492"],"medium"),
    ("What is the deepest lake in the world?",     ["Lake Baikal","Baikal"],"medium"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (lazy)
# ─────────────────────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None

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

def get_embedding(text, layer=LAYER):
    import torch
    _load_model()
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.hidden_states[layer][0, -1, :].numpy()

# ─────────────────────────────────────────────────────────────────────────────
# Bank cache — all FACTUAL_QA, no GPT-2 answer check needed
# ─────────────────────────────────────────────────────────────────────────────
def build_bank(force=False):
    """Build or load ReasonBank. Returns list of {q, emb}."""
    if not force and os.path.exists(CACHE_FILE):
        print(f"Loading bank cache ({CACHE_FILE})...", flush=True)
        with open(CACHE_FILE) as f:
            data = json.load(f)
        bank = [{'q': d['q'], 'emb': np.array(d['emb'])} for d in data]
        print(f"  ReasonBank: {len(bank)} verified facts loaded", flush=True)
        return bank

    print(f"Building ReasonBank ({len(FACTUAL_QA)} facts)...", flush=True)
    _load_model()
    bank = []
    for i, (q, _, _) in enumerate(FACTUAL_QA):
        emb = get_embedding(q)
        bank.append({'q': q, 'emb': emb})
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(FACTUAL_QA)}] done", flush=True)

    save_data = [{'q': b['q'], 'emb': b['emb'].tolist()} for b in bank]
    with open(CACHE_FILE, 'w') as f:
        json.dump(save_data, f)
    print(f"Bank built: {len(bank)} facts. Saved → {CACHE_FILE}", flush=True)
    return bank

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory curvature + token entropy (Research A + B)
# ─────────────────────────────────────────────────────────────────────────────
def compute_trajectory(prompt, n_tokens=N_GEN_TOKENS, layer=LAYER):
    """
    Generate n_tokens from prompt; collect hidden state at each step.

    Returns:
      states      : (n_tokens+1, 768)  hidden states
      kappa_mean  : float  mean trajectory curvature (0=straight, 1=random-walk, 2=oscillation)
      entropy_mean: float  mean next-token entropy (nats) — high = uncertain generation
    """
    import torch
    _load_model()
    p = f"Q: {prompt}\nA:"
    inputs = _tokenizer(p, return_tensors='pt', truncation=True, max_length=100)
    input_ids = inputs['input_ids']

    states   = []
    entropies = []
    with torch.no_grad():
        out = _model(**inputs)
        states.append(out.hidden_states[layer][0, -1, :].numpy())
        generated_ids = input_ids.clone()
        for _ in range(n_tokens):
            out2 = _model(generated_ids, output_hidden_states=True)
            logits = out2.logits[0, -1, :]
            # Token entropy
            log_probs = torch.log_softmax(logits, dim=-1)
            probs     = torch.exp(log_probs)
            ent       = float(-torch.sum(probs * log_probs))
            entropies.append(ent)
            next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            states.append(out2.hidden_states[layer][0, -1, :].numpy())

    states = np.array(states)
    steps  = np.diff(states, axis=0)
    norms  = np.linalg.norm(steps, axis=1, keepdims=True) + 1e-9
    steps_norm = steps / norms

    kappas = []
    for t in range(1, len(steps_norm)):
        cos_sim = float(np.clip(np.dot(steps_norm[t], steps_norm[t-1]), -1.0, 1.0))
        kappas.append(1.0 - cos_sim)

    kappa_mean   = float(np.mean(kappas))   if kappas   else 1.0
    entropy_mean = float(np.mean(entropies)) if entropies else 0.0

    return states, kappa_mean, entropy_mean

# ─────────────────────────────────────────────────────────────────────────────
# γ-based score (Unified γ(r,d) law in PCA3D space)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_pca3d(bank):
    global _pca3d_model, _bank_3d
    if _pca3d_model is None:
        from sklearn.decomposition import PCA
        bank_embs = np.array([b['emb'] for b in bank])
        _pca3d_model = PCA(n_components=3, random_state=42)
        _bank_3d = _pca3d_model.fit_transform(bank_embs)
    return _pca3d_model, _bank_3d


def compute_gamma_score(prompt_emb, bank):
    """
    γ hallucination score via unified law in PCA3D space.
    γ_H = 1 - exp(-K_GAMMA * max(d_min_3d - RTH_GAMMA, 0))
    """
    pca3d, bank_3d = _ensure_pca3d(bank)
    q3d = pca3d.transform(prompt_emb[np.newaxis, :])[0]
    dists_3d = np.linalg.norm(bank_3d - q3d, axis=1)
    d_min_3d = float(dists_3d.min())
    gamma_h  = float(1.0 - np.exp(-K_GAMMA * max(d_min_3d - RTH_GAMMA, 0.0)))
    return gamma_h, d_min_3d


# ─────────────────────────────────────────────────────────────────────────────
# Risk computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk(prompt_emb, bank, entropy_mean=None):
    bank_embs = np.array([b['emb'] for b in bank])
    dists  = np.linalg.norm(bank_embs - prompt_emb[np.newaxis, :], axis=1)
    d_min  = float(np.min(dists))
    nn_idx = int(np.argmin(dists))

    risk_d = float(sigmoid(A_CALIB * (d_min - RC_CALIB)))

    # Composite: fuse entropy signal for non-bank prompts only
    if entropy_mean is not None and d_min >= BANK_HIT_THRESHOLD:
        risk_h = float(sigmoid(B_CALIB * (entropy_mean - HC_CALIB)))
        risk   = W_D * risk_d + W_H * risk_h
    else:
        risk   = risk_d
    risk = float(np.clip(risk, 0.0, 1.0))

    n_bank = len(bank_embs)
    if n_bank > 1:
        sample_idx = np.random.choice(n_bank, min(n_bank, 60), replace=False)
        intra = [float(np.min(np.linalg.norm(
                     np.delete(bank_embs, idx, axis=0) - bank_embs[idx], axis=1)))
                 for idx in sample_idx]
        ood_score = float(np.mean(d_min > np.array(intra)))
    else:
        ood_score = risk

    n_support = int(np.sum(dists < EPS_STAR))

    gamma_h, d_min_3d = compute_gamma_score(prompt_emb, bank)

    return {
        'd_min':     d_min,
        'd_min_3d':  d_min_3d,
        'risk':      risk,
        'risk_d':    risk_d,
        'gamma_h':   gamma_h,
        'ood':       ood_score,
        'n_support': n_support,
        'nn_q':      bank[nn_idx]['q'],
    }

# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────
def risk_bar(risk, width=20):
    filled = int(risk * width)
    bar = '█' * filled + '░' * (width - filled)
    if   risk < 0.35: label = 'LOW'
    elif risk < 0.65: label = 'MEDIUM'
    elif risk < 0.85: label = 'HIGH'
    else:             label = 'VERY HIGH'
    return bar, label

def print_report(prompt, metrics, kappa_mean, entropy_mean):
    risk    = metrics['risk']
    d_min   = metrics['d_min']
    d_min_3d = metrics.get('d_min_3d', 0.0)
    gamma_h = metrics.get('gamma_h', 0.0)
    ood     = metrics['ood']
    n_sup   = metrics['n_support']
    nn_q    = metrics['nn_q']
    bar, label = risk_bar(risk)

    support_str = 'dense' if n_sup >= 5 else ('sparse' if n_sup >= 1 else 'none')
    risk_d  = metrics.get('risk_d', risk)
    geo_pct = risk_d * 100
    ent_risk = float(sigmoid(B_CALIB * (entropy_mean - HC_CALIB))) * 100
    gbar, glabel = risk_bar(gamma_h)

    print()
    print('─' * 62)
    print('  HALLUCINATION RADAR v0.4')
    print('─' * 62)
    print(f'  Prompt : "{prompt[:54]}{"..." if len(prompt)>54 else ""}"')
    print(f'  d_min  : {d_min:.1f}  (r_c={RC_CALIB})    '
          f'd_min_3d : {d_min_3d:.3f}  (r_th={RTH_GAMMA})')
    print('─' * 62)
    print(f'  Composite Risk (U12)  : {risk*100:.0f}%  {bar} {label}')
    print(f'    ├ Geometry  (65%)   : {geo_pct:.0f}%  (768-dim sigmoid)')
    print(f'    └ Entropy   (35%)   : {ent_risk:.0f}%  (H={entropy_mean:.2f} nats)')
    print(f'  γ Score (universal)   : {gamma_h*100:.0f}%  {gbar} {glabel}')
    print(f'    └ 1-exp(-k·max(d_3d-r_th,0))  k={K_GAMMA}, r_th={RTH_GAMMA}')
    print(f'  OOD score             : {ood:.2f}')
    print(f'  Nearest support       : {support_str}  ({n_sup} neighbors within ε={EPS_STAR:.0f})')
    print('─' * 62)
    print(f'  Nearest bank Q : "{nn_q[:56]}"')
    print('─' * 62)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def visualize(prompts_data, bank, out_path='/home/yoiyoi/hallucination_radar.png'):
    """PCA scatter: bank + query prompts with generation trajectories."""
    all_embs = [b['emb'] for b in bank]
    for pd in prompts_data:
        all_embs.append(pd['emb'])
        all_embs.extend(pd['trajectory_states'])

    all_embs = np.array(all_embs)
    from sklearn.decomposition import PCA
    pca    = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_embs)

    n_bank = len(bank)
    bank_2d = all_2d[:n_bank]

    offset = n_bank
    prompt_2d_list = []
    traj_2d_list   = []
    for pd in prompts_data:
        n_traj = len(pd['trajectory_states'])
        prompt_2d_list.append(all_2d[offset])
        traj_2d_list.append(all_2d[offset: offset + 1 + n_traj])
        offset += 1 + n_traj

    n_prompts = len(prompts_data)
    ncols     = min(n_prompts, 4)
    nrows     = math.ceil(n_prompts / ncols) + 1

    risk_cmap = plt.cm.RdYlGn_r
    norm      = Normalize(vmin=0, vmax=1)

    # Each cell = 2 sub-columns (PCA | gauge), overview row spans all
    fig = plt.figure(figsize=(6 * ncols, 4.5 * nrows), facecolor='#0f172a')

    def _style(ax, title=''):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.grid(True, color='#334155', alpha=0.5, linewidth=0.5)
        if title:
            ax.set_title(title, color='#e2e8f0', fontsize=8, pad=4)

    def _base(ax):
        ax.scatter(bank_2d[:, 0], bank_2d[:, 1],
                   c='#10b981', s=14, alpha=0.25, label='Bank', zorder=2)

    def draw_gauge(ax, risk, label=''):
        """Semi-circle risk gauge."""
        ax.set_facecolor('#1e293b')
        ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.15, 1.3)
        ax.axis('off')

        # Arc background gradient
        theta = np.linspace(np.pi, 0, 200)
        for j in range(len(theta) - 1):
            t = j / (len(theta) - 2)
            color = risk_cmap(t)
            ax.plot([np.cos(theta[j]), np.cos(theta[j+1])],
                    [np.sin(theta[j]), np.sin(theta[j+1])],
                    color=color, linewidth=10, alpha=0.7, solid_capstyle='round')

        # Needle
        angle = np.pi * (1 - risk)
        nx = 0.78 * np.cos(angle)
        ny = 0.78 * np.sin(angle)
        needle_color = risk_cmap(risk)
        ax.annotate('', xy=(nx, ny), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=needle_color,
                                   lw=2.5, mutation_scale=15))
        ax.plot(0, 0, 'o', color='#e2e8f0', markersize=6, zorder=5)

        # Risk text
        bar, lbl = risk_bar(risk, width=8)
        lbl_color = '#ef4444' if risk > 0.65 else ('#f59e0b' if risk > 0.35 else '#10b981')
        ax.text(0, -0.08, f'{risk*100:.0f}%', ha='center', va='top',
                fontsize=18, fontweight='bold', color=lbl_color)
        ax.text(0, -0.22, lbl, ha='center', va='top', fontsize=9, color=lbl_color)
        if label:
            short = label[:34] + ('…' if len(label) > 34 else '')
            ax.text(0, 1.22, f'"{short}"', ha='center', va='top',
                    fontsize=7, color='#cbd5e1', wrap=True)

    # ── Row 0: PCA overview (full width) ──────────────────────────────────────
    ax_ov = fig.add_subplot(nrows, 1, 1)
    _base(ax_ov)
    for i, (p2d, pd) in enumerate(zip(prompt_2d_list, prompts_data)):
        risk  = pd['metrics']['risk']
        color = risk_cmap(norm(risk))
        ax_ov.scatter(p2d[0], p2d[1], c=[color], s=150,
                      edgecolors='white', linewidths=0.8, zorder=5)
        ax_ov.annotate(f'P{i+1}', (p2d[0], p2d[1]),
                       textcoords='offset points', xytext=(5, 5),
                       fontsize=8, color='#e2e8f0')
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_ov, fraction=0.015, pad=0.01)
    cb.set_label('Risk', color='#94a3b8', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=7)
    ax_ov.legend(fontsize=7, loc='lower right',
                 facecolor='#1e293b', edgecolor='#334155',
                 labelcolor='#94a3b8')
    _style(ax_ov, f'Embedding Space Overview  (GPT-2 layer-{LAYER} · PCA · {len(bank)}-fact ReasonBank)')
    ax_ov.set_xlabel('PC1', color='#94a3b8')
    ax_ov.set_ylabel('PC2', color='#94a3b8')

    # ── Rows 1+: PCA + gauge per prompt ───────────────────────────────────────
    for i, (pd, p2d, t2d) in enumerate(zip(prompts_data, prompt_2d_list, traj_2d_list)):
        row = (i // ncols) + 1
        col =  i  % ncols

        # Split each cell into left (PCA) and right (gauge) using gridspec
        left_idx  = row * ncols * 2 + col * 2 + 1
        right_idx = row * ncols * 2 + col * 2 + 2
        # use add_subplot with a GridSpec instead
        gs_inner = fig.add_gridspec if False else None  # fallback: use subplot2grid
        ax_pca = plt.subplot2grid((nrows, ncols * 2), (row, col * 2),
                                  fig=fig)
        ax_g   = plt.subplot2grid((nrows, ncols * 2), (row, col * 2 + 1),
                                  fig=fig)

        # PCA panel
        _base(ax_pca)
        if len(t2d) > 1:
            colors_t = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(t2d) - 1))
            for j in range(len(t2d) - 1):
                ax_pca.plot(t2d[j:j+2, 0], t2d[j:j+2, 1], '-',
                            color=colors_t[j], linewidth=1.8, alpha=0.85, zorder=3)
            ax_pca.scatter(t2d[1:, 0], t2d[1:, 1], c=range(len(t2d)-1),
                           cmap='YlOrRd', s=20, alpha=0.6, zorder=4, vmin=0, vmax=len(t2d))
        risk  = pd['metrics']['risk']
        color = risk_cmap(norm(risk))
        ax_pca.scatter(p2d[0], p2d[1], c=[color], s=200,
                       edgecolors='white', linewidths=1.5, zorder=6, marker='*')
        _style(ax_pca, f'P{i+1} trajectory')
        ax_pca.set_xlabel('PC1', color='#94a3b8')
        ax_pca.set_ylabel('PC2', color='#94a3b8')

        # Gauge panel
        draw_gauge(ax_g, risk, label=pd['prompt'])

    # Hide unused slots
    # (subplot2grid handles this automatically for the PCA/gauge pairs)

    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Hallucination Radar v0.3  ·  Theorem U12  ·  Composite Score: Geometry + Entropy',
                 color='#f1f5f9', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f"Visualization saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# γ comparison visualization
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TIER = {
    # Safe (bank members)
    "What is the capital of France?":                     'safe',
    "Who discovered penicillin?":                         'safe',
    "In what year did World War II end?":                 'safe',
    # Clearly risky
    "Explain the Voynich manuscript's linguistic structure.": 'risky',
    "Describe the grammar of the Elvish language Sindarin.":  'risky',
    "Describe a Shakespeare play about Mars colonization.":   'risky',
    # Borderline
    "What was Einstein's 1931 quantum biology paper?":    'borderline',
    "What is the cuisine of the lost city of Atlantis?":  'borderline',
}
TIER_COLOR = {'safe': '#10b981', 'borderline': '#f59e0b', 'risky': '#ef4444'}


def visualize_gamma(prompts_data, bank,
                    out_path='/home/yoiyoi/hallucination_gamma.png'):
    """
    3-panel γ comparison figure:
      Panel 1: PCA3D scatter (bank + queries), colored by γ score
      Panel 2: γ score vs composite risk scatter, labeled by tier
      Panel 3: γ(d_min_3d) master curve + query points
    """
    _, bank_3d = _ensure_pca3d(bank)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0f172a')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, color='#334155', alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

    risk_cmap = plt.cm.RdYlGn_r

    # ── Panel 1: PCA3D scatter ───────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(bank_3d[:, 0], bank_3d[:, 1],
               c='#10b981', s=10, alpha=0.2, label='Bank', zorder=2)

    for pd in prompts_data:
        pca3d, _ = _ensure_pca3d(bank)
        q3d = pca3d.transform(pd['emb'][np.newaxis, :])[0]
        gam = pd['metrics'].get('gamma_h', 0.0)
        tier = PROMPT_TIER.get(pd['prompt'], 'borderline')
        c = TIER_COLOR[tier]
        ax.scatter(q3d[0], q3d[1], c=[risk_cmap(gam)], s=140,
                   edgecolors=c, linewidths=2.0, zorder=5, marker='*')
        short = pd['prompt'][:20] + '…'
        ax.annotate(short, (q3d[0], q3d[1]),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=5.5, color='#cbd5e1')

    ax.set_xlabel('PC1', color='#94a3b8')
    ax.set_ylabel('PC2', color='#94a3b8')
    ax.set_title('PCA3D embedding space\n(★ colored by γ score, ring = tier)',
                 color='#e2e8f0', fontsize=9)
    for tier, c in TIER_COLOR.items():
        ax.scatter([], [], c=c, s=60, label=tier)
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

    # ── Panel 2: γ vs composite risk scatter ────────────────────────────────
    ax = axes[1]
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, xs, '--', color='#475569', linewidth=1.2, alpha=0.7,
            label='γ = risk (diagonal)')

    for pd in prompts_data:
        risk  = pd['metrics']['risk']
        gam   = pd['metrics'].get('gamma_h', 0.0)
        tier  = PROMPT_TIER.get(pd['prompt'], 'borderline')
        c     = TIER_COLOR[tier]
        ax.scatter(risk, gam, c=[c], s=80, edgecolors='white',
                   linewidths=0.8, zorder=5)
        ax.annotate(pd['prompt'][:18] + '…', (risk, gam),
                    textcoords='offset points', xytext=(4, 2),
                    fontsize=5, color='#94a3b8')

    ax.set_xlabel('Composite Risk (U12 sigmoid)', color='#94a3b8')
    ax.set_ylabel('γ Score (unified law)', color='#94a3b8')
    ax.set_title('γ Score vs Composite Risk\n(agreement = on diagonal)',
                 color='#e2e8f0', fontsize=9)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    for tier, c in TIER_COLOR.items():
        ax.scatter([], [], c=c, s=50, label=tier)
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

    # ── Panel 3: master curve γ(d_min_3d) + query points ────────────────────
    ax = axes[2]
    d_range = np.linspace(0, 25, 300)
    gamma_curve = 1.0 - np.exp(-K_GAMMA * np.maximum(d_range - RTH_GAMMA, 0.0))
    ax.plot(d_range, gamma_curve, '-', color='white', linewidth=2.5,
            label=f'γ=1-exp(-{K_GAMMA}·max(d-{RTH_GAMMA},0))', zorder=2)
    ax.axvline(RTH_GAMMA, color='#f59e0b', linestyle=':', linewidth=1.5,
               alpha=0.8, label=f'r_th={RTH_GAMMA}')

    for pd in prompts_data:
        d3d  = pd['metrics'].get('d_min_3d', 0.0)
        gam  = pd['metrics'].get('gamma_h', 0.0)
        tier = PROMPT_TIER.get(pd['prompt'], 'borderline')
        c    = TIER_COLOR[tier]
        ax.scatter(d3d, gam, c=[c], s=80, edgecolors='white',
                   linewidths=0.8, zorder=5)

    ax.set_xlabel('d_min_3d  (PCA3D distance to nearest bank)', color='#94a3b8')
    ax.set_ylabel('γ(d_min_3d)', color='#94a3b8')
    ax.set_title(r'Master curve: $\gamma = 1-e^{-k\,\max(d-r_{th},0)}$' + '\n'
                 'H(d) — hallucination via unified law',
                 color='#e2e8f0', fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    for tier, c in TIER_COLOR.items():
        ax.scatter([], [], c=c, s=50, label=tier)
    ax.legend(fontsize=7, facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8')

    fig.suptitle(r'Unified $\gamma(r,d)$ Hallucination Score  ·  '
                 r'Radar v0.4  ·  $H(d)=1-e^{-k\cdot\max(d-r_{th},0)}$',
                 color='#f1f5f9', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    args      = sys.argv[1:]

    if '--build' in args:
        build_bank(force=True)
        print("Done.")
        return

    do_gamma  = '--gamma' in args
    user_args = [a for a in args if not a.startswith('--')]
    prompts   = user_args if user_args else DEMO_PROMPTS

    bank = build_bank(force=False)
    print(f"\nReasonBank: {len(bank)} verified facts", flush=True)
    print(f"Analysing {len(prompts)} prompt(s)...\n", flush=True)

    prompts_data = []
    for prompt in prompts:
        emb     = get_embedding(prompt)
        traj_states, kappa_mean, entropy_mean = compute_trajectory(prompt)
        metrics = compute_risk(emb, bank, entropy_mean=entropy_mean)

        print_report(prompt, metrics, kappa_mean, entropy_mean)

        prompts_data.append({
            'prompt':            prompt,
            'emb':               emb,
            'metrics':           metrics,
            'trajectory_states': list(traj_states),
            'kappa_mean':        kappa_mean,
            'entropy_mean':      entropy_mean,
        })

    visualize(prompts_data, bank)

    if do_gamma or not user_args:
        visualize_gamma(prompts_data, bank)

    print()
    print('─' * 75)
    print('  SUMMARY')
    print('─' * 75)
    print(f'  {"Prompt":40s} {"Risk":>6} {"γ_H":>6} {"OOD":>5} {"H(nats)":>7}')
    print(f'  {"─"*40} {"─"*6} {"─"*6} {"─"*5} {"─"*7}')
    for pd in prompts_data:
        r   = pd['metrics']['risk']
        gam = pd['metrics'].get('gamma_h', 0.0)
        o   = pd['metrics']['ood']
        h   = pd['entropy_mean']
        flag = '⚠' if r > 0.5 or gam > 0.5 else ' '
        agree = '✓' if abs(r - gam) < 0.3 else '≠'
        print(f'  {flag}{pd["prompt"][:39]:<39} {r*100:>5.0f}% {gam*100:>5.0f}% {o:>5.2f} {h:>7.2f} {agree}')
    print('─' * 75)
    print()
    # γ vs composite agreement
    diffs = [abs(pd['metrics']['risk'] - pd['metrics'].get('gamma_h', 0.0))
             for pd in prompts_data]
    print(f'  γ ↔ Risk agreement:  mean|Δ|={np.mean(diffs):.3f}  max|Δ|={np.max(diffs):.3f}')


if __name__ == '__main__':
    main()
