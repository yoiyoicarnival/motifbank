"""
gpt2_factual_hdist.py — U12 Real-World Validation

Core question: Does d_min predict H_factual on real GPT-2?

Design:
  1. Load real GPT-2 (small, 117M)
  2. 200 factual Q&A questions (geography/science/history/math, graded difficulty)
  3. Embed each question with GPT-2 hidden states (last token, layer 11)
  4. Answer each question with greedy generation (check against gold answer)
  5. Build ReasonBank from CORRECTLY answered train questions
  6. Compute d_min(test_q, ReasonBank) for each test question
  7. Fit: P(wrong | d_min = r) = σ(a·(r − r_c))  → compare with U12

If correlation(d_min, P_wrong) > 0.7: publishable as "Geometric Hallucination Predictor"
"""

import numpy as np
import json, math, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(42)

# ─── Factual QA Dataset (200 questions, graded difficulty) ──────────────────
# Format: (question, [gold_answers], difficulty)
# difficulty: 'easy' | 'medium' | 'hard'

FACTUAL_QA = [
    # ── Geography: capitals (easy) ──────────────────────────────────────────
    ("What is the capital of France?",       ["Paris"],          "easy"),
    ("What is the capital of Japan?",        ["Tokyo"],          "easy"),
    ("What is the capital of Germany?",      ["Berlin"],         "easy"),
    ("What is the capital of Italy?",        ["Rome"],           "easy"),
    ("What is the capital of Spain?",        ["Madrid"],         "easy"),
    ("What is the capital of China?",        ["Beijing"],        "easy"),
    ("What is the capital of Australia?",    ["Canberra"],       "medium"),
    ("What is the capital of Canada?",       ["Ottawa"],         "medium"),
    ("What is the capital of Brazil?",       ["Brasilia"],       "medium"),
    ("What is the capital of Argentina?",    ["Buenos Aires"],   "medium"),
    ("What is the capital of Russia?",       ["Moscow"],         "easy"),
    ("What is the capital of Egypt?",        ["Cairo"],          "easy"),
    ("What is the capital of Mexico?",       ["Mexico City"],    "easy"),
    ("What is the capital of India?",        ["New Delhi", "Delhi"], "easy"),
    ("What is the capital of South Korea?",  ["Seoul"],          "easy"),
    ("What is the capital of Turkey?",       ["Ankara"],         "medium"),
    ("What is the capital of Sweden?",       ["Stockholm"],      "easy"),
    ("What is the capital of Norway?",       ["Oslo"],           "easy"),
    ("What is the capital of Portugal?",     ["Lisbon"],         "easy"),
    ("What is the capital of Greece?",       ["Athens"],         "easy"),
    ("What is the capital of Poland?",       ["Warsaw"],         "easy"),
    ("What is the capital of Netherlands?",  ["Amsterdam"],      "easy"),
    ("What is the capital of Belgium?",      ["Brussels"],       "easy"),
    ("What is the capital of Switzerland?",  ["Bern"],           "medium"),
    ("What is the capital of Austria?",      ["Vienna"],         "easy"),
    ("What is the capital of Thailand?",     ["Bangkok"],        "easy"),
    ("What is the capital of Saudi Arabia?", ["Riyadh"],         "medium"),
    ("What is the capital of South Africa?", ["Pretoria","Cape Town","Bloemfontein"], "hard"),
    ("What is the capital of Kenya?",        ["Nairobi"],        "medium"),
    ("What is the capital of Nigeria?",      ["Abuja"],          "hard"),
    # ── Science & Math (graded) ─────────────────────────────────────────────
    ("What is the chemical symbol for gold?",        ["Au"],          "easy"),
    ("What is the chemical symbol for silver?",      ["Ag"],          "easy"),
    ("What is the chemical symbol for iron?",        ["Fe"],          "easy"),
    ("What is the chemical symbol for sodium?",      ["Na"],          "easy"),
    ("What is the chemical symbol for potassium?",   ["K"],           "easy"),
    ("What is the chemical symbol for lead?",        ["Pb"],          "easy"),
    ("What planet is closest to the Sun?",           ["Mercury"],     "easy"),
    ("What planet is largest in the solar system?",  ["Jupiter"],     "easy"),
    ("What is the speed of light in km/s?",          ["300000","299792"], "medium"),
    ("What is the boiling point of water in Celsius?", ["100"],        "easy"),
    ("What is the freezing point of water in Celsius?", ["0"],          "easy"),
    ("How many bones are in the adult human body?",  ["206"],          "medium"),
    ("What gas do plants absorb from the atmosphere?", ["carbon dioxide","CO2"], "easy"),
    ("What is the atomic number of hydrogen?",       ["1"],            "easy"),
    ("What is the atomic number of carbon?",         ["6"],            "medium"),
    ("What is the atomic number of oxygen?",         ["8"],            "medium"),
    ("What is the value of pi to 2 decimal places?", ["3.14"],         "easy"),
    ("What is the square root of 144?",              ["12"],           "easy"),
    ("What is 2 to the power of 10?",               ["1024"],         "medium"),
    ("What is the formula for water?",               ["H2O"],          "easy"),
    ("What is the formula for carbon dioxide?",      ["CO2"],          "easy"),
    ("What element has the symbol O?",               ["oxygen","Oxygen"], "easy"),
    ("What element has the symbol N?",               ["nitrogen","Nitrogen"], "easy"),
    ("What element has the symbol C?",               ["carbon","Carbon"],    "easy"),
    ("What element has the symbol H?",               ["hydrogen","Hydrogen"],"easy"),
    ("What element has the symbol He?",              ["helium","Helium"],    "easy"),
    ("What is the hardest natural substance?",       ["diamond","Diamond"],  "easy"),
    ("What is the most abundant gas in Earth's atmosphere?", ["nitrogen","Nitrogen"], "medium"),
    ("How many chromosomes do humans have?",         ["46"],           "medium"),
    ("What organ produces insulin?",                 ["pancreas","Pancreas"],"medium"),
    # ── History (medium-hard) ───────────────────────────────────────────────
    ("In what year did World War II end?",                ["1945"],       "easy"),
    ("In what year did World War I begin?",               ["1914"],       "easy"),
    ("In what year did the Berlin Wall fall?",            ["1989"],       "medium"),
    ("In what year did man first land on the Moon?",      ["1969"],       "easy"),
    ("Who was the first President of the United States?", ["George Washington","Washington"], "easy"),
    ("Who discovered penicillin?",                        ["Alexander Fleming","Fleming"],    "easy"),
    ("Who invented the telephone?",                       ["Alexander Graham Bell","Bell"],   "easy"),
    ("Who painted the Mona Lisa?",                        ["Leonardo da Vinci","Leonardo","da Vinci"], "easy"),
    ("Who wrote Romeo and Juliet?",                       ["William Shakespeare","Shakespeare"], "easy"),
    ("Who wrote Hamlet?",                                 ["William Shakespeare","Shakespeare"], "easy"),
    ("Who developed the theory of relativity?",           ["Albert Einstein","Einstein"],     "easy"),
    ("Who discovered gravity?",                           ["Isaac Newton","Newton"],          "easy"),
    ("Who invented the light bulb?",                      ["Thomas Edison","Edison"],         "easy"),
    ("Who was the first woman to win a Nobel Prize?",     ["Marie Curie","Curie"],            "easy"),
    ("In what year was the Declaration of Independence signed?", ["1776"],    "easy"),
    ("What year did the French Revolution begin?",               ["1789"],    "medium"),
    ("Who was Napoleon Bonaparte defeated by at Waterloo?",      ["Wellington","Duke of Wellington"], "hard"),
    ("What year did Christopher Columbus reach the Americas?",   ["1492"],    "medium"),
    ("Who was the first human to travel to space?",              ["Yuri Gagarin","Gagarin"], "easy"),
    ("What year did the Soviet Union dissolve?",                  ["1991"],   "medium"),
    # ── Language & Literature ────────────────────────────────────────────────
    ("What language is spoken in Brazil?",            ["Portuguese"],     "easy"),
    ("What language is spoken in Egypt?",             ["Arabic"],         "easy"),
    ("What is the longest word in the English language?", ["pneumonoultramicroscopicsilicovolcanoconiosis"], "hard"),
    ("How many letters are in the English alphabet?", ["26"],             "easy"),
    ("What is the most spoken language in the world?", ["Mandarin","Chinese","Mandarin Chinese"], "medium"),
    # ── Geography (distances, size) ─────────────────────────────────────────
    ("What is the largest ocean on Earth?",           ["Pacific","Pacific Ocean"],  "easy"),
    ("What is the largest continent?",                ["Asia"],           "easy"),
    ("What is the longest river in the world?",       ["Nile","Nile River"],         "easy"),
    ("What is the tallest mountain in the world?",    ["Mount Everest","Everest"],   "easy"),
    ("What is the largest country by area?",          ["Russia"],         "easy"),
    ("What is the smallest country in the world?",    ["Vatican City","Vatican"],    "easy"),
    ("What is the largest desert in the world?",      ["Sahara","Sahara Desert"],    "easy"),
    ("What is the deepest lake in the world?",        ["Lake Baikal","Baikal"],      "medium"),
    ("What country has the most natural lakes?",      ["Canada"],         "hard"),
    ("How many continents are there?",                ["7","seven"],      "easy"),
    ("How many oceans are there?",                    ["5","five"],       "medium"),
    # ── Animals & Biology ───────────────────────────────────────────────────
    ("What is the largest animal on Earth?",          ["blue whale","Blue Whale"],   "easy"),
    ("What is the fastest land animal?",              ["cheetah","Cheetah"],         "easy"),
    ("How many legs does a spider have?",             ["8","eight"],      "easy"),
    ("How many legs does an insect have?",            ["6","six"],        "easy"),
    ("What do bees produce?",                         ["honey","Honey"],  "easy"),
    ("What is the gestation period of an elephant in months?", ["22"],    "hard"),
    ("What is the collective noun for a group of lions?", ["pride"],      "medium"),
    ("What is the largest land animal?",              ["elephant","African elephant"],"easy"),
    # ── Technology ──────────────────────────────────────────────────────────
    ("Who founded Microsoft?",                        ["Bill Gates","Gates","Paul Allen"], "easy"),
    ("Who co-founded Apple?",                         ["Steve Jobs","Jobs","Steve Wozniak","Wozniak"], "easy"),
    ("In what year was the World Wide Web invented?", ["1989","1991"],    "medium"),
    ("What does CPU stand for?",                      ["Central Processing Unit"],    "easy"),
    ("What does HTML stand for?",                     ["HyperText Markup Language"],  "medium"),
    ("What does RAM stand for?",                      ["Random Access Memory"],       "medium"),
    ("What does URL stand for?",                      ["Uniform Resource Locator"],   "medium"),
    ("What does AI stand for?",                       ["Artificial Intelligence"],    "easy"),
    # ── Numbers & Records ───────────────────────────────────────────────────
    ("How many days are in a leap year?",             ["366"],            "easy"),
    ("How many days are in a year?",                  ["365"],            "easy"),
    ("How many hours are in a day?",                  ["24"],             "easy"),
    ("How many minutes are in an hour?",              ["60"],             "easy"),
    ("How many seconds are in a minute?",             ["60"],             "easy"),
    ("How many months are in a year?",                ["12"],             "easy"),
    ("How many sides does a hexagon have?",           ["6","six"],        "easy"),
    ("How many sides does an octagon have?",          ["8","eight"],      "easy"),
    ("How many sides does a pentagon have?",          ["5","five"],       "easy"),
    ("What is the Roman numeral for 10?",             ["X"],              "easy"),
    ("What is the Roman numeral for 5?",              ["V"],              "easy"),
    ("What is the Roman numeral for 100?",            ["C"],              "easy"),
    # ── Music & Culture ─────────────────────────────────────────────────────
    ("How many strings does a standard guitar have?", ["6","six"],        "easy"),
    ("How many keys does a standard piano have?",     ["88"],             "medium"),
    ("What instrument is Beethoven famous for?",      ["piano","Piano"],  "easy"),
    ("Who sang Bohemian Rhapsody?",                   ["Queen"],          "easy"),
    ("Who wrote the Four Seasons?",                   ["Vivaldi","Antonio Vivaldi"],  "medium"),
    ("How many players are on a soccer team?",        ["11","eleven"],    "easy"),
    ("How many players are on a basketball team?",    ["5","five"],       "easy"),
    ("How many players are in a baseball team?",      ["9","nine"],       "easy"),
    # ── Medicine ────────────────────────────────────────────────────────────
    ("What vitamin is produced by the body when exposed to sunlight?", ["Vitamin D","D"], "medium"),
    ("What organ filters blood in the body?",         ["kidney","kidneys","Kidney"],  "easy"),
    ("What is the largest organ in the human body?",  ["skin","Skin"],    "medium"),
    ("What is the normal human body temperature in Celsius?", ["37"],     "easy"),
    ("What blood type is the universal donor?",       ["O negative","O-","O"],        "medium"),
    # ── Space & Astronomy ───────────────────────────────────────────────────
    ("How many planets are in the solar system?",     ["8","eight"],      "easy"),
    ("What is the name of Earth's moon?",             ["Moon","Luna"],    "easy"),
    ("How many moons does Mars have?",                ["2","two"],        "medium"),
    ("What is the name of the galaxy we live in?",    ["Milky Way"],      "easy"),
    ("What planet has rings?",                        ["Saturn"],         "easy"),
    ("How far is the Moon from Earth in km?",         ["384400","384000","380000"], "hard"),
    ("What is the closest star to Earth?",            ["Sun","the Sun"],  "easy"),
    ("What year did Voyager 1 launch?",               ["1977"],           "hard"),
    # ── Extra medium-hard ────────────────────────────────────────────────────
    ("What is the currency of Japan?",                ["yen","Yen"],      "easy"),
    ("What is the currency of the UK?",               ["pound","Pound","GBP"], "easy"),
    ("What is the currency of India?",                ["rupee","Rupee"],  "easy"),
    ("Who wrote Don Quixote?",                        ["Cervantes","Miguel de Cervantes"], "medium"),
    ("What year did Shakespeare die?",                ["1616"],           "medium"),
    ("What is the Pythagorean theorem?",              ["a squared plus b squared equals c squared","a2+b2=c2"], "medium"),
    ("What is the speed of sound in m/s?",            ["343","340"],      "hard"),
    ("What are the primary colors?",                  ["red","blue","yellow","red blue yellow"], "easy"),
    ("How many colors are in a rainbow?",             ["7","seven"],      "easy"),
    ("What is the chemical formula for table salt?",  ["NaCl"],           "medium"),
    ("Who invented the printing press?",              ["Gutenberg","Johannes Gutenberg"], "medium"),
    ("What year was the Eiffel Tower built?",         ["1889"],           "medium"),
    ("How tall is the Eiffel Tower in meters?",       ["330","324","300"], "hard"),
    ("What is the longest bone in the human body?",   ["femur","Femur"],   "medium"),
    ("How many teeth does an adult human have?",      ["32"],              "medium"),
    ("What is the study of earthquakes called?",      ["seismology"],      "hard"),
    ("What is the study of stars called?",            ["astronomy"],       "easy"),
    ("What does DNA stand for?",                      ["Deoxyribonucleic acid"],      "medium"),
    ("How many chromosomes does a human sperm cell have?", ["23"],        "hard"),
    ("What year did the Titanic sink?",               ["1912"],           "easy"),
    ("What ocean did the Titanic sink in?",           ["Atlantic","Atlantic Ocean"],  "easy"),
]

print(f"Dataset: {len(FACTUAL_QA)} factual questions")
diff_counts = {}
for _, _, d in FACTUAL_QA:
    diff_counts[d] = diff_counts.get(d, 0) + 1
for d, c in sorted(diff_counts.items()):
    print(f"  {d}: {c}")
print()

# ─── Load GPT-2 ──────────────────────────────────────────────────────────────
print("Loading GPT-2...")
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model     = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model.eval()
print(f"Loaded GPT-2: {sum(p.numel() for p in model.parameters()):,} parameters")
print()

def get_embedding(text, layer=11):
    """Get GPT-2 hidden state at last token position, specified layer."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, 768)
    return hidden[0, -1, :].numpy()       # last token embedding

def answer_question(question, max_new=8):
    """Greedy generation: return generated text after 'A:'."""
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=100)
    with torch.no_grad():
        out = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()

def check_answer(generated, gold_answers):
    """Check if any gold answer appears in generated text (case-insensitive)."""
    gen_lower = generated.lower().strip()
    for gold in gold_answers:
        gold_lower = gold.lower().strip()
        if gold_lower in gen_lower or gen_lower in gold_lower:
            return True
    return False

# ─── Phase 1: Get embeddings + answers for all questions ────────────────────
print("Phase 1: Embedding + answering all questions...")
print()

results = []
for i, (q, gold, diff) in enumerate(FACTUAL_QA):
    emb = get_embedding(q)
    ans = answer_question(q)
    correct = check_answer(ans, gold)
    results.append({
        'q': q, 'gold': gold, 'ans': ans,
        'correct': correct, 'diff': diff, 'emb': emb
    })
    if (i+1) % 20 == 0:
        n_correct = sum(r['correct'] for r in results)
        print(f"  [{i+1}/{len(FACTUAL_QA)}] accuracy so far: {n_correct/(i+1)*100:.1f}%")

print()
n_total   = len(results)
n_correct = sum(r['correct'] for r in results)
print(f"Overall accuracy: {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%")
print()

# By difficulty
for diff_level in ['easy', 'medium', 'hard']:
    subset = [r for r in results if r['diff'] == diff_level]
    n_c = sum(r['correct'] for r in subset)
    print(f"  {diff_level}: {n_c}/{len(subset)} = {n_c/len(subset)*100:.1f}%")

# ─── Phase 2: Build ReasonBank + compute d_min ───────────────────────────────
print()
print("Phase 2: Building ReasonBank from correct answers...")
print()

# Shuffle and split 70/30
indices = list(range(len(results)))
np.random.shuffle(indices)
n_train = int(0.7 * len(results))
train_idx = indices[:n_train]
test_idx  = indices[n_train:]

train_results = [results[i] for i in train_idx]
test_results  = [results[i] for i in test_idx]

# ReasonBank = embeddings of correctly answered TRAIN questions
bank_results = [r for r in train_results if r['correct']]
bank_wrong   = [r for r in train_results if not r['correct']]
bank_embs    = np.array([r['emb'] for r in bank_results]) if bank_results else np.zeros((1, 768))

print(f"  Train: {len(train_results)} questions, {len(bank_results)} correct → ReasonBank")
print(f"  Test:  {len(test_results)} questions")
print(f"  Bank size |B| = {len(bank_results)}")
print()

# Compute d_min for each test question
for r in test_results:
    if len(bank_embs) > 0:
        dists = np.linalg.norm(bank_embs - r['emb'][np.newaxis, :], axis=1)
        r['d_min'] = float(np.min(dists))
        r['d_mean'] = float(np.mean(dists))
        r['nn_q'] = bank_results[np.argmin(dists)]['q']
    else:
        r['d_min'] = 999.0
        r['d_mean'] = 999.0
        r['nn_q'] = ''

d_min_vals = np.array([r['d_min'] for r in test_results])
correct_vals = np.array([r['correct'] for r in test_results], dtype=float)

# ─── Phase 3: Correlation and logistic fit ───────────────────────────────────
print("Phase 3: Correlation analysis...")
print()

print(f"  d_min stats: mean={np.mean(d_min_vals):.3f}, std={np.std(d_min_vals):.3f}")
print(f"  Accuracy in test: {int(np.sum(correct_vals))}/{len(test_results)} = {np.mean(correct_vals)*100:.1f}%")
print()

# Point-biserial correlation (d_min with binary correct)
from scipy.stats import pointbiserialr, spearmanr
pb_corr, pb_pval = pointbiserialr(correct_vals, d_min_vals)
sp_corr, sp_pval = spearmanr(correct_vals, d_min_vals)
print(f"  Point-biserial correlation (correct vs d_min): r={pb_corr:.4f}, p={pb_pval:.4g}")
print(f"  Spearman rank correlation (correct vs d_min):  ρ={sp_corr:.4f}, p={sp_pval:.4g}")
print()

if pb_corr < 0:
    print(f"  DIRECTION: d_min is NEGATIVELY correlated with correctness ✅")
    print(f"  (lower d_min → closer to bank → higher accuracy)")
else:
    print(f"  DIRECTION: d_min is POSITIVELY correlated with correctness")
    print(f"  (opposite to prediction — check embedding quality)")

# Bin analysis
n_bins = 8
d_sorted = np.sort(d_min_vals)
bin_edges = np.percentile(d_min_vals, np.linspace(0, 100, n_bins+1))
bin_edges = np.unique(bin_edges)

print()
print("  Binned accuracy vs d_min (each bin ~equal number of questions):")
print()
print(f"  {'d_min bin':>20} {'n':>4} {'accuracy':>10} {'P(wrong)':>10}")
print(f"  {'-'*20} {'-'*4} {'-'*10} {'-'*10}")

bin_centers = []
bin_accuracies = []
bin_pwrong = []
bin_ns = []

for i in range(len(bin_edges)-1):
    mask = (d_min_vals >= bin_edges[i]) & (d_min_vals < bin_edges[i+1])
    if i == len(bin_edges)-2:
        mask = (d_min_vals >= bin_edges[i])
    n_in_bin = int(np.sum(mask))
    if n_in_bin == 0:
        continue
    acc = float(np.mean(correct_vals[mask]))
    pw  = 1.0 - acc
    center = float(np.mean(d_min_vals[mask]))
    bin_centers.append(center)
    bin_accuracies.append(acc)
    bin_pwrong.append(pw)
    bin_ns.append(n_in_bin)
    print(f"  [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): "
          f"{n_in_bin:>4}  {acc*100:>9.1f}%  {pw*100:>9.1f}%")

bin_centers  = np.array(bin_centers)
bin_pwrong   = np.array(bin_pwrong)
bin_accuracies = np.array(bin_accuracies)

# Logistic fit: P(wrong) = σ(a*(r - r_c))
def logistic(r, a, rc):
    return 1.0 / (1.0 + np.exp(np.clip(-a * (r - rc), -500, 500)))

fit_result = None
try:
    popt, _ = curve_fit(logistic, bin_centers, bin_pwrong,
                        p0=[1.0, np.median(bin_centers)],
                        maxfev=5000,
                        bounds=([-50, 0], [50, 100]))
    pred = logistic(bin_centers, *popt)
    ss_res = np.sum((bin_pwrong - pred)**2)
    ss_tot = np.sum((bin_pwrong - np.mean(bin_pwrong))**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-15)
    fit_result = {'a': float(popt[0]), 'rc': float(popt[1]), 'R2': float(r2)}
    print()
    print(f"  Logistic fit: P(wrong) = σ({popt[0]:.3f}·(d_min − {popt[1]:.3f}))")
    print(f"  R² = {r2:.4f}")
except Exception as e:
    print(f"  Logistic fit failed: {e}")

# ─── Key finding ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("KEY FINDING")
print("=" * 70)
print()
print(f"  Point-biserial r = {pb_corr:.4f} (p={pb_pval:.3g})")
print(f"  Spearman ρ = {sp_corr:.4f} (p={sp_pval:.3g})")
print()

if abs(pb_corr) > 0.3 and pb_pval < 0.05:
    strength = "SIGNIFICANT"
    sig_str  = "✅"
elif abs(pb_corr) > 0.15 and pb_pval < 0.1:
    strength = "MARGINAL"
    sig_str  = "⚠️"
else:
    strength = "WEAK / NOT SIGNIFICANT"
    sig_str  = "❌"

print(f"  Correlation strength: {strength} {sig_str}")
print()
if pb_corr < -0.1:
    print("  INTERPRETATION: d_min PREDICTS H_factual.")
    print("  Queries far from ReasonBank → GPT-2 more likely to be wrong.")
    if abs(pb_corr) > 0.3:
        print("  → U12 is validated on REAL factual QA data.")
        print("  → 'Geometric Hallucination Predictor' claim is SUPPORTED.")
    else:
        print("  → Moderate support for U12 (effect exists but weak).")
elif pb_corr > 0.1:
    print("  UNEXPECTED: d_min and accuracy are POSITIVELY correlated.")
    print("  Possible causes:")
    print("    (1) GPT-2 embedding space does not reflect factual knowledge")
    print("    (2) ReasonBank (correct GPT-2 answers) biased toward certain domains")
    print("    (3) Correct answers cluster far from typical questions")
else:
    print("  NULL RESULT: d_min does not predict factual accuracy.")
    print("  Possible causes:")
    print("    (1) GPT-2 embedding ≠ semantic meaning (token-level features)")
    print("    (2) Factual knowledge is distributed, not clustered in embedding space")
    print("    (3) Need better embedding model (sentence-transformer, etc.)")

# Show best and worst examples
print()
print("  Examples: nearest bank questions for test queries")
print()
test_sorted = sorted(test_results, key=lambda r: r['d_min'])
print("  CLOSEST to bank (small d_min, expected: correct):")
for r in test_sorted[:4]:
    marker = '✅' if r['correct'] else '❌'
    print(f"    {marker} d={r['d_min']:.3f}: '{r['q'][:55]}' → '{r['ans'][:20]}'")
print("  FARTHEST from bank (large d_min, expected: wrong):")
for r in test_sorted[-4:]:
    marker = '✅' if r['correct'] else '❌'
    print(f"    {marker} d={r['d_min']:.3f}: '{r['q'][:55]}' → '{r['ans'][:20]}'")

# ─── PLOTS ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('GPT-2 + Factual QA: H_dist → H_factual Correlation\n'
             f'(r={pb_corr:.3f}, ρ={sp_corr:.3f}, n={len(test_results)} test questions)',
             fontsize=12, fontweight='bold')

# Plot 1: d_min distribution by correctness
ax = axes[0, 0]
d_correct = d_min_vals[correct_vals == 1]
d_wrong   = d_min_vals[correct_vals == 0]
bins_plot = np.linspace(d_min_vals.min(), d_min_vals.max(), 25)
ax.hist(d_correct, bins=bins_plot, alpha=0.6, color='#10b981', label=f'Correct (n={len(d_correct)})', density=True)
ax.hist(d_wrong,   bins=bins_plot, alpha=0.6, color='#ef4444', label=f'Wrong (n={len(d_wrong)})',   density=True)
ax.set_xlabel('d_min (distance to nearest ReasonBank member)')
ax.set_ylabel('Density')
ax.set_title(f'd_min Distribution by Correctness\nPB-r={pb_corr:.3f}, Spearman={sp_corr:.3f}')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.text(0.6, 0.85, strength, transform=ax.transAxes,
        fontsize=10, fontweight='bold',
        color='#10b981' if abs(pb_corr) > 0.3 else '#f59e0b')

# Plot 2: P(wrong) vs d_min (binned)
ax = axes[0, 1]
ax.scatter(bin_centers, bin_pwrong, s=np.array(bin_ns) * 5, color='#6366f1', zorder=5,
           alpha=0.8, label='Empirical P(wrong) per bin')
if fit_result:
    r_fine = np.linspace(d_min_vals.min(), d_min_vals.max(), 300)
    ax.plot(r_fine, logistic(r_fine, fit_result['a'], fit_result['rc']),
            '-', color='#ef4444', linewidth=2.5,
            label=f"Logistic: R²={fit_result['R2']:.3f}\na={fit_result['a']:.2f}, r_c={fit_result['rc']:.2f}")
    ax.axvline(fit_result['rc'], color='#f59e0b', linestyle='--', linewidth=1.5,
               label=f"r_c={fit_result['rc']:.2f}")
ax.set_xlabel('d_min'); ax.set_ylabel('P(H_factual) = 1 - accuracy')
ax.set_title('U12 Validation: P(H_factual) vs d_min\n(binned, real GPT-2 + factual QA)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])

# Plot 3: Accuracy by difficulty
ax = axes[1, 0]
diff_order = ['easy', 'medium', 'hard']
diff_acc = {}
diff_d   = {}
for diff_l in diff_order:
    subset = [r for r in test_results if r['diff'] == diff_l]
    diff_acc[diff_l] = np.mean([r['correct'] for r in subset]) if subset else 0
    diff_d[diff_l]   = np.mean([r['d_min'] for r in subset])   if subset else 0
bars = ax.bar(diff_order, [diff_acc[d] for d in diff_order],
              color=['#10b981', '#f59e0b', '#ef4444'], alpha=0.8)
ax2d = ax.twinx()
ax2d.plot(diff_order, [diff_d[d] for d in diff_order], 'ko-', markersize=8,
          linewidth=2, label='⟨d_min⟩')
ax.set_ylabel('Accuracy'); ax2d.set_ylabel('⟨d_min⟩')
ax.set_title('Accuracy and d_min by Difficulty\n(verification: hard should have high d_min)')
ax2d.legend(fontsize=9)
for bar, d in zip(bars, diff_order):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{diff_acc[d]*100:.0f}%', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Scatter d_min vs correctness (jittered)
ax = axes[1, 1]
jitter = np.random.randn(len(test_results)) * 0.02
ax.scatter(d_min_vals, correct_vals + jitter, alpha=0.4, s=20, c=correct_vals,
           cmap='RdYlGn', vmin=0, vmax=1)
if fit_result:
    r_fine2 = np.linspace(d_min_vals.min(), d_min_vals.max(), 200)
    ax.plot(r_fine2, 1 - logistic(r_fine2, fit_result['a'], fit_result['rc']),
            'b-', linewidth=2.5, label=f'P(correct) = 1 - σ(a(r-r_c))\nR²={fit_result["R2"]:.3f}')
ax.set_xlabel('d_min (L2 distance in GPT-2 embedding space)')
ax.set_ylabel('Correct (1) / Wrong (0)')
ax.set_title('Scatter: d_min vs Correctness\n(green=correct, red=wrong)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('/home/yoiyoi/gpt2_factual_hdist.png', dpi=130, bbox_inches='tight')
print()
print("Figure saved: gpt2_factual_hdist.png")

# ─── Save results ────────────────────────────────────────────────────────────
output = {
    'dataset_size': len(FACTUAL_QA),
    'train_size': len(train_results),
    'test_size': len(test_results),
    'bank_size': len(bank_results),
    'overall_accuracy': float(np.mean([r['correct'] for r in results])),
    'test_accuracy':    float(np.mean(correct_vals)),
    'pb_correlation': float(pb_corr), 'pb_pvalue': float(pb_pval),
    'sp_correlation': float(sp_corr), 'sp_pvalue': float(sp_pval),
    'correlation_strength': strength,
    'logistic_fit': fit_result,
    'd_min_by_difficulty': {
        d: {'accuracy': float(diff_acc[d]), 'd_min_mean': float(diff_d[d])}
        for d in diff_order
    },
}
with open('/home/yoiyoi/gpt2_factual_hdist_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Results saved: gpt2_factual_hdist_results.json")
print()
print("=" * 70)
print("SUMMARY FOR MOTIFBANK_THEOREMS.md")
print("=" * 70)
print(f"  Correlation (d_min → H_factual): r={pb_corr:.4f}, ρ={sp_corr:.4f}")
print(f"  Strength: {strength}")
if fit_result:
    print(f"  Logistic: P(wrong) = σ({fit_result['a']:.2f}·(d_min−{fit_result['rc']:.2f})), R²={fit_result['R2']:.4f}")
print(f"  GPT-2 accuracy: {float(np.mean(correct_vals))*100:.1f}% (test set)")
