#!/usr/bin/env python3
"""
calibrate_sbert.py — sentence-transformer で FACTUAL_QA を再評価し
  1) バンク精度を計測
  2) d_min 分布 (correct vs wrong) を出力
  3) logistic 回帰で A_CALIB, RC_CALIB を推定

Usage:
  OMP_NUM_THREADS=1 python3 calibrate_sbert.py
"""
import json, os, sys, numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar, minimize

CACHE_FILE = '/home/yoiyoi/radar_sbert_cache.json'

# ── factual QA (hallucination_radar.py から転記) ──────────────────────────────
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

# Hallucination-prone prompts (ground truth: wrong/unknown)
HALLUCINATION_PROMPTS = [
    "What was Einstein's 1931 quantum biology paper?",
    "Explain the Voynich manuscript's linguistic structure.",
    "What did Marie Curie discover in 1924?",
    "Who invented the Internet in 1965?",
    "What is the capital of the Kingdom of Lemuria?",
    "What theorem did Euler prove in 1743 about the soul?",
    "What novel did Shakespeare write about time travel?",
    "What was Napoleon's doctoral dissertation on?",
    "What did Newton discover about electromagnetism in 1680?",
    "What is the boiling point of dark matter?",
]

def check_answer(gen, golds):
    g = gen.lower().strip()
    return any(gold.lower().strip() in g or g in gold.lower().strip() for gold in golds)

def main():
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    print("Loading sentence-transformer (all-MiniLM-L6-v2)...", flush=True)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists(CACHE_FILE):
        print(f"Loading cache: {CACHE_FILE}", flush=True)
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        bank = [{'q': d['q'], 'emb': np.array(d['emb']), 'correct': d['correct']} for d in cache['bank']]
    else:
        print("Loading GPT-2 for answer generation...", flush=True)
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        mdl = GPT2LMHeadModel.from_pretrained('gpt2')
        mdl.eval()

        bank = []
        questions = [q for q, _, _ in FACTUAL_QA]
        print(f"Encoding {len(questions)} questions with SBERT...", flush=True)
        embs = sbert.encode(questions, batch_size=64, show_progress_bar=True)

        for i, ((q, gold, diff), emb) in enumerate(zip(FACTUAL_QA, embs)):
            prompt = f"Q: {q}\nA:"
            inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=100)
            with torch.no_grad():
                out = mdl.generate(inputs['input_ids'], max_new_tokens=8,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            ok = check_answer(ans, gold)
            bank.append({'q': q, 'emb': emb.tolist(), 'correct': ok, 'diff': diff, 'ans': ans})

        # Encode hallucination prompts
        hall_embs = sbert.encode(HALLUCINATION_PROMPTS, batch_size=32)

        cache = {
            'bank': bank,
            'hall': [{'q': q, 'emb': e.tolist()} for q, e in zip(HALLUCINATION_PROMPTS, hall_embs)]
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"Cache saved: {CACHE_FILE}", flush=True)

    n_correct = sum(b['correct'] for b in bank)
    print(f"\nBank accuracy: {n_correct}/{len(bank)} = {n_correct/len(bank)*100:.1f}%")

    bank_correct = [b for b in bank if b['correct']]
    bank_embs = np.array([b['emb'] for b in bank_correct])

    # d_min for each correct bank member (leave-one-out)
    dmins_correct = []
    for i, b in enumerate(bank_correct):
        others = np.delete(bank_embs, i, axis=0)
        d = float(np.min(np.linalg.norm(others - b['emb'], axis=1)))
        dmins_correct.append(d)

    # d_min for hallucination prompts
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    hall_data = cache.get('hall', [])
    dmins_hall = []
    for h in hall_data:
        emb = np.array(h['emb'])
        d = float(np.min(np.linalg.norm(bank_embs - emb, axis=1)))
        dmins_hall.append(d)

    print(f"\nd_min stats (correct bank LOO): mean={np.mean(dmins_correct):.3f} ± {np.std(dmins_correct):.3f}")
    print(f"d_min stats (hallucination):    mean={np.mean(dmins_hall):.3f} ± {np.std(dmins_hall):.3f}")

    # Infer logistic: P(hallucination) = sigmoid(a*(d - rc))
    # Fit using combined data: label=0 for correct bank, label=1 for hallucination
    all_d = np.array(dmins_correct + dmins_hall)
    all_y = np.array([0]*len(dmins_correct) + [1]*len(dmins_hall))

    def neg_log_lik(params):
        a, rc = params
        p = sigmoid(a * (all_d - rc))
        p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(all_y * np.log(p) + (1-all_y) * np.log(1-p))

    res = minimize(neg_log_lik, [0.3, np.median(all_d)], method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-5})
    a_fit, rc_fit = res.x
    print(f"\nCalibration fit:")
    print(f"  A_CALIB  = {a_fit:.4f}")
    print(f"  RC_CALIB = {rc_fit:.4f}")
    print(f"  P(H|d=RC) = 0.50  ← transition point")
    print(f"  NLL      = {res.fun:.4f}")

    # EPS_STAR = RC_CALIB + 1/A_CALIB (≈ 84th percentile of sigmoid)
    eps_star = rc_fit + 1.0 / a_fit
    print(f"  EPS_STAR = {eps_star:.1f}  (suggested)")

    # Save result
    result = {
        'n_bank': len(bank), 'n_correct': n_correct, 'accuracy': n_correct/len(bank),
        'A_CALIB': float(a_fit), 'RC_CALIB': float(rc_fit), 'EPS_STAR': float(eps_star),
        'dmins_correct_mean': float(np.mean(dmins_correct)),
        'dmins_correct_std':  float(np.std(dmins_correct)),
        'dmins_hall_mean':    float(np.mean(dmins_hall)),
        'dmins_hall_std':     float(np.std(dmins_hall)),
    }
    with open('/home/yoiyoi/sbert_calibration.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: /home/yoiyoi/sbert_calibration.json")


if __name__ == '__main__':
    main()
