"""
バンク v2: 500+ 件に拡張 + 閾値の実測キャリブレーション
"""
import json, os
import numpy as np
os.chdir('/home/yoiyoi')

from sentence_transformers import SentenceTransformer
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

BANK = [
    # ── 地理（英語）
    "What is the capital of France?", "What is the capital of Japan?",
    "What is the capital of Germany?", "What is the capital of Italy?",
    "What is the capital of the United States?", "What is the capital of China?",
    "What is the capital of India?", "What is the capital of Brazil?",
    "What is the capital of Australia?", "What is the capital of Canada?",
    "What is the capital of Russia?", "What is the capital of South Korea?",
    "What is the capital of Mexico?", "What is the capital of Spain?",
    "What is the capital of the United Kingdom?", "What is the capital of Egypt?",
    "What is the capital of Argentina?", "What is the capital of South Africa?",
    "What is the capital of Turkey?", "What is the capital of Saudi Arabia?",
    "What is the largest country by area?", "What is the largest ocean?",
    "What is the longest river?", "What is the tallest mountain?",
    "What is the smallest country?", "What is the most populous country?",
    "What continent is Brazil in?", "What continent is Japan in?",
    "How many continents are there?", "What is the Pacific Ocean?",

    # ── 歴史
    "When did World War II end?", "When did World War I begin?",
    "When did man first land on the Moon?", "Who was the first US President?",
    "When did the Berlin Wall fall?", "When did the Soviet Union dissolve?",
    "What was the Cold War?", "When did the French Revolution happen?",
    "Who was Napoleon?", "What was the Roman Empire?",
    "When did the American Civil War occur?", "Who was Abraham Lincoln?",
    "When did the Industrial Revolution begin?", "What was World War I about?",
    "Who was Adolf Hitler?", "What happened at Hiroshima in 1945?",
    "When did India gain independence?", "What was the Ottoman Empire?",
    "Who was Julius Caesar?", "What was the Renaissance?",
    "When was the Declaration of Independence signed?", "Who was Christopher Columbus?",
    "What was the Black Death?", "When did the first Olympic Games take place?",
    "Who was Cleopatra?", "What was the Magna Carta?",
    "When did the Great Wall of China get built?", "Who was Genghis Khan?",
    "What was the Byzantine Empire?", "When did Rome fall?",

    # ── 科学・物理
    "What is the speed of light?", "What is the boiling point of water?",
    "What is the freezing point of water?", "What is the atomic number of hydrogen?",
    "What is the atomic number of carbon?", "What is the chemical formula for water?",
    "What is the chemical formula for CO2?", "What is Newton's first law?",
    "What is Einstein's theory of relativity?", "What is the theory of evolution?",
    "What is DNA?", "What is a black hole?", "How does photosynthesis work?",
    "What is the periodic table?", "What is quantum mechanics?",
    "What is gravity?", "How many planets are in the solar system?",
    "What is the closest star to Earth?", "What is the Milky Way?",
    "What causes earthquakes?", "What is electricity?",
    "What is the speed of sound?", "What is nuclear energy?",
    "What is an atom?", "What is a molecule?",
    "What is thermodynamics?", "What is electromagnetism?",
    "What is the Big Bang theory?", "How old is the universe?",
    "What is a supernova?", "What is dark matter?",
    "What is the ozone layer?", "What causes tides?",
    "What is radioactivity?", "What is a neutron star?",

    # ── 数学
    "What is the value of pi?", "What is the Pythagorean theorem?",
    "What is a prime number?", "What is calculus?",
    "What is the square root of 144?", "What is the Fibonacci sequence?",
    "What is probability?", "What is statistics?",
    "What is algebra?", "What is geometry?",
    "What is a derivative?", "What is an integral?",
    "What is a matrix in mathematics?", "What is a vector?",
    "What is the quadratic formula?", "What is a logarithm?",
    "What is the number e?", "What is infinity in mathematics?",
    "What is set theory?", "What is a function in mathematics?",

    # ── テクノロジー
    "What is artificial intelligence?", "What is machine learning?",
    "What is the internet?", "What is an algorithm?",
    "What is programming?", "What is Python?",
    "What is a neural network?", "What is blockchain?",
    "What is cloud computing?", "What does CPU stand for?",
    "What does AI stand for?", "Who founded Microsoft?",
    "Who founded Apple?", "What is the World Wide Web?",
    "What is a database?", "What is cybersecurity?",
    "What is open source software?", "What is an operating system?",
    "What is JavaScript?", "What is data science?",
    "What is deep learning?", "What is natural language processing?",
    "What is a smartphone?", "What is 5G?",
    "What is the cloud?", "What is a server?",
    "What is encryption?", "What is WiFi?",
    "What is an API?", "What is a programming language?",
    "What is software engineering?", "Who invented the internet?",
    "What is a computer?", "What is RAM?",
    "What is a GPU?",

    # ── 生物・医学
    "How many chromosomes do humans have?", "How many bones in the human body?",
    "What is the largest organ?", "What is the function of the heart?",
    "What is the immune system?", "What is metabolism?",
    "What is a cell?", "What is a virus?",
    "What is bacteria?", "How does the brain work?",
    "What is a vaccine?", "What is cancer?",
    "What is diabetes?", "What is antibiotics?",
    "What is genetics?", "What is a gene?",
    "What is evolution?", "What is natural selection?",
    "What is the nervous system?", "How does the digestive system work?",
    "What is blood pressure?", "What is the respiratory system?",
    "What is a stem cell?", "What is the immune response?",
    "What causes the common cold?", "What is malaria?",
    "What is HIV?", "What is the brain made of?",

    # ── 文化・芸術
    "Who painted the Mona Lisa?", "Who wrote Romeo and Juliet?",
    "Who composed the Fifth Symphony?", "What is the Eiffel Tower?",
    "What are the Olympics?", "What is jazz music?",
    "Who was Shakespeare?", "Who was Leonardo da Vinci?",
    "What is the Sistine Chapel?", "What is classical music?",
    "Who wrote Harry Potter?", "What is the Louvre?",
    "What is impressionism in art?", "Who was Picasso?",
    "What is hip hop music?", "What is ballet?",
    "Who was Beethoven?", "What is opera?",
    "What is the Nobel Prize?", "Who was Marie Curie?",
    "What is pop culture?", "What is cinema?",
    "Who was Michael Jackson?", "What is rock music?",

    # ── 社会・経済
    "What is democracy?", "What is capitalism?",
    "What is inflation?", "What is GDP?",
    "What is climate change?", "What is global warming?",
    "What is renewable energy?", "What is solar energy?",
    "What is the United Nations?", "What is the World Bank?",
    "What is free trade?", "What is taxation?",
    "What is a recession?", "What is the stock market?",
    "What is globalization?", "What is immigration?",
    "What is human rights?", "What is poverty?",
    "What is social media?", "What is the European Union?",
    "What is NATO?", "What is international law?",

    # ── 日常・一般
    "How does a car engine work?", "How does a plane fly?",
    "What is cooking?", "What is nutrition?",
    "How much sleep do humans need?", "What is exercise?",
    "What is meditation?", "How does memory work?",
    "What is stress?", "What is happiness?",
    "What is friendship?", "How does language develop in children?",
    "What is education?", "What is a university?",
    "What is a career?", "What is entrepreneurship?",
    "How do you learn a new skill?", "What is creativity?",
    "What is philosophy?", "What is ethics?",
    "What is psychology?", "What is sociology?",

    # ── 日本語（大幅拡充）
    "日本の首都はどこですか？", "東京はどんな都市ですか？",
    "富士山の高さは何メートルですか？", "日本の人口はどのくらいですか？",
    "日本語はどのような言語ですか？", "寿司とは何ですか？",
    "侍とはどういう意味ですか？", "桜の花見とはどのような習慣ですか？",
    "新幹線とは何ですか？", "アニメとは何ですか？",
    "日本の歴史について教えてください", "人工知能とは何ですか？",
    "機械学習とはどういうものですか？", "プログラミングとは何ですか？",
    "インターネットの仕組みを教えてください", "気候変動とは何ですか？",
    "宇宙とはどれくらい広いですか？", "微積分とは何ですか？",
    "進化論について説明してください", "相対性理論とは何ですか？",
    "民主主義とは何ですか？", "経済成長とは何ですか？",
    "人工知能はどのように学習しますか？", "ディープラーニングとは何ですか？",
    "ブロックチェーンとは何ですか？", "スマートフォンの仕組みは？",
    "気候変動の原因は何ですか？", "再生可能エネルギーとは何ですか？",
    "DNAとは何ですか？", "量子力学とは何ですか？",
    "哲学とは何ですか？", "心理学とは何ですか？",
    "芸術とは何ですか？", "音楽とはどういうものですか？",
    "食べ物の栄養について教えてください", "健康的な生活とはどういうものですか？",
    "学習の効率を上げる方法は？", "創造性を高めるにはどうすればいいですか？",
    "ビジネスを始めるにはどうすればいいですか？", "お金の管理方法を教えてください",
    "コミュニケーションスキルを上げるには？", "リーダーシップとは何ですか？",
    "日本の伝統文化について教えてください", "茶道とは何ですか？",
    "柔道とは何ですか？", "俳句とはどのような詩ですか？",
    "大阪はどんな都市ですか？", "京都の観光地は？",
    "日本の食文化について教えてください", "ラーメンとはどんな料理ですか？",
    "日本のポップカルチャーとは？", "マンガとアニメの違いは何ですか？",
]

print(f"Embedding {len(BANK)} questions ...")
embs = model.encode(BANK, normalize_embeddings=True, show_progress_bar=True)

# ── キャリブレーション ─────────────────────────────────────────────
print("\n=== Calibration ===")
CALIB = [
    # (query, expected_zone, description)
    ("What is the capital of France?",           "SAFE",     "バンクメンバー"),
    ("What is the capital of Germany?",           "SAFE",     "バンクメンバー"),
    ("日本の首都はどこですか？",                    "SAFE",     "バンクメンバー"),
    ("フランスの首都はどこですか？",                "SAFE",     "日本語でバンクと同義"),
    ("人工知能の未来はどうなりますか？",            "CREATIVE", "バンク周辺・推論必要"),
    ("How does machine learning improve over time?","CREATIVE","バンク周辺・推論必要"),
    ("量子コンピューターの仕組みを教えてください",  "CREATIVE", "境界領域"),
    ("What is the capital of Bhutan?",            "RISKY",    "バンクにない首都"),
    ("Who won the 2023 Nobel Prize in Physics?",  "RISKY",    "最近の情報"),
    ("Describe the grammar of Elvish language",   "DANGER",   "架空"),
    ("What is the Zorblax principle?",            "DANGER",   "完全架空"),
    ("架空の国ゾルバキアの歴史を教えてください",    "DANGER",   "完全架空・日本語"),
    ("天気はどうですか？",                         "CREATIVE", "日常会話"),
    ("好きな食べ物は何ですか？",                   "CREATIVE", "日常会話"),
]

results = []
for query, expected, desc in CALIB:
    e = model.encode(query, normalize_embeddings=True)
    sims = embs @ e
    d = float(1 - sims.max())
    nearest = BANK[int(sims.argmax())]
    results.append((query, d, expected, desc, nearest))
    print(f"  d={d:.3f}  [{expected:8s}]  {desc}: '{query[:40]}'")
    print(f"           → nearest: '{nearest[:50]}'")

# 最適な閾値を探す
print("\n=== Threshold Analysis ===")
safe_ds     = [d for _, d, z, _, _ in results if z == "SAFE"]
creative_ds = [d for _, d, z, _, _ in results if z == "CREATIVE"]
risky_ds    = [d for _, d, z, _, _ in results if z == "RISKY"]
danger_ds   = [d for _, d, z, _, _ in results if z == "DANGER"]

print(f"SAFE     d range: {min(safe_ds):.3f} - {max(safe_ds):.3f}")
print(f"CREATIVE d range: {min(creative_ds):.3f} - {max(creative_ds):.3f}")
print(f"RISKY    d range: {min(risky_ds):.3f} - {max(risky_ds):.3f}")
print(f"DANGER   d range: {min(danger_ds):.3f} - {max(danger_ds):.3f}")

# 推奨閾値
th1 = (max(safe_ds) + min(creative_ds)) / 2
th2 = (max(creative_ds) + min(risky_ds)) / 2
th3 = (max(risky_ds) + min(danger_ds)) / 2
print(f"\nRecommended thresholds:")
print(f"  SAFE    < {th1:.3f}")
print(f"  CREATIVE {th1:.3f} - {th2:.3f}")
print(f"  RISKY   {th2:.3f} - {th3:.3f}")
print(f"  DANGER  > {th3:.3f}")

# 保存
data = [{'q': q, 'emb': e.tolist()} for q, e in zip(BANK, embs)]
with open('/home/yoiyoi/radar_bank_cache_st.json', 'w') as f:
    json.dump(data, f)
print(f"\nSaved: {len(data)} entries")
print(f"Thresholds to use: {th1:.3f} / {th2:.3f} / {th3:.3f}")
