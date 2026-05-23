import json, numpy as np
from scipy.special import expit as sigmoid
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

with open('/home/yoiyoi/radar_bank_cache.json') as f:
    data = json.load(f)
bank_embs = np.array([d['emb'] for d in data])

tok = GPT2Tokenizer.from_pretrained('gpt2')
mdl = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
mdl.eval()

A, RC = 0.1044, 83.08

def analyse(text):
    inp = tok(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        out = mdl(**inp)
    emb = out.hidden_states[11][0, -1, :].numpy()
    dist = float(np.min(np.linalg.norm(bank_embs - emb, axis=1)))
    risk = float(sigmoid(A * (dist - RC)))
    ids = tok(f'Q: {text}\nA:', return_tensors='pt', truncation=True, max_length=100)['input_ids']
    ents = []
    for _ in range(20):
        with torch.no_grad():
            o2 = mdl(ids, output_hidden_states=True)
        lp = torch.log_softmax(o2.logits[0, -1, :], dim=-1)
        ents.append(float(-(torch.exp(lp) * lp).sum()))
        ids = torch.cat([ids, o2.logits[0, -1, :].argmax().reshape(1, 1)], dim=1)
    H = float(np.mean(ents))
    print(f'd={dist:6.1f} risk={risk*100:4.0f}% H={H:.2f}  {text[:60]}')

PROMPTS = [
    "What is the capital of France?",
    "Who discovered penicillin?",
    "In what year did World War II end?",
    "Explain the Voynich manuscript linguistic structure.",
    "What is the cuisine of the lost city of Atlantis?",
    "What did Socrates write about quantum mechanics?",
    "Describe Shakespeare play about Mars colonization.",
    "What is the capital of Lemuria?",
    "What was Einstein's 1931 quantum biology paper?",
    "What novel did Shakespeare write about time travel?",
    "Who invented the telephone?",
    "What is the boiling point of dark matter?",
    "Describe the grammar of the Elvish language Sindarin.",
]

print(f'{"d":>7} {"risk":>6} {"H":>5}  prompt')
print('-' * 72)
for p in PROMPTS:
    analyse(p)
