"""
GPT-2 small で residual_geometry.py を実際に動かすテスト。
CUDA なしでも動く (CPU, ~few minutes)
"""
import sys
sys.path.insert(0, "/home/yoiyoi")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ── utils (residual_geometry.py からコピー) ────────────────────────────────────

def extract_hidden_states(model, tokenizer, texts, device):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0).squeeze(1).cpu().float().numpy()
        results.append(hs)
    return results

def trajectory_geometry(H):
    n = H.shape[0]
    kappa = np.zeros(n)
    cos_theta = np.ones(n)
    for t in range(1, n - 1):
        v1 = H[t] - H[t-1]
        v2 = H[t+1] - H[t]
        second_diff = H[t+1] - 2*H[t] + H[t-1]
        kappa[t] = np.linalg.norm(second_diff) / (np.linalg.norm(v2)**2 + 1e-8)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            cos_theta[t] = np.dot(v1, v2) / (n1 * n2)
    return kappa, cos_theta

def participation_ratio(H):
    H_c = H - H.mean(axis=0)
    _, s, _ = np.linalg.svd(H_c, full_matrices=False)
    s2 = s**2
    return float(s2.sum()**2 / (s2**2).sum())

def generate_hmm_beliefs(seq_len, n_states=4, n_obs=8, seed=42):
    rng = np.random.default_rng(seed)
    T_mat = rng.dirichlet(np.ones(n_states), size=n_states)
    E_mat = rng.dirichlet(np.ones(n_obs), size=n_states)
    state = rng.integers(n_states)
    obs = []
    for _ in range(seq_len - 1):
        obs.append(rng.choice(n_obs, p=E_mat[state]))
        state = rng.choice(n_states, p=T_mat[state])
    obs.append(rng.choice(n_obs, p=E_mat[state]))
    b = np.ones(n_states) / n_states
    beliefs = np.zeros((seq_len, n_states))
    for t in range(seq_len):
        b = b * E_mat[:, obs[t]]
        b /= b.sum() + 1e-12
        beliefs[t] = b
        if t < seq_len - 1:
            b = T_mat.T @ b
    return obs, beliefs

# ── main ───────────────────────────────────────────────────────────────────────

model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print("Loading GPT-2 small ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)

texts = [
    "The capital of France is Paris. The capital of Germany is Berlin.",
    "Water is composed of hydrogen and oxygen atoms bonded together.",
    "The Eiffel Tower is located in Tokyo and was built in 1889.",  # factual error
]

print("Extracting hidden states ...")
results = {}
for text in texts:
    hs_list = extract_hidden_states(model, tokenizer, [text], device)
    hs = hs_list[0]  # [n_layers+1, seq, hidden]
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    )
    results[text[:40]] = {"hs": hs, "tokens": tokens}
    print(f"  '{text[:40]}...' -> shape {hs.shape}, tokens={len(tokens)}")

# ── geometry per text ──────────────────────────────────────────────────────────
print("\nComputing geometry ...")
fig, axes = plt.subplots(len(texts), 3, figsize=(15, 4 * len(texts)))

for row, (label, data) in enumerate(results.items()):
    hs  = data["hs"]    # [n_layers+1, seq, hidden]
    tks = data["tokens"]
    n_layers, seq_len, hidden = hs.shape

    # PR per layer
    pr_per_layer = np.array([participation_ratio(hs[l]) for l in range(n_layers)])

    # kappa per layer (last token as proxy for "most processed")
    kappas = []
    cos_thetas = []
    for l in range(n_layers):
        k, c = trajectory_geometry(hs[l])
        kappas.append(k)
        cos_thetas.append(c)
    kappas    = np.array(kappas)     # [n_layers, seq]
    cos_thetas = np.array(cos_thetas)

    # belief readout
    _, true_beliefs = generate_hmm_beliefs(seq_len)
    r2_list = []
    for l in range(n_layers):
        X = hs[l]
        reg = Ridge(alpha=1.0).fit(X, true_beliefs)
        r2 = r2_score(true_beliefs, reg.predict(X), multioutput="variance_weighted")
        r2_list.append(r2)
    r2_arr = np.array(r2_list)

    # plot
    ax0 = axes[row, 0]
    im = ax0.imshow(kappas, aspect="auto", origin="lower", cmap="hot")
    ax0.set_title(f"κ: {label}", fontsize=8)
    ax0.set_ylabel("layer")
    plt.colorbar(im, ax=ax0)

    ax1 = axes[row, 1]
    ax1.plot(pr_per_layer, marker=".")
    ax1.set_title("PR (eff dim) per layer")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("PR")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[row, 2]
    ax2.plot(r2_arr, marker=".", color="darkorange")
    ax2.set_title(f"Belief R² (best: L{np.argmax(r2_arr)}, {r2_arr.max():.3f})")
    ax2.set_xlabel("layer")
    ax2.set_ylabel("R²")
    ax2.grid(True, alpha=0.3)

    print(f"  {label}: best belief R²={r2_arr.max():.4f} at layer {np.argmax(r2_arr)}, "
          f"PR=[{pr_per_layer.min():.1f},{pr_per_layer.max():.1f}], "
          f"max_kappa={kappas.max():.4f}")

plt.tight_layout()
fig.savefig("/home/yoiyoi/gpt2_geometry.png", dpi=150, bbox_inches="tight")
print("\nSaved: ~/gpt2_geometry.png")

# ── factual error sentence comparison ─────────────────────────────────────────
# Compare correct vs incorrect sentence at last layer
print("\n── Curvature comparison: correct vs factual error ──")
correct  = "The Eiffel Tower is located in Paris and was built in 1889."
incorrect = "The Eiffel Tower is located in Tokyo and was built in 1889."

hs_c = extract_hidden_states(model, tokenizer, [correct],  device)[0]
hs_i = extract_hidden_states(model, tokenizer, [incorrect], device)[0]

print("Layer | kappa_sum_correct | kappa_sum_incorrect | PR_correct | PR_incorrect")
print("-" * 78)
for l in range(hs_c.shape[0]):
    kc, _ = trajectory_geometry(hs_c[l])
    ki, _ = trajectory_geometry(hs_i[l])
    prc = participation_ratio(hs_c[l])
    pri = participation_ratio(hs_i[l])
    if l % 3 == 0 or l == hs_c.shape[0] - 1:
        print(f"  L{l:2d} | {kc.sum():.4f}              | {ki.sum():.4f}               | {prc:.1f}       | {pri:.1f}")

print("\nDone.")
