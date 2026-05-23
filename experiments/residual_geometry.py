"""
residual_geometry.py
Residual stream extraction + belief geometry + trajectory geometry
Compatible: Mistral-7B, Qwen-series, and similar decoder-only models

Usage:
    python residual_geometry.py [model_name_or_path]
    python residual_geometry.py mistralai/Mistral-7B-v0.1
    python residual_geometry.py Qwen/Qwen2.5-7B-Instruct
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from typing import Optional


# ── 1. Model Loading ──────────────────────────────────────────────────────────

def load_model(model_name: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model, device


# ── 2. Residual Stream Extraction ─────────────────────────────────────────────

def extract_hidden_states(model, tokenizer, texts: list, device: str) -> list:
    """
    output_hidden_states=True で各テキストの全層 hidden state を返す。
    Returns: list of np.ndarray [n_layers+1, seq, hidden]  (embedding 含む)
    """
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1,) each [1, seq, hidden]
        hs = torch.stack(out.hidden_states, dim=0)  # [n_layers+1, 1, seq, hidden]
        results.append(hs.squeeze(1).cpu().float().numpy())  # [n_layers+1, seq, hidden]
    return results


def _find_layers(model):
    """Mistral / Qwen / GPT-2 などのアーキテクチャで transformer block を探す。"""
    candidates = [
        "model.layers",          # Mistral, Llama, Qwen2
        "transformer.h",         # GPT-2, GPT-J
        "model.decoder.layers",  # OPT
        "gpt_neox.layers",       # Pythia
    ]
    for attr in candidates:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("transformer block が見つかりません。モデル構造を確認してください。")


def extract_hook_states(model, tokenizer, text: str, device: str) -> np.ndarray:
    """
    hook ベースで各 block の出力 (resid_post 相当) を取る。
    output_hidden_states と比べて embedding 層を含まない点に注意。
    Returns: np.ndarray [n_layers, seq, hidden]
    """
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            activations[name] = x.detach().float().cpu()
        return hook

    layers = _find_layers(model)
    hooks = [block.register_forward_hook(make_hook(f"L{i}"))
             for i, block in enumerate(layers)]

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    keys = sorted(activations.keys(), key=lambda k: int(k[1:]))
    return torch.stack([activations[k].squeeze(0) for k in keys], dim=0).numpy()
    # [n_layers, seq, hidden]


# ── 3. Trajectory Geometry ────────────────────────────────────────────────────

def trajectory_geometry(H: np.ndarray):
    """
    H: [seq, hidden]
    Returns:
        kappa     [seq]  -- 離散曲率 (interior tokens のみ非ゼロ)
        cos_theta [seq]  -- 接線ベクトル間の cos (turning angle)
    """
    n = H.shape[0]
    kappa = np.zeros(n)
    cos_theta = np.ones(n)   # 境界はまっすぐ扱い

    for t in range(1, n - 1):
        v1 = H[t] - H[t - 1]
        v2 = H[t + 1] - H[t]
        second_diff = H[t + 1] - 2 * H[t] + H[t - 1]

        denom_k = np.linalg.norm(v2) ** 2 + 1e-8
        kappa[t] = np.linalg.norm(second_diff) / denom_k

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            cos_theta[t] = np.dot(v1, v2) / (n1 * n2)

    return kappa, cos_theta


def layerwise_geometry(hs: np.ndarray):
    """
    hs: [n_layers, seq, hidden]
    Returns kappas, cos_thetas: [n_layers, seq]
    """
    kappas = []
    cos_thetas = []
    for l in range(hs.shape[0]):
        k, c = trajectory_geometry(hs[l])
        kappas.append(k)
        cos_thetas.append(c)
    return np.array(kappas), np.array(cos_thetas)


def participation_ratio(H: np.ndarray) -> float:
    """
    PR = (sum eigenvalues)^2 / sum(eigenvalues^2)
    residual stream の effective dimension を測る。
    H: [seq, hidden]
    """
    H_centered = H - H.mean(axis=0)
    _, s, _ = np.linalg.svd(H_centered, full_matrices=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def layerwise_pr(hs: np.ndarray) -> np.ndarray:
    """hs: [n_layers, seq, hidden] -> PR per layer [n_layers]"""
    return np.array([participation_ratio(hs[l]) for l in range(hs.shape[0])])


# ── 4. Belief State Linear Readout ────────────────────────────────────────────

def fit_belief_readout(hs: np.ndarray, true_beliefs: np.ndarray,
                       alpha: float = 1e3, n_splits: int = 5,
                       pca_dim: int = 64):
    """
    hs:           [n_layers, seq, hidden]
    true_beliefs: [seq, belief_dim]

    seq_len << hidden_dim なので PCA で次元削減してから Ridge。
    seq が短い (< n_splits) ときは LOO 相当に fallback。
    Returns:
        r2_per_layer: [n_layers]   -- cross-validated R²
        regs:         list of fitted Ridge regressors (full data)
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold

    seq_len = hs.shape[1]
    actual_splits = min(n_splits, seq_len)

    r2_per_layer = []
    regs = []

    for l in range(hs.shape[0]):
        X = hs[l]  # [seq, hidden]

        # PCA → Ridge pipeline で過学習を抑制
        # CV 時の train fold サイズに合わせて n_comp を決める
        train_size = seq_len * (actual_splits - 1) // actual_splits
        n_comp = min(pca_dim, train_size - 1, X.shape[1])
        n_comp = max(n_comp, 1)

        pipe = Pipeline([
            ("pca", PCA(n_components=n_comp)),
            ("ridge", Ridge(alpha=alpha)),
        ])

        if actual_splits >= 2 and n_comp >= 1 and train_size > n_comp:
            cv = KFold(n_splits=actual_splits, shuffle=True, random_state=0)
            y_pred = cross_val_predict(pipe, X, true_beliefs, cv=cv)
            r2 = r2_score(true_beliefs, y_pred, multioutput="variance_weighted")
        else:
            r2 = float("nan")

        pipe.fit(X, true_beliefs)
        r2_per_layer.append(r2)
        regs.append(pipe)

    return np.array(r2_per_layer), regs


# ── 5. Synthetic HMM Belief States ────────────────────────────────────────────

def generate_hmm_beliefs(seq_len: int, n_states: int = 4,
                          n_obs: int = 8, seed: int = 42):
    """
    minimal HMM で observation 列と belief state を生成。
    論文の synthetic process に対応する supervision source。
    Returns:
        obs:     list[int]  length seq_len
        beliefs: np.ndarray [seq_len, n_states]
    """
    rng = np.random.default_rng(seed)
    T_mat = rng.dirichlet(np.ones(n_states), size=n_states)   # [S, S]
    E_mat = rng.dirichlet(np.ones(n_obs),    size=n_states)   # [S, obs]

    state = rng.integers(n_states)
    states = [state]
    obs = []
    for _ in range(seq_len - 1):
        o = rng.choice(n_obs, p=E_mat[state])
        obs.append(o)
        state = rng.choice(n_states, p=T_mat[state])
        states.append(state)
    obs.append(rng.choice(n_obs, p=E_mat[state]))

    # forward algorithm
    beliefs = np.zeros((seq_len, n_states))
    b = np.ones(n_states) / n_states
    for t in range(seq_len):
        b = b * E_mat[:, obs[t]]
        b /= b.sum() + 1e-12
        beliefs[t] = b
        if t < seq_len - 1:
            b = T_mat.T @ b

    return obs, beliefs


# ── 6. Visualization ──────────────────────────────────────────────────────────

def plot_geometry(kappas: np.ndarray, cos_thetas: np.ndarray,
                  pr_per_layer: np.ndarray, tokens: list, title: str = ""):
    n_layers, seq = kappas.shape
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 1, hspace=0.4)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(kappas, aspect="auto", origin="lower",
                     extent=[0, seq, 0, n_layers], cmap="hot")
    ax0.set_ylabel("Layer")
    ax0.set_title(f"Curvature κ   {title}")
    plt.colorbar(im0, ax=ax0)

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(cos_thetas, aspect="auto", origin="lower",
                     extent=[0, seq, 0, n_layers],
                     cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_ylabel("Layer")
    ax1.set_title("cos θ  (turning angle)")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(pr_per_layer, marker="o", color="steelblue")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("PR (eff. dim)")
    ax2.set_title("Participation Ratio per layer")
    ax2.grid(True, alpha=0.3)

    display_tokens = tokens[:seq] if len(tokens) >= seq else tokens
    if len(display_tokens) <= 40:
        for ax in [ax0, ax1]:
            ax.set_xticks(np.arange(len(display_tokens)) + 0.5)
            ax.set_xticklabels(display_tokens, rotation=45, ha="right", fontsize=7)

    return fig


def plot_belief_r2(r2_per_layer: np.ndarray, title: str = "Belief readout R²"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r2_per_layer, marker="o", color="darkorange")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (variance weighted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── 7. Main ───────────────────────────────────────────────────────────────────

def readout_entropy(model, tokenizer, text: str, device: str,
                    n_pca: int = 50, n_splits: int = 5):
    """
    Practical belief proxy: predict next-token entropy from residual stream.
    Requires seq > n_splits * 4 tokens. Returns NaN array otherwise.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hs = torch.stack(out.hidden_states, dim=0).squeeze(1).cpu().float().numpy()
    seq_len = hs.shape[1]

    if seq_len < n_splits * 4:
        return np.full(hs.shape[0], float("nan"))

    logits = out.logits[0].cpu().float().numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1, keepdims=True)

    r2_list = []
    for l in range(hs.shape[0]):
        X = hs[l]
        n_comp = max(1, min(n_pca, seq_len // n_splits - 1, X.shape[1]))
        pipe = Pipeline([("pca", PCA(n_components=n_comp)),
                         ("ridge", Ridge(alpha=1e3))])
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        preds, trues = [], []
        for tr, te in cv.split(X):
            pipe.fit(X[tr], entropy[tr])
            preds.append(pipe.predict(X[te]))
            trues.append(entropy[te])
        r2_list.append(r2_score(np.vstack(trues), np.vstack(preds)))

    return np.array(r2_list)


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    prompt = (sys.argv[2] if len(sys.argv) > 2
              else "The history of artificial intelligence began in")

    print(f"[1/5] Loading {model_name} ...")
    tokenizer, model, device = load_model(model_name)

    print("[2/5] Generating long text ...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    long_text = tokenizer.decode(gen[0], skip_special_tokens=True)
    seq_len = len(tokenizer(long_text)["input_ids"])
    print(f"  seq_len = {seq_len} tokens")

    print("[3/5] Extracting hidden states ...")
    hs_list = extract_hidden_states(model, tokenizer, [long_text], device)
    hs = hs_list[0]  # [n_layers+1, seq, hidden]
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(long_text, return_tensors="pt")["input_ids"][0].tolist()
    )

    print("[4/5] Computing geometry + entropy readout ...")
    kappas, cos_thetas = layerwise_geometry(hs)
    pr_per_layer = layerwise_pr(hs)
    r2_entropy = readout_entropy(model, tokenizer, long_text, device)
    best_l = int(np.nanargmax(r2_entropy))
    print(f"  PR: [{pr_per_layer.min():.2f}, {pr_per_layer.max():.2f}]")
    print(f"  Entropy R² best: {r2_entropy[best_l]:.4f} @ L{best_l}")
    print(f"  R² per layer: {r2_entropy.round(3)}")

    print("[5/5] Saving plots ...")
    short_name = model_name.split("/")[-1]
    fig1 = plot_geometry(kappas, cos_thetas, pr_per_layer, tokens[:40],
                         title=f"({short_name})")
    fig1.savefig("geometry.png", dpi=150, bbox_inches="tight")

    fig2 = plot_belief_r2(r2_entropy, title=f"Entropy readout R² ({short_name})")
    fig2.savefig("entropy_r2.png", dpi=150, bbox_inches="tight")
    print("  -> geometry.png, entropy_r2.png")
    print("Done.")
