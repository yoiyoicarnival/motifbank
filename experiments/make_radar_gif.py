#!/usr/bin/env python3
"""
make_radar_gif.py — Generate demo GIF for Hallucination Radar README

Story: Prompt enters GPT-2 embedding manifold → trajectory unfolds →
       risk score reveals whether the model knows the answer.

Output: hallucination_radar_demo.gif
"""

import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

CACHE_FILE = '/home/yoiyoi/radar_bank_cache.json'

# ── Two showcase prompts ──────────────────────────────────────────────────────
DEMO = [
    {
        'prompt':  "What is the capital of France?",
        'label':   "Safe — bank hit",
        'risk':    0.00,
        'color':   '#10b981',   # green
    },
    {
        'prompt':  "Explain the Voynich manuscript's linguistic structure.",
        'label':   "Danger — hallucination",
        'risk':    0.87,
        'color':   '#ef4444',   # red
    },
]

def load_bank():
    with open(CACHE_FILE) as f:
        data = json.load(f)
    return [np.array(d['emb']) for d in data]

def get_embeddings_and_trajectories():
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    print("Loading GPT-2 for GIF generation...", flush=True)
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
    mdl.eval()

    def embed(text):
        inputs = tok(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            out = mdl(**inputs)
        return out.hidden_states[11][0, -1, :].numpy()

    def trajectory(text, n=16):
        p = f"Q: {text}\nA:"
        inputs = tok(p, return_tensors='pt', truncation=True, max_length=100)
        ids = inputs['input_ids']
        states = []
        with torch.no_grad():
            out = mdl(**inputs)
            states.append(out.hidden_states[11][0, -1, :].numpy())
            for _ in range(n):
                out2 = mdl(ids, output_hidden_states=True)
                nxt  = out2.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                ids  = torch.cat([ids, nxt], dim=1)
                states.append(out2.hidden_states[11][0, -1, :].numpy())
        return np.array(states)

    results = []
    for d in DEMO:
        print(f"  Computing: {d['prompt'][:50]}", flush=True)
        emb  = embed(d['prompt'])
        traj = trajectory(d['prompt'])
        results.append({'emb': emb, 'traj': traj})
    return results

def make_gif(bank_embs, results, out='hallucination_radar_demo.gif'):
    # PCA on all vectors
    all_embs = list(bank_embs)
    for r in results:
        all_embs.append(r['emb'])
        all_embs.extend(r['traj'])
    all_embs = np.array(all_embs)

    pca    = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_embs)

    n_bank = len(bank_embs)
    bank_2d = all_2d[:n_bank]

    offset = n_bank
    for r in results:
        n_traj = len(r['traj'])
        r['emb_2d']  = all_2d[offset]
        r['traj_2d'] = all_2d[offset: offset + 1 + n_traj]
        offset += 1 + n_traj

    # ── Animation: 3 panels side by side ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_bank, ax_safe, ax_danger = axes
    risk_cmap = plt.cm.RdYlGn_r
    norm      = Normalize(0, 1)

    N_FRAMES = 40  # total animation frames

    def draw_bank(ax):
        ax.scatter(bank_2d[:, 0], bank_2d[:, 1],
                   c='#10b981', s=12, alpha=0.30, zorder=2)
        ax.set_title('ReasonBank\n(123 verified facts)', fontsize=10)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.2)

    def init():
        for ax in axes:
            ax.cla()
        draw_bank(ax_bank)
        draw_bank(ax_safe)
        draw_bank(ax_danger)
        return []

    def animate(frame):
        for ax in [ax_safe, ax_danger]:
            ax.cla()
        draw_bank(ax_safe)
        draw_bank(ax_danger)

        n_traj = len(results[0]['traj_2d'])  # 17 points
        traj_frame = min(int(frame / N_FRAMES * n_traj) + 1, n_traj)

        # Safe prompt
        r0 = results[0]
        risk0 = DEMO[0]['risk']
        if frame > 0:
            t0 = r0['traj_2d'][:traj_frame]
            if len(t0) > 1:
                ax_safe.plot(t0[:, 0], t0[:, 1], '-', color='#f97316',
                             linewidth=2, alpha=0.8, zorder=3)
                ax_safe.scatter(t0[1:, 0], t0[1:, 1], c='#f97316', s=22, alpha=0.6)
            ax_safe.scatter(r0['emb_2d'][0], r0['emb_2d'][1],
                            c=[risk_cmap(norm(risk0))], s=220,
                            edgecolors='black', linewidths=1.5, zorder=6, marker='*')
        risk_pct0 = int(risk0 * 100 * min(frame / (N_FRAMES * 0.5), 1))
        ax_safe.set_title(
            f'"{DEMO[0]["prompt"][:35]}"\n'
            f'  Risk: {risk_pct0}% ✓ LOW',
            fontsize=9, color='#10b981')
        ax_safe.set_xlabel('PC1'); ax_safe.set_ylabel('PC2')
        ax_safe.grid(True, alpha=0.2)

        # Dangerous prompt (appears after frame N_FRAMES//3)
        r1 = results[1]
        risk1 = DEMO[1]['risk']
        if frame > N_FRAMES // 3:
            f_shifted = frame - N_FRAMES // 3
            tf = min(int(f_shifted / (N_FRAMES * 0.67) * n_traj) + 1, n_traj)
            t1 = r1['traj_2d'][:tf]
            if len(t1) > 1:
                ax_danger.plot(t1[:, 0], t1[:, 1], '-', color='#f97316',
                               linewidth=2, alpha=0.8, zorder=3)
                ax_danger.scatter(t1[1:, 0], t1[1:, 1], c='#f97316', s=22, alpha=0.6)
            ax_danger.scatter(r1['emb_2d'][0], r1['emb_2d'][1],
                              c=[risk_cmap(norm(risk1))], s=220,
                              edgecolors='black', linewidths=1.5, zorder=6, marker='*')
            progress = min((frame - N_FRAMES // 3) / (N_FRAMES * 0.5), 1.0)
            risk_pct1 = int(risk1 * 100 * progress)
            label_color = '#ef4444' if risk_pct1 > 50 else '#f59e0b'
            ax_danger.set_title(
                f'"{DEMO[1]["prompt"][:35]}"\n'
                f'  Risk: {risk_pct1}% ⚠ {"VERY HIGH" if risk_pct1 > 50 else "..."}',
                fontsize=9, color=label_color)
        else:
            ax_danger.set_title(
                f'"{DEMO[1]["prompt"][:35]}"\n  Risk: ...', fontsize=9)
        ax_danger.set_xlabel('PC1'); ax_danger.set_ylabel('PC2')
        ax_danger.grid(True, alpha=0.2)

        fig.suptitle('Hallucination Radar — GPT-2 Embedding Space\n'
                     'How far is the prompt from verified knowledge?',
                     fontsize=11, fontweight='bold')
        return []

    ani = animation.FuncAnimation(
        fig, animate, frames=N_FRAMES, init_func=init,
        interval=100, blit=False)

    writer = animation.PillowWriter(fps=12)
    ani.save(out, writer=writer)
    plt.close(fig)
    print(f"GIF saved: {out}")


if __name__ == '__main__':
    import os
    os.chdir('/home/yoiyoi')
    bank_embs = load_bank()
    print(f"Bank loaded: {len(bank_embs)} facts")
    results   = get_embeddings_and_trajectories()
    make_gif(bank_embs, results)
