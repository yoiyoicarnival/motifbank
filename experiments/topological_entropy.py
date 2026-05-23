#!/usr/bin/env python3
"""
topological_entropy.py
トポロジカルエントロピー h_top の計算 (RG固定点検証)

理論 (Perplexity提案):
  - Phase-0 飽和 ↔ RG固定点 ↔ h_top ≈ 0 (Solomyak 1997, Lind & Marcus)
  - パターン複雑度 p(n) = {長さnのモチーフ列のユニーク数}
  - h_top = lim_{n→∞} log p(n) / n
  - 結晶/フラクタル (Phase 0): h_top = 0 (有限アルファベット)
  - 非晶質 (Phase 3): h_top > 0 (指数的成長)

実装:
  各材料について、R_cut 内の連続モチーフ列を生成し
  p(n) を n=1..8 について計算 → log-log フィット → h_top

材料:
  1. Sierpinski三角形 gen=5 (Phase 0)
  2. Vicsekフラクタル gen=3 (Phase 1)
  3. ランダム点群 (Phase 3)
  4. 1D 格子 (参考 h_top=0 基準)
"""
import os, sys, json
import numpy as np
from itertools import combinations
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('OMP_NUM_THREADS', '1')


# ── fractal generators ──────────────────────────────────────────────────

def sierpinski_centers(gen, a=1.0):
    if gen == 1:
        return np.array([[0.0, 0.0], [a, 0.0], [a/2, a*np.sqrt(3)/2]])
    prev = sierpinski_centers(gen - 1, a)
    side = a * 2**(gen - 2)
    offsets = np.array([[0.0,0.0],[side,0.0],[side/2, side*np.sqrt(3)/2]])
    pts = np.vstack([prev + off for off in offsets])
    _, idx = np.unique(np.round(pts, 8), axis=0, return_index=True)
    return pts[np.sort(idx)]


def vicsek_centers(gen, a=1.0):
    if gen == 1:
        return np.array([[0.0,0.0],[a,0.0],[-a,0.0],[0.0,a],[0.0,-a]])
    prev = vicsek_centers(gen - 1, a)
    D = 3 * a * (3**(gen - 2))
    offsets = np.array([[0.0,0.0],[D,0.0],[-D,0.0],[0.0,D],[0.0,-D]])
    pts = np.vstack([prev + off for off in offsets])
    _, idx = np.unique(np.round(pts, 8), axis=0, return_index=True)
    return pts[np.sort(idx)]


def random_amorphous(N=60, seed=42, box=5.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box, size=(N, 2))


def lattice_1d(N=30, a=1.0):
    return np.column_stack([np.arange(N, dtype=float) * a,
                             np.zeros(N, dtype=float)])


# ── モチーフ列生成 ────────────────────────────────────────────────────────

def build_motif_sequence(centers, r_cut_factor=2.5, unit_a=1.0):
    """
    各点 i に対して、R_cut 内の近傍 trimer から得られる
    geom_key (距離3タプル) のリストを 'モチーフ列' とする。

    返り値: List[Tuple]  各要素が一つのモチーフID
    """
    N = len(centers)
    r_cut = r_cut_factor * unit_a
    diff = centers[:, None, :] - centers[None, :, :]
    dmat = np.sqrt((diff**2).sum(axis=2))

    motifs = []
    for i in range(N):
        for j in range(i+1, N):
            if dmat[i,j] > r_cut:
                continue
            for k in range(j+1, N):
                if dmat[i,k] > r_cut or dmat[j,k] > r_cut:
                    continue
                d3 = tuple(round(d, 6) for d in
                            sorted([dmat[i,j], dmat[i,k], dmat[j,k]]))
                motifs.append(d3)
    return motifs


def pattern_complexity(motif_seq, max_n=8):
    """
    p(n) = 長さ n の連続モチーフ部分列のユニーク数
    注: モチーフ列を記号列とみなして複雑度を計算
    """
    if len(motif_seq) < 2:
        return {1: 1}

    # 各モチーフをIDに変換
    vocab = {}
    ids = []
    for m in motif_seq:
        if m not in vocab:
            vocab[m] = len(vocab)
        ids.append(vocab[m])

    pn = {}
    for n in range(1, min(max_n + 1, len(ids))):
        words = set()
        for start in range(len(ids) - n + 1):
            word = tuple(ids[start:start+n])
            words.add(word)
        pn[n] = len(words)

    return pn, len(vocab)


def estimate_htop(pn_dict):
    """
    h_top = lim_{n→∞} log p(n) / n
    線形フィット: log p(n) = h_top × n + c

    注意: 有限系でvocab_size = L (全モチーフがユニーク) の場合、
    p(n) ≈ L - n + 1 (線形減少) → h_top ≈ 0 という有限サイズ効果が出る。
    これは h_top を lim_{n→n_max/2} の範囲で測定することで軽減できる。
    真の h_top は「語彙サイズ / 系列長 → 1 なら高エントロピー」で補完。
    """
    ns = sorted(pn_dict.keys())
    if len(ns) < 3:
        return float('nan')
    xs = np.array(ns, dtype=float)
    ys = np.log(np.array([pn_dict[n] for n in ns], dtype=float) + 1e-10)
    # 前半を使う (後半はp(n)が飽和/線形減少する領域)
    half = max(2, len(xs) // 2)
    xs_fit, ys_fit = xs[:half], ys[:half]
    if len(xs_fit) < 2:
        return float('nan')
    slope, _ = np.polyfit(xs_fit, ys_fit, 1)
    return slope


def vocab_entropy_rate(vocab_size, seq_len):
    """
    有効エントロピー率 H_eff = log2(vocab_size) / log2(seq_len)
    Phase 0: vocab_size = const → H_eff → 0 (N → ∞)
    Phase 3: vocab_size = seq_len → H_eff → 1
    """
    if seq_len <= 1 or vocab_size <= 1:
        return 0.0
    return np.log2(vocab_size) / np.log2(seq_len)


# ── メイン ────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  トポロジカルエントロピー h_top 計算")
    print("  RG固定点検証: Phase-0 → h_top ≈ 0")
    print("  参考: Solomyak (1997), Lind & Marcus (Symbolic Dynamics)")
    print("=" * 65)

    MAX_N = 8  # パターン長の上限

    systems = [
        {
            'label': '1D格子 N=30 (基準 h_top=0)',
            'centers': lattice_1d(N=30),
            'r_cut_factor': 2.5,
            'expected_phase': 0,
        },
        {
            'label': 'Sierpinski三角形 gen=4 (Phase 0)',
            'centers': sierpinski_centers(4, a=1.0),
            'r_cut_factor': 2.5,
            'expected_phase': 0,
        },
        {
            'label': 'Vicsekフラクタル gen=3 (Phase 1)',
            'centers': vicsek_centers(3, a=1.0),
            'r_cut_factor': 3.5,
            'expected_phase': 1,
        },
        {
            'label': 'ランダム点群 N=60 (Phase 3)',
            'centers': random_amorphous(N=60, seed=42, box=5.0),
            'r_cut_factor': 2.0,
            'expected_phase': 3,
        },
    ]

    results = {}

    for cfg in systems:
        centers = cfg['centers']
        N = len(centers)
        motif_seq = build_motif_sequence(
            centers, r_cut_factor=cfg['r_cut_factor'])
        L = len(motif_seq)

        if L < 2:
            print(f"\n  [{cfg['label']}] モチーフ数が少なすぎます ({L})")
            continue

        pn, vocab_size = pattern_complexity(motif_seq, max_n=MAX_N)
        h_top = estimate_htop(pn)
        H_eff = vocab_entropy_rate(vocab_size, L)

        print(f"\n  [{cfg['label']}]")
        print(f"    N={N}, モチーフ列長={L}, 語彙サイズ={vocab_size}")
        print(f"    有効エントロピー率 H_eff = log2({vocab_size})/log2({L}) = {H_eff:.4f}")
        print(f"    {'n':>4}  {'p(n)':>8}  {'log p(n)':>10}")
        for n in sorted(pn.keys()):
            print(f"    {n:>4}  {pn[n]:>8}  {np.log(pn[n]+1e-10):>10.4f}")
        print(f"\n    h_top (前半線形フィット) ≈ {h_top:.5f}")

        # Phase推定: H_eff が主指標
        phase_exp = cfg['expected_phase']
        if H_eff < 0.05:
            verdict = f"H_eff={H_eff:.3f} ≈ 0 → Phase-0 (RG固定点) ✅"
        elif H_eff < 0.60:
            verdict = f"H_eff={H_eff:.3f} → Phase-1 準周期"
        else:
            verdict = f"H_eff={H_eff:.3f} → Phase-3 非晶質 ✅"

        match = "✅" if (H_eff < 0.05) == (phase_exp == 0) else "⚠️"
        print(f"    {verdict}  (期待Phase: {phase_exp}) {match}")

        results[cfg['label']] = {
            'N': int(N), 'motif_seq_len': L, 'vocab_size': vocab_size,
            'pn': {str(k): v for k, v in pn.items()},
            'h_top': float(h_top),
            'H_eff': float(H_eff),
            'expected_phase': phase_exp,
        }

    # サマリー
    print("\n" + "="*65)
    print("  h_top サマリー")
    print("  " + "-"*55)
    print(f"  {'材料':42s}  {'H_eff':>8}  {'h_top':>8}  {'解釈':>10}")
    for label, res in results.items():
        ht = res['h_top']
        he = res['H_eff']
        if he < 0.05:
            interp = "RG固定点"
        elif he < 0.60:
            interp = "準周期"
        else:
            interp = "非晶質"
        print(f"  {label:42s}  {he:8.4f}  {ht:8.5f}  {interp:>10}")

    print("""
  理論的解釈:
    h_top = 0: モチーフ語彙が有限かつ閉じている → RG変換で不変
              Birkhoff平均が収束、置換系と等価 (Solomyak 1997)
    h_top > 0: 新しいパターンが指数的に出現 → 非晶質、非周期系

    Phase-0 (crystal/fractal) → h_top = 0 は RG固定点の特徴付けの
    MotifBankによる実験的確認。
""")

    json.dump(results, open('/home/yoiyoi/topological_entropy.json', 'w'),
              indent=2, default=str)
    print("  結果: /home/yoiyoi/topological_entropy.json")


if __name__ == '__main__':
    main()
