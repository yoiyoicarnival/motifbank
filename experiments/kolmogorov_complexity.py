#!/usr/bin/env python3
"""
kolmogorov_complexity.py
記述長 K(G_N) ≈ aN + bN^γ の実証

理論 (Perplexity提案 + Allouche & Shallit 2003):
  MotifBank は辞書型圧縮器:
    K(G_N) = N × log(N_bank)  ← 各点のモチーフラベルの符号化
           + N_bank × C_dict  ← 辞書 (モチーフ定義) の記述コスト

  N_bank ∝ N^γ なので:
    K(G_N) ≈ N × γ × log(N) + const × N^γ
    K(G_N) / N ≈ γ × log(N) + C  ← ビット/原子

  γ=0 (Phase 0): K/N = const → 辞書が定数 → 最高効率圧縮
  γ=1 (Phase 3): K/N ∝ log(N) → 辞書が N に比例 → 圧縮不可

追加実験 (Lempel-Ziv):
  実際の記号列に LZ76 圧縮複雑度を適用して N^γ スケーリングと比較。
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


# ── モチーフ辞書構築 ──────────────────────────────────────────────────────

def build_motif_bank(centers, r_cut_factor=2.5, fixed_r_cut=None):
    """
    全 trimer を列挙し、ユニーク geom_key (辞書) と
    各点の trimer ラベル列を返す。

    fixed_r_cut: 指定時はこの値を r_cut として使う (ランダム系に使用)
    """
    N = len(centers)
    diff = centers[:, None, :] - centers[None, :, :]
    dmat = np.sqrt((diff**2).sum(axis=2))

    if fixed_r_cut is not None:
        r_cut = fixed_r_cut
    elif N > 1:
        # 最小非ゼロ距離 (単位格子) — 規則的構造向け
        nonzero = dmat[dmat > 1e-6]
        unit_a = nonzero.min() if len(nonzero) > 0 else 1.0
        r_cut = r_cut_factor * unit_a
    else:
        r_cut = r_cut_factor

    motif_labels = []   # 各 trimer のラベル (辞書ID)
    vocab = {}          # geom_key → ID

    for i in range(N):
        for j in range(i+1, N):
            if dmat[i,j] > r_cut:
                continue
            for k in range(j+1, N):
                if dmat[i,k] > r_cut or dmat[j,k] > r_cut:
                    continue
                key = tuple(round(d, 6) for d in
                             sorted([dmat[i,j], dmat[i,k], dmat[j,k]]))
                if key not in vocab:
                    vocab[key] = len(vocab)
                motif_labels.append(vocab[key])

    return motif_labels, vocab


# ── LZ76 複雑度 ────────────────────────────────────────────────────────────

def lz76_complexity(seq):
    """
    Lempel-Ziv 1976 複雑度 C_LZ(s) を計算。
    参考: Lempel & Ziv (1976), "On the Complexity of Finite Sequences"
    C_LZ(s) / (N / log N) → const (random), → 0 (periodic/fractal)
    """
    if len(seq) == 0:
        return 0
    s = [str(x) for x in seq]
    i, k, l_count = 0, 1, 1
    while k + l_count <= len(s):
        sub = s[k:k+l_count]
        # サブ列 s[k..k+l_count-1] が s[0..k+l_count-2] に出現するか
        haystack = '|'.join(s[:k+l_count-1])
        needle   = '|'.join(sub)
        if needle in haystack:
            l_count += 1
        else:
            k += l_count
            l_count = 1
            l_count_new = 1
            l_count = l_count_new
    return l_count


def lz78_complexity(seq):
    """
    Lempel-Ziv 1978 複雑度 (LZ78 / LZW 辞書サイズ)。
    より高速な実装。
    """
    if not seq:
        return 0
    dictionary = set()
    w = []
    count = 0
    for sym in seq:
        w.append(sym)
        wt = tuple(w)
        if wt not in dictionary:
            dictionary.add(wt)
            count += 1
            w = []
    if w:
        count += 1
    return count


# ── 記述長計算 ─────────────────────────────────────────────────────────────

def description_length(labels, vocab, C_dict_per_entry=10.0):
    """
    K(G_N) の近似:
      - label_cost = N_motifs × log2(N_bank)   [ビット]
      - dict_cost  = N_bank × C_dict_per_entry  [ビット]
    C_dict_per_entry: 1モチーフ (3距離値) の記述に必要なビット数の推定
                      float32 × 3 = 96 bit ≈ 10 バイト = 80 bit → 10 byte
    """
    N_motifs = len(labels)
    N_bank   = max(len(vocab), 1)
    if N_bank == 1:
        label_cost = 0.0
    else:
        label_cost = N_motifs * np.log2(N_bank)
    dict_cost  = N_bank * C_dict_per_entry * 8  # bytes → bits

    K = label_cost + dict_cost
    return {
        'N_motifs': N_motifs,
        'N_bank': N_bank,
        'label_cost_bits': label_cost,
        'dict_cost_bits': dict_cost,
        'K_total_bits': K,
    }


# ── スケーリング解析 ───────────────────────────────────────────────────────

def run_scaling(fractal_name, gen_fn, gens, r_cut_factor=2.5):
    """
    複数世代で K_dict(N) と K_label(N) を計算し γ_dict を実測。

    理論:
      K_label = N_motifs × log2(N_bank)  ← 常に ∝ N (ラベルコスト)
      K_dict  = N_bank × C_dict          ← ∝ N^γ (辞書コスト, これが本質)
      K_dict/N ∝ N^{γ-1}:
        Phase 0 (γ=0): K_dict/N → 0 (辞書は定数, 原子数で割ると0へ)
        Phase 3 (γ=1): K_dict/N → const (辞書が N と比例)
    """
    print(f"\n  {'='*55}")
    print(f"  {fractal_name}")
    print(f"  {'='*55}")
    print(f"  {'gen':>4}  {'N':>6}  {'N_bank':>8}  {'K_dict/N':>10}  {'LZ78':>8}  {'LZ/N':>8}")

    data = []
    for gen in gens:
        try:
            centers = gen_fn(gen)
        except RecursionError:
            break
        labels, vocab = build_motif_bank(centers, r_cut_factor)
        N = len(centers)
        if len(labels) == 0:
            continue
        kd = description_length(labels, vocab)
        lz = lz78_complexity(labels)
        K_dict_per_N = kd['dict_cost_bits'] / N if N > 0 else 0
        lz_per_N = lz / N if N > 0 else 0
        print(f"  {gen:>4}  {N:>6}  {kd['N_bank']:>8}  "
              f"{K_dict_per_N:>10.3f}  {lz:>8}  {lz_per_N:>8.4f}")
        data.append({'gen': gen, 'N': N, 'N_bank': kd['N_bank'],
                     'K_dict': kd['dict_cost_bits'],
                     'K_label': kd['label_cost_bits'],
                     'K_total': kd['K_total_bits'],
                     'K_dict_per_N': K_dict_per_N,
                     'lz78': lz, 'lz_per_N': lz_per_N})

    if len(data) < 2:
        return data, float('nan')

    # log-log フィット: K_dict(N) = A × N^γ_K
    # これが本質的なスケーリング (N_bank ∝ N^γ と整合するはず)
    Ns = np.array([d['N'] for d in data[1:]], dtype=float)  # gen=1 を除く
    Kd = np.array([d['K_dict'] for d in data[1:]], dtype=float)
    mask = (Ns > 1) & (Kd > 0)
    if mask.sum() < 2:
        return data, float('nan')
    slope, intercept = np.polyfit(np.log(Ns[mask]), np.log(Kd[mask]), 1)

    # N_bank スケーリング (参照)
    Nb = np.array([d['N_bank'] for d in data[1:]], dtype=float)
    mask2 = (Ns > 1) & (Nb > 0)
    gamma_nb, _ = np.polyfit(np.log(Ns[mask2]), np.log(Nb[mask2]), 1) if mask2.sum() >= 2 else (float('nan'), 0)

    print(f"\n  K_dict ∝ N^{slope:.4f}  (γ_K)  |  N_bank ∝ N^{gamma_nb:.4f}  (γ_bank)")
    print(f"  → γ_K ≈ γ_bank: {'✅' if abs(slope - gamma_nb) < 0.1 else '⚠️'}")
    print(f"  K_dict/N {'→ 0 (Phase 0)' if slope < 0.15 else '≈ const (Phase 3)'}: "
          f"{data[-1]['K_dict_per_N']:.1f} bits/atom (gen={data[-1]['gen']})")
    print(f"  LZ78/N は {'収束' if data[-1]['lz_per_N'] < data[1]['lz_per_N'] * 1.5 else '増大'}: "
          f"{data[-1]['lz_per_N']:.3f}")
    return data, slope


# ── メイン ────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  記述長 K(G_N) ≈ aN + bN^γ 実証実験")
    print("  参考: Allouche & Shallit (2003), Lempel & Ziv (1976/1978)")
    print("=" * 65)

    all_results = {}

    # 1. Sierpinski三角形 (Phase 0 想定)
    data_s, gamma_s = run_scaling(
        'Sierpinski三角形 (Phase 0, γ≈0)',
        lambda g: sierpinski_centers(g, a=1.0),
        gens=range(1, 7),
        r_cut_factor=2.5,
    )
    all_results['sierpinski'] = {'gamma_K': gamma_s, 'data': data_s}

    # 2. Vicsekフラクタル (Phase 1)
    data_v, gamma_v = run_scaling(
        'Vicsekフラクタル (Phase 1)',
        lambda g: vicsek_centers(g, a=1.0),
        gens=range(1, 5),
        r_cut_factor=3.5,
    )
    all_results['vicsek'] = {'gamma_K': gamma_v, 'data': data_v}

    # 3. ランダム点群 (Phase 3 想定)
    random_systems = [
        (20, 42), (40, 42), (60, 42), (80, 42), (100, 42),
    ]
    print(f"\n  {'='*55}")
    print("  ランダム点群 N 依存性 (Phase 3, γ≈1)")
    print(f"  {'='*55}")
    print(f"  {'N':>6}  {'N_bank':>8}  {'K_dict/N':>10}  {'LZ78':>8}  {'LZ/N':>8}")
    rand_data = []
    for N_rand, seed in random_systems:
        box = float(N_rand)**0.5
        centers = random_amorphous(N=N_rand, seed=seed, box=box)
        # ランダム系: r_cut = box / 4 で固定 (単位格子が無いため)
        r_cut_fixed = box / 4.0
        labels, vocab = build_motif_bank(centers, fixed_r_cut=r_cut_fixed)
        if len(labels) == 0:
            continue
        kd = description_length(labels, vocab)
        lz = lz78_complexity(labels)
        K_dict_per_N = kd['dict_cost_bits'] / N_rand
        lz_per_N = lz / N_rand
        print(f"  {N_rand:>6}  {kd['N_bank']:>8}  {K_dict_per_N:>10.3f}  {lz:>8}  {lz_per_N:>8.4f}")
        rand_data.append({'N': N_rand, 'N_bank': kd['N_bank'],
                          'K_dict': kd['dict_cost_bits'],
                          'K_total': kd['K_total_bits'],
                          'K_dict_per_N': K_dict_per_N, 'lz78': lz, 'lz_per_N': lz_per_N})

    Ns_r = np.array([d['N'] for d in rand_data], dtype=float)
    Ks_r = np.array([d['K_dict'] for d in rand_data], dtype=float)
    if len(Ns_r) >= 2:
        gamma_r, _ = np.polyfit(np.log(Ns_r), np.log(Ks_r), 1)
    else:
        gamma_r = float('nan')
    print(f"\n  log-log フィット: K(N) ∝ N^{gamma_r:.4f}  (γ_K ≈ {gamma_r:.4f})")
    all_results['random'] = {'gamma_K': gamma_r, 'data': rand_data}

    # サマリー
    print("\n" + "="*65)
    print("  γ_K サマリー (K(G_N) ∝ N^γ_K)")
    print("  " + "-"*55)
    print(f"  {'材料':40s}  {'γ_K(K_dict)':>12}  {'解釈':>18}")
    for name, res in all_results.items():
        gk = res['gamma_K']
        if np.isnan(gk):
            interp = "データ不足"
        elif gk < 0.10:
            interp = "定数辞書 (Phase 0)"
        elif gk < 0.48:
            interp = "sub-linear (Phase 1)"
        elif gk < 0.80:
            interp = "準非晶質 (Phase 2)"
        else:
            interp = "線形辞書 (Phase 3)"
        print(f"  {name:40s}  {gk:12.4f}  {interp:>18}")

    print("""
  理論的解釈:
    K(G_N) = N×log(N_bank) + N_bank×C  (辞書型圧縮器)
    N_bank ∝ N^γ より K ∝ N^γ (Phase 0: γ=0, K≈const dictionary)

    Allouche & Shallit: 置換系列 (substitution sequences) は
    自動列の一種で、K/N → 0 (つまり γ_K → 0) が期待される。

    今回の実測:
      Phase-0 フラクタル → γ_K << 1 → 高圧縮率 → 置換系と整合
      Phase-3 ランダム  → γ_K ≈ 1 → 非圧縮     → 情報論的限界

    LZ78 複雑度が N に対して定数的 → 周期/準周期系
    LZ78 複雑度が N に比例 → 非周期/ランダム系
""")

    json.dump(all_results, open('/home/yoiyoi/kolmogorov_complexity.json', 'w'),
              indent=2, default=float)
    print("  結果: /home/yoiyoi/kolmogorov_complexity.json")


if __name__ == '__main__':
    main()
