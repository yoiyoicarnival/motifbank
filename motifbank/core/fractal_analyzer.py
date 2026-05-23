"""
fractal_analyzer.py  --  Fractal Motif Analyzer (Layer 2)
フラクタル系のgeom_keyリストを列挙し、DBとの重複を分析する

使い方:
  python3 fractal_analyzer.py --type carpet --gen 2 --db motif_db.sqlite
  python3 fractal_analyzer.py --type sierpinski --gen 5 --db motif_db.sqlite
"""
import numpy as np, argparse, json, time
from itertools import combinations
from pathlib import Path

A = 0.75  # デフォルトスケール

def h3_pos(cx, cy, a=A):
    R = a / np.sqrt(3)
    return np.array([[cx, cy+R], [cx-a/2, cy-R/2], [cx+a/2, cy-R/2]])

def geom_key_from_centers(centers_list, a=A):
    all_h = np.vstack([h3_pos(c[0], c[1], a) for c in centers_list])
    n = len(all_h)
    dists = tuple(sorted(
        round(np.linalg.norm(all_h[i]-all_h[j]), 4)
        for i in range(n) for j in range(i+1, n)
    ))
    return str(dists)

# ── フラクタル中心座標 ─────────────────────────────────────────
def sierpinski_centers(gen, a=A):
    if gen == 1:
        return [(0,0),(a,0),(a/2, a*np.sqrt(3)/2)]
    prev = sierpinski_centers(gen-1, a)
    side = a * 2**(gen-2)
    offs = [(0,0),(side,0),(side/2, side*np.sqrt(3)/2)]
    pts = [(px+ox, py+oy) for (ox,oy) in offs for (px,py) in prev]
    seen = set(); uniq = []
    for p in pts:
        k = (round(p[0],6), round(p[1],6))
        if k not in seen: seen.add(k); uniq.append(p)
    return uniq

def vicsek_centers(gen, a=A):
    if gen == 1: return [(0,0),(a,0),(-a,0),(0,a),(0,-a)]
    prev = vicsek_centers(gen-1, a)
    D3 = 3*a
    offs = [(0,0),(D3,0),(-D3,0),(0,D3),(0,-D3)]
    pts = [(px+ox, py+oy) for (ox,oy) in offs for (px,py) in prev]
    seen = set(); uniq = []
    for p in pts:
        k = (round(p[0],6), round(p[1],6))
        if k not in seen: seen.add(k); uniq.append(p)
    return uniq

def carpet_centers(gen, a=A):
    OFFS8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
    if gen == 1: return [(a*ox, a*oy) for (ox,oy) in OFFS8]
    prev = carpet_centers(gen-1, a)
    D3 = 3*a
    pts = [(D3*ox+px, D3*oy+py) for (ox,oy) in OFFS8 for (px,py) in prev]
    seen = set(); uniq = []
    for p in pts:
        k = (round(p[0],6), round(p[1],6))
        if k not in seen: seen.add(k); uniq.append(p)
    return uniq

FRACTAL_REGISTRY = {
    'sierpinski': sierpinski_centers,
    'vicsek': vicsek_centers,
    'carpet': carpet_centers,
}

def analyze_fractal(fractal_type: str, gen: int, a: float = A,
                    db_path: str = None, verbose: bool = True) -> dict:
    """
    フラクタル系のユニークgeom_keyを列挙し、DB重複を分析する。
    """
    if fractal_type not in FRACTAL_REGISTRY:
        raise ValueError(f"Unknown fractal: {fractal_type}. Known: {list(FRACTAL_REGISTRY.keys())}")

    t0 = time.time()
    centers = FRACTAL_REGISTRY[fractal_type](gen, a)
    N = len(centers)
    n_total = N*(N-1)*(N-2)//6

    if verbose:
        print(f"\n{'='*60}")
        print(f"Fractal Analyzer: {fractal_type} Gen{gen}")
        print(f"  N = {N} clusters")
        print(f"  Total trimers = {n_total:,}")
        print(f"  Enumerating unique classes...")

    # ユニーク trimer クラスを列挙
    trimer_keys = {}
    pair_keys = {}

    for i, j in combinations(range(N), 2):
        k = geom_key_from_centers([centers[i], centers[j]], a)
        if k not in pair_keys:
            pair_keys[k] = (i, j)

    for i, j, k_ in combinations(range(N), 3):
        k = geom_key_from_centers([centers[i], centers[j], centers[k_]], a)
        if k not in trimer_keys:
            trimer_keys[k] = (i, j, k_)

    n_unique_pairs = len(pair_keys)
    n_unique_trimers = len(trimer_keys)
    reuse_rate = n_total / n_unique_trimers if n_unique_trimers > 0 else 0

    result = {
        'fractal': fractal_type,
        'gen': gen,
        'A': a,
        'N': N,
        'n_total_trimers': n_total,
        'n_unique_pairs': n_unique_pairs,
        'n_unique_trimers': n_unique_trimers,
        'reuse_rate': reuse_rate,
        'pair_keys': list(pair_keys.keys()),
        'trimer_keys': list(trimer_keys.keys()),
        'enum_time_s': time.time() - t0,
    }

    if verbose:
        print(f"  Unique pair classes:   {n_unique_pairs:,}")
        print(f"  Unique trimer classes: {n_unique_trimers:,}")
        print(f"  Reuse率: {reuse_rate:.1f}x (圧縮比)")
        print(f"  Enumeration time: {time.time()-t0:.2f}s")

    # DB重複チェック
    if db_path and Path(db_path).exists():
        import sys; sys.path.insert(0, '/tmp')
        from motif_db import MotifDB
        db = MotifDB(db_path)
        coverage = db.coverage(list(trimer_keys.keys()))
        db.close()

        result['db_coverage'] = coverage
        cost_est = coverage['new'] * 62  # seconds per class on PC

        if verbose:
            print(f"\n  DB Coverage:")
            print(f"    Already computed: {coverage['cached']:,} / {coverage['total']:,} "
                  f"({coverage['coverage_pct']:.1f}%)")
            print(f"    New computation needed: {coverage['new']:,} classes")
            print(f"    Estimated time: {cost_est/3600:.1f}h (@ 62s/class on i7)")
            print(f"    Estimated cost: ${coverage['new'] * 0.01:.2f} (@ $0.01/class)")
    else:
        cost_est = n_unique_trimers * 62
        if verbose:
            print(f"\n  (No DB specified)")
            print(f"  Full computation estimate: {cost_est/3600:.1f}h")

    return result


def print_transferability_matrix(fractal_specs: list, a: float = A):
    """複数フラクタルのクラス集合の重複行列を表示。"""
    print(f"\n{'='*60}")
    print("Transferability Matrix")
    print(f"{'='*60}")

    datasets = {}
    for (ftype, gen) in fractal_specs:
        name = f"{ftype[:4]}G{gen}"
        res = analyze_fractal(ftype, gen, a, verbose=False)
        datasets[name] = (set(res['trimer_keys']), res['n_unique_trimers'])
        print(f"  {name}: {res['n_unique_trimers']} unique trimer classes (N={res['N']})")

    print(f"\n{'':15s}", end='')
    names = list(datasets.keys())
    for n in names:
        print(f"{n:>10s}", end='')
    print()

    for n1 in names:
        s1, c1 = datasets[n1]
        print(f"{n1:15s}", end='')
        for n2 in names:
            s2, c2 = datasets[n2]
            if n1 == n2:
                print(f"{'---':>10s}", end='')
            else:
                overlap = len(s1 & s2)
                pct = overlap / c1 * 100
                print(f"{pct:>9.1f}%", end='')
        print()
    print("(各行: その行のクラスのうち、列のクラスと一致する割合)")


if __name__ == '__main__':
    import sys

    # デモ: 複数フラクタルの分析
    print("\n" + "="*60)
    print("FRACTAL MOTIF ANALYZER デモ")
    print("="*60)

    # 各フラクタルの分析
    specs = [
        ('sierpinski', 2),
        ('sierpinski', 3),
        ('sierpinski', 4),
        ('vicsek', 1),
        ('vicsek', 2),
        ('carpet', 2),
    ]

    for ftype, gen in specs:
        r = analyze_fractal(ftype, gen, verbose=True)
        print(f"  → reuse率: {r['reuse_rate']:.1f}x")

    # 転用可能性行列
    print_transferability_matrix(specs)

    # ビジネスシミュレーション
    print(f"\n{'='*60}")
    print("ビジネスシミュレーション")
    print(f"{'='*60}")
    print("""
仮定: 1000人のユーザーが Sierpinski Gen4 を計算したい
  ユーザー1人あたりの通常コスト: 376 classes × 62s = 6.5時間
  1000人合計: 6,500時間 = 270日分の計算

  データベース構築コスト (1回):
    376 unique classes × 62s = 6.5時間
    電気代換算: ~$0.37

  データベース利用時の節約:
    1000ユーザー × (6.5h - 0秒) ≈ 6,500時間 節約
    計算コスト節約額: 1000 × $0.37 = $370

  情報財の価値:
    「1度計算した情報は、無限回再利用可能」
    → データベースの価値は ユーザー数 × 単価 で増大
""")
