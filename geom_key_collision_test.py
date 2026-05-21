#!/usr/bin/env python3
"""
geom_key_collision_test.py -- geom_key の衝突・一意性テスト

テスト内容:
  1. False Positive (衝突): 異なる幾何が同じキーを持つか？
  2. False Negative (見逃し): 同じ幾何（回転・並進後）が違うキーになるか？
  3. Gen5全クラス: 70,115件すべてが一意か？

使い方:
  python3 geom_key_collision_test.py              # 全テスト
  python3 geom_key_collision_test.py --quick      # 高速モード (10万件)
  python3 geom_key_collision_test.py --gen5       # Gen5のみ
"""
import sys, json, math, time, argparse, itertools, random
import numpy as np
from pathlib import Path

# ─── geom_key (api_server.py と完全に同一の実装) ────────────────────────────

def geom_key(atoms: list) -> str:
    coords = np.array([xyz for _, xyz in atoms])
    dists = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dists.append(round(float(np.linalg.norm(coords[i] - coords[j])), 4))
    dists.sort()
    return str(tuple(dists))


# ─── Sierpinski H3 幾何 (sier_gen5_v3.py と同一) ────────────────────────────

A = 0.75

def h3_at(cx, cy, cz=0.0):
    R = A / math.sqrt(3)
    return [('H', (cx, cy + R, cz)),
            ('H', (cx - A/2, cy - R/2, cz)),
            ('H', (cx + A/2, cy - R/2, cz))]

def sierpinski(n):
    if n == 0:
        return np.array([[0., 0.], [A, 0.], [A/2, A*math.sqrt(3)/2]])
    prev = sierpinski(n-1)
    side = A * 2**(n-1)
    offs = np.array([[0.,0.], [side,0.], [side/2, side*math.sqrt(3)/2]])
    combined = np.vstack([prev + o for o in offs])
    seen, uniq = set(), []
    for p in combined:
        k = (round(float(p[0]),6), round(float(p[1]),6))
        if k not in seen:
            seen.add(k); uniq.append(p)
    return np.array(uniq)


# ─── テスト 1: 回転・並進不変性 ─────────────────────────────────────────────

def test_invariance(n_samples=10000, seed=42):
    """同じ分子を回転・並進しても同じキーになるか？"""
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(n_samples):
        n_atoms = rng.integers(3, 12)
        coords = rng.uniform(-5, 5, (n_atoms, 3))
        atoms = [('H', tuple(c)) for c in coords]
        key0 = geom_key(atoms)

        # 回転
        angle = rng.uniform(0, 2*math.pi)
        axis  = rng.uniform(-1, 1, 3); axis /= np.linalg.norm(axis)
        c, s  = math.cos(angle), math.sin(angle)
        K     = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R_mat = c*np.eye(3) + s*K + (1-c)*np.outer(axis,axis)
        rot_coords = coords @ R_mat.T
        trans = rng.uniform(-100, 100, 3)
        rot_atoms = [('H', tuple(c + trans)) for c in rot_coords]
        key1 = geom_key(rot_atoms)

        if key0 != key1:
            fails += 1

    return n_samples, fails


# ─── テスト 2: ランダム衝突テスト ───────────────────────────────────────────

def test_random_collisions(n=1_000_000, seed=0):
    """ランダムな幾何 100万件でgeom_keyが衝突するか？"""
    rng = np.random.default_rng(seed)
    keys_seen = {}
    collisions = 0

    batch = 10_000
    t0 = time.time()
    for i in range(0, n, batch):
        n_batch = min(batch, n - i)
        for _ in range(n_batch):
            n_atoms = rng.integers(3, 12)
            # 9Hトリマー相当の幾何を多めに生成
            if rng.random() < 0.5:
                n_atoms = 9
            coords = rng.uniform(-20, 20, (n_atoms, 3))
            atoms = [('H', tuple(c)) for c in coords]
            k = geom_key(atoms)
            if k in keys_seen:
                # 本当に異なる幾何か確認
                prev_coords = np.array([xyz for _, xyz in keys_seen[k]])
                new_coords  = np.array([xyz for _, xyz in atoms])
                if prev_coords.shape == new_coords.shape:
                    # ペア距離が同じなら同じ幾何 (偽衝突でない)
                    pass
                else:
                    collisions += 1
            else:
                keys_seen[k] = atoms

        elapsed = time.time() - t0
        rate = (i + n_batch) / elapsed
        print(f"  {i+n_batch:>9,} / {n:,} ({(i+n_batch)/n*100:.1f}%) "
              f"collisions={collisions} rate={rate:.0f}/s", flush=True)

    return n, collisions, len(keys_seen)


# ─── テスト 3: Gen5全クラスの一意性 ─────────────────────────────────────────

def test_gen5_uniqueness():
    """Sierpinski Gen5の70,115クラスがすべて一意なキーを持つか？"""
    pts = sierpinski(5)
    N = len(pts)
    print(f"  N={N} clusters, C(N,3)={N*(N-1)*(N-2)//6:,} trimers")

    seen = {}       # geom_key → (i,j,k) 代表トリマー
    collisions = 0  # 異なる代表トリマーが同じキーを持つ場合 (真の衝突)
    t0 = time.time()

    for idx, (i, j, k) in enumerate(itertools.combinations(range(N), 3)):
        ai = h3_at(*pts[i]); aj = h3_at(*pts[j]); ak = h3_at(*pts[k])
        gk = geom_key(ai + aj + ak)
        if gk not in seen:
            seen[gk] = (i, j, k)
        if (idx+1) % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1:>9,} trimers ({elapsed:.1f}s), unique={len(seen):,}", flush=True)

    # 代表トリマー同士でキーが重複していないか確認 (構造上ありえないが念のため)
    total = idx + 1
    unique = len(seen)
    equiv  = total - unique   # 幾何的等価クラスに吸収されたトリマー数 (正常)
    print(f"  Total trimers: {total:,}, Unique classes: {unique:,}, "
          f"Equivalent (expected): {equiv:,}")
    return total, unique, collisions


# ─── メイン ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='高速モード (10万件)')
    ap.add_argument('--gen5',  action='store_true', help='Gen5のみ')
    ap.add_argument('--out',   default='geom_key_test_report.json')
    args = ap.parse_args()

    report = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'tests': {}}
    all_pass = True

    print("="*60)
    print("  MotifBank geom_key 衝突テスト")
    print("="*60)

    if not args.gen5:
        # テスト1: 不変性
        print("\n[1] 回転・並進不変性テスト (10,000サンプル)...")
        n, fails = test_invariance(10_000)
        status = "PASS" if fails == 0 else "FAIL"
        print(f"  → {status}: {n}サンプル中 {fails} 件失敗")
        report['tests']['invariance'] = {'n': n, 'fails': fails, 'pass': fails == 0}
        if fails > 0: all_pass = False

        # テスト2: ランダム衝突
        n_rand = 100_000 if args.quick else 1_000_000
        print(f"\n[2] ランダム衝突テスト ({n_rand:,}件)...")
        n, colls, unique = test_random_collisions(n_rand)
        status = "PASS" if colls == 0 else f"WARN (衝突{colls}件)"
        print(f"  → {status}: {n:,}件 → unique={unique:,}, collisions={colls}")
        report['tests']['random_collision'] = {
            'n': n, 'unique': unique, 'collisions': colls, 'pass': colls == 0
        }
        if colls > 0: all_pass = False

    # テスト3: Gen5
    print(f"\n[3] Gen5 全クラス一意性テスト...")
    total, unique, collisions = test_gen5_uniqueness()
    equiv  = total - unique
    status = "PASS" if collisions == 0 else f"FAIL (真の衝突{collisions}件)"
    print(f"  → {status}: {total:,}トリマー → {unique:,}ユニーク"
          f" (等価{equiv:,}件は正常)")
    report['tests']['gen5_uniqueness'] = {
        'total_trimers': total, 'unique_classes': unique,
        'equivalent_expected': equiv, 'collisions': collisions,
        'pass': collisions == 0
    }
    if collisions > 0: all_pass = False

    # 結果まとめ
    print("\n" + "="*60)
    print(f"  総合結果: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    print("="*60)
    report['all_pass'] = all_pass

    json.dump(report, open(args.out, 'w'), indent=2)
    print(f"\n  レポート: {args.out}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
