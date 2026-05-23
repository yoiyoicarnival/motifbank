#!/usr/bin/env python3
"""
motif_granularity.py — motif記述子の粒度最適化実験

問題:
  Si(OH)4 (9原子) から pairwise_desc = 72dim を構築中。
  「どの粒度が精度支配するか？」を実験で解明する。

比較対象:
  [D1] pairwise_dist (36dim): ペア距離のみ
  [D2] coulomb (36dim):       Coulomb 1/r² のみ
  [D3] pairwise_desc (72dim): 距離 + Coulomb [現状]
  [D4] sorted_distances (8dim): 元素別距離ヒストグラム (Si-O×4, O-H×4)
  [D5] moments (18dim): 各ペアタイプの統計モーメント (mean, std, skew)
  [D6] extended+context (72+k dim): 隣接フラグメント中心間距離を追加

それぞれ GP + NW(Matérn) で MAE を比較し、
「どの情報が精度支配しているか」を特定する。

Usage:
  OMP_NUM_THREADS=1 python3 motif_granularity.py
"""

import os, sys, json, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import skew as scipy_skew
from sklearn.preprocessing import StandardScaler

from motifbank_cli import geom_key
from motifbank_ml import GPEnergyPredictor, farthest_point_sampling
from real_gp_benchmark import generate_sioh4_fragments, CACHE_FILE, KCAL
from world_encoder import pairwise_desc


# ─────────────────────────────────────────────────
# §1 各種記述子
# ─────────────────────────────────────────────────

ATOM_TYPES = ["Si", "O", "O", "O", "O", "H", "H", "H", "H"]
N_ATOMS = 9


def build_desc_d1(mol):
    """[D1] ペア距離のみ (36dim)"""
    coords = np.array(mol)
    n = len(coords)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(coords[i] - coords[j]))
    return np.array(sorted(dists), dtype=np.float64)   # sort でinvariant


def build_desc_d2(mol):
    """[D2] Coulomb 1/r² のみ (36dim)"""
    coords = np.array(mol)
    n = len(coords)
    # 核電荷: Si=14, O=8, H=1
    charges = np.array([14, 8, 8, 8, 8, 1, 1, 1, 1], dtype=np.float64)
    col = []
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(coords[i] - coords[j])
            col.append(charges[i] * charges[j] / max(r**2, 1e-6))
    return np.array(sorted(col), dtype=np.float64)


def build_desc_d3(mol):
    """[D3] 距離 + Coulomb = pairwise_desc (72dim) [現状]"""
    return pairwise_desc(mol)


def build_desc_d4(mol):
    """[D4] 元素別距離ソート (8dim): Si-O×4最近傍, O-H×4最近傍"""
    coords = np.array(mol)
    # Si-O4 (idx 1-4), O-H ペア (O_idx=1-4, H_idx=5-8)
    si = coords[0]
    o4 = coords[1:5]
    h4 = coords[5:9]
    sio_dists = sorted([np.linalg.norm(si - o) for o in o4])
    # 各O に最近傍のH を割り当て
    oh_dists = []
    for oc in o4:
        oh_dists.append(min(np.linalg.norm(oc - hc) for hc in h4))
    oh_dists.sort()
    return np.array(sio_dists + oh_dists, dtype=np.float64)


def build_desc_d5(mol):
    """[D5] ペアタイプ別統計モーメント (18dim): 3タイプ × (mean,std,skew)"""
    coords = np.array(mol)
    si, o4, h4 = coords[0], coords[1:5], coords[5:9]

    sio = [np.linalg.norm(si - o) for o in o4]
    oh  = [np.linalg.norm(o - h) for o in o4 for h in h4]
    oo  = [np.linalg.norm(o4[i] - o4[j]) for i in range(4) for j in range(i+1, 4)]

    feats = []
    for grp in [sio, oh, oo]:
        a = np.array(grp)
        feats += [a.mean(), a.std(), float(scipy_skew(a)) if len(a)>2 else 0.0]
    return np.array(feats, dtype=np.float64)


def build_desc_d6(mol, neighbor_centers, k=5):
    """[D6] pairwise_desc + 隣接フラグメント中心間距離 (72+k dim)"""
    base = build_desc_d3(mol)
    center = np.mean(np.array(mol), axis=0)
    if len(neighbor_centers) == 0:
        return np.concatenate([base, np.zeros(k)])
    dists_to_nbr = np.array([np.linalg.norm(center - c) for c in neighbor_centers])
    dists_to_nbr.sort()
    nbr_feats = dists_to_nbr[:k]
    if len(nbr_feats) < k:
        nbr_feats = np.pad(nbr_feats, (0, k - len(nbr_feats)), constant_values=10.0)
    return np.concatenate([base, nbr_feats])


DESCRIPTORS = {
    "D1:dist(36)":     build_desc_d1,
    "D2:coulomb(36)":  build_desc_d2,
    "D3:full(72)":     build_desc_d3,
    "D4:elem_sorted(8)": build_desc_d4,
    "D5:moments(18)":  build_desc_d5,
}


# ─────────────────────────────────────────────────
# §2 NW Matérn 予測器 (gp_free_predictor.py から流用)
# ─────────────────────────────────────────────────

class NWMatern:
    def __init__(self, length_scale=1.0):
        self.l  = length_scale
        self.X  = None
        self.E  = None

    def fit(self, X, E):
        self.X = np.asarray(X, dtype=np.float64)
        self.E = np.asarray(E, dtype=np.float64)

    def predict(self, x, k=50):
        d2     = np.sum((self.X - x) ** 2, axis=1)
        k_eff  = min(k, len(self.E))
        top_i  = np.argpartition(d2, k_eff)[:k_eff]
        r      = np.sqrt(np.maximum(d2[top_i], 0.0)) / self.l
        sqrt5r = np.sqrt(5.0) * r
        w      = (1.0 + sqrt5r + sqrt5r**2/3.0) * np.exp(-sqrt5r)
        ws     = w.sum()
        if ws < 1e-300:
            return float(self.E[top_i].mean()), 0.0
        w /= ws
        mu    = float(np.dot(w, self.E[top_i]))
        sigma = float(np.sqrt(np.dot(w, (self.E[top_i] - mu)**2)))
        return mu, sigma


def tune_l_nw(X_tr, E_tr, X_val, E_val, k=50):
    best_l, best_mae = 1.0, float("inf")
    nw = NWMatern()
    nw.fit(X_tr, E_tr)
    for l in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        nw.l = l
        ps = [nw.predict(X_val[i], k=k)[0] for i in range(len(E_val))]
        mae = np.mean(np.abs(np.array(ps) - E_val)) * KCAL
        if mae < best_mae:
            best_mae, best_l = mae, l
    return best_l, best_mae


# ─────────────────────────────────────────────────
# §3 メイン実験
# ─────────────────────────────────────────────────

def exp_granularity():
    print("=" * 65)
    print("motif 粒度最適化: descriptor 比較実験")
    print("=" * 65)

    # データ読み込み
    cache = json.load(open(CACHE_FILE))
    mols_all, _, _ = generate_sioh4_fragments(n=600, seed=42)
    mols, energies, gkeys = [], [], []
    for mol in mols_all:
        k = str(geom_key(mol))
        if k in cache:
            mols.append(mol); energies.append(cache[k]); gkeys.append(k)

    n_total = len(mols)
    print(f"\n  データ: {n_total} fragments")

    # FPS で train/test 分割 (D3 の pairwise_desc で)
    X_pd = np.array([pairwise_desc(m) for m in mols])
    n_train = min(max(120, int(n_total * 0.7)), n_total - max(30, int(n_total * 0.2)))
    idx_tr  = farthest_point_sampling(X_pd, n_train, seed=42)
    idx_te  = [i for i in range(n_total) if i not in set(idx_tr)]
    mols_tr = [mols[i] for i in idx_tr]
    mols_te = [mols[i] for i in idx_te]
    y_tr    = np.array([energies[i] for i in idx_tr])
    y_te    = np.array([energies[i] for i in idx_te])
    N_te    = len(idx_te)
    n_val   = min(40, n_train // 4)

    print(f"  train={n_train}, test={N_te}")

    # フラグメント中心 (D6 用)
    centers_tr = [np.mean(np.array(m), axis=0) for m in mols_tr]

    results = {}

    # ── [A] GP baseline (D3 使用) ──
    print("\n  [A] GP baseline (D3:pairwise_desc 72dim)")
    scl = StandardScaler()
    X_gp_tr = scl.fit_transform(X_pd[idx_tr])
    X_gp_te = scl.transform(X_pd[idx_te])
    gp = GPEnergyPredictor()
    t0 = time.time()
    gp.fit(list(X_pd[idx_tr]), list(y_tr), n_atoms_list=[9]*n_train, verbose=False)
    t_gp_fit = time.time() - t0
    t0 = time.time()
    preds_gp = [gp.predict(X_pd[idx_te[i]], n_atoms=9)[0] for i in range(N_te)]
    t_gp_inf = (time.time() - t0) / N_te * 1000
    mae_gp = np.mean(np.abs(np.array(preds_gp) - y_te)) * KCAL
    print(f"  MAE={mae_gp:.3f} kcal  fit={t_gp_fit:.1f}s  infer={t_gp_inf:.2f}ms")
    results["[A] GP (D3)"] = {"mae": mae_gp, "dim": 72}

    # ── 各 descriptor で NW Matérn ──
    print("\n  === NW Matérn 5/2 比較 ===")

    for name, desc_fn in DESCRIPTORS.items():
        # 全モルの descriptor 計算
        try:
            Xs = np.array([desc_fn(m) for m in mols])
        except Exception as e:
            print(f"  {name}: エラー {e}")
            continue
        dim = Xs.shape[1]

        # 正規化
        scl_d = StandardScaler().fit(Xs[idx_tr])
        X_tr_d = scl_d.transform(Xs[idx_tr])
        X_te_d = scl_d.transform(Xs[idx_te])

        # τ チューニング
        X_sub_tr = X_tr_d[:n_train - n_val]
        X_sub_val = X_tr_d[n_train - n_val:]
        y_sub_tr = y_tr[:n_train - n_val]
        y_sub_val = y_tr[n_train - n_val:]
        best_l, _ = tune_l_nw(X_sub_tr, y_sub_tr, X_sub_val, y_sub_val,
                               k=min(50, len(y_sub_tr)-1))

        nw = NWMatern(length_scale=best_l)
        nw.fit(X_tr_d, y_tr)
        k = min(50, n_train - 1)
        t0 = time.time()
        preds = [nw.predict(X_te_d[i], k=k)[0] for i in range(N_te)]
        t_inf = (time.time() - t0) / N_te * 1000
        mae = np.mean(np.abs(np.array(preds) - y_te)) * KCAL
        print(f"  {name:25s}: MAE={mae:.3f} kcal  l={best_l:.1f}  "
              f"{t_inf:.3f}ms/call  ({dim}dim)")
        results[name] = {"mae": mae, "dim": dim}

    # ── [D6] extended + context ──
    print("\n  D6:full+context(72+5)")
    Xs6 = np.array([build_desc_d6(mols[i], centers_tr if i in set(idx_tr) else centers_tr)
                    for i in range(n_total)])
    dim6 = Xs6.shape[1]
    scl6 = StandardScaler().fit(Xs6[idx_tr])
    X6_tr = scl6.transform(Xs6[idx_tr])
    X6_te = scl6.transform(Xs6[idx_te])
    X6_sub_tr = X6_tr[:n_train - n_val]
    X6_sub_val = X6_tr[n_train - n_val:]
    best_l6, _ = tune_l_nw(X6_sub_tr, y_sub_tr, X6_sub_val, y_sub_val,
                            k=min(50, len(y_sub_tr)-1))
    nw6 = NWMatern(length_scale=best_l6)
    nw6.fit(X6_tr, y_tr)
    t0 = time.time()
    preds6 = [nw6.predict(X6_te[i], k=min(50, n_train-1))[0] for i in range(N_te)]
    t6 = (time.time() - t0) / N_te * 1000
    mae6 = np.mean(np.abs(np.array(preds6) - y_te)) * KCAL
    print(f"  D6:full+context(77)      : MAE={mae6:.3f} kcal  l={best_l6:.1f}  "
          f"{t6:.3f}ms/call")
    results["D6:full+context(77)"] = {"mae": mae6, "dim": dim6}

    # ── GP on D4 も比較 ──
    print("\n  [A4] GP on D4:elem_sorted (8dim)")
    X_d4 = np.array([build_desc_d4(m) for m in mols])
    gp4 = GPEnergyPredictor()
    t0 = time.time()
    gp4.fit(list(X_d4[idx_tr]), list(y_tr), n_atoms_list=[9]*n_train, verbose=False)
    t_gp4_fit = time.time() - t0
    t0 = time.time()
    preds_gp4 = [gp4.predict(X_d4[idx_te[i]], n_atoms=9)[0] for i in range(N_te)]
    t_gp4_inf = (time.time() - t0) / N_te * 1000
    mae_gp4 = np.mean(np.abs(np.array(preds_gp4) - y_te)) * KCAL
    print(f"  MAE={mae_gp4:.3f} kcal  fit={t_gp4_fit:.1f}s  infer={t_gp4_inf:.2f}ms")
    results["[A4] GP (D4)"] = {"mae": mae_gp4, "dim": 8}

    # ── [A4+] GP on D4+Coulomb (16dim) ──
    print("\n  [A4+] GP on D4+Coulomb (16dim)")
    def build_desc_d4c(mol):
        coords = np.array(mol)
        si = coords[0]; o4 = coords[1:5]; h4 = coords[5:9]
        # 距離
        sio_d = sorted([np.linalg.norm(si - o) for o in o4])
        oh_d  = sorted([min(np.linalg.norm(o - h) for h in h4) for o in o4])
        # Coulomb (元素電荷: Si=14, O=8, H=1)
        sio_c = sorted([14*8 / max(np.linalg.norm(si - o)**2, 1e-6) for o in o4])
        oh_c  = sorted([8*1 / max(min(np.linalg.norm(o - h) for h in h4)**2, 1e-6) for o in o4])
        return np.array(sio_d + oh_d + sio_c + oh_c, dtype=np.float64)
    X_d4c = np.array([build_desc_d4c(m) for m in mols])
    gp4c = GPEnergyPredictor()
    t0 = time.time()
    gp4c.fit(list(X_d4c[idx_tr]), list(y_tr), n_atoms_list=[9]*n_train, verbose=False)
    t_gp4c = time.time() - t0
    t0 = time.time()
    preds_gp4c = [gp4c.predict(X_d4c[idx_te[i]], n_atoms=9)[0] for i in range(N_te)]
    t_gp4c_inf = (time.time() - t0) / N_te * 1000
    mae_gp4c = np.mean(np.abs(np.array(preds_gp4c) - y_te)) * KCAL
    print(f"  MAE={mae_gp4c:.3f} kcal  fit={t_gp4c:.1f}s  infer={t_gp4c_inf:.2f}ms")
    results["[A4+] GP (D4+Coulomb 16dim)"] = {"mae": mae_gp4c, "dim": 16}
    # NW on D4+Coulomb
    scl4c = StandardScaler().fit(X_d4c[idx_tr])
    X4c_tr = scl4c.transform(X_d4c[idx_tr]); X4c_te = scl4c.transform(X_d4c[idx_te])
    best_l4c, _ = tune_l_nw(X4c_tr[:n_train-n_val], y_sub_tr,
                             X4c_tr[n_train-n_val:], y_sub_val, k=min(50,n_train-n_val-1))
    nw4c = NWMatern(best_l4c)
    nw4c.fit(X4c_tr, y_tr)
    t0 = time.time()
    preds_nw4c = [nw4c.predict(X4c_te[i], k=min(50,n_train-1))[0] for i in range(N_te)]
    t_nw4c = (time.time() - t0) / N_te * 1000
    mae_nw4c = np.mean(np.abs(np.array(preds_nw4c) - y_te)) * KCAL
    print(f"  NW MAE={mae_nw4c:.3f} kcal  l={best_l4c:.1f}  {t_nw4c:.3f}ms")
    results["D4+Coulomb NW(16dim)"] = {"mae": mae_nw4c, "dim": 16}

    # ── D7: Si-O only (4dim) ──
    print("\n  D7:SiO_only(4dim)")
    def build_desc_d7(mol):
        coords = np.array(mol)
        si = coords[0]; o4 = coords[1:5]
        return np.array(sorted([np.linalg.norm(si - o) for o in o4]), dtype=np.float64)
    X_d7 = np.array([build_desc_d7(m) for m in mols])
    scl7 = StandardScaler().fit(X_d7[idx_tr])
    X7_tr = scl7.transform(X_d7[idx_tr]); X7_te = scl7.transform(X_d7[idx_te])
    best_l7, _ = tune_l_nw(X7_tr[:n_train-n_val], y_sub_tr,
                            X7_tr[n_train-n_val:], y_sub_val, k=min(50,n_train-n_val-1))
    nw7 = NWMatern(length_scale=best_l7)
    nw7.fit(X7_tr, y_tr)
    t0 = time.time()
    preds7 = [nw7.predict(X7_te[i], k=min(50, n_train-1))[0] for i in range(N_te)]
    t7 = (time.time() - t0) / N_te * 1000
    mae7 = np.mean(np.abs(np.array(preds7) - y_te)) * KCAL
    print(f"  D7:SiO_only(4)           : MAE={mae7:.3f} kcal  l={best_l7:.1f}  "
          f"{t7:.3f}ms/call")
    results["D7:SiO_only(4)"] = {"mae": mae7, "dim": 4}

    # ── サマリ ──
    print("\n" + "=" * 65)
    print("★ 粒度最適化: 精度ランキング")
    print("=" * 65)
    sorted_r = sorted(results.items(), key=lambda x: x[1]["mae"])
    for name, r in sorted_r:
        marker = " ← 最良" if name == sorted_r[0][0] else ""
        print(f"  {name:30s}: MAE={r['mae']:.3f} kcal  ({r['dim']}dim){marker}")

    nw_results = {k: v for k, v in results.items() if not k.startswith("[A")}
    nw_best = min(nw_results.items(), key=lambda x: x[1]["mae"])
    print(f"\n  GP (D3 72dim):    MAE={mae_gp:.3f} kcal (ARD Matérn)")
    print(f"  GP (D4  8dim):    MAE={mae_gp4:.3f} kcal (ARD Matérn)")
    print(f"  NW best: {nw_best[0]:20s}: MAE={nw_best[1]['mae']:.3f} kcal ({nw_best[1]['dim']}dim)")
    print(f"  NW/GP gap (D4):   {nw_best[1]['mae'] - mae_gp4:+.3f} kcal")

    return results


if __name__ == "__main__":
    r = exp_granularity()
    out = os.path.join(_DIR, "motif_granularity_results.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\n結果保存: motif_granularity_results.json")
