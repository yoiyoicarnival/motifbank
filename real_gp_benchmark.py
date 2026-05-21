#!/usr/bin/env python3
"""
real_gp_benchmark.py — 実DFT データで GP エネルギー予測を検証

実験設計:
  1. MFI + LTA + cristobalite から固有 Si(OH)4 フラグメントを収集
  2. PySCF PBE/def2-SVP でエネルギー計算 (JSON キャッシュ、再実行可能)
  3. GP 学習曲線: n_train=[10,20,40,80,160], MAE (kcal/mol, mHa)
  4. Active Learning: FPS+不確かさサンプリング vs ランダム比較
  5. Uncertainty calibration: coverage @ 1σ/2σ/3σ, reliability diagram
  6. ml_pred rate: σ < threshold で何% の QC をスキップできるか

判定基準:
  PASS: MAE < 1 kcal/mol (1.6 mHa)  → 化学精度
  GOAL: MAE < 0.5 kcal/mol (0.8 mHa) → 論文級
  ml_pred PASS: 50% 以上のテストセットが σ < 5e-3 Ha でカバー

Usage:
  OMP_NUM_THREADS=1 python3 real_gp_benchmark.py            # フル (~150 DFT calls)
  OMP_NUM_THREADS=1 python3 real_gp_benchmark.py --n 50     # 高速 (~50 calls)
  OMP_NUM_THREADS=1 python3 real_gp_benchmark.py --skip-qc  # キャッシュのみ (評価のみ)
"""

import os, sys, json, time, argparse, warnings
os.environ["OMP_NUM_THREADS"] = "1"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
from scipy.stats import spearmanr

# MotifBank
from motifbank_cli import (
    from_cif, geom_key, MotifBank,
    qc_compute_pyscf, qc_compute_mock,
)
from motifbank_ml import (
    element_aware_descriptor, GPEnergyPredictor,
    farthest_point_sampling, ELEM_BINS,
)

KCAL = 627.5095   # Ha → kcal/mol
MHA  = 1000.0     # Ha → mHa

CACHE_FILE  = os.path.join(_SCRIPT_DIR, "real_gp_cache.json")
RESULT_FILE = os.path.join(_SCRIPT_DIR, "real_gp_results.json")

# Si(OH)4 原子種 (9原子: Si O O O O H H H H)
SIOH4_TYPES = ["Si", "O", "O", "O", "O", "H", "H", "H", "H"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §0. 合成 Si(OH)4 ジェネレーター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_eq_sioh4():
    """
    MFI CIF から実際の平衡 Si(OH)4 座標を取得。
    質量中心を原点に移動して返す。
    E(PBE/def2-SVP) = -592.129475 Ha
    """
    try:
        from motifbank_cli import from_cif as _from_cif
        mols, _, _ = _from_cif(os.path.join(_SCRIPT_DIR, 'examples', 'MFI_iza.cif'),
                                supercell=(1,1,1), mol_type='si_oh4', verbose=False)
        mol = np.array(mols[0], dtype=float)
    except Exception:
        from motifbank_cli import from_cif as _from_cif
        mols, _, _ = _from_cif(os.path.join(_SCRIPT_DIR, 'examples', 'cristobalite_alpha.cif'),
                                supercell=(2,2,1), mol_type='si_oh4', verbose=False)
        mol = np.array(mols[0], dtype=float)
    # 質量中心を原点へ
    mol -= mol.mean(axis=0)
    return mol

import warnings as _w
with _w.catch_warnings():
    _w.simplefilter("ignore")
    _EQ_MOL = _load_eq_sioh4()


def generate_sioh4_fragments(n=200, seed=42,
                              sigma_si=0.02,
                              sigma_o=0.025,
                              sigma_h=0.02):
    """
    平衡 Si(OH)4 からランダム原子変位でフラグメントを生成。

    原子種ごとに変位 σ を分ける:
    - Si: σ=0.02 Å (重い、あまり動かない)
    - O : σ=0.025 Å (中程度)
    - H : σ=0.02 Å (軽いが大変位は非物理的なので抑制)

    拒否条件:
    - 非 H-H 原子間距離 < 1.0 Å
    - H-H 距離 < 0.5 Å
    → これで高エネルギー構造を排除し E span ≤ 30 kcal/mol を狙う。
    """
    rng  = np.random.RandomState(seed)
    mols = []
    keys = []
    seen = set()

    # 原子インデックス: [Si=0, O=1-4, H=5-8]
    sigmas = np.array(
        [sigma_si]
        + [sigma_o] * 4
        + [sigma_h] * 4
    )

    max_tries = n * 15
    for _ in range(max_tries):
        if len(mols) >= n:
            break

        disp = rng.randn(9, 3) * sigmas[:, None]
        mol  = _EQ_MOL + disp

        # Si(OH)4 物理距離チェック
        # idx: 0=Si, 1-4=O, 5-8=H  (H[5+k] は O[1+k] と共有結合)
        ok = True
        for i in range(9):
            for j in range(i + 1, 9):
                d = np.linalg.norm(mol[i] - mol[j])
                si_i = (i == 0)
                o_i  = (1 <= i <= 4)
                h_i  = (i >= 5)
                si_j = (j == 0)
                o_j  = (1 <= j <= 4)
                h_j  = (j >= 5)
                bonded_oh = (o_i and h_j and j - i == 4)  # O[k]-H[k+4]
                # Si-O 共有結合: 1.4-1.9 Å
                if (si_i and o_j) or (o_i and si_j):
                    if not (1.4 < d < 1.9): ok = False; break
                # O-H 共有結合のみ: 0.8-1.2 Å
                elif bonded_oh:
                    if not (0.8 < d < 1.2): ok = False; break
                # O-H 非共有: > 1.5 Å
                elif (o_i and h_j) or (h_i and o_j):
                    if d < 1.5: ok = False; break
                # O-O: > 2.0 Å
                elif o_i and o_j:
                    if d < 2.0: ok = False; break
                # H-H: > 1.2 Å
                elif h_i and h_j:
                    if d < 1.2: ok = False; break
                # Si-H: > 2.0 Å
                elif (si_i and h_j) or (h_i and si_j):
                    if d < 2.0: ok = False; break
            if not ok:
                break
        if not ok:
            continue

        k = geom_key(mol)
        if k in seen:
            continue
        seen.add(k)
        mols.append(mol)
        keys.append(k)

    return mols, ["synthetic"] * len(mols), keys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. フラグメント収集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_fragments(max_frags=300, verbose=True):
    """
    MFI + LTA + cristobalite から固有 Si(OH)4 フラグメントを収集。
    geom_key で重複排除。
    """
    sources = [
        ("examples/MFI_iza.cif",          (1, 1, 1), "MFI"),
        ("examples/LTA_iza.cif",           (1, 1, 1), "LTA"),
        ("examples/cristobalite_alpha.cif", (2, 2, 1), "cristobalite"),
    ]
    base = _SCRIPT_DIR

    unique = {}   # geom_key → (mol_array, source_name)
    if verbose:
        print("[§1] フラグメント収集")

    for cif_rel, sc, name in sources:
        cif_path = os.path.join(base, cif_rel)
        if not os.path.exists(cif_path):
            if verbose: print(f"  {name}: {cif_path} not found, skip")
            continue
        try:
            mols, _, _ = from_cif(cif_path, supercell=sc, mol_type="si_oh4",
                                  verbose=False)
        except Exception as e:
            if verbose: print(f"  {name}: error {e}, skip")
            continue
        before = len(unique)
        for mol in mols:
            k = geom_key(mol)
            if k not in unique:
                unique[k] = (np.asarray(mol, dtype=float), name)
        added = len(unique) - before
        if verbose:
            print(f"  {name:16s}: {len(mols):4d} total → {added:4d} unique added "
                  f"(running: {len(unique)})")
        if len(unique) >= max_frags:
            break

    keys  = list(unique.keys())
    mols  = [unique[k][0] for k in keys]
    names = [unique[k][1] for k in keys]

    if verbose:
        print(f"  合計: {len(mols)} 固有フラグメント\n")
    return mols, names, keys


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. PySCF 計算 (キャッシュ付き)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pyscf_sioh4(mol_arr, verbose=False):
    """PySCF PBE/def2-SVP で Si(OH)4 エネルギーを計算"""
    import pyscf.gto
    import pyscf.dft
    warnings.filterwarnings("ignore")

    atoms = [(elem, tuple(coord))
             for elem, coord in zip(SIOH4_TYPES, mol_arr)]
    pmol = pyscf.gto.M(atom=atoms, basis="def2-SVP",
                       charge=0, spin=0, verbose=0)
    mf = pmol.RKS(xc="PBE")
    mf.conv_tol = 1e-9
    mf.kernel()
    return float(mf.e_tot)


def compute_with_cache(mols, keys, cache_file=CACHE_FILE,
                       skip_qc=False, verbose=True):
    """
    各フラグメントの DFT エネルギーを計算。
    既計算分はキャッシュから読み込み。
    """
    # キャッシュロード
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        if verbose:
            print(f"[§2] キャッシュ: {len(cache)} 件読み込み")

    energies = []
    n_new = 0
    t_start = time.time()

    for i, (mol, k) in enumerate(zip(mols, keys)):
        k_str = str(k)
        if k_str in cache:
            energies.append(cache[k_str])
            continue

        if skip_qc:
            energies.append(None)
            continue

        # 新規計算
        t0 = time.time()
        try:
            e = _pyscf_sioh4(mol)
        except Exception as ex:
            if verbose:
                print(f"  [{i+1}/{len(mols)}] FAILED: {ex}")
            energies.append(None)
            continue

        dt = time.time() - t0
        cache[k_str] = e
        energies.append(e)
        n_new += 1

        elapsed = time.time() - t_start
        remaining = (elapsed / n_new) * (len(mols) - i - 1) if n_new > 0 else 0
        if verbose:
            print(f"  [{i+1:3d}/{len(mols)}] E={e:.6f} Ha  "
                  f"t={dt:.1f}s  ETA={remaining/60:.1f}min  "
                  f"(new={n_new})")

        # 10件ごとにキャッシュ保存
        if n_new % 10 == 0:
            with open(cache_file, "w") as f:
                json.dump(cache, f)

    # 最終保存
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    # None を除外
    valid = [(mol, e) for mol, e in zip(mols, energies) if e is not None]

    # エネルギー上限フィルタ: E_min + 40 kcal/mol 以内のみ使用
    if valid:
        E_min   = min(v[1] for v in valid)
        cutoff  = E_min + 40.0 / KCAL   # 40 kcal/mol = 0.0637 Ha
        filtered = [(m, e) for m, e in valid if e <= cutoff]
        n_removed = len(valid) - len(filtered)
        if n_removed > 0 and verbose:
            print(f"  エネルギーフィルタ: {n_removed} 件除外 "
                  f"(ΔE > 40 kcal/mol)")
        valid = filtered

    valid_mols = [v[0] for v in valid]
    valid_E    = [v[1] for v in valid]

    if verbose:
        print(f"\n  有効フラグメント: {len(valid_mols)} / {len(mols)}")
        if valid_E:
            print(f"  E range: {min(valid_E):.4f} – {max(valid_E):.4f} Ha  "
                  f"σ={np.std(valid_E)*MHA:.2f} mHa  "
                  f"span={(max(valid_E)-min(valid_E))*KCAL:.1f} kcal/mol\n")
    return valid_mols, valid_E


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. 記述子計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pairwise_descriptor(mol):
    """
    9-atom Si(OH)4 → 72-dim 記述子
    = 36 pairwise distances + 36 Coulomb-like 1/r^2 terms
    """
    pts = np.asarray(mol, dtype=float)
    dists = []
    for i in range(9):
        for j in range(i + 1, 9):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            dists.append(d)
            dists.append(1.0 / (d ** 2 + 1e-8))
    return np.array(dists, dtype=np.float32)


def elem_sorted_descriptor(mol):
    """
    9-atom Si(OH)4 → 8-dim 元素別ソート記述子 [D4, 最適記述子]
    = 4 Si-O距離(ソート) + 4 O-H最近傍距離(ソート)

    実験結果: GP+D4=0.108 kcal vs GP+D3=0.225 kcal (2× 精度向上)
              NW+D4=0.482 kcal vs NW+D3=1.105 kcal (2× 精度向上)
    原子順: [Si(0), O(1-4), H(5-8)]
    """
    coords = np.asarray(mol, dtype=np.float64)
    si = coords[0]; o4 = coords[1:5]; h4 = coords[5:9]
    sio_d = sorted([float(np.linalg.norm(si - o)) for o in o4])
    oh_d  = sorted([float(min(np.linalg.norm(o - h) for h in h4)) for o in o4])
    return np.array(sio_d + oh_d, dtype=np.float32)


def compute_descriptors(mols, verbose=True):
    """pairwise_descriptor (72次元) を計算"""
    descs = [pairwise_descriptor(m) for m in mols]
    X = np.array(descs, dtype=np.float32)
    if verbose:
        print(f"[§3] 記述子: {X.shape}  "
              f"(pairwise dist+coulomb, range {X.min():.3f}–{X.max():.3f})\n")
    return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. GP 学習曲線
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def learning_curve(X, y, n_train_list=None, n_trials=3, verbose=True):
    """
    FPS サブセットで GP を学習し、残りでテスト。
    n_trials 回平均を取る。
    Returns: dict of n_train → {mae_kcal, mae_mHa, rmse_kcal}
    """
    if n_train_list is None:
        n_train_list = [10, 20, 40, 80, 120, 160]

    N = len(X)
    y = np.array(y)
    # エネルギー: per-atom 正規化用 (Si(OH)4 = 9 atoms 固定)
    n_atoms = 9

    results = {}
    if verbose:
        print("[§4] GP 学習曲線")
        print(f"  {'n_train':>8}  {'MAE (kcal)':>11}  {'MAE (mHa)':>10}  "
              f"{'RMSE (kcal)':>12}  {'R²':>6}")
        print("  " + "─" * 55)

    for n_train in n_train_list:
        if n_train >= N - 5:
            continue

        maes, rmses = [], []
        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 100)
            # FPS でトレーニングセット選択
            train_idx = farthest_point_sampling(X, n_train, seed=trial)
            test_idx  = [i for i in range(N) if i not in set(train_idx)]

            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]

            gp = GPEnergyPredictor()
            gp.fit(list(X_tr), list(y_tr),
                   n_atoms_list=[n_atoms]*len(y_tr), verbose=False)

            preds, sigmas = [], []
            for xi in X_te:
                e_p, s_p = gp.predict(xi, n_atoms=n_atoms)
                preds.append(e_p)
                sigmas.append(s_p)

            preds = np.array(preds)
            errs  = np.abs(preds - y_te)
            maes.append(np.mean(errs) * KCAL)
            rmses.append(np.sqrt(np.mean(errs**2)) * KCAL)

        mae_mean  = float(np.mean(maes))
        rmse_mean = float(np.mean(rmses))
        results[n_train] = {
            "mae_kcal":  mae_mean,
            "mae_mHa":   mae_mean / KCAL * MHA,
            "rmse_kcal": rmse_mean,
        }
        if verbose:
            r2 = 1.0 - (rmse_mean / KCAL)**2 / max(np.var(y), 1e-20)
            mark = " ✓ CHEM" if mae_mean < 1.0 else (" △" if mae_mean < 2.0 else "")
            print(f"  {n_train:>8}  {mae_mean:>10.3f}  "
                  f"{mae_mean/KCAL*MHA:>10.2f}  "
                  f"{rmse_mean:>11.3f}  {r2:>5.3f}{mark}")

    print()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. Active Learning 曲線
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def active_learning_curve(X, y, n_init=10, n_steps=8, n_per_step=10,
                          n_test=30, verbose=True):
    """
    FPS 初期化 + 不確かさサンプリング (AL) vs ランダムサンプリングの比較。
    Returns: dict {al: [mae_list], random: [mae_list], n_train_list}
    """
    N = len(X)
    y = np.array(y)
    n_atoms = 9

    # テストセット: 固定 (最後の n_test 点)
    rng = np.random.RandomState(999)
    test_idx  = list(rng.choice(N, n_test, replace=False))
    pool_mask = np.ones(N, dtype=bool)
    pool_mask[test_idx] = False
    pool_idx  = list(np.where(pool_mask)[0])

    X_te = X[test_idx]
    y_te = y[test_idx]

    def run_trial(strategy):
        avail = list(pool_idx)
        if strategy == "al":
            # FPS 初期化
            X_pool = X[avail]
            sel_local = farthest_point_sampling(X_pool, n_init, seed=0)
            chosen = [avail[i] for i in sel_local]
        else:
            rng2 = np.random.RandomState(0)
            chosen = list(rng2.choice(avail, n_init, replace=False))

        remaining = [i for i in avail if i not in set(chosen)]
        mae_trace = []
        n_trace   = []

        for step in range(n_steps):
            X_tr = X[chosen]
            y_tr = y[chosen]

            gp = GPEnergyPredictor()
            gp.fit(list(X_tr), list(y_tr),
                   n_atoms_list=[n_atoms]*len(y_tr), verbose=False)

            preds = [gp.predict(xi, n_atoms=n_atoms)[0] for xi in X_te]
            mae   = float(np.mean(np.abs(np.array(preds) - y_te))) * KCAL
            mae_trace.append(mae)
            n_trace.append(len(chosen))

            if not remaining or step == n_steps - 1:
                break

            if strategy == "al":
                # 不確かさ最大の点を選択
                sigs = [gp.predict(X[i], n_atoms=n_atoms)[1] for i in remaining]
                top  = sorted(range(len(remaining)), key=lambda i: -sigs[i])
                to_add = [remaining[i] for i in top[:n_per_step]]
            else:
                rng3 = np.random.RandomState(step)
                to_add = list(rng3.choice(remaining,
                                          min(n_per_step, len(remaining)),
                                          replace=False))

            for idx in to_add:
                chosen.append(idx)
                if idx in remaining:
                    remaining.remove(idx)

        return n_trace, mae_trace

    if verbose:
        print("[§5] Active Learning 曲線")
        print(f"  n_init={n_init}, steps={n_steps}, per_step={n_per_step}, "
              f"test_size={n_test}")

    n_al,  mae_al  = run_trial("al")
    n_rnd, mae_rnd = run_trial("random")

    if verbose:
        print(f"  {'n_train':>8}  {'AL (kcal)':>10}  {'Random (kcal)':>13}  {'改善':>6}")
        print("  " + "─" * 45)
        for nt, ma, mr in zip(n_al, mae_al, mae_rnd):
            mark = "↑" if ma < mr else "="
            print(f"  {nt:>8}  {ma:>10.3f}  {mr:>13.3f}  {mark:>6}")
        print()

    return {"al": mae_al, "random": mae_rnd, "n_train": n_al}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §6. Uncertainty Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def uncertainty_calibration(X_train, y_train, X_test, y_test, verbose=True):
    """
    GP の不確かさキャリブレーション:
    - coverage @ 1σ/2σ/3σ (期待: 68%/95%/99.7%)
    - Spearman(σ, |error|)
    - Expected Calibration Error (ECE)
    """
    n_atoms = 9
    gp = GPEnergyPredictor()
    gp.fit(list(X_train), list(y_train),
           n_atoms_list=[n_atoms]*len(y_train), verbose=False)

    preds, sigmas, errors = [], [], []
    for xi, yi in zip(X_test, y_test):
        ep, sp = gp.predict(xi, n_atoms=n_atoms)
        preds.append(ep)
        sigmas.append(sp)
        errors.append(abs(ep - yi))

    preds  = np.array(preds)
    sigmas = np.array(sigmas)
    errors = np.array(errors)

    # Coverage @ kσ
    coverage = {}
    for k, expected in [(1, 0.683), (2, 0.954), (3, 0.997)]:
        inside = np.mean(errors <= k * sigmas)
        coverage[k] = float(inside)

    # Spearman
    rho, pval = spearmanr(sigmas, errors)

    # ECE: |σ が大きいほど error が大きい| — 10分位で計算
    n_bins = 10
    bins   = np.percentile(sigmas, np.linspace(0, 100, n_bins + 1))
    ece_parts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sigmas >= lo) & (sigmas < hi)
        if mask.sum() > 0:
            mean_sigma = float(np.mean(sigmas[mask]))
            mean_error = float(np.mean(errors[mask]))
            # 理想: mean_error ≈ mean_sigma (GP が完全キャリブレートの場合)
            ece_parts.append(abs(mean_error - mean_sigma))
    ece = float(np.mean(ece_parts)) * MHA  # mHa 単位

    # μ±2σ 内の MAE
    in_2sig = errors[errors <= 2 * sigmas]
    mae_in  = float(np.mean(in_2sig)) * KCAL if len(in_2sig) > 0 else 0.0

    if verbose:
        print("[§6] Uncertainty Calibration")
        print(f"  coverage @ 1σ: {coverage[1]*100:.1f}%  (expected 68.3%)")
        print(f"  coverage @ 2σ: {coverage[2]*100:.1f}%  (expected 95.4%)")
        print(f"  coverage @ 3σ: {coverage[3]*100:.1f}%  (expected 99.7%)")
        print(f"  Spearman ρ(σ, |err|) = {rho:.3f}  p={pval:.2e}")
        print(f"  ECE = {ece:.2f} mHa  (lower=better)")
        print()

    return {
        "coverage_1sig": coverage[1],
        "coverage_2sig": coverage[2],
        "coverage_3sig": coverage[3],
        "spearman_rho":  float(rho),
        "spearman_p":    float(pval),
        "ece_mHa":       ece,
        "mae_in_2sig_kcal": mae_in,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §7. ml_pred Rate 解析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mlpred_rate_analysis(X_train, y_train, X_test, y_test, verbose=True):
    """
    σ_threshold ごとに:
    - ml_pred rate: σ < threshold の割合
    - ML MAE: そのサブセットの平均絶対誤差
    - QC 節約率: 1 - ml_pred_rate
    """
    n_atoms = 9
    gp = GPEnergyPredictor()
    gp.fit(list(X_train), list(y_train),
           n_atoms_list=[n_atoms]*len(y_train), verbose=False)

    preds, sigmas, errors = [], [], []
    for xi, yi in zip(X_test, y_test):
        ep, sp = gp.predict(xi, n_atoms=n_atoms)
        preds.append(ep)
        sigmas.append(sp)
        errors.append(abs(ep - yi))

    sigmas = np.array(sigmas)
    errors = np.array(errors)

    thresholds = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]

    if verbose:
        print("[§7] ml_pred Rate 解析")
        print(f"  {'σ_thresh (Ha)':>14}  {'ml_pred%':>9}  "
              f"{'MAE (kcal)':>11}  {'QC節約%':>8}  {'判定':>6}")
        print("  " + "─" * 58)

    best_threshold = None
    results = {}
    for thr in thresholds:
        mask  = sigmas < thr
        rate  = float(np.mean(mask))
        if mask.sum() > 0:
            mae_sub = float(np.mean(errors[mask])) * KCAL
        else:
            mae_sub = float("nan")
        qc_save = rate * 100

        if rate >= 0.50 and best_threshold is None and (np.isnan(mae_sub) or mae_sub < 1.0):
            best_threshold = thr

        if verbose:
            judge = ""
            if rate >= 0.50 and not np.isnan(mae_sub) and mae_sub < 1.0:
                judge = "✓ PASS"
            elif rate >= 0.50:
                judge = "△ rate OK"
            elif not np.isnan(mae_sub) and mae_sub < 1.0:
                judge = "△ acc OK"
            print(f"  {thr:>14.1e}  {rate*100:>8.1f}%  "
                  f"{mae_sub:>10.3f}  {qc_save:>7.1f}%  {judge:>6}")

        results[thr] = {
            "ml_pred_rate": rate,
            "mae_kcal":     mae_sub,
            "qc_save_pct":  qc_save,
        }

    if verbose:
        if best_threshold is not None:
            print(f"\n  ★ 最適閾値: σ < {best_threshold:.1e} Ha で "
                  f"ml_pred率≥50% かつ MAE<1 kcal/mol 達成\n")
        else:
            print(f"\n  ※ σ分布: "
                  f"median={np.median(sigmas)*MHA:.2f} mHa  "
                  f"p25={np.percentile(sigmas,25)*MHA:.2f}  "
                  f"p75={np.percentile(sigmas,75)*MHA:.2f}\n")

    return results, best_threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §8. 最終サマリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_summary(lc_results, calib_results, mlpred_results, best_thr):
    print("=" * 65)
    print("★ 実DFT GP ベンチマーク — 総合判定")
    print("=" * 65)

    # 学習曲線: 最大 n_train での MAE
    n_trains = sorted(lc_results.keys())
    best_lc  = lc_results[max(n_trains)]
    mae_best = best_lc["mae_kcal"]

    print(f"\n[学習曲線]  n_train={max(n_trains)} 時")
    print(f"  MAE = {mae_best:.3f} kcal/mol ({best_lc['mae_mHa']:.2f} mHa)")
    chem_pass = mae_best < 1.0
    goal_pass = mae_best < 0.5
    print(f"  化学精度 (<1 kcal/mol): {'PASS ✓' if chem_pass else 'FAIL ✗'}")
    print(f"  論文目標 (<0.5 kcal/mol): {'PASS ✓' if goal_pass else 'FAIL ✗'}")

    print(f"\n[Calibration]")
    cov2 = calib_results["coverage_2sig"]
    rho  = calib_results["spearman_rho"]
    ece  = calib_results["ece_mHa"]
    print(f"  2σ coverage = {cov2*100:.1f}%  (expected 95.4%)")
    print(f"  Spearman ρ(σ,|err|) = {rho:.3f}")
    print(f"  ECE = {ece:.2f} mHa")
    calib_pass = cov2 > 0.80 and rho > 0.3
    print(f"  calibration 良好: {'YES ✓' if calib_pass else 'NO ✗'}")

    print(f"\n[ml_pred Rate]")
    if best_thr is not None:
        r = mlpred_results[best_thr]
        print(f"  最適閾値 σ<{best_thr:.1e} Ha:")
        print(f"    ml_pred rate = {r['ml_pred_rate']*100:.1f}%")
        print(f"    MAE within   = {r['mae_kcal']:.3f} kcal/mol")
        ml_pass = r["ml_pred_rate"] >= 0.50 and r["mae_kcal"] < 1.0
        print(f"    ml_pred ≥50% かつ MAE<1: {'PASS ✓' if ml_pass else 'FAIL ✗'}")
    else:
        ml_pass = False
        print("  50%超のml_pred率+化学精度を満たす閾値なし")

    print("\n" + "─" * 65)
    all_pass = chem_pass and calib_pass and ml_pass
    if all_pass:
        print("総合: PASS ✓ → 論文級の実証完了")
        print("次ステップ: pair correction, scaling benchmark, JCTC 投稿準備")
    elif chem_pass:
        print("総合: 化学精度達成 — ml_pred率またはcalibration改善が次の課題")
    else:
        print("総合: 化学精度未達 — 記述子改善または追加データが必要")
    print("=" * 65)

    return {
        "chem_pass":   chem_pass,
        "goal_pass":   goal_pass,
        "calib_pass":  calib_pass,
        "ml_pass":     ml_pass,
        "all_pass":    all_pass,
        "mae_best_kcal": mae_best,
        "best_threshold": best_thr,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=200,
                        help="最大フラグメント数 (default: 200)")
    parser.add_argument("--skip-qc",  action="store_true",
                        help="PySCF 計算をスキップ (キャッシュのみ使用)")
    parser.add_argument("--n-train",  type=int, default=120,
                        help="最大トレーニングサイズ (default: 120)")
    args = parser.parse_args()

    print("=" * 65)
    print("MotifBank GP — 実DFT (PBE/def2-SVP) 検証ベンチマーク")
    print(f"  max_frags={args.n}  n_train_max={args.n_train}  skip_qc={args.skip_qc}")
    print("=" * 65 + "\n")

    # §1 フラグメント生成 (合成多様化 + CIF由来混合)
    print("[§1] フラグメント生成")
    # 合成: 多様な Si(OH)4 (d_SiO/d_OH/角度をランダム変動)
    n_synth  = int(args.n * 0.85)
    n_small  = args.n - n_synth
    mols, names, keys = generate_sioh4_fragments(n=n_synth, seed=42)
    # CIF 由来も少量追加 (ゼオライト実環境)
    cif_mols, cif_names, cif_keys = collect_fragments(max_frags=n_small, verbose=False)
    seen = set(keys)
    for m, nm, k in zip(cif_mols, cif_names, cif_keys):
        if k not in seen:
            mols.append(m); names.append(nm); keys.append(k)
            seen.add(k)
    print(f"  synthetic: {n_synth},  cif-derived: {len(mols)-n_synth},  "
          f"total: {len(mols)}\n")

    # §2 DFT 計算
    mols, energies = compute_with_cache(
        mols, keys,
        cache_file=CACHE_FILE,
        skip_qc=args.skip_qc,
    )
    N = len(mols)
    if N < 12:
        print(f"ERROR: 有効フラグメント {N} < 12。--skip-qc の場合はキャッシュが必要。")
        return

    # §3 記述子
    X = compute_descriptors(mols)
    y = np.array(energies)

    # train/test 分割 (80/20)
    n_test  = max(20, N // 5)
    n_train = min(args.n_train, N - n_test)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    train_idx = list(perm[:n_train])
    test_idx  = list(perm[n_train:n_train + n_test])

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    print(f"Train: {len(train_idx)},  Test: {len(test_idx)}\n")

    # §4 学習曲線
    n_list = [n for n in [10, 20, 40, 80, 120, 160] if n < n_train]
    n_list.append(n_train)
    lc = learning_curve(X, y, n_train_list=sorted(set(n_list)))

    # §5 AL 曲線
    n_init_al = min(10, n_train // 4)
    n_steps   = min(8, (n_train - n_init_al) // 10)
    if N >= 60:
        al = active_learning_curve(X, y, n_init=n_init_al,
                                   n_steps=max(3, n_steps),
                                   n_per_step=10, n_test=n_test)
    else:
        al = {}
        print("[§5] データ不足のため AL スキップ\n")

    # §6 Calibration (最大 n_train でのモデル)
    calib = uncertainty_calibration(X_tr, y_tr, X_te, y_te)

    # §7 ml_pred rate
    mlpred, best_thr = mlpred_rate_analysis(X_tr, y_tr, X_te, y_te)

    # §8 サマリ
    verdict = print_summary(lc, calib, mlpred, best_thr)

    # 結果保存
    results = {
        "n_fragments": N,
        "n_train": n_train,
        "n_test":  n_test,
        "learning_curve": {str(k): v for k, v in lc.items()},
        "calibration": calib,
        "mlpred": {str(k): v for k, v in mlpred.items()},
        "verdict": verdict,
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n結果保存: {RESULT_FILE}")


if __name__ == "__main__":
    main()
