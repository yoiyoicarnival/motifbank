#!/usr/bin/env python3
"""
motifbank_cli.py — MotifBank コマンドラインツール
MBE Phase分類 + フラグメントバンクの実践的実装

使い方:
  python3 motifbank_cli.py demo                    # 内蔵デモ実行
  python3 motifbank_cli.py classify  INPUT.json    # Phase分類
  python3 motifbank_cli.py build     INPUT.json    # バンク構築
  python3 motifbank_cli.py mbe       INPUT.json    # MBE計算 (バンク使用)
  python3 motifbank_cli.py benchmark INPUT.cif     # CIF直接ベンチマーク
  python3 motifbank_cli.py status    BANK.json     # バンク統計表示

INPUT.json フォーマット (3種類):
  {"system": "ice2d",  "nx": 6, "ny": 6}                       # 内蔵ビルダー
  {"cif": "path/to/ice_Ih.cif", "supercell": [2,2,1]}          # CIF入力
  {"molecules": [[[x,y,z],...], ...], "atom_types": [["O","H","H"],...]}  # 座標直接

注意: OMP_NUM_THREADS=1 で実行すること (σ_software=0 保証)
"""

import os, sys, json, time, itertools, argparse
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. コア: geom_key + バンク操作
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECIMAL   = 1      # 距離の丸め精度 (0.1Å)
GAMMA_C   = 0.48   # Phase 1/2 境界
R_CUT_DEF = 6.0    # デフォルトカットオフ (Å)

def geom_key(mol_list, decimal=DECIMAL):
    """分子リスト → 幾何構造キー (距離タプル、元素非依存)"""
    pts = np.vstack(mol_list)
    d = [round(float(np.linalg.norm(pts[i] - pts[j])), decimal)
         for i, j in itertools.combinations(range(len(pts)), 2)]
    return tuple(sorted(d))

def dist_vec(mol_list):
    """分子リスト → ソート済み距離ベクトル"""
    pts = np.vstack(mol_list)
    return np.array(sorted(
        np.linalg.norm(pts[i] - pts[j])
        for i, j in itertools.combinations(range(len(pts)), 2)
    ))

def rmsd(d1, d2):
    return float(np.sqrt(np.mean((d1 - d2) ** 2)))

def com(mol):
    """重心 (Center of Mass)"""
    return np.mean(mol, axis=0)

def cutoff_trimers(mols, r_cut=R_CUT_DEF, max_t=500_000):
    """R_cut 以内の全トリマー index を列挙 (近傍グリッド高速版)"""
    n = len(mols)
    coms = np.array([com(m) for m in mols])

    cs = r_cut
    grid = {}
    for i, c in enumerate(coms):
        cell = (int(c[0]//cs), int(c[1]//cs), int(c[2]//cs))
        grid.setdefault(cell, []).append(i)

    nbrs = [[] for _ in range(n)]
    for i, c in enumerate(coms):
        ci = (int(c[0]//cs), int(c[1]//cs), int(c[2]//cs))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((ci[0]+dx, ci[1]+dy, ci[2]+dz), []):
                        if j > i and np.linalg.norm(c - coms[j]) < r_cut:
                            nbrs[i].append(j)

    idxs = []
    for i in range(n):
        ni = nbrs[i]
        for p in range(len(ni)):
            j = ni[p]
            for q in range(p+1, len(ni)):
                k = ni[q]
                if np.linalg.norm(coms[j] - coms[k]) < r_cut:
                    idxs.append((i, j, k))
                    if len(idxs) >= max_t:
                        return idxs
    return idxs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. バンク: 保存・読み込み・照会
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MotifBank:
    """
    フラグメントエネルギーバンク

    構造:
      key (geom_key タプル) → {
        "energy_Ha": float,   # QC エネルギー (Hartree)
        "dist_vec":  list,    # 距離ベクトル (soft matching 用)
        "hits":      int,     # 再利用回数
        "source":    str      # "qc_computed" / "mock"
      }
    """
    def __init__(self, path=None):
        self.data   = {}
        self.path   = path
        self.n_hit  = 0
        self.n_miss = 0
        if path and os.path.exists(path):
            self.load(path)

    def load(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.data = {tuple(json.loads(k)): v for k, v in raw.items()}
        print(f"  [bank] loaded {len(self.data)} entries from {path}")

    def save(self, path=None):
        path = path or self.path
        if not path:
            raise ValueError("save path not specified")
        serial = {json.dumps(list(k)): v for k, v in self.data.items()}
        with open(path, 'w') as f:
            json.dump(serial, f, indent=2)
        print(f"  [bank] saved {len(self.data)} entries -> {path}")

    def query_exact(self, key):
        rec = self.data.get(key)
        if rec:
            self.n_hit += 1
            rec["hits"] = rec.get("hits", 0) + 1
            return rec["energy_Ha"]
        self.n_miss += 1
        return None

    def query_soft(self, mol_list, eps=0.10):
        """RMSD soft matching 照会 (同じ距離次元のエントリのみ比較)"""
        dv = dist_vec(mol_list)
        ndv = len(dv)
        for k, rec in self.data.items():
            bv = np.array(rec["dist_vec"])
            if len(bv) == ndv and rmsd(dv, bv) < eps:
                self.n_hit += 1
                rec["hits"] = rec.get("hits", 0) + 1
                return rec["energy_Ha"]
        self.n_miss += 1
        return None

    def store(self, key, mol_list, energy_Ha, source="qc_computed"):
        dv = dist_vec(mol_list).tolist()
        self.data[key] = {
            "energy_Ha": energy_Ha,
            "dist_vec":  dv,
            "hits":      0,
            "source":    source
        }

    def stats(self):
        total = self.n_hit + self.n_miss
        reuse = self.n_hit / max(total, 1)
        return {
            "n_entries":    len(self.data),
            "n_hit":        self.n_hit,
            "n_miss":       self.n_miss,
            "reuse_frac":   round(reuse, 4),
        }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. Phase 分類エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify(mols, mols_2x=None, r_cut=R_CUT_DEF, T_K=300.0,
             eps_match=0.10, label="system", verbose=True):
    """
    Phase 0/1/2/3 分類 + ROI 推定

    mols_2x: 2倍サイズのスーパーセル (gamma 軌跡計算用、省略可)
    """
    t0 = time.perf_counter()

    trims = cutoff_trimers(mols, r_cut)
    Nt = len(trims)
    if Nt == 0:
        return {"error": "no trimers within R_cut", "label": label}

    bank_keys = {}
    for i, j, k in trims:
        key = geom_key([mols[i], mols[j], mols[k]])
        bank_keys[key] = dist_vec([mols[i], mols[j], mols[k]])
    N_bank = len(bank_keys)
    compress = Nt / max(N_bank, 1)
    H_ratio = np.log(N_bank) / np.log(max(Nt, 2))

    gamma_traj = None
    if mols_2x is not None:
        trims_2x = cutoff_trimers(mols_2x, r_cut)
        keys_2x = {geom_key([mols_2x[i], mols_2x[j], mols_2x[k]])
                   for i, j, k in trims_2x}
        N2 = len(keys_2x); Nt2 = len(trims_2x)
        if Nt2 > Nt and N2 > 0 and N_bank > 0:
            g = np.log(N2 / N_bank) / np.log(Nt2 / Nt)
            if np.isfinite(g):
                gamma_traj = float(g)

    crystal_flag = (gamma_traj is not None and
                    gamma_traj < H_ratio * 0.95 and H_ratio > 0.40)
    gamma = gamma_traj if gamma_traj is not None else H_ratio

    if crystal_flag and gamma < 0.30:
        phase, basis = 0, "Phase 0: N_unique 飽和 (完全再利用)"
    elif gamma < GAMMA_C:
        phase, basis = 1, "Phase 1: sub-linear 成長 (Bank 有効)"
    elif gamma < 0.80:
        phase, basis = 2, "Phase 2: super-linear (速度のみ)"
    else:
        phase, basis = 3, "Phase 3: ランダム (Bank 無効)"

    delta_T = 0.05 * np.sqrt(max(T_K, 1) / 273.0)
    reuse = _estimate_reuse(mols, bank_keys, r_cut, delta_T, eps_match)

    roi       = max(0.0, (1.0 - 1.0 / compress) * reuse)
    roi_amort = float(reuse)
    alpha_pred = max(0.0, 2.546 * gamma - 0.101)

    if phase <= 1 and roi > 0.60:
        strategy = "DEPLOY"
    elif roi > 0.20:
        strategy = "SPEED"
    else:
        strategy = "SKIP"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = {
        "label":        label,
        "phase":        phase,
        "basis":        basis,
        "gamma":        round(gamma, 4),
        "gamma_traj":   round(gamma_traj, 4) if gamma_traj is not None else None,
        "H_ratio":      round(H_ratio, 4),
        "crystal_flag": crystal_flag,
        "alpha_pred":   round(alpha_pred, 3),
        "N_bank":       N_bank,
        "Nt_cut":       Nt,
        "compress":     round(compress, 1),
        "T_K":          T_K,
        "delta_T_A":    round(delta_T, 3),
        "eps_match_A":  eps_match,
        "reuse":        round(reuse, 3),
        "roi_pct":      round(roi * 100, 1),
        "roi_amort_pct":round(roi_amort * 100, 1),
        "strategy":     strategy,
        "elapsed_ms":   round(elapsed_ms, 1),
    }

    if verbose:
        _print_classify(result)
    return result

def _estimate_reuse(mols, bank_keys, r_cut, delta_T, eps_match, n_seeds=3):
    bank_vecs = list(bank_keys.values())
    if not bank_vecs:
        return 0.0
    hits = misses = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        mols_T = [m + rng.normal(0, delta_T, m.shape) for m in mols]
        trims_T = cutoff_trimers(mols_T, r_cut, max_t=5000)
        for i, j, k in trims_T:
            dv = dist_vec([mols_T[i], mols_T[j], mols_T[k]])
            if any(rmsd(dv, bv) < eps_match for bv in bank_vecs):
                hits += 1
            else:
                misses += 1
    total = hits + misses
    return hits / max(total, 1)

def _print_classify(r):
    print(f"\n{'='*60}")
    print(f"  MotifBank 分類: {r['label']}")
    print(f"{'='*60}")
    print(f"  Phase:        {r['phase']}  ---  {r['basis']}")
    print(f"  gamma (cutoff): {r['gamma']:.4f}  (gamma_traj={r['gamma_traj']})")
    print(f"  alpha 予測:   {r['alpha_pred']:.3f}  ({'MBE収束' if r['alpha_pred']<1 else 'MBE発散'})")
    print(f"  Crystal flag: {r['crystal_flag']}")
    print(f"  N_bank:       {r['N_bank']} 型  ({r['compress']:.1f}x 圧縮)")
    print(f"  T={r['T_K']:.0f}K reuse:  {r['reuse']*100:.0f}%  (eps={r['eps_match_A']:.2f}A)")
    print(f"  ROI (初回):   {r['roi_pct']:.1f}%")
    print(f"  ROI (2回目+): {r['roi_amort_pct']:.1f}%")
    print(f"  戦略:         * {r['strategy']}")
    print(f"  判定時間:     {r['elapsed_ms']:.1f} ms")
    print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. QC バックエンド
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def qc_compute_mock(mol_list, atom_types_list=None):
    """
    Mock QC (Lennard-Jones 近似)
    atom_types_list は無視 (座標のみ使用)
    """
    pts = np.vstack(mol_list)
    dists = [np.linalg.norm(pts[i] - pts[j])
             for i, j in itertools.combinations(range(len(pts)), 2)]
    mock_e = sum(4e-3 * ((2.5/d)**12 - (2.5/d)**6) for d in dists if d > 0.1)
    return float(mock_e)


def qc_compute_pyscf(mol_list, atom_types_list=None,
                     basis='sto-3g', method='hf', charge=0, spin=0):
    """
    PySCF による実 QC 計算 (HF/MP2/CCSD)

    mol_list: list of ndarray (N*3, Angstrom)
    atom_types_list: list of lists, e.g. [["O","H","H"], ["O","H","H"]]
                     None -> H2O (3原子) を仮定
    basis: 'sto-3g' / '6-31g' / 'cc-pvdz' など
    method: 'hf' / 'mp2' / 'ccsd'
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        raise ImportError("pip install pyscf が必要です")

    # デフォルト: 3原子 = H2O
    if atom_types_list is None:
        total_atoms = sum(len(np.array(m).reshape(-1, 3)) for m in mol_list)
        if total_atoms % 3 == 0:
            atom_types_list = [["O", "H", "H"] for _ in mol_list]
        else:
            raise ValueError(
                f"atom_types_list が未指定で原子数={total_atoms} (3の倍数でない)。"
                " atom_types を指定するか CIF 入力を使用してください。"
            )

    # PySCF atom string 構築 (Angstrom, セミコロン区切り)
    parts = []
    for mol, atypes in zip(mol_list, atom_types_list):
        mol_arr = np.array(mol)
        if mol_arr.ndim == 1:
            mol_arr = mol_arr.reshape(1, 3)
        for sym, coord in zip(atypes, mol_arr):
            parts.append(f"{sym} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}")
    atom_str = "; ".join(parts)

    mol_pyscf = gto.M(
        atom=atom_str,
        basis=basis,
        unit='Angstrom',
        charge=charge,
        spin=spin,
        verbose=0,
    )
    mol_pyscf.max_memory = 4000  # MB

    method_lower = method.lower()
    if method_lower == 'hf':
        mf = scf.RHF(mol_pyscf)
        mf.conv_tol = 1e-9   # determinism 保証
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"HF 未収束: {atom_str[:80]}...")
        return float(mf.e_tot)

    elif method_lower == 'mp2':
        from pyscf import mp
        mf = scf.RHF(mol_pyscf)
        mf.conv_tol = 1e-9
        mf.kernel()
        pt = mp.MP2(mf).run()
        return float(pt.e_tot)

    elif method_lower == 'ccsd':
        from pyscf import cc
        mf = scf.RHF(mol_pyscf)
        mf.conv_tol = 1e-9
        mf.kernel()
        mycc = cc.CCSD(mf).run()
        return float(mycc.e_tot)

    else:
        raise ValueError(f"未対応メソッド: {method}  (hf / mp2 / ccsd)")


def make_qc_func(backend='mock', basis='sto-3g', method='hf',
                 charge=0, spin=0):
    """QC 関数を設定から生成するファクトリ"""
    if backend == 'mock':
        return qc_compute_mock
    elif backend == 'pyscf':
        def _pyscf(mol_list, atom_types_list=None,
                   charge=charge, spin=spin):
            # charge/spin はデフォルト値を使うが、run_mbe から上書き可能
            return qc_compute_pyscf(mol_list, atom_types_list,
                                    basis=basis, method=method,
                                    charge=charge, spin=spin)
        _pyscf.__name__ = f'pyscf_{method}_{basis}'
        return _pyscf
    else:
        raise ValueError(f"未知のバックエンド: {backend}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. MBE エンジン (正しい3体定式化)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_mbe(mols, bank, r_cut=R_CUT_DEF, eps_match=0.10,
            qc_func=None, atom_types_list=None,
            charge_per_mol=0, spin_per_mol=0,
            verbose=True, memory_saving=False):
    """
    MBE 計算 (bank 使用)

    エネルギー定義:
      E_mono[i]   = E(i)
      de2[ij]     = E(ij) - E(i) - E(j)
      de3[ijk]    = E(ijk) - E(ij) - E(ik) - E(jk) + E(i) + E(j) + E(k)
                  = E(ijk) - de2(ij) - de2(ik) - de2(jk) - E(i) - E(j) - E(k)
      E_total     = sum(E_mono) + sum(de2) + sum(de3)

    atom_types_list: [["O","H","H"], ...] -- pyscf バックエンド使用時に必須
    charge_per_mol:  各分子の電荷 (H3+ なら +1, H2O なら 0)
    spin_per_mol:    各分子の 2S (通常 0 = singlet)
    """
    if qc_func is None:
        qc_func = qc_compute_mock

    def _qc(indices):
        mol_sub = [mols[i] for i in indices]
        total_charge = charge_per_mol * len(indices)
        total_spin   = spin_per_mol   * len(indices)
        if atom_types_list is not None:
            at_sub = [atom_types_list[i] for i in indices]
            # charge/spin を渡す (pyscf バックエンドのみ有効)
            try:
                return qc_func(mol_sub, at_sub,
                               charge=total_charge, spin=total_spin)
            except TypeError:
                return qc_func(mol_sub, at_sub)
        return qc_func(mol_sub)

    coms = np.array([com(m) for m in mols])
    n = len(mols)
    t0 = time.perf_counter()

    # -- モノマー --
    E_mono = {}
    for i, mol in enumerate(mols):
        key = geom_key([mol])
        e = bank.query_exact(key)
        if e is None:
            e = _qc([i])
            bank.store(key, [mol], e, source="qc_computed")
        E_mono[i] = e

    # -- 2体 --
    E_2body = 0.0
    pair_de2 = {}   # (i,j) -> de2(ij)、3体計算で再利用
    n_pairs = hit_pairs = 0

    for i, j in itertools.combinations(range(n), 2):
        if np.linalg.norm(coms[i] - coms[j]) < r_cut:
            key = geom_key([mols[i], mols[j]])
            de2 = bank.query_soft([mols[i], mols[j]], eps_match)
            if de2 is None:
                e_ij = _qc([i, j])
                de2 = e_ij - E_mono[i] - E_mono[j]
                bank.store(key, [mols[i], mols[j]], de2)
            else:
                hit_pairs += 1
            if not memory_saving:
                pair_de2[(i, j)] = de2
            E_2body += de2
            n_pairs += 1

    # -- 3体 --
    E_3body = 0.0
    n_trims = hit_trims = 0
    trims = cutoff_trimers(mols, r_cut)

    if memory_saving:
        def _get_pde2(a, b):
            ia, ib = min(a, b), max(a, b)
            gk = geom_key([mols[ia], mols[ib]])
            rec = bank.data.get(gk)
            if rec:
                return rec["energy_Ha"]
            dv = dist_vec([mols[ia], mols[ib]])
            for rec2 in bank.data.values():
                bv = np.array(rec2["dist_vec"])
                if len(bv) == len(dv) and rmsd(dv, bv) < eps_match:
                    return rec2["energy_Ha"]
            return 0.0
    else:
        def _get_pde2(a, b):
            return pair_de2.get((min(a, b), max(a, b)), 0.0)

    for i, j, k in trims:
        key = geom_key([mols[i], mols[j], mols[k]])
        de3 = bank.query_soft([mols[i], mols[j], mols[k]], eps_match)
        if de3 is None:
            e_ijk = _qc([i, j, k])
            # de3 = E(ijk) - de2(ij) - de2(ik) - de2(jk) - Ei - Ej - Ek
            de3 = (e_ijk
                   - _get_pde2(i, j) - _get_pde2(i, k) - _get_pde2(j, k)
                   - E_mono[i] - E_mono[j] - E_mono[k])
            bank.store(key, [mols[i], mols[j], mols[k]], de3)
        else:
            hit_trims += 1
        E_3body += de3
        n_trims += 1

    elapsed = time.perf_counter() - t0
    roi_actual = (hit_pairs + hit_trims) / max(n_pairs + n_trims, 1)
    E_sum_mono = sum(E_mono.values())

    result = {
        "E_mono_Ha":    round(E_sum_mono, 8),
        "E_2body_Ha":   round(E_2body, 8),
        "E_3body_Ha":   round(E_3body, 8),
        "E_total_Ha":   round(E_sum_mono + E_2body + E_3body, 8),
        "n_mols":       n,
        "n_pairs":      n_pairs,
        "n_trims":      n_trims,
        "hit_pairs":    hit_pairs,
        "hit_trims":    hit_trims,
        "roi_actual":   round(roi_actual, 3),
        "elapsed_s":    round(elapsed, 3),
        "bank_stats":   bank.stats(),
    }

    if verbose:
        _print_mbe(result)
    return result

def _print_mbe(r):
    e2 = r['E_2body_Ha']; e3 = r['E_3body_Ha']; em = r['E_mono_Ha']
    ratio = abs(e3) / max(abs(e2), 1e-12) * 100
    print(f"\n{'='*55}")
    print(f"  MBE 計算結果  ({r['n_mols']} 分子)")
    print(f"{'='*55}")
    print(f"  E_mono:  {em:>16.8f} Ha")
    print(f"  E_2body: {e2:>16.8f} Ha")
    print(f"  E_3body: {e3:>16.8f} Ha  ({ratio:.1f}% of 2体)")
    print(f"  E_total: {r['E_total_Ha']:>16.8f} Ha")
    print(f"  ペア:    {r['n_pairs']} 個  bank={r['hit_pairs']} "
          f"({r['hit_pairs']/max(r['n_pairs'],1)*100:.0f}%)")
    print(f"  トリマー:{r['n_trims']} 個  bank={r['hit_trims']} "
          f"({r['hit_trims']/max(r['n_trims'],1)*100:.0f}%)")
    print(f"  実 ROI:  {r['roi_actual']*100:.1f}%")
    print(f"  計算時間:{r['elapsed_s']:.3f} s")
    bs = r['bank_stats']
    print(f"  バンク:  {bs['n_entries']} 型 "
          f"(累計 hit={bs['n_hit']}, miss={bs['n_miss']})")
    print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §6. CIF 読み込み (ASE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def from_cif(cif_path, supercell=(1, 1, 1), mol_type="auto",
             r_OH_max=1.3, verbose=True):
    """
    CIF ファイル -> 分子リスト

    Returns:
      mols:            list of ndarray (N*3, Angstrom)
      atom_types_list: list of lists of str
      label:           str

    mol_type:
      "auto"   -> 元素組成から自動判定
      "h2o"    -> O + 最近傍 2H でグループ化
      "h3plus" -> H 原子を3個ずつグループ化 (H3+)
      "sio4"   -> Si + 最近傍 4O でグループ化
      "atoms"  -> 全原子を1原子分子として返す (汎用)
    """
    try:
        from ase.io import read as ase_read
    except ImportError:
        raise ImportError("pip install ase が必要です (pip install ase)")

    atoms = ase_read(cif_path)
    if supercell != (1, 1, 1):
        atoms = atoms * supercell

    positions = np.array(atoms.get_positions())
    symbols   = list(atoms.get_chemical_symbols())
    label     = os.path.splitext(os.path.basename(cif_path))[0]

    if verbose:
        from collections import Counter
        cnt = Counter(symbols)
        print(f"  [CIF] {cif_path}: {len(positions)} atoms {dict(cnt)}")

    # -- 自動判定 --
    if mol_type == "auto":
        sym_set = set(symbols)
        if sym_set <= {"O", "H"}:
            mol_type = "h2o"
        elif sym_set == {"H"}:
            mol_type = "h3plus"
        elif "Si" in sym_set and "O" in sym_set:
            mol_type = "sio4"
        else:
            mol_type = "atoms"
            if verbose:
                print(f"  [CIF] mol_type=atoms (汎用モード: 1原子=1分子)")

    # -- H2O: O + 最近傍 2H --
    if mol_type == "h2o":
        O_idx = [i for i, s in enumerate(symbols) if s == 'O']
        H_idx = [i for i, s in enumerate(symbols) if s == 'H']
        if not O_idx:
            raise ValueError("H2O モードですが O 原子がありません")
        mols, atypes = [], []
        used_H = set()
        for oi in O_idx:
            op = positions[oi]
            h_dists = sorted(
                [(np.linalg.norm(op - positions[hi]), hi)
                 for hi in H_idx if hi not in used_H]
            )
            # r_OH_max 以内の H を最大 2 個取る
            h_picked = [d[1] for d in h_dists if d[0] <= r_OH_max][:2]
            if len(h_picked) < 2 and h_dists:
                # 距離制限内に足りない場合は最近傍 2 個で補完
                h_picked = [d[1] for d in h_dists[:2]]
            for hi in h_picked:
                used_H.add(hi)
            coords = np.array([positions[oi]] + [positions[hi] for hi in h_picked])
            mols.append(coords)
            atypes.append(["O"] + ["H"] * len(h_picked))
        return mols, atypes, label

    # -- H3+: H を3個ずつグループ --
    elif mol_type == "h3plus":
        if len(positions) % 3 != 0:
            raise ValueError(f"H原子数 {len(positions)} が3の倍数ではありません")
        mols, atypes = [], []
        for k in range(len(positions) // 3):
            mols.append(positions[3*k:3*k+3].copy())
            atypes.append(["H", "H", "H"])
        return mols, atypes, label

    # -- SiO4: Si + 最近傍 4O --
    elif mol_type == "sio4":
        Si_idx = [i for i, s in enumerate(symbols) if s == 'Si']
        O_idx  = [i for i, s in enumerate(symbols) if s == 'O']
        mols, atypes = [], []
        for si in Si_idx:
            sp = positions[si]
            o_dists = sorted(
                [(np.linalg.norm(sp - positions[oi]), oi)
                 for oi in O_idx]
            )[:4]
            o_picked = [d[1] for d in o_dists]
            coords = np.array([positions[si]] + [positions[oi] for oi in o_picked])
            mols.append(coords)
            atypes.append(["Si"] + ["O"] * len(o_picked))
        return mols, atypes, label

    elif mol_type == "si_oh4":
        # H-capped Si(OH)4: Si + 4O + 4H (中性, charge=0)
        # H を O-Si 方向の延長上 0.96Å に配置 → dangling bond をキャップ
        Si_idx = [i for i, s in enumerate(symbols) if s == 'Si']
        O_idx  = [i for i, s in enumerate(symbols) if s == 'O']
        OH_BOND = 0.96   # Å
        mols, atypes = [], []
        for si in Si_idx:
            sp = positions[si]
            o_dists = sorted(
                [(np.linalg.norm(sp - positions[oi]), oi)
                 for oi in O_idx]
            )[:4]
            o_coords = [positions[d[1]] for d in o_dists]
            h_coords = []
            for oc in o_coords:
                # Si→O 方向の延長上に H を配置 (もう一方の Si の代わり)
                v = oc - sp
                v_unit = v / np.linalg.norm(v)
                h_coords.append(oc + v_unit * OH_BOND)
            coords = np.array([sp] + o_coords + h_coords)
            mols.append(coords)
            atypes.append(["Si"] + ["O"] * 4 + ["H"] * 4)
        return mols, atypes, label

    # -- 汎用: 1原子 = 1分子 --
    else:  # "atoms"
        mols   = [positions[i:i+1].copy() for i in range(len(positions))]
        atypes = [[symbols[i]] for i in range(len(positions))]
        return mols, atypes, label


def from_ase(ase_atoms, mol_type="auto", r_OH_max=1.3, label=None, verbose=True):
    """
    ASE Atoms オブジェクト → 分子リスト

    from_cif の ASE オブジェクト版。MD スナップショットや
    VASP/QE 出力を直接渡す場合に使用。

    例:
      from ase.io import read
      atoms = read("md_snapshot.xyz")
      mols, atypes, label = from_ase(atoms)
    """
    positions = np.array(ase_atoms.get_positions())
    symbols   = list(ase_atoms.get_chemical_symbols())
    label_str = label or "ase_input"

    if verbose:
        from collections import Counter
        cnt = Counter(symbols)
        print(f"  [ASE] {len(positions)} atoms {dict(cnt)}")

    # _from_positions_symbols を共有
    import tempfile, os as _os
    try:
        from ase.io import write as ase_write
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmppath = tmp.name
        ase_write(tmppath, ase_atoms, format="extxyz")
        mols, atypes, _ = from_cif(tmppath, mol_type=mol_type,
                                    r_OH_max=r_OH_max, verbose=False)
        _os.unlink(tmppath)
    except Exception:
        # フォールバック: positions/symbols から直接処理
        mols, atypes = _split_molecules(positions, symbols,
                                         mol_type=mol_type, r_OH_max=r_OH_max)
    return mols, atypes, label_str


def _split_molecules(positions, symbols, mol_type="auto", r_OH_max=1.3):
    """positions + symbols リストから分子を分割 (from_cif の内部ロジック共有)"""
    sym_set = set(symbols)
    if mol_type == "auto":
        if sym_set <= {"O", "H"}:
            mol_type = "h2o"
        elif sym_set == {"H"}:
            mol_type = "h3plus"
        else:
            mol_type = "atoms"

    if mol_type == "h2o":
        O_idx = [i for i, s in enumerate(symbols) if s == 'O']
        H_idx = [i for i, s in enumerate(symbols) if s == 'H']
        mols, atypes = [], []
        used_H = set()
        for oi in O_idx:
            op = positions[oi]
            h_dists = sorted(
                [(np.linalg.norm(op - positions[hi]), hi)
                 for hi in H_idx if hi not in used_H]
            )
            h_picked = [d[1] for d in h_dists if d[0] <= r_OH_max][:2]
            if len(h_picked) < 2 and h_dists:
                h_picked = [d[1] for d in h_dists[:2]]
            for hi in h_picked:
                used_H.add(hi)
            coords = np.array([positions[oi]] + [positions[hi] for hi in h_picked])
            mols.append(coords)
            atypes.append(["O"] + ["H"] * len(h_picked))
        return mols, atypes
    elif mol_type == "h3plus":
        mols   = [positions[3*k:3*k+3].copy() for k in range(len(positions)//3)]
        atypes = [["H","H","H"]] * len(mols)
        return mols, atypes
    else:
        mols   = [positions[i:i+1].copy() for i in range(len(positions))]
        atypes = [[symbols[i]] for i in range(len(positions))]
        return mols, atypes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §7. 内蔵システムビルダー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_ice2d(nx, ny, a=2.76):
    """2D 六方晶 H2O モデル"""
    rOH = 0.9572; angle = 104.52 * np.pi / 180
    lv1 = np.array([a*np.sqrt(3), 0.])
    lv2 = np.array([a*np.sqrt(3)/2, a*3/2])
    bp  = [np.zeros(2), np.array([a*np.sqrt(3)/2, a/2])]
    mols = []
    for i in range(nx):
        for j in range(ny):
            o = i*lv1 + j*lv2
            for kb, b in enumerate(bp):
                p = o + b; phi = kb * np.pi / 3
                Ox, Oy = p
                mol = np.array([
                    [Ox, Oy, 0.],
                    [Ox + rOH*np.cos(phi+angle/2), Oy + rOH*np.sin(phi+angle/2), 0.],
                    [Ox + rOH*np.cos(phi-angle/2), Oy + rOH*np.sin(phi-angle/2), 0.],
                ])
                mols.append(mol)
    return mols

def build_h3plus(Ox, Oy, Oz=0.0, a=0.75):
    """H3+ 正三角形分子"""
    r = a / np.sqrt(3)
    return np.array([
        [Ox + r,       Oy,        Oz],
        [Ox - r/2,     Oy + a/2,  Oz],
        [Ox - r/2,     Oy - a/2,  Oz],
    ])

def build_carpet(gen, A_lat=2.5, A_h3=0.75):
    """Sierpinski Carpet H3+"""
    def _carpet_positions(n, scale):
        if n == 0:
            return [np.array([0.0, 0.0, 0.0])]
        prev = _carpet_positions(n-1, scale/3)
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                offsets.append(np.array([dx, dy, 0.0]) * scale / 3)
        return [p + off for p in prev for off in offsets]

    positions = _carpet_positions(gen, A_lat * 3**gen)
    mols = []
    for pos in positions:
        mols.append(build_h3plus(pos[0], pos[1], pos[2], a=A_h3))
    return mols

def build_mof_pore(n_pores, d_inter=12.0, n_per_pore=8, a=2.76):
    """MOF ポアモデル (H2O リング)"""
    rOH = 0.9572; angle = 104.52 * np.pi / 180
    r_ring = a * n_per_pore / (2 * np.pi)
    n_side = int(np.ceil(np.sqrt(n_pores)))
    mols = []
    for ix in range(n_side):
        for iy in range(n_side):
            if len(mols) // n_per_pore >= n_pores:
                break
            cx, cy = ix * d_inter, iy * d_inter
            for k in range(n_per_pore):
                theta = 2 * np.pi * k / n_per_pore
                phi = theta + np.pi / 2
                Ox = cx + r_ring * np.cos(theta)
                Oy = cy + r_ring * np.sin(theta)
                mol = np.array([
                    [Ox, Oy, 0.],
                    [Ox + rOH*np.cos(phi+angle/2), Oy + rOH*np.sin(phi+angle/2), 0.],
                    [Ox + rOH*np.cos(phi-angle/2), Oy + rOH*np.sin(phi-angle/2), 0.],
                ])
                mols.append(mol)
    return mols

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §8. ファイル I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_input(path):
    """
    INPUT.json を読み込んで molecules + config を返す

    Returns:
      mols:   list of ndarray
      mols_2x: list of ndarray or None
      cfg:    dict (R_cut, T_K, eps_match, bank_file, atom_types_list, label, ...)
    """
    with open(path) as f:
        cfg = json.load(f)

    atom_types_list = None
    mols_2x = None

    sys_name = cfg.get("system")

    # -- 内蔵ビルダー --
    if sys_name == "ice2d":
        nx, ny = cfg.get("nx", 5), cfg.get("ny", 5)
        mols = build_ice2d(nx, ny)
        mols_2x = build_ice2d(nx*2, ny*2)
        label = f"ice2d_{nx}x{ny}"
        atom_types_list = [["O", "H", "H"]] * len(mols)

    elif sys_name == "carpet":
        gen = cfg.get("gen", 2)
        mols = build_carpet(gen)
        mols_2x = build_carpet(gen + 1) if gen < 4 else None
        label = f"carpet_gen{gen}"
        atom_types_list = [["H", "H", "H"]] * len(mols)
        # H3+ は電荷 +1 (デフォルト上書き)
        cfg.setdefault("charge_per_mol", 1)
        cfg.setdefault("spin_per_mol",   0)

    elif sys_name == "mof":
        n = cfg.get("n_pores", 4)
        mols = build_mof_pore(n)
        mols_2x = build_mof_pore(n * 4)
        label = f"mof_{n}pores"
        atom_types_list = [["O", "H", "H"]] * len(mols)

    # -- CIF 入力 --
    elif "cif" in cfg:
        sc = tuple(cfg.get("supercell", [1, 1, 1]))
        mols, atom_types_list, label_auto = from_cif(
            cfg["cif"],
            supercell=sc,
            mol_type=cfg.get("mol_type", "auto"),
        )
        label = cfg.get("label", label_auto)
        # 2倍セル: xy を 2 倍にして再読み
        sc2 = (sc[0]*2, sc[1]*2, sc[2])
        try:
            mols_2x, _, _ = from_cif(cfg["cif"], supercell=sc2,
                                      mol_type=cfg.get("mol_type", "auto"),
                                      verbose=False)
        except Exception:
            mols_2x = None

    # -- 座標直接指定 --
    elif "molecules" in cfg:
        mols = [np.array(m) for m in cfg["molecules"]]
        if "atom_types" in cfg:
            atom_types_list = cfg["atom_types"]
        if "molecules_2x" in cfg:
            mols_2x = [np.array(m) for m in cfg["molecules_2x"]]
        label = cfg.get("label", "custom")

    else:
        raise ValueError(f"不明な入力形式: {list(cfg.keys())}")

    return mols, mols_2x, {
        "label":           cfg.get("label", label),
        "R_cut":           cfg.get("R_cut", R_CUT_DEF),
        "T_K":             cfg.get("T_K", 300.0),
        "eps_match":       cfg.get("eps_match", 0.10),
        "bank_file":       cfg.get("bank_file", None),
        "atom_types_list": atom_types_list,
        "qc_backend":      cfg.get("qc_backend", "mock"),
        "qc_basis":        cfg.get("qc_basis", "sto-3g"),
        "qc_method":       cfg.get("qc_method", "hf"),
        "charge_per_mol":  cfg.get("charge_per_mol", 0),
        "spin_per_mol":    cfg.get("spin_per_mol",   0),
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §9. CLI コマンド
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_classify(args):
    mols, mols_2x, cfg = load_input(args.input)
    result = classify(mols, mols_2x,
                      r_cut=cfg["R_cut"], T_K=cfg["T_K"],
                      eps_match=cfg["eps_match"],
                      label=cfg["label"])
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  -> 結果保存: {args.output}")

def cmd_build(args):
    mols, _, cfg = load_input(args.input)
    bank_path = args.bank or cfg.get("bank_file") or "motifbank.json"

    bank = MotifBank(path=bank_path if os.path.exists(bank_path) else None)
    bank.path = bank_path

    backend = args.qc if args.qc else cfg["qc_backend"]
    cpm     = cfg.get("charge_per_mol", 0)
    spm     = cfg.get("spin_per_mol",   0)
    qc_func = make_qc_func(backend, cfg["qc_basis"], cfg["qc_method"])
    atl     = cfg["atom_types_list"]

    n_before = len(bank.data)
    # run_mbe 経由で bank を構築することで de3 値が正しく格納される
    run_mbe(mols, bank,
            r_cut=cfg["R_cut"],
            eps_match=cfg["eps_match"],
            qc_func=qc_func,
            atom_types_list=atl,
            charge_per_mol=cpm,
            spin_per_mol=spm,
            verbose=True)
    n_added = len(bank.data) - n_before
    print(f"  新規追加: {n_added} 型, 合計: {len(bank.data)} 型")
    bank.save()

def cmd_mbe(args):
    mols, _, cfg = load_input(args.input)
    bank_path = args.bank or cfg.get("bank_file") or "motifbank.json"
    bank = MotifBank(path=bank_path if os.path.exists(bank_path) else None)
    bank.path = bank_path

    backend = args.qc if args.qc else cfg["qc_backend"]
    cpm     = cfg.get("charge_per_mol", 0)
    spm     = cfg.get("spin_per_mol",   0)
    qc_func = make_qc_func(backend, cfg["qc_basis"], cfg["qc_method"],
                            charge=cpm, spin=spm)
    atl     = cfg["atom_types_list"]

    result = run_mbe(mols, bank,
                     r_cut=cfg["R_cut"],
                     eps_match=cfg["eps_match"],
                     qc_func=qc_func,
                     atom_types_list=atl,
                     charge_per_mol=cpm,
                     spin_per_mol=spm)
    bank.save()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  -> 結果保存: {args.output}")

def cmd_benchmark(args):
    """
    CIF ファイルの MBE ベンチマーク
    - 小セル -> bank 構築 (warm-up)
    - 大セル (2x) -> MBE 実行、ROI / 高速化倍率を測定
    """
    cif_path = args.cif
    sc       = tuple(int(x) for x in args.supercell.split(','))
    backend  = args.qc
    basis    = args.basis
    method   = args.method
    r_cut    = args.r_cut

    print(f"\n{'='*65}")
    print(f"  MotifBank ベンチマーク")
    print(f"  CIF:     {cif_path}")
    print(f"  sc:      {sc},  R_cut={r_cut}A")
    print(f"  backend: {backend}/{method}/{basis}")
    print(f"{'='*65}")

    # 小セル -> Phase 分類 + bank 構築
    mols_s, atl_s, label = from_cif(cif_path, supercell=sc)
    sc2 = (sc[0]*2, sc[1]*2, sc[2])
    mols_2x, atl_2x, _ = from_cif(cif_path, supercell=sc2, verbose=False)

    print(f"\n[1/3] Phase 分類  ({len(mols_s)} 分子)")
    cr = classify(mols_s, mols_2x, r_cut=r_cut, label=label)

    bank_path = args.bank or f"{label}_bank.json"
    bank = MotifBank(path=bank_path if os.path.exists(bank_path) else None)
    bank.path = bank_path
    cpm     = getattr(args, 'charge_per_mol', 0)
    spm     = getattr(args, 'spin_per_mol',   0)
    qc_func = make_qc_func(backend, basis, method)

    # bank warm-up (小セル) — run_mbe 経由で de3 値を正しく格納
    print(f"\n[2/3] Bank 構築  (small cell {len(mols_s)} 分子, backend={backend}, charge/mol={cpm})")
    t0 = time.perf_counter()
    n_before = len(bank.data)
    run_mbe(mols_s, bank, r_cut=r_cut,
            qc_func=qc_func, atom_types_list=atl_s,
            charge_per_mol=cpm, spin_per_mol=spm,
            verbose=False)
    t_build = time.perf_counter() - t0
    added = len(bank.data) - n_before
    print(f"  Bank 構築: {added} 型追加 (合計 {len(bank.data)} 型), {t_build:.2f}s")

    # 大セル MBE
    print(f"\n[3/3] MBE 実行  ({len(mols_2x)} 分子, 2x cell)")
    mbe_r = run_mbe(mols_2x, bank, r_cut=r_cut,
                    qc_func=qc_func, atom_types_list=atl_2x,
                    charge_per_mol=cpm, spin_per_mol=spm)
    bank.save()

    speedup = 1.0 / max(1.0 - mbe_r["roi_actual"], 1e-9)
    print(f"\n  * 実測 ROI: {mbe_r['roi_actual']*100:.1f}%,  "
          f"高速化 {speedup:.1f}x")
    print(f"  E_total = {mbe_r['E_total_Ha']:.6f} Ha  "
          f"({mbe_r['E_total_Ha'] * 627.5095:.2f} kcal/mol)")
    print()

    if args.output:
        out = {"classify": cr, "mbe": mbe_r}
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  -> 結果保存: {args.output}")

def cmd_status(args):
    if not os.path.exists(args.bank):
        print(f"  エラー: {args.bank} が見つかりません")
        return
    bank = MotifBank(args.bank)
    print(f"\n  MotifBank 統計: {args.bank}")
    print(f"  ─────────────────────────────")
    print(f"  登録型数:     {len(bank.data)}")
    if bank.data:
        energies = [v["energy_Ha"] for v in bank.data.values()]
        hits     = [v.get("hits", 0) for v in bank.data.values()]
        sources  = {}
        for v in bank.data.values():
            s = v.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
        print(f"  エネルギー範囲: [{min(energies):.6f}, {max(energies):.6f}] Ha")
        print(f"  平均再利用回数: {np.mean(hits):.1f}")
        print(f"  ソース: {sources}")
    print()

def cmd_demo(args):
    print("\n" + "="*65)
    print("  MotifBank デモ実行")
    print("="*65)

    planner_results = {}

    cases = [
        ("2D ice (4x4->6x6)", build_ice2d(4, 4),  build_ice2d(6, 6)),
        ("Carpet Gen1->Gen2", build_carpet(1),     build_carpet(2)),
        ("MOF 4->9 pores",   build_mof_pore(4),   build_mof_pore(9)),
    ]

    for label, mols, mols_2x in cases:
        r = classify(mols, mols_2x, label=label, verbose=True)
        planner_results[label] = r

    print("="*70)
    print("  デモ サマリー")
    print("="*70)
    print(f"  {'材料':<20s} {'Phase':>5s} {'gamma':>7s} {'reuse':>7s} "
          f"{'N_bank':>7s} {'ROI1st':>7s} {'戦略':<8s}")
    print("  " + "-"*65)
    for label, r in planner_results.items():
        print(f"  {r['label']:<20s} {r['phase']:>5d} {r['gamma']:>7.4f} "
              f"{r['reuse']*100:>6.0f}% {r['N_bank']:>7d} "
              f"{r['roi_pct']:>5.1f}%  {r['strategy']}")
    print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §10. エントリーポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    p = argparse.ArgumentParser(
        description='MotifBank CLI -- MBE フラグメントバンク',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
コマンド例:
  python3 motifbank_cli.py demo
  python3 motifbank_cli.py classify  ice_input.json
  python3 motifbank_cli.py build     ice_input.json --bank ice_bank.json
  python3 motifbank_cli.py mbe       ice_input.json --bank ice_bank.json --qc pyscf
  python3 motifbank_cli.py benchmark ice_Ih.cif --sc 2,2,1 --qc pyscf
  python3 motifbank_cli.py status    ice_bank.json

INPUT.json の例:
  {"system": "ice2d",  "nx": 5,  "ny": 5}
  {"system": "carpet", "gen": 2}
  {"cif": "/tmp/ice_Ih.cif", "supercell": [2,2,1], "qc_backend": "pyscf"}
  {"molecules": [[[0,0,0],[1,0,0],[0.5,0.87,0]]], "atom_types": [["O","H","H"]]}
        """
    )
    sub = p.add_subparsers(dest='cmd')

    # classify
    pc = sub.add_parser('classify', help='Phase分類 + ROI推定')
    pc.add_argument('input',  help='入力 JSON ファイル')
    pc.add_argument('--output', '-o', help='結果 JSON 出力先')

    # build
    pb = sub.add_parser('build', help='バンク構築')
    pb.add_argument('input', help='入力 JSON ファイル')
    pb.add_argument('--bank', '-b', help='バンクファイルパス (デフォルト: motifbank.json)')
    pb.add_argument('--qc', choices=['mock', 'pyscf'], default=None,
                    help='QC バックエンド (省略時は JSON の qc_backend を使用)')

    # mbe
    pm = sub.add_parser('mbe', help='MBE 計算 (バンク使用)')
    pm.add_argument('input', help='入力 JSON ファイル')
    pm.add_argument('--bank', '-b', help='バンクファイルパス')
    pm.add_argument('--output', '-o', help='結果 JSON 出力先')
    pm.add_argument('--qc', choices=['mock', 'pyscf'], default=None,
                    help='QC バックエンド')

    # benchmark
    pbm = sub.add_parser('benchmark', help='CIF 直接 MBE ベンチマーク')
    pbm.add_argument('cif', help='CIF ファイルパス')
    pbm.add_argument('--sc', '--supercell', dest='supercell', default='1,1,1',
                     help='スーパーセル 例: 2,2,1 (デフォルト: 1,1,1)')
    pbm.add_argument('--r-cut', type=float, default=R_CUT_DEF,
                     help=f'カットオフ距離 A (デフォルト: {R_CUT_DEF})')
    pbm.add_argument('--qc', choices=['mock', 'pyscf'], default='mock',
                     help='QC バックエンド')
    pbm.add_argument('--basis', default='sto-3g', help='基底関数 (pyscf 用)')
    pbm.add_argument('--method', choices=['hf', 'mp2', 'ccsd'], default='hf',
                     help='計算レベル (pyscf 用)')
    pbm.add_argument('--bank', '-b', help='バンクファイルパス')
    pbm.add_argument('--output', '-o', help='結果 JSON 出力先')
    pbm.add_argument('--charge-per-mol', dest='charge_per_mol', type=int, default=0,
                     help='各分子の電荷 (H3+ なら 1, H2O なら 0)')
    pbm.add_argument('--spin-per-mol',   dest='spin_per_mol',   type=int, default=0,
                     help='各分子の 2S (通常 0)')

    # status
    ps = sub.add_parser('status', help='バンク統計')
    ps.add_argument('bank', help='バンク JSON ファイル')

    # demo
    sub.add_parser('demo', help='内蔵デモ実行')

    args = p.parse_args()

    if args.cmd == 'classify':
        cmd_classify(args)
    elif args.cmd == 'build':
        cmd_build(args)
    elif args.cmd == 'mbe':
        cmd_mbe(args)
    elif args.cmd == 'benchmark':
        cmd_benchmark(args)
    elif args.cmd == 'status':
        cmd_status(args)
    elif args.cmd == 'demo':
        cmd_demo(args)
    else:
        p.print_help()

if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')  # determinism 保証
    main()
