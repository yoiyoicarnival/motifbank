#!/usr/bin/env python3
"""
polymer_ml.py — MotifBank ML × TrajBank 高分子統合エンジン

motifbank_ml.py (結晶/ゼオライト ML) と traj_bank.py (MD軌跡再利用) を
高分子シミュレーション向けに統合する。

主要コンポーネント:
  1. PolymerDescriptor  : Gly/ペプチド向け element_aware 記述子
                          (N-C / C=O / N-H-backbone の3チャンネル, 192次元)
  2. TrajBankML         : TrajBank + GPEnergyPredictor の統合
                          cold→exact / warm→GP予測 / hot→QC の3段階クエリ
  3. ConformationalSampler: FPS-seeded コンフォメーション多様性サンプリング
  4. benchmark_polymer  : TrajBank(既存) vs TrajBankML(ML強化版) の比較

使い方:
  OMP_NUM_THREADS=1 python3 polymer_ml.py          # フルベンチマーク
  OMP_NUM_THREADS=1 python3 polymer_ml.py --quick  # 高速モード (N=4, steps=10)
  OMP_NUM_THREADS=1 python3 polymer_ml.py --demo   # デモのみ
"""

import os, sys, time, argparse
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/yoiyoi")

import numpy as np

# MotifBank コア
from motifbank_cli import (
    MotifBank, geom_key, dist_vec, classify, run_mbe,
    qc_compute_mock, make_qc_func,
)
# MotifBank ML 拡張
from motifbank_ml import (
    rdf_descriptor,
    GPEnergyPredictor, EnsembleEnergyPredictor,
    AdaptiveEpsilon, FragmentCluster,
    farthest_point_sampling,
    MLBank, _store_with_mol,
    ELEM_BINS, EMBED_DIM,
)
# TrajBank
from traj_bank import (
    TrajBank, RegionTracker, UnsupervisedMotifCluster,
    build_gly_peptide, perturb_peptide,
    ApplicabilityPredictor,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. PolymerDescriptor — Gly ペプチド向け element_aware 記述子
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Gly 残基の原子順: [N, Cα, C(carbonyl), O, HN, Hα1, Hα2]
GLY_UNIT_SIZE = 7
GLY_ATOM_ORDER = ['N', 'C', 'C', 'O', 'H', 'H', 'H']

def infer_atom_types_gly(mol_list):
    """
    Gly 残基 mol_list の原子種推定
    from traj_bank: 各残基 = [N, Cα, C, O, HN, Hα1, Hα2] (7原子)
    index%7: 0=N, 1=C(α), 2=C(carbonyl), 3=O, 4-6=H
    """
    types = []
    for i in range(len(mol_list)):
        r = i % GLY_UNIT_SIZE
        if r == 0:
            types.append('N')
        elif r in (1, 2):
            types.append('C')
        elif r == 3:
            types.append('O')
        else:
            types.append('H')
    return types

def polymer_descriptor(mol_list, n_bins=ELEM_BINS, r_max=8.0, atom_types=None):
    """
    Gly ペプチド向け element_aware RDF 記述子 (3チャンネル・192次元)

    Channel 0: Heavy-Heavy (N-C, C-C, C=O, N-O — ペプチド骨格の幾何)
    Channel 1: Heavy-H   (N-H, C-H — 水素結合・CH結合)
    Channel 2: H-H       (水素間非共有距離 — コンフォメーション感度)

    mol_list は以下いずれでもよい:
      - list of residues: [(7,3) ndarray, ...]  ← build_gly_peptide の出力
      - list of atoms:    [(3,) ndarray, ...]   ← フラットな座標リスト
    """
    # residue配列 (2D) をフラット原子リストに展開
    flat_pts   = []
    flat_types = []
    for item in mol_list:
        arr = np.asarray(item, dtype=np.float32)
        if arr.ndim == 2:          # residue (n_atoms, 3)
            flat_pts.append(arr)
            n_a = arr.shape[0]
            flat_types.extend([GLY_ATOM_ORDER[k % GLY_UNIT_SIZE] for k in range(n_a)])
        else:                       # single atom (3,)
            flat_pts.append(arr.reshape(1, 3))

    pts = np.concatenate(flat_pts, axis=0)  # (N_atoms, 3)
    n   = len(pts)

    if atom_types is not None:
        type_list = atom_types
    elif flat_types:
        type_list = flat_types
    else:
        type_list = infer_atom_types_gly(list(range(n)))

    hist_hh = np.zeros(n_bins, dtype=np.float32)  # Heavy-Heavy
    hist_xh = np.zeros(n_bins, dtype=np.float32)  # Heavy-H
    hist_hh2= np.zeros(n_bins, dtype=np.float32)  # H-H

    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(pts[i] - pts[j]))
            if r <= 0 or r > r_max:
                continue
            b  = min(int(r / r_max * n_bins), n_bins - 1)
            ti = type_list[i]
            tj = type_list[j]
            hi = (ti == 'H')
            hj = (tj == 'H')
            if hi and hj:
                hist_hh2[b] += 1.0
            elif hi or hj:
                hist_xh[b]  += 1.0
            else:
                hist_hh[b]  += 1.0

    for h in (hist_hh, hist_xh, hist_hh2):
        s = h.sum()
        if s > 0:
            h /= s

    return np.concatenate([hist_hh, hist_xh, hist_hh2])

def polymer_batch(residues_list):
    """list of mol_list → (N, 192) ndarray"""
    return np.vstack([polymer_descriptor(r) for r in residues_list])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. PolymerBank — ポリマー対応 MLBank
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PolymerBank(MLBank):
    """
    MLBank をポリマー (Gly ペプチド) 向けに拡張。
    - 記述子: polymer_descriptor (Heavy-Heavy / Heavy-H / H-H)
    - 原子種: infer_atom_types_gly
    - GP: per-atom 正規化でモノマー/ダイマー/トリマーのスケール差を吸収
    """

    def __init__(self, path=None, sigma_threshold=1e-3):
        super().__init__(path, sigma_threshold)

    def _desc(self, mol_list):
        """Gly ペプチド → polymer_descriptor (192次元)"""
        return polymer_descriptor(mol_list)

    def _embed(self, mol_list):
        """polymer_descriptor → autoencoder 32次元埋め込み"""
        import torch
        desc = polymer_descriptor(mol_list)
        xt   = torch.tensor(desc, dtype=torch.float32).unsqueeze(0)
        if self.autoencoder is not None:
            self.autoencoder.eval()
            with torch.no_grad():
                return self.autoencoder.encode(xt).squeeze(0).numpy()
        return desc[:EMBED_DIM] if len(desc) >= EMBED_DIM else \
               np.pad(desc, (0, EMBED_DIM - len(desc)))

    def train_ml_polymer(self, verbose=True):
        """polymer_descriptor ベースで ML モデルを学習"""
        import torch
        from motifbank_ml import train_autoencoder, FragmentCluster, AdaptiveEpsilon

        mols_list = [v['mol'] for v in self.data.values() if 'mol' in v]
        energies  = [v['energy_Ha'] for v in self.data.values() if 'mol' in v]

        if len(mols_list) < 8:
            if verbose:
                print(f"  [PolymerBank] entries={len(mols_list)} < 8, skip ML training")
            return self

        if verbose:
            print(f"\n[PolymerBank] ML学習: {len(mols_list)} ペプチドフラグメント ...")

        descs = np.array([polymer_descriptor(m) for m in mols_list], dtype=np.float32)

        if verbose: print("  (1/3) Autoencoder (unsupervised, 192→32)...")
        self.autoencoder = train_autoencoder(descs, epochs=200, verbose=False)

        self.autoencoder.eval()
        with torch.no_grad():
            Xt     = torch.tensor(descs)
            embeds = self.autoencoder.encode(Xt).numpy()

        if verbose: print("  (2/3) Clustering...")
        self.cluster.n_clusters = 'auto'
        self.cluster.fit(embeds, energies)
        if verbose: self.cluster.report()

        if verbose: print("  (3/3) GP predictor (Matern 5/2, per-atom)...")
        n_atoms_list = [len(m) for m in mols_list]
        self.predictor = GPEnergyPredictor()
        self.predictor.fit(list(descs), energies,
                           n_atoms_list=n_atoms_list, verbose=verbose)

        self.adaptive_eps.fit(mols_list, energies)
        if verbose:
            eps = self.adaptive_eps.safe_epsilon(0.95)
            print(f"  Adaptive ε(95%) = {eps:.4f} Å")
            print("[PolymerBank] 学習完了\n")
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. TrajBankML — TrajBank + GP 統合エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrajBankML:
    """
    TrajBank + GPEnergyPredictor の統合。
    3段階クエリ:
      1. cold + exact match  → バンクヒット (QC不要)
      2. warm + GP予測       → ML予測 (σ < threshold で採用)
      3. hot                 → 実QC実行 → バンクに追加 → GPを再訓練

    TrajBank との違い:
      - PolymerBank (MLBank) を内部バンクとして使用
      - 蓄積データから GP を定期学習
      - GP が十分精度のとき、soft-miss でも QC をスキップ
    """

    def __init__(self, r_cut=6.0, eps_init=0.10,
                 atom_types_list=None, qc_func=None,
                 sigma_threshold=5e-3, ml_train_every=15,
                 verbose=True):
        self.r_cut            = r_cut
        self.eps              = eps_init
        self.atl              = atom_types_list
        self.qc_func          = qc_func or qc_compute_mock
        self.bank             = PolymerBank(sigma_threshold=sigma_threshold)
        self.tracker          = None
        self.cluster          = UnsupervisedMotifCluster(eps_global=eps_init)
        self.sigma_threshold  = sigma_threshold
        self.ml_train_every   = ml_train_every
        self.step_n           = 0
        self._mols_prev       = None
        self.history          = []
        self.verbose          = verbose
        # 統計
        self.stats = {'exact': 0, 'ml_pred': 0, 'qc': 0, 'soft': 0}

    def _store(self, k, mol, energy):
        """mol 込みでバンクに保存"""
        _store_with_mol(self.bank, k, mol, energy)

    def step(self, mols, label=None):
        """
        1ステップ: mols (各残基の座標リスト) を受け取りエネルギーを返す。
        Returns: dict {E_total_Ha, qc_calls, source_counts, step_time}
        """
        t0    = time.perf_counter()
        mols  = [np.asarray(m, dtype=float) for m in mols]
        n     = len(mols)
        label = label or f"step{self.step_n}"

        # RegionTracker 更新
        if self.tracker is None:
            self.tracker = RegionTracker(n, eps_cold=self.eps * 0.5)
        if self._mols_prev is not None:
            self.tracker.update(self._mols_prev, mols)
        cold_set = self.tracker.cold_mask()

        # 各フラグメントを3段階でクエリ
        energies = []
        qc_calls = 0
        sources  = []

        for i, mol in enumerate(mols):
            k  = geom_key(mol)
            dv = dist_vec(mol)

            # 1. Exact match
            cached = self.bank.query_exact(k)
            if cached is not None:
                energies.append(cached)
                sources.append('exact')
                self.stats['exact'] += 1
                continue

            # 2. Soft match (adaptive ε)
            eps = self.bank.adaptive_eps.safe_epsilon(0.95) \
                  if self.bank.adaptive_eps.fitted else self.eps
            soft = self.bank.query_soft(mol, eps=eps)
            if soft is not None:
                energies.append(soft)
                sources.append('soft')
                self.stats['soft'] += 1
                continue

            # 3. GP 予測 (十分なデータがある場合)
            if self.bank.predictor.trained:
                desc   = self.bank._desc(mol)
                e_pred, sigma = self.bank.predictor.predict(desc, n_atoms=len(mol))
                if sigma < self.sigma_threshold:
                    energies.append(e_pred)
                    sources.append('ml_pred')
                    self.stats['ml_pred'] += 1
                    continue

            # 4. QC 実行
            e = self.qc_func(mol)
            self._store(k, mol, e)
            energies.append(e)
            sources.append('qc')
            self.stats['qc'] += 1
            qc_calls += 1

        E_total = float(sum(energies))

        # 定期 ML 再訓練
        if (self.step_n + 1) % self.ml_train_every == 0:
            n_entries = len(self.bank.data)
            if n_entries >= 8:
                if self.verbose:
                    print(f"  [TrajBankML] step {self.step_n}: ML再訓練 (bank={n_entries})...")
                self.bank.train_ml_polymer(verbose=False)

        dt = time.perf_counter() - t0
        rec = {
            'step': self.step_n, 'label': label,
            'E_total_Ha': E_total, 'qc_calls': qc_calls,
            'source_counts': dict(zip(*np.unique(sources, return_counts=True))),
            'step_time': dt,
            'bank_size': len(self.bank.data),
        }
        self.history.append(rec)
        self._mols_prev = [m.copy() for m in mols]
        self.step_n += 1
        return rec

    def train_ml_now(self):
        """手動で ML 学習をトリガー"""
        self.bank.train_ml_polymer(verbose=self.verbose)

    def summary(self):
        if not self.history:
            return
        total_qc = sum(r['qc_calls'] for r in self.history)
        total_mol = sum(sum(r['source_counts'].values()) for r in self.history)
        saved = total_mol - total_qc
        print(f"\n[TrajBankML] {self.step_n} ステップ完了")
        print(f"  total molecule queries: {total_mol}")
        print(f"  QC calls: {total_qc}  saved: {saved}  ROI: {saved/max(total_mol,1)*100:.1f}%")
        all_src = {}
        for r in self.history:
            for s, c in r['source_counts'].items():
                all_src[s] = all_src.get(s, 0) + c
        for s, c in sorted(all_src.items(), key=lambda x: -x[1]):
            print(f"  {s:12s}: {c:5d} ({c/max(total_mol,1)*100:.1f}%)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. ConformationalSampler — FPS コンフォメーションサンプリング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConformationalSampler:
    """
    ペプチドのコンフォメーション空間を FPS (farthest-point sampling) で
    効率的に探索し、初期 bank 構築コストを削減する。

    使い方:
      sampler = ConformationalSampler(n_res=8)
      seed_confs = sampler.fps_seed(n_select=10)
      for conf in seed_confs:
          e = qc_func(conf)
          bank.store(geom_key(conf), conf, e)
    """

    def __init__(self, n_res=8, phi_range=(-180, 180), psi_range=(-180, 180),
                 n_grid=50, seed=42):
        self.n_res    = n_res
        self.phi_range = phi_range
        self.psi_range = psi_range
        self.n_grid   = n_grid
        self.rng      = np.random.RandomState(seed)

    def _sample_confs(self, n_confs):
        """φ/ψ 空間をランダムサンプリングしてコンフォメーションリストを生成"""
        confs = []
        for _ in range(n_confs):
            phi = self.rng.uniform(*self.phi_range)
            psi = self.rng.uniform(*self.psi_range)
            res = build_gly_peptide(self.n_res, phi=phi, psi=psi)
            confs.append((phi, psi, [np.asarray(r) for r in res]))
        return confs

    def fps_seed(self, n_select=10, pool_size=200):
        """
        FPS でコンフォメーション空間を均等にカバーするシードを選択。
        φ/ψ 空間のユークリッド距離でサンプリング (descriptor空間より高速)。
        """
        confs = self._sample_confs(pool_size)
        angles = np.array([[phi, psi] for phi, psi, _ in confs])
        # 角度空間で FPS (torus距離は無視、線形近似)
        selected_idx = farthest_point_sampling(angles, n_select, seed=42)
        return [(confs[i][2], confs[i][0], confs[i][1]) for i in selected_idx]

    def phi_psi_grid(self, n_phi=5, n_psi=5):
        """φ/ψ グリッド点でのコンフォメーション (Ramachandran セクション)"""
        phis = np.linspace(-180, 180, n_phi, endpoint=False)
        psis = np.linspace(-180, 180, n_psi, endpoint=False)
        results = []
        for phi in phis:
            for psi in psis:
                res = build_gly_peptide(self.n_res, phi=phi, psi=psi)
                results.append(([np.asarray(r) for r in res], phi, psi))
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. ベンチマーク比較
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_traj_bank(mols_traj, qc_func, eps=0.10, r_cut=6.0):
    """既存 TrajBank でトラジェクトリを処理"""
    tb  = TrajBank(r_cut=r_cut, eps_init=eps, qc_func=qc_func,
                   bank_path=None, cluster_interval=5)
    qc_total, mol_total = 0, 0
    E_traj = []
    for step, mols in enumerate(mols_traj):
        res = tb.step(mols, label=f"s{step}", verbose=False)
        # TrajBank.step rec: 'E_Ha', 'roi' (fraction cached)
        E_traj.append(res.get('E_Ha', res.get('E_total_Ha', 0.0)))
        n = len(mols)
        roi = res.get('roi', 0.0)
        qc_total  += max(1, round(n * (1.0 - roi)))
        mol_total += n
    return {'qc': qc_total, 'mol': mol_total, 'E': E_traj}

def _run_traj_bank_ml(mols_traj, qc_func, eps=0.10, r_cut=6.0,
                      ml_train_every=10):
    """TrajBankML でトラジェクトリを処理"""
    tb = TrajBankML(r_cut=r_cut, eps_init=eps, qc_func=qc_func,
                    sigma_threshold=5e-3, ml_train_every=ml_train_every,
                    verbose=False)
    E_traj = []

    # Phase 0: FPS シードで初期 bank を構築
    sampler = ConformationalSampler(n_res=len(mols_traj[0]))
    seeds = sampler.fps_seed(n_select=min(8, len(mols_traj[0])*2), pool_size=100)
    for residues, phi, psi in seeds:
        for res in residues:
            k = geom_key(res)
            if k not in tb.bank.data:
                e = qc_func(res)
                tb._store(k, res, e)
    if len(tb.bank.data) >= 8:
        tb.bank.train_ml_polymer(verbose=False)

    for step, mols in enumerate(mols_traj):
        rec = tb.step(mols, label=f"s{step}")
        E_traj.append(rec['E_total_Ha'])

    qc_total  = tb.stats['qc']
    mol_total = sum(sum(r['source_counts'].values()) for r in tb.history)
    return {'qc': qc_total, 'mol': mol_total, 'E': E_traj,
            'stats': tb.stats, 'history': tb.history}

def benchmark_polymer(n_res=4, n_steps=30, sigma=0.01, seed=42,
                      phi=-60., psi=-40., verbose=True):
    """
    Gly_n ペプチド擬似 MD: TrajBank vs TrajBankML の比較

    Args:
      n_res   : 残基数 (少ないほど高速)
      n_steps : MDステップ数
      sigma   : 1ステップ変位 (Å)
      phi/psi : 初期構造二面角 (α-helix: -60/-40, β-sheet: -120/130)
    """
    print("=" * 65)
    print(f"高分子シミュレーション ML ベンチマーク")
    print(f"Gly{n_res}  φ={phi}°  ψ={psi}°  steps={n_steps}  σ={sigma}Å")
    print("=" * 65)

    rng = np.random.RandomState(seed)
    base = build_gly_peptide(n_res, phi=phi, psi=psi)
    base = [np.asarray(r, dtype=float) for r in base]

    # トラジェクトリ生成
    mols_traj = [base]
    curr = base
    for _ in range(n_steps - 1):
        curr = perturb_peptide(curr, sigma)
        mols_traj.append(curr)

    qc_mock = qc_compute_mock

    # 参照: 全ステップ新規QC
    E_ref = []
    for mols in mols_traj:
        E_ref.append(sum(qc_mock(m) for m in mols))

    print(f"\n  N_residues = {n_res},  N_steps = {n_steps}")
    print(f"  E_ref range: {min(E_ref):.3f} – {max(E_ref):.3f} Ha  "
          f"σ = {np.std(E_ref):.4f} Ha")

    # ── [A] 既存 TrajBank ──
    print(f"\n[A] 既存 TrajBank (exact + soft match)...")
    t0 = time.perf_counter()
    res_tb = _run_traj_bank(mols_traj, qc_mock, eps=0.10)
    t_tb   = time.perf_counter() - t0
    roi_tb = 1 - res_tb['qc'] / max(res_tb['mol'], 1)
    mae_tb = float(np.mean(np.abs(np.array(res_tb['E']) - np.array(E_ref))))

    # ── [B] TrajBankML (GP 統合) ──
    print(f"\n[B] TrajBankML (FPS-seed + GP 予測)...")
    t0 = time.perf_counter()
    res_ml = _run_traj_bank_ml(mols_traj, qc_mock, eps=0.10,
                                ml_train_every=max(5, n_steps//6))
    t_ml   = time.perf_counter() - t0
    roi_ml = 1 - res_ml['qc'] / max(res_ml['mol'], 1)
    mae_ml = float(np.mean(np.abs(np.array(res_ml['E']) - np.array(E_ref))))

    # ── 結果表示 ──
    print(f"\n{'─'*65}")
    print(f"{'指標':30s}  {'TrajBank':>12}  {'TrajBankML':>12}  {'改善':>8}")
    print(f"{'─'*65}")

    def row(label, a, b, fmt='.1f', unit='', higher_better=True):
        if isinstance(a, float):
            sa = f"{a:{fmt}}{unit}"
            sb = f"{b:{fmt}}{unit}"
        else:
            sa, sb = str(a), str(b)
        if higher_better:
            mark = "↑" if b > a else ("=" if b == a else "↓")
        else:
            mark = "↑" if b < a else ("=" if b == a else "↓")
        print(f"  {label:28s}  {sa:>12}  {sb:>12}  {mark:>8}")

    row("QC コール数",         res_tb['qc'],  res_ml['qc'],  fmt='d',   higher_better=False)
    row("ROI (QC 節約率)",     roi_tb*100,    roi_ml*100,    unit='%',  higher_better=True)
    row("MAE vs ref (Ha)",     mae_tb,        mae_ml,        fmt='.4f', higher_better=False)
    row("経過時間 (s)",         t_tb,          t_ml,          fmt='.2f', higher_better=False)
    row("Bank size",           len([]),       res_ml.get('stats', {}).get('qc', 0),
        fmt='d', higher_better=False)

    # src 内訳
    print(f"\n  [TrajBankML] クエリ内訳:")
    stats = res_ml.get('stats', {})
    total = sum(stats.values())
    for s, c in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {s:12s}: {c:5d} ({c/max(total,1)*100:.1f}%)")

    return {
        'traj_bank': res_tb, 'traj_bank_ml': res_ml,
        'E_ref': E_ref,
        'roi_tb': roi_tb, 'roi_ml': roi_ml,
        'mae_tb': mae_tb, 'mae_ml': mae_ml,
    }


def descriptor_quality_check():
    """polymer_descriptor の化学的正確性を検証"""
    print("\n[記述子品質チェック]")

    # α-helix (φ=-60, ψ=-40) モノマー
    res_helix = build_gly_peptide(4, phi=-60., psi=-40.)
    res_beta  = build_gly_peptide(4, phi=-120., psi=130.)

    mono_h = [np.asarray(res_helix[0])]   # 1残基のみ
    mono_b = [np.asarray(res_beta[0])]

    desc_h = polymer_descriptor(mono_h)
    desc_b = polymer_descriptor(mono_b)

    # 1残基の重原子-重原子チャンネル (channel 0): N-Cα, Cα-C, C=O 等
    hh_h = desc_h[:ELEM_BINS]
    xh_h = desc_h[ELEM_BINS:2*ELEM_BINS]

    # N-Cα 結合距離 (~1.46 Å) がチャンネル 0 に現れるか
    peak_hh = int(np.argmax(hh_h)) / ELEM_BINS * 8.0
    peak_xh = int(np.argmax(xh_h)) / ELEM_BINS * 8.0
    print(f"  Heavy-Heavy ピーク: {peak_hh:.2f} Å (期待: N-Cα ~1.46 Å 付近)")
    print(f"  Heavy-H ピーク   : {peak_xh:.2f} Å (期待: N-H ~1.01 Å 付近)")

    # α-helix vs β-sheet の記述子距離
    all_res_h = [np.asarray(r) for r in res_helix]
    all_res_b = [np.asarray(r) for r in res_beta]
    d_h = polymer_descriptor(all_res_h)
    d_b = polymer_descriptor(all_res_b)
    diff = float(np.linalg.norm(d_h - d_b))
    print(f"  α-helix vs β-sheet 記述子距離: {diff:.4f}  "
          f"({'明確に区別 ✓' if diff > 0.1 else '区別不十分 ✗'})")

    # flat RDF との比較
    d_rdf_h = rdf_descriptor(all_res_h)
    d_rdf_b = rdf_descriptor(all_res_b)
    diff_rdf = float(np.linalg.norm(d_rdf_h - d_rdf_b))
    better = diff > diff_rdf
    print(f"  polymer_desc 距離: {diff:.4f}  flat_rdf 距離: {diff_rdf:.4f}  "
          f"({'polymer_desc がより識別力高い ✓' if better else 'flat_rdf の方が大きい △'})")

    return diff > 0.1


def conformational_coverage_test(n_res=4):
    """FPS-seed が Ramachandran 空間をカバーする検証"""
    print("\n[コンフォメーション空間カバレッジ]")
    sampler = ConformationalSampler(n_res=n_res)

    fps_seeds = sampler.fps_seed(n_select=12, pool_size=100)
    # ランダムシード
    rng = np.random.RandomState(0)
    all_confs = sampler._sample_confs(100)
    rand_idx = list(rng.choice(100, 12, replace=False))
    rand_seeds = [all_confs[i] for i in rand_idx]

    fps_angles  = np.array([[phi, psi] for _, phi, psi in fps_seeds])
    rand_angles = np.array([[phi, psi] for phi, psi, _ in rand_seeds])
    all_angles  = np.array([[phi, psi] for phi, psi, _ in all_confs])

    def coverage(selected, pool):
        dists = np.array([np.min(np.linalg.norm(pool - s, axis=1))
                          for s in selected])
        return float(np.mean(dists))  # 平均最近傍距離 (大きいほど多様)

    cov_fps  = coverage(fps_angles, all_angles)
    cov_rand = coverage(rand_angles, all_angles)
    print(f"  FPS-seed coverage : {cov_fps:.2f}°  (12点でφ/ψ空間をカバー)")
    print(f"  Random-seed coverage: {cov_rand:.2f}°")
    better = cov_fps > cov_rand
    print(f"  FPS が {'より広く ✓' if better else '同程度'}カバー "
          f"({cov_fps/max(cov_rand,1e-3):.2f}×)")
    return better


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick',   action='store_true', help='高速モード (N=4, steps=15)')
    ap.add_argument('--demo',    action='store_true', help='記述子品質確認のみ')
    ap.add_argument('--n_res',   type=int, default=6)
    ap.add_argument('--n_steps', type=int, default=30)
    ap.add_argument('--helix',   action='store_true', help='α-helix (デフォルト)')
    ap.add_argument('--beta',    action='store_true', help='β-sheet モード')
    args = ap.parse_args()

    phi, psi = -60., -40.  # α-helix デフォルト
    if args.beta:
        phi, psi = -120., 130.
        print("モード: β-sheet")
    else:
        print("モード: α-helix")

    print("=" * 65)
    print("MotifBank ML × TrajBank — 高分子シミュレーション統合エンジン")
    print("=" * 65)

    # 記述子品質チェック
    ok1 = descriptor_quality_check()

    # コンフォメーションカバレッジ
    ok2 = conformational_coverage_test(n_res=4)

    if args.demo:
        print(f"\n記述子チェック: {'PASS' if ok1 else 'WARN'}")
        print(f"カバレッジチェック: {'PASS' if ok2 else 'WARN'}")
        return

    # ベンチマーク
    n_res   = 4 if args.quick else args.n_res
    n_steps = 15 if args.quick else args.n_steps

    result = benchmark_polymer(
        n_res=n_res, n_steps=n_steps, sigma=0.01,
        phi=phi, psi=psi, verbose=True
    )

    # 統合サマリ
    print(f"\n{'='*65}")
    print("統合サマリ: MotifBank ML × 高分子シミュレーション")
    print(f"{'='*65}")
    print(f"  記述子   : polymer_descriptor (Heavy-Heavy/Heavy-H/H-H, 192次元)")
    print(f"  予測器   : GPEnergyPredictor (Matérn 5/2, per-atom 正規化)")
    print(f"  シード   : FPS (farthest-point sampling) でコンフォメーション多様性確保")
    print(f"  統合効果 : ROI {result['roi_tb']*100:.1f}% → {result['roi_ml']*100:.1f}%  "
          f"(TrajBank→TrajBankML)")
    print()

if __name__ == '__main__':
    main()
