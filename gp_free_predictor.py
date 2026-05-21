#!/usr/bin/env python3
"""
gp_free_predictor.py — "GP を消す" 設計: retrieval + kernel smoother + tiny correction

設計思想:
  GP の役割は 2 つだった:
    (1) エネルギー補間 (neighbor の加重平均) → Nadaraya-Watson で代替 (O(k), <0.1ms)
    (2) 不確かさ推定 (posterior variance)   → 距離ベースヒューリスティックで代替

  結果として:
    [旧] encoder → FAISS → GP(360ms) → (μ, σ)
    [新] encoder → FAISS → NW(<0.1ms) + 微小残差NN(1ms) → (μ, σ)

  3 段階ルーティング:
    Level 0: cosine_sim > 0.99  → exact retrieval  (0 computation)
    Level 1: cosine_sim > 0.70  → NW kernel smooth  (<0.1ms)
    Level 2: cosine_sim < 0.70  → DFT キュー        (16s, novel fragment)

γ 接続:
  γ 小 (結晶) → bank hit 率 高 → Level 0 が支配 → cost ≈ 0
  γ 大 (非晶) → Level 2 が支配 → cost ∝ N_novel (unavoidable)

Usage:
  OMP_NUM_THREADS=1 python3 gp_free_predictor.py
"""

import os, sys, json, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

from motifbank_cli import geom_key
from motifbank_ml import GPEnergyPredictor, farthest_point_sampling
from real_gp_benchmark import generate_sioh4_fragments, CACHE_FILE, KCAL
from world_encoder import (
    pairwise_desc, MolecularEncoder, pretrain, INPUT_DIM, EMBED_DIM,
)


# ─────────────────────────────────────────────────
# §1 Nadaraya-Watson kernel smoother (GP 代替)
# ─────────────────────────────────────────────────

class NWRegressor:
    """
    Nadaraya-Watson カーネル回帰 (コサイン類似度ベース):
      Ê(x) = Σ_i softmax(sim(z,z_i)/τ) × E_i
      σ²(x) = Σ_i softmax(sim(z,z_i)/τ) × (E_i - Ê(x))²
    """
    def __init__(self, tau=0.15):
        self.tau = tau
        self.Z   = None   # (N, D) L2-normalized
        self.E   = None   # (N,) energies

    def fit(self, Z, E):
        self.Z = np.asarray(Z, dtype=np.float32)
        self.E = np.asarray(E, dtype=np.float64)

    def predict(self, z, k=50):
        assert self.Z is not None
        sims = self.Z @ z.astype(np.float32)
        k_eff = min(k, len(self.E))
        top_idx = np.argpartition(-sims, k_eff)[:k_eff]
        sims_k  = sims[top_idx].astype(np.float64)
        E_k     = self.E[top_idx]
        w = np.exp((sims_k - sims_k.max()) / self.tau)
        w /= w.sum()
        mu    = float(np.dot(w, E_k))
        sigma = float(np.sqrt(np.dot(w, (E_k - mu) ** 2)))
        return mu, sigma


class NWDescriptorRegressor:
    """
    NW smoother using pairwise_desc directly (same features as GP).
    RBF kernel (Gaussian) in descriptor space: sim = exp(-||x-x_i||²/(2l²))
    l (length scale) is tuned once on validation set.
    """
    def __init__(self, length_scale=1.0):
        self.l  = length_scale
        self.X  = None   # (N, D) raw descriptors (standardized)
        self.E  = None

    def fit(self, X, E):
        self.X = np.asarray(X, dtype=np.float64)
        self.E = np.asarray(E, dtype=np.float64)

    def predict(self, x, k=50):
        dists2 = np.sum((self.X - x.astype(np.float64)) ** 2, axis=1)
        k_eff  = min(k, len(self.E))
        top_idx = np.argpartition(dists2, k_eff)[:k_eff]
        d2_k   = dists2[top_idx]
        E_k    = self.E[top_idx]
        w = np.exp(-d2_k / (2 * self.l ** 2))
        w_sum = w.sum()
        if w_sum < 1e-300:   # all neighbors too far → uniform
            w = np.ones(k_eff, dtype=np.float64) / k_eff
        else:
            w /= w_sum
        mu    = float(np.dot(w, E_k))
        sigma = float(np.sqrt(np.dot(w, (E_k - mu) ** 2)))
        return mu, sigma


# ─────────────────────────────────────────────────
# §2 残差補正NN (tiny: 32→16→1, ~600 params)
# ─────────────────────────────────────────────────

class ResidualCorrector(nn.Module):
    """
    NW 回帰の残差を学習: Δ = E_true - E_NW
    入力: 32-dim encoder embedding
    出力: スカラー補正値 (Ha)

    学習後は E_pred = E_NW + Δ_nn → 精度向上
    """
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.GELU(),
            nn.Linear(16, 8),         nn.GELU(),
            nn.Linear(8, 1),
        )
        self._y_mean = 0.0
        self._y_std  = 1.0

    def forward(self, z):
        return self.net(z)

    def fit(self, Z_train, E_nw_train, E_true_train, n_epochs=500, lr=3e-3):
        """Z: (N,D) embeddings, E_nw: NW予測, E_true: DFT真値"""
        residuals = (E_true_train - E_nw_train).astype(np.float32)
        self._y_mean = float(residuals.mean())
        self._y_std  = float(residuals.std()) + 1e-8
        y_norm = (residuals - self._y_mean) / self._y_std

        Zt = torch.tensor(Z_train, dtype=torch.float32)
        yt = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        for _ in range(n_epochs):
            pred = self(Zt)
            loss = F.mse_loss(pred, yt)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    def correct(self, z_np):
        """z_np: (D,) numpy array → scalar correction in Ha"""
        self.eval()
        with torch.no_grad():
            zt = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)
            delta_norm = self(zt).item()
        return delta_norm * self._y_std + self._y_mean


# ─────────────────────────────────────────────────
# §3 完全ルーティング推論 (3 レベル)
# ─────────────────────────────────────────────────

class RoutingPredictor:
    """
    3 レベル計算ルーティング:

      Level 0 (exact)  : max_sim > sim_exact → bank エネルギーをそのまま返す
      Level 1 (fast)   : max_sim > sim_novel → NW 補間 (+残差NN)
      Level 2 (DFT)    : max_sim < sim_novel → DFT キューに追加

    γ 小の系 (結晶) → Level 0 が圧倒的多数 → cost → O(1)
    γ 大の系 (非晶) → Level 2 が増加 → cost ∝ N_novel

    これが「計算量を novelty/uncertainty で決める」の実装。
    """
    def __init__(self, sim_exact=0.995, sim_novel=0.70):
        self.sim_exact  = sim_exact
        self.sim_novel  = sim_novel
        # 以下は fit() で設定
        self.encoder    = None
        self.scaler     = None
        self.nw         = None
        self.residual   = None
        self._Z_train   = None   # (N, D) L2-normalized
        self._E_train   = None
        self._mols_train = None

    def fit(self, encoder, scaler, mols_train, energies_train,
            nw_tau=0.15, train_residual=True, n_atoms=9):
        self.encoder = encoder
        self.scaler  = scaler
        encoder.eval()

        # 埋め込みを事前計算
        Zs = []
        for mol in mols_train:
            x = scaler.transform(pairwise_desc(mol).reshape(1, -1))
            xt = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                z = encoder(xt).numpy()[0]   # L2-normalized
            Zs.append(z)
        Z  = np.array(Zs, dtype=np.float32)
        E  = np.array(energies_train, dtype=np.float64)
        self._Z_train     = Z
        self._E_train     = E
        self._mols_train  = mols_train

        # NW フィット
        self.nw = NWRegressor(tau=nw_tau)
        self.nw.fit(Z, E)

        # 残差補正NN (オプション)
        if train_residual:
            E_nw_tr = np.array([self.nw.predict(Z[i], k=min(50, len(E)-1))[0]
                                 for i in range(len(E))])
            self.residual = ResidualCorrector()
            self.residual.fit(Z, E_nw_tr, E, n_epochs=500)

        # FAISS index (L2-norm 済みベクトルで inner product = cosine)
        if FAISS_OK:
            Z_idx = Z.copy()
            faiss.normalize_L2(Z_idx)
            self._faiss_index = faiss.IndexFlatIP(Z.shape[1])
            self._faiss_index.add(Z_idx)
        else:
            self._faiss_index = None

    def _embed(self, mol):
        x = self.scaler.transform(pairwise_desc(mol).reshape(1, -1))
        xt = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            self.encoder.eval()
            z = self.encoder(xt).numpy()[0]
        return z   # already L2-normalized by encoder

    def predict(self, mol, n_atoms=9):
        """
        Returns:
          (energy_Ha, uncertainty_Ha, level, sim_max)
          level: 0=exact, 1=nw, 1r=nw+residual, 2=dft_needed
        """
        z = self._embed(mol)
        z_idx = z.reshape(1, -1).astype(np.float32)

        # 最近傍の cosine similarity
        if self._faiss_index is not None:
            sims, idxs = self._faiss_index.search(z_idx, 1)
            max_sim = float(sims[0, 0])
            nn_idx  = int(idxs[0, 0])
        else:
            sims = self._Z_train @ z
            nn_idx  = int(np.argmax(sims))
            max_sim = float(sims[nn_idx])

        # ── Level 0: exact match ──
        if max_sim >= self.sim_exact:
            return float(self._E_train[nn_idx]), 0.0, 0, max_sim

        # ── Level 2: OOD → DFT ──
        if max_sim < self.sim_novel:
            return None, None, 2, max_sim

        # ── Level 1: NW 補間 ──
        mu, sigma = self.nw.predict(z, k=min(50, len(self._E_train)-1))
        if self.residual is not None:
            delta = self.residual.correct(z)
            mu += delta
            return mu, sigma, "1r", max_sim
        return mu, sigma, 1, max_sim

    def route_stats(self, mols_test):
        """テストセット全体のルーティング分布を返す"""
        counts = {0: 0, 1: 0, "1r": 0, 2: 0}
        sims_all = []
        for mol in mols_test:
            _, _, lvl, sim = self.predict(mol)
            counts[lvl] = counts.get(lvl, 0) + 1
            sims_all.append(sim)
        return counts, sims_all


# ─────────────────────────────────────────────────
# §4 τ 最適化 (NW の唯一のハイパーパラメータ)
# ─────────────────────────────────────────────────

def tune_tau(Z_tr, E_tr, Z_val, E_val, taus=None, k=50):
    """NW の温度 τ をバリデーションセットで探索 (1 回だけ)"""
    if taus is None:
        taus = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    best_tau, best_mae = taus[0], float("inf")
    nw_tmp = NWRegressor()
    nw_tmp.fit(Z_tr, E_tr)
    for tau in taus:
        nw_tmp.tau = tau
        preds = [nw_tmp.predict(Z_val[i], k=k)[0] for i in range(len(Z_val))]
        mae = np.mean(np.abs(np.array(preds) - E_val)) * KCAL
        if mae < best_mae:
            best_mae, best_tau = mae, tau
    return best_tau, best_mae


# ─────────────────────────────────────────────────
# §5 メイン比較実験
# ─────────────────────────────────────────────────

def exp_gp_free():
    print("=" * 65)
    print("GP を消す: retrieval + kernel smoother + tiny correction")
    print("=" * 65)

    cache = json.load(open(CACHE_FILE))
    mols_all, _, _ = generate_sioh4_fragments(n=600, seed=42)
    mols, energies, gkeys = [], [], []
    for mol in mols_all:
        k = str(geom_key(mol))
        if k in cache:
            mols.append(mol); energies.append(cache[k]); gkeys.append(k)

    X_pd = np.array([pairwise_desc(m) for m in mols])
    n_total = len(mols)
    # n_train: use up to 70% of available data, capped for fair comparison
    n_train = min(max(120, int(n_total * 0.7)), n_total - max(30, int(n_total * 0.2)))
    idx_tr  = farthest_point_sampling(X_pd, n_train, seed=42)
    idx_te  = [i for i in range(len(mols)) if i not in set(idx_tr)]
    mols_tr = [mols[i] for i in idx_tr]
    mols_te = [mols[i] for i in idx_te]
    y_tr    = np.array([energies[i] for i in idx_tr])
    y_te    = np.array([energies[i] for i in idx_te])
    N_te    = len(idx_te)

    print(f"\n  データ: {len(mols)} fragments, train={n_train}, test={N_te}")

    # ── encoder 事前学習 ──
    print("\n  encoder pre-training ...", end=" ", flush=True)
    t0 = time.time()
    encoder, scaler, _ = pretrain(mols_tr, [gkeys[i] for i in idx_tr],
                                  n_epochs=500, verbose=False)
    t_pre = time.time() - t0
    print(f"{t_pre:.0f}s")

    # encoder embeddings 計算
    def embed_mols(ms):
        encoder.eval()
        Z = []
        for mol in ms:
            x = scaler.transform(pairwise_desc(mol).reshape(1,-1))
            xt = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                Z.append(encoder(xt).numpy()[0])
        return np.array(Z, dtype=np.float32)

    Z_tr = embed_mols(mols_tr)
    Z_te = embed_mols(mols_te)

    # ── [A] GP ベースライン ──
    print("\n  [A] Global GP (baseline)")
    gp = GPEnergyPredictor()
    t0 = time.time()
    gp.fit(list(X_pd[idx_tr]), list(y_tr), n_atoms_list=[9]*n_train, verbose=False)
    t_gp_fit = time.time() - t0
    t0 = time.time()
    preds_a = [gp.predict(X_pd[i], n_atoms=9)[0] for i in idx_te]
    t_gp_inf = (time.time() - t0) / N_te * 1000
    mae_a = np.mean(np.abs(np.array(preds_a) - y_te)) * KCAL
    print(f"  MAE={mae_a:.3f} kcal  fit={t_gp_fit:.1f}s  infer={t_gp_inf:.1f}ms/call")

    # ── descriptor 正規化 (GP と同一特徴量) ──
    from sklearn.preprocessing import StandardScaler as _SS
    desc_scaler = _SS().fit(X_pd[idx_tr])
    X_tr_sc = desc_scaler.transform(X_pd[idx_tr])
    X_te_sc = desc_scaler.transform(X_pd[idx_te])

    # ── [B2] NW on pairwise_desc (GPと同一特徴量) ──
    print("\n  [B2] NW on pairwise_desc (same features as GP)")
    # length_scale チューニング
    n_val = 40
    X_sub_tr  = X_tr_sc[:n_train - n_val]
    X_sub_val = X_tr_sc[n_train - n_val:]
    y_sub_tr  = y_tr[:n_train - n_val]
    y_sub_val = y_tr[n_train - n_val:]
    best_l, best_l_mae = 1.0, float("inf")
    nw_d_tmp = NWDescriptorRegressor()
    nw_d_tmp.fit(X_sub_tr, y_sub_tr)
    for ls in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        nw_d_tmp.l = ls
        ps = [nw_d_tmp.predict(X_sub_val[i], k=50)[0]
              for i in range(len(y_sub_val))]
        mae_v = np.mean(np.abs(np.array(ps) - y_sub_val)) * KCAL
        if mae_v < best_l_mae:
            best_l_mae, best_l = mae_v, ls
    print(f"  l={best_l:.1f}  val_MAE={best_l_mae:.3f} kcal")
    nw_d = NWDescriptorRegressor(length_scale=best_l)
    nw_d.fit(X_tr_sc, y_tr)
    t0 = time.time()
    preds_b2, sigs_b2 = [], []
    for i in range(N_te):
        mu, sig = nw_d.predict(X_te_sc[i], k=50)
        preds_b2.append(mu); sigs_b2.append(sig)
    t_nw_d = (time.time() - t0) / N_te * 1000
    mae_b2 = np.mean(np.abs(np.array(preds_b2) - y_te)) * KCAL
    print(f"  MAE={mae_b2:.3f} kcal  infer={t_nw_d:.3f}ms/call  "
          f"speedup vs GP: {t_gp_inf/t_nw_d:.0f}×")

    # ── τ チューニング for embed-NW (1 回だけ) ──
    print("\n  NW(embed) τ チューニング ...", end=" ", flush=True)
    idx_sub_tr  = list(range(n_train - n_val))
    idx_sub_val = list(range(n_train - n_val, n_train))
    best_tau, best_val_mae = tune_tau(
        Z_tr[idx_sub_tr], y_tr[idx_sub_tr],
        Z_tr[idx_sub_val], y_tr[idx_sub_val],
    )
    print(f"τ={best_tau:.3f}  val_MAE={best_val_mae:.3f} kcal")

    # ── [B] NW on encoder embedding (GP なし) ──
    print("\n  [B] NW kernel smoother on encoder embedding")
    nw = NWRegressor(tau=best_tau)
    nw.fit(Z_tr, y_tr)
    t0 = time.time()
    preds_b, sigs_b = [], []
    for z in Z_te:
        mu, sig = nw.predict(z, k=50)
        preds_b.append(mu); sigs_b.append(sig)
    t_nw = (time.time() - t0) / N_te * 1000
    mae_b = np.mean(np.abs(np.array(preds_b) - y_te)) * KCAL
    print(f"  MAE={mae_b:.3f} kcal  infer={t_nw:.3f}ms/call  "
          f"speedup vs GP: {t_gp_inf/t_nw:.0f}×")

    # ── [C2] NW-desc + 残差NN (72-dim input) ──
    print("\n  [C2] NW(desc) + 残差NN (72-dim input, 500 epoch)")

    class _DescCorrector(nn.Module):
        def __init__(self, in_dim=72):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32), nn.GELU(),
                nn.Linear(32, 16),     nn.GELU(),
                nn.Linear(16, 1),
            )
            self._ym, self._ys = 0.0, 1.0

        def forward(self, x):
            return self.net(x)

        def fit(self, X_sc, E_nw_tr, E_true, n_epochs=500, lr=3e-3):
            res = (E_true - E_nw_tr).astype(np.float32)
            self._ym = float(res.mean()); self._ys = float(res.std()) + 1e-8
            y_n = (res - self._ym) / self._ys
            Xt = torch.tensor(X_sc, dtype=torch.float32)
            yt = torch.tensor(y_n, dtype=torch.float32).unsqueeze(1)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            for _ in range(n_epochs):
                loss = nn.functional.mse_loss(self(Xt), yt)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        def correct(self, x_sc):
            self.eval()
            with torch.no_grad():
                xt = torch.tensor(x_sc, dtype=torch.float32).unsqueeze(0)
                d = self(xt).item()
            return d * self._ys + self._ym

    E_nw_d_tr = np.array([nw_d.predict(X_tr_sc[i], k=min(50, n_train-1))[0]
                           for i in range(n_train)])
    corr2 = _DescCorrector(in_dim=X_tr_sc.shape[1])
    t0 = time.time()
    corr2.fit(X_tr_sc, E_nw_d_tr, y_tr)
    t_rc2_fit = time.time() - t0
    t0 = time.time()
    preds_c2 = [preds_b2[i] + corr2.correct(X_te_sc[i]) for i in range(N_te)]
    t_rc2_inf = (time.time() - t0) / N_te * 1000
    mae_c2 = np.mean(np.abs(np.array(preds_c2) - y_te)) * KCAL
    n_p_c2 = sum(p.numel() for p in corr2.parameters())
    print(f"  MAE={mae_c2:.3f} kcal  fit={t_rc2_fit:.1f}s  "
          f"infer={t_nw_d+t_rc2_inf:.3f}ms/call  ({n_p_c2} params)")

    # ── [C] NW-embed + 残差NN ──
    print("\n  [C] NW(embed) + 残差補正NN (32-dim input, 500 epoch)")
    E_nw_tr = np.array([nw.predict(Z_tr[i], k=min(50, n_train-1))[0]
                         for i in range(n_train)])
    corrector = ResidualCorrector()
    t0 = time.time()
    corrector.fit(Z_tr, E_nw_tr, y_tr)
    t_rc_fit = time.time() - t0

    t0 = time.time()
    preds_c = [preds_b[i] + corrector.correct(Z_te[i]) for i in range(N_te)]
    t_rc_inf = (time.time() - t0) / N_te * 1000
    mae_c = np.mean(np.abs(np.array(preds_c) - y_te)) * KCAL
    n_params_rc = sum(p.numel() for p in corrector.parameters())
    print(f"  MAE={mae_c:.3f} kcal  fit={t_rc_fit:.1f}s  infer={t_nw+t_rc_inf:.3f}ms/call  "
          f"({n_params_rc} params)")

    # ── [D] RoutingPredictor (完全ルーティング) ──
    print("\n  [D] RoutingPredictor (3-level routing)")
    router = RoutingPredictor(sim_exact=0.995, sim_novel=0.70)
    t0 = time.time()
    router.fit(encoder, scaler, mols_tr, y_tr, nw_tau=best_tau)
    t_router_fit = time.time() - t0

    preds_d, sigs_d, levels_d, sims_d = [], [], [], []
    t0 = time.time()
    for mol in mols_te:
        mu, sig, lvl, sim = router.predict(mol)
        preds_d.append(mu if mu is not None else float(np.mean(y_tr)))
        sigs_d.append(sig if sig is not None else 1.0)
        levels_d.append(lvl); sims_d.append(sim)
    t_router_inf = (time.time() - t0) / N_te * 1000

    mae_d = np.mean(np.abs(np.array(preds_d) - y_te)) * KCAL
    level_counts = {k: levels_d.count(k) for k in [0, 1, "1r", 2]}
    print(f"  MAE={mae_d:.3f} kcal  infer={t_router_inf:.2f}ms/call")
    print(f"  ルーティング: L0(exact)={level_counts.get(0,0)} "
          f"L1(NW)={level_counts.get(1,0)} "
          f"L1r(NW+corr)={level_counts.get('1r',0)} "
          f"L2(DFT)={level_counts.get(2,0)}")

    # ── σ 較正チェック ──
    print("\n  σ 較正チェック (NW uncertainty):")
    errs = np.abs(np.array(preds_b) - y_te) * KCAL
    sigs_kcal = np.array(sigs_b) * KCAL
    from scipy.stats import spearmanr
    rho, p = spearmanr(sigs_kcal, errs)
    print(f"  ρ(σ, |err|) = {rho:+.3f}  p={p:.3e}  "
          f"{'✓ 有意' if p < 0.05 else '✗ 非有意'}")
    # σ 閾値別の誤検知率
    for thresh_kcal in [0.5, 1.0, 2.0]:
        flag = sigs_kcal > thresh_kcal
        if flag.sum() > 0:
            prec = (errs[flag] > 0.5).mean()  # 実際に誤差が大きい割合
            print(f"  σ>{thresh_kcal:.1f}: {flag.sum()}/{N_te} flagged, "
                  f"precision={prec:.2f}")

    # ── サマリ ──
    print("\n" + "=" * 65)
    print("★ GP 削除の効果")
    print("=" * 65)
    print(f"  [A]   Global GP                     : MAE={mae_a:.3f} kcal  {t_gp_inf:.1f}ms/call")
    print(f"  [B2]  NW on pairwise_desc (l-tuned) : MAE={mae_b2:.3f} kcal  {t_nw_d:.3f}ms/call  "
          f"({t_gp_inf/t_nw_d:.0f}× faster)")
    print(f"  [C2]  NW(desc) + 残差NN ({n_p_c2}p)    : MAE={mae_c2:.3f} kcal  "
          f"{t_nw_d+t_rc2_inf:.3f}ms/call")
    print(f"  [B]   NW on encoder embed (τ-tuned) : MAE={mae_b:.3f} kcal  {t_nw:.3f}ms/call  "
          f"({t_gp_inf/t_nw:.0f}× faster)")
    print(f"  [C]   NW(embed) + 残差NN ({n_params_rc}p)   : MAE={mae_c:.3f} kcal  "
          f"{t_nw+t_rc_inf:.3f}ms/call")
    print(f"  [D]   RoutingPredictor              : MAE={mae_d:.3f} kcal  "
          f"{t_router_inf:.2f}ms/call")

    nw_best_mae = min(mae_b, mae_b2)
    nw_best_ms  = t_nw_d if mae_b2 < mae_b else t_nw
    print(f"""
  GP を消した効果:
  ─────────────────────────────────────────────
  速度: {t_gp_inf:.1f}ms → {nw_best_ms:.3f}ms  ({t_gp_inf/nw_best_ms:.0f}× speedup)
  精度: {mae_a:.3f} vs {nw_best_mae:.3f} kcal (差 {nw_best_mae-mae_a:+.3f} kcal)
  パラメータ: GP 自動 (sklearn) → NW l/τ=1個 + 残差NN {n_params_rc}個

  スケーリング則の変化:
  ─────────────────────────────────────────────
  [旧] cost ∝ N_system  (N 個の GP fit)
  [新] cost ∝ N_novel   (Level 2 ルーティング回数)
       N_novel = γ × N → γ小(結晶)なら O(1)

  γ → ルーティング → 計算量:
    Phase 0 (γ≈0): Level 0 支配 → cost = N_bank_sat × T_QC (定数)
    Phase 2/3 (γ>γ_c): Level 2 多発 → cost ∝ N
""")

    return {
        "mae_a": mae_a, "mae_b2": mae_b2, "mae_c2": mae_c2,
        "mae_b": mae_b, "mae_c": mae_c, "mae_d": mae_d,
        "infer_ms_gp": t_gp_inf, "infer_ms_nw_desc": t_nw_d, "infer_ms_nw": t_nw,
        "speedup_desc": t_gp_inf / t_nw_d, "speedup_embed": t_gp_inf / t_nw,
        "tau_best": best_tau, "l_best": best_l, "routing": level_counts,
    }


if __name__ == "__main__":
    r = exp_gp_free()
    out = os.path.join(_DIR, "gp_free_results.json")
    with open(out, "w") as f:
        json.dump({k: (v if not isinstance(v, dict) else
                       {str(kk): vv for kk, vv in v.items()})
                   for k, v in r.items()}, f, indent=2)
    print(f"\n結果保存: gp_free_results.json")
