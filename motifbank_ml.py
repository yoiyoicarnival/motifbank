#!/usr/bin/env python3
"""
motifbank_ml.py — MotifBank ML拡張
教師なし学習 + アクティブラーニングによるフラグメントバンク深化

主要コンポーネント:
  1. RDF Descriptor        : 可変サイズフラグメント → 固定長特徴量 (教師なし)
  2. FragmentAutoencoder   : 幾何 → 32次元埋め込み (再構成損失 + triplet)
  3. AdaptiveEpsilon       : bankデータから最適 soft-matching 閾値を学習
  4. EnsemblePredictor     : 埋め込み → エネルギー + 不確実性 (5モデルアンサンブル)
  5. ActiveLearner         : 不確実性ドリブンQCサンプリング
  6. MLBank                : MotifBank + ML クエリを統合

使い方:
  OMP_NUM_THREADS=1 python3 motifbank_ml.py --demo
  OMP_NUM_THREADS=1 python3 motifbank_ml.py --train  BANK.json
  OMP_NUM_THREADS=1 python3 motifbank_ml.py --active BANK.json --budget 50
"""

import os, sys, json, time, argparse, itertools
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from motifbank_cli import (
    MotifBank, geom_key, dist_vec, from_cif,
    qc_compute_mock, classify, R_CUT_DEF,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §1. RDF Descriptor — 可変サイズフラグメント → 固定長
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RDF_BINS   = 64    # 距離ヒストグラムのビン数
RDF_R_MAX  = 8.0   # 最大距離 (Å)
EMBED_DIM  = 32    # 埋め込み次元
ELEM_BINS  = 64    # 元素ペア別ヒストグラムのビン数
ELEM_DIM   = ELEM_BINS * 3  # Si-Si + Si-O + O-O → 192次元

def rdf_descriptor(mol_list, n_bins=RDF_BINS, r_max=RDF_R_MAX):
    """フラグメント → 固定長 RDF ヒストグラム (元素非依存・後方互換)"""
    pts = np.vstack(mol_list)
    dists = [np.linalg.norm(pts[i] - pts[j])
             for i, j in itertools.combinations(range(len(pts)), 2)]
    if not dists:
        return np.zeros(n_bins)
    bins = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(dists, bins=bins, density=False)
    norm = max(hist.sum(), 1)
    return hist.astype(np.float32) / norm

def infer_atom_types_sioh4(mol_list):
    """si_oh4 mol_list: 各 SiO4 ユニット = [Si, O, O, O, O] → index%5==0 が Si"""
    return ['Si' if i % 5 == 0 else 'O' for i in range(len(mol_list))]

def element_aware_descriptor(mol_list, n_bins=ELEM_BINS, r_max=RDF_R_MAX,
                               atom_types=None):
    """
    元素ペア別 RDF 記述子 (Si-Si / Si-O / O-O の 3 チャンネル)
    si_oh4 フォーマット前提: index%5==0 = Si, その他 = O
    出力: (3*n_bins,) = 192次元 固定長
    """
    pts = np.array(mol_list, dtype=np.float32)
    n   = len(pts)
    if atom_types is None:
        atom_types = infer_atom_types_sioh4(mol_list)

    hist_ss = np.zeros(n_bins, dtype=np.float32)
    hist_so = np.zeros(n_bins, dtype=np.float32)
    hist_oo = np.zeros(n_bins, dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(pts[i] - pts[j]))
            if r <= 0 or r > r_max:
                continue
            b  = min(int(r / r_max * n_bins), n_bins - 1)
            ti = atom_types[i]
            tj = atom_types[j]
            if ti == tj == 'Si':
                hist_ss[b] += 1.0
            elif ti == tj == 'O':
                hist_oo[b] += 1.0
            else:
                hist_so[b] += 1.0

    for h in (hist_ss, hist_so, hist_oo):
        s = h.sum()
        if s > 0:
            h /= s

    return np.concatenate([hist_ss, hist_so, hist_oo])

def rdf_batch(mol_list_list):
    """フラグメントリスト → (N, ELEM_DIM) ndarray (element-aware)"""
    return np.vstack([element_aware_descriptor(m) for m in mol_list_list])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §2. FragmentAutoencoder — 教師なし幾何埋め込み
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FragmentAutoencoder(nn.Module):
    def __init__(self, in_dim=ELEM_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),       nn.ReLU(),
            nn.Linear(128, in_dim),   nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(descriptors, epochs=300, lr=3e-3, batch=64, verbose=True):
    """
    教師なし自己符号化器学習
    損失 = 再構成 MSE + triplet 損失 (幾何距離が近いペアは埋め込みも近く)
    """
    X = torch.tensor(descriptors, dtype=torch.float32)
    n = len(X)
    model = FragmentAutoencoder(in_dim=X.shape[1])
    opt   = Adam(model.parameters(), lr=lr)
    ds    = TensorDataset(X)
    loader = DataLoader(ds, batch_size=min(batch, n), shuffle=True)

    # ── triplet マイニング: geom距離ベースでポジティブ/ネガティブを選ぶ ──
    # RDF のペアワイズ距離を事前計算 (小規模なら O(N²) でOK)
    with torch.no_grad():
        Xn = F.normalize(X, dim=1)
        dist_mat = torch.cdist(Xn, Xn)  # (N, N)

    best_loss = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        recon_losses = []
        trip_losses  = []
        for (xb,) in loader:
            # ── 再構成損失 ──
            xr, z = model(xb)
            recon = F.mse_loss(xr, xb)

            # ── triplet 損失: バッチ内でアンカー・ポジ・ネガを選ぶ ──
            # ポジ = RDF距離が最小のペア / ネガ = 最大のペア
            if len(xb) >= 3:
                zn = F.normalize(z, dim=1)
                ed = torch.cdist(zn, zn)  # (B, B) 埋め込み距離
                # バッチ内のRDF距離
                idx = [ds.tensors[0].tolist().index(list(xb[i].tolist()))
                       if xb[i].tolist() in ds.tensors[0].tolist() else 0
                       for i in range(len(xb))]
                # 簡略化: バッチ内ランダムtriplet
                B = len(xb)
                a_idx = torch.arange(B)
                p_idx = (a_idx + 1) % B
                n_idx = (a_idx + B // 2) % B
                trip = F.relu(
                    ed[a_idx, p_idx] - ed[a_idx, n_idx] + 0.3
                ).mean()
            else:
                trip = torch.tensor(0.0)

            loss = recon + 0.1 * trip
            opt.zero_grad(); loss.backward(); opt.step()
            recon_losses.append(recon.item())
            trip_losses.append(trip.item())

        if ep % 50 == 0 and verbose:
            print(f"  ep {ep:3d}  recon={np.mean(recon_losses):.4f}  "
                  f"trip={np.mean(trip_losses):.4f}")

        avg = np.mean(recon_losses)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if verbose:
        print(f"  → best recon loss: {best_loss:.4f}")
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §3. AdaptiveEpsilon — bankデータから最適閾値を学習
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveEpsilon:
    """
    bank内の全ペア (geom_dist, |ΔE|) から
    P(|ΔE| < tol | geom_dist = d) をフィット
    → 指定精度 tol で安全な最大 ε を返す
    """
    def __init__(self, tol_ha=1e-4):
        self.tol_ha  = tol_ha    # 許容エネルギー誤差 (Ha)
        self.iso_reg = None
        self.d_max   = 1.0
        self.fitted  = False

    def fit(self, bank_mols, bank_energies):
        """
        bank_mols:    list of mol_list
        bank_energies: list of float (Ha)
        """
        n = len(bank_mols)
        if n < 4:
            self.fitted = False
            return self

        dvecs = [dist_vec(m) for m in bank_mols]
        pairs_d  = []
        pairs_de = []
        for i in range(n):
            for j in range(i+1, min(i+50, n)):  # 近傍だけ
                d1, d2 = dvecs[i], dvecs[j]
                if len(d1) == len(d2):
                    gd = float(np.sqrt(np.mean((d1 - d2)**2)))  # RMSD
                    de = abs(bank_energies[i] - bank_energies[j])
                    pairs_d.append(gd)
                    pairs_de.append(de)

        if len(pairs_d) < 10:
            self.fitted = False
            return self

        # ラベル: |ΔE| < tol → safe=1, else safe=0
        d_arr   = np.array(pairs_d)
        safe    = (np.array(pairs_de) < self.tol_ha).astype(float)
        # 単調回帰: d が小さいほど safe 確率は高い
        order   = np.argsort(d_arr)
        self.iso_reg = IsotonicRegression(increasing=False, out_of_bounds='clip')
        self.iso_reg.fit(d_arr[order], safe[order])
        self.d_max   = d_arr.max()
        self.fitted  = True
        return self

    def safe_epsilon(self, min_prob=0.95):
        """P(safe | d) >= min_prob を満たす最大 d を返す"""
        if not self.fitted:
            return 0.10   # デフォルト
        d_grid = np.linspace(0, self.d_max, 500)
        prob   = self.iso_reg.predict(d_grid)
        safe_d = d_grid[prob >= min_prob]
        return float(safe_d.max()) if len(safe_d) > 0 else 0.05

    def prob_safe(self, geom_dist):
        if not self.fitted:
            return 1.0 if geom_dist < 0.10 else 0.0
        return float(self.iso_reg.predict([geom_dist])[0])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4. EnsembleEnergyPredictor — 不確実性推定付きエネルギー予測
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EnergyMLP(nn.Module):
    def __init__(self, in_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.SiLU(),
            nn.Linear(64, 32),     nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EnsembleEnergyPredictor:
    """
    5モデルアンサンブル: 埋め込み → (E_pred, σ_pred)
    σ が大きい → QC 計算が必要 (アクティブラーニングトリガー)
    """
    N_MODELS = 5

    def __init__(self, embed_dim=EMBED_DIM):
        self.models  = [EnergyMLP(embed_dim) for _ in range(self.N_MODELS)]
        self.scaler  = StandardScaler()
        self.e_mean  = 0.0
        self.e_std   = 1.0
        self.trained = False

    def fit(self, embeddings, energies, epochs=500, lr=1e-3, verbose=True):
        if len(embeddings) < 4:
            return self
        X = np.array(embeddings, dtype=np.float32)
        y = np.array(energies,   dtype=np.float32)
        # ロバスト正規化: 外れ値に対して中央値±IQRを使う
        p25, p75 = np.percentile(y, [25, 75])
        self.e_mean = float(np.median(y))
        self.e_std  = float(max(p75 - p25, 1e-8))
        y_norm = np.clip((y - self.e_mean) / self.e_std, -5, 5)

        Xt = torch.tensor(X)
        yt = torch.tensor(y_norm)
        ds = TensorDataset(Xt, yt)

        for mi, model in enumerate(self.models):
            opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            # ブートストラップサンプリングでアンサンブルの多様性を確保
            idx = np.random.choice(len(X), len(X), replace=True)
            dl  = DataLoader(TensorDataset(Xt[idx], yt[idx]),
                             batch_size=min(32, len(idx)), shuffle=True)
            best_l, best_s = float('inf'), None
            for ep in range(epochs):
                model.train()
                losses = []
                for xb, yb in dl:
                    pred = model(xb)
                    loss = F.mse_loss(pred, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    losses.append(loss.item())
                avg = np.mean(losses)
                if avg < best_l:
                    best_l = avg
                    best_s = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_s)
            if verbose:
                print(f"  model {mi+1}/{self.N_MODELS}  loss={best_l:.5f}")

        self.trained = True
        return self

    def predict(self, embedding):
        """
        Returns: (e_pred_Ha, sigma_Ha)
        """
        if not self.trained:
            return 0.0, float('inf')
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(x).item())
        preds = np.array(preds)
        e_pred = float(preds.mean()) * self.e_std + self.e_mean
        sigma  = float(preds.std())  * self.e_std
        return e_pred, sigma


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §4b. GPEnergyPredictor — Gaussian Process (小データ特化)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GPEnergyPredictor:
    """
    Gaussian Process エネルギー予測器
    <100 点で MLP Ensemble より高精度 (Perplexity推奨)
    pipeline: 入力記述子 → StandardScaler → PCA → Matérn 5/2 GP
    per-atom 正規化でモノマー/ペア/トリマー混在に対応
    """

    def __init__(self):
        self.gpr      = None
        self.pca      = None
        self.scaler_x = StandardScaler()
        self.e_mean   = 0.0
        self.e_std    = 1.0
        self.trained  = False

    def fit(self, descriptors, energies, n_atoms_list=None, verbose=True):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            Matern, WhiteKernel, ConstantKernel as C,
        )
        from sklearn.decomposition import PCA

        n = len(descriptors)
        if n < 4:
            return self

        X = np.array(descriptors, dtype=np.float64)
        y = np.array(energies,    dtype=np.float64)

        # per-atom 正規化: モノマー(5原子)とトリマー(15原子)のスケール差を吸収
        na = np.array(n_atoms_list, dtype=np.float64) if n_atoms_list else np.ones(n)
        y_pa = y / na
        self.e_mean = float(y_pa.mean())
        self.e_std  = float(max(y_pa.std(), 1e-8))
        y_norm = (y_pa - self.e_mean) / self.e_std

        # PCA で次元削減: curse-of-dimensionality 回避
        X_s = self.scaler_x.fit_transform(X)
        n_comp = min(n - 1, X.shape[1], 20)
        if n_comp < X.shape[1]:
            self.pca = PCA(n_components=n_comp, random_state=42)
            X_red = self.pca.fit_transform(X_s)
        else:
            self.pca = None
            X_red = X_s

        # ARD Matérn 5/2 kernel (各 PCA 軸に独立 length-scale)
        ls0    = np.ones(X_red.shape[1])
        kernel = (
            C(1.0, (1e-6, 1e6)) *
            Matern(length_scale=ls0, length_scale_bounds=(1e-4, 1e6), nu=2.5) +
            WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e4))
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3,
            normalize_y=False, alpha=0.0,
        )
        import warnings as _w
        from sklearn.exceptions import ConvergenceWarning as _CW
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=_CW)
            self.gpr.fit(X_red, y_norm)
        self.trained = True

        if verbose:
            y_hat, _ = self.gpr.predict(X_red, return_std=True)
            mae_pa   = float(np.mean(np.abs(y_hat - y_norm))) * self.e_std
            print(f"  GP train MAE = {mae_pa:.4e} Ha/atom  n={n}")

        return self

    def _transform(self, descriptor):
        X = np.array(descriptor, dtype=np.float64).reshape(1, -1)
        X_s = self.scaler_x.transform(X)
        if self.pca is not None:
            X_s = self.pca.transform(X_s)
        return X_s

    def predict(self, descriptor, n_atoms=None):
        """Returns: (e_pred_Ha, sigma_Ha)"""
        if not self.trained:
            return 0.0, float('inf')
        X_red = self._transform(descriptor)
        y_norm, sig_norm = self.gpr.predict(X_red, return_std=True)
        na    = float(n_atoms) if n_atoms else 1.0
        e_pa  = float(y_norm[0]) * self.e_std + self.e_mean
        sig_pa = float(sig_norm[0]) * self.e_std
        return e_pa * na, sig_pa * na


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5. FragmentCluster — 教師なしクラスタリング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FragmentCluster:
    """
    銀行エントリをクラスタリングして:
    1. 自然なフラグメントファミリーを発見
    2. 各クラスタの代表エネルギーを計算
    3. 異常フラグメント(DBSCAN ノイズ)を検出
    """
    def __init__(self, n_clusters='auto'):
        self.n_clusters = n_clusters
        self.labels_    = None
        self.centers_   = None
        self.cluster_energies_ = None

    def fit(self, embeddings, energies=None, max_k=20):
        X = np.array(embeddings)
        if len(X) < 3:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self

        # ── K選択: エルボー法 (inertia の変化率) ──
        if self.n_clusters == 'auto':
            k_range = range(2, min(max_k, len(X)))
            inertias = []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                km.fit(X)
                inertias.append(km.inertia_)
            if len(inertias) >= 3:
                diffs  = np.diff(inertias)
                diffs2 = np.diff(diffs)
                elbow  = int(np.argmax(diffs2)) + 2  # +2: k_range start + diff offset
                k_best = max(2, min(elbow, max_k))
            else:
                k_best = 2
            self.n_clusters = k_best

        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_  = km.fit_predict(X)
        self.centers_ = km.cluster_centers_

        if energies is not None:
            energies = np.array(energies)
            self.cluster_energies_ = {
                c: float(energies[self.labels_ == c].mean())
                for c in range(self.n_clusters)
                if (self.labels_ == c).any()
            }

        # ── DBSCAN で外れ値検出 ──
        from sklearn.neighbors import NearestNeighbors
        if len(X) >= 5:
            nbrs = NearestNeighbors(n_neighbors=min(5, len(X)-1)).fit(X)
            dists, _ = nbrs.kneighbors(X)
            eps_auto = float(np.percentile(dists[:, -1], 90))
            db = DBSCAN(eps=eps_auto, min_samples=2).fit(X)
            self.outlier_mask_ = db.labels_ == -1
        else:
            self.outlier_mask_ = np.zeros(len(X), dtype=bool)

        return self

    def report(self):
        n_out = self.outlier_mask_.sum() if hasattr(self, 'outlier_mask_') else 0
        print(f"  FragmentCluster: {self.n_clusters} clusters, "
              f"{n_out} outliers (DBSCAN)")
        if self.cluster_energies_:
            for c, e in sorted(self.cluster_energies_.items()):
                cnt = (self.labels_ == c).sum()
                print(f"    cluster {c:2d}: {cnt:4d} fragments, "
                      f"mean E = {e:.4f} Ha")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §5b. Farthest-Point Sampling — core-set 初期シード選択
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def farthest_point_sampling(X, n_select, seed=42):
    """
    貪欲 farthest-point sampling (core-set)
    現在の選択集合から最も遠い点を順次追加する。
    K-means centroid より多様性が高く、能動学習の初期シードに最適。
    O(N * n_select) で効率的。
    """
    n = len(X)
    if n_select >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(0, n))]
    min_dists = np.full(n, np.inf)

    for _ in range(n_select - 1):
        last = selected[-1]
        dists = np.linalg.norm(X - X[last], axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -np.inf  # 選択済みを除外
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return selected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §6. MLBank — MotifBank + ML を統合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MLBank(MotifBank):
    """
    MotifBankをML機能で拡張:
    1. 正確マッチ (geom_key) → 従来通り
    2. 適応的 soft match (adaptive ε) → より賢い再利用
    3. ML予測 (不確実性 < threshold) → QCコール不要
    4. QCコール (不確実性 > threshold) → 計算してbank更新
    """
    def __init__(self, path=None, sigma_threshold=1e-3):
        super().__init__(path)
        self.autoencoder  = None
        self.predictor    = GPEnergyPredictor()       # GP (primary)
        self.ensemble     = EnsembleEnergyPredictor() # MLP ensemble (fallback/comparison)
        self.adaptive_eps = AdaptiveEpsilon(tol_ha=1e-4)
        self.cluster      = FragmentCluster()
        self.sigma_threshold = sigma_threshold   # Ha
        self._mol_cache   = []
        self.stats = {'exact': 0, 'soft': 0, 'ml_pred': 0, 'qc': 0}

    def _desc(self, mol_list):
        """フラグメント → 元素別 RDF 記述子 (192次元)"""
        return element_aware_descriptor(mol_list)

    def _embed(self, mol_list):
        """フラグメント → 32次元埋め込み (autoencoder 経由)"""
        desc = element_aware_descriptor(mol_list)
        xt   = torch.tensor(desc, dtype=torch.float32).unsqueeze(0)
        if self.autoencoder is not None:
            self.autoencoder.eval()
            with torch.no_grad():
                return self.autoencoder.encode(xt).squeeze(0).numpy()
        return desc[:EMBED_DIM] if len(desc) >= EMBED_DIM else \
               np.pad(desc, (0, EMBED_DIM - len(desc)))

    def train_ml(self, verbose=True):
        """bank データから ML モデルを学習"""
        if len(self.data) < 8:
            print("  [ML] bank entries too few (need ≥8). skipping.")
            return self

        mols_list = [v['mol'] for v in self.data.values() if 'mol' in v]
        energies  = [v['energy_Ha'] for v in self.data.values() if 'mol' in v]

        if len(mols_list) < 8:
            print("  [ML] not enough mol data in bank. skipping.")
            return self

        print(f"\n[ML] Training on {len(mols_list)} bank fragments...")

        # (1) Element-aware descriptor (Si-Si/Si-O/O-O 3チャンネル)
        descs = np.array([element_aware_descriptor(m) for m in mols_list], dtype=np.float32)

        # (2) Autoencoder (教師なし, 可視化・soft-match 用)
        print("  (1/4) Autoencoder (unsupervised, 192→32)...")
        self.autoencoder = train_autoencoder(descs, epochs=200, verbose=verbose)

        # (3) 埋め込み (クラスタリング用)
        self.autoencoder.eval()
        with torch.no_grad():
            Xt     = torch.tensor(descs)
            embeds = self.autoencoder.encode(Xt).numpy()

        # (4) クラスタリング (教師なし)
        print("  (2/4) Clustering (unsupervised)...")
        self.cluster.fit(embeds, energies)
        self.cluster.report()

        # (5) Adaptive epsilon
        print("  (3/4) Adaptive epsilon...")
        self.adaptive_eps.fit(mols_list, energies)
        eps_95 = self.adaptive_eps.safe_epsilon(min_prob=0.95)
        eps_99 = self.adaptive_eps.safe_epsilon(min_prob=0.99)
        print(f"    ε(95% safe) = {eps_95:.4f} Å  ε(99% safe) = {eps_99:.4f} Å  "
              f"(current fixed = 0.10 Å)")

        # (6) GP predictor (element-aware desc + PCA → GP)
        print("  (4/4) GP energy predictor (Matern 5/2, per-atom normalization)...")
        n_atoms_list = [len(m) for m in mols_list]
        self.predictor = GPEnergyPredictor()
        self.predictor.fit(list(descs), energies,
                           n_atoms_list=n_atoms_list, verbose=verbose)

        # (6b) Ensemble as fallback/comparison
        self.ensemble.fit(list(embeds), energies, epochs=200, verbose=False)

        print("[ML] Training complete.\n")
        return self

    def query_ml(self, mol_list, qc_func=None):
        """
        拡張クエリ: exact → soft(adaptive ε) → ML予測 → QC
        Returns: (energy_Ha, source, sigma)
        """
        k   = geom_key(mol_list)
        dv  = dist_vec(mol_list)

        # 1. Exact match
        cached = self.query_exact(k)
        if cached is not None:
            self.stats['exact'] += 1
            return cached, 'exact', 0.0

        # 2. Adaptive soft match
        eps = self.adaptive_eps.safe_epsilon(min_prob=0.95) \
              if self.adaptive_eps.fitted else 0.10
        soft = self.query_soft(mol_list, eps=eps)
        if soft is not None:
            self.stats['soft'] += 1
            return soft, f'soft(ε={eps:.3f})', 0.0

        # 3. ML prediction (GP on element-aware descriptor)
        desc    = self._desc(mol_list)
        n_atoms = len(mol_list)
        e_pred, sigma = self.predictor.predict(desc, n_atoms=n_atoms)
        if self.predictor.trained and sigma < self.sigma_threshold:
            self.stats['ml_pred'] += 1
            return e_pred, f'ml_pred(σ={sigma:.2e})', sigma

        # 4. QC call (必要なときだけ)
        if qc_func is not None:
            e_qc = qc_func(mol_list)
            self.store(k, mol_list, e_qc)
            self.stats['qc'] += 1
            return e_qc, 'qc', 0.0

        return e_pred, f'ml_uncertain(σ={sigma:.2e})', sigma

    def print_stats(self):
        total = sum(self.stats.values())
        if total == 0:
            return
        print("\n[MLBank query stats]")
        for src, cnt in self.stats.items():
            print(f"  {src:12s}: {cnt:5d} ({cnt/total*100:.1f}%)")
        saved = (self.stats['exact'] + self.stats['soft'] +
                 self.stats['ml_pred'])
        print(f"  → QC calls saved: {saved}/{total} ({saved/total*100:.1f}%)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §7. UMAP 可視化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def visualize_fragment_space(bank, title="Fragment Embedding Space", save="frag_space.png"):
    """bank の埋め込み空間を UMAP + エネルギー彩色で可視化"""
    try:
        import umap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("  [vis] umap/matplotlib not found, skipping.")
        return

    mols_list = [v['mol'] for v in bank.data.values() if 'mol' in v]
    energies  = [v['energy_Ha'] for v in bank.data.values() if 'mol' in v]
    if len(mols_list) < 5:
        print("  [vis] not enough data for UMAP.")
        return

    descs = np.array([rdf_descriptor(m) for m in mols_list], dtype=np.float32)

    # UMAP 次元削減
    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(descs)-1),
                        min_dist=0.1, random_state=42)
    xy = reducer.fit_transform(descs)

    # エネルギー彩色
    E = np.array(energies)
    norm = Normalize(vmin=E.min(), vmax=E.max())
    sm   = ScalarMappable(cmap='viridis', norm=norm)
    colors = sm.to_rgba(E)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: エネルギー彩色
    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=E, cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label='Energy (Ha)')
    axes[0].set_title(f'{title}\n(colored by energy)', fontsize=10)
    axes[0].set_xlabel('UMAP-1'); axes[0].set_ylabel('UMAP-2')

    # 右: クラスタ彩色
    if bank.cluster.labels_ is not None and len(bank.cluster.labels_) == len(descs):
        labels = bank.cluster.labels_
        n_cl   = len(set(labels))
        cmap   = plt.cm.get_cmap('tab20', n_cl)
        axes[1].scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, s=30, alpha=0.7)
        # 外れ値を × でマーク
        if hasattr(bank.cluster, 'outlier_mask_'):
            out = bank.cluster.outlier_mask_
            axes[1].scatter(xy[out, 0], xy[out, 1], marker='x', c='red',
                            s=80, linewidths=2, label='outlier', zorder=5)
        axes[1].set_title(f'{title}\n(colored by cluster, k={n_cl})', fontsize=10)
        axes[1].set_xlabel('UMAP-1'); axes[1].set_ylabel('UMAP-2')
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches='tight')
    print(f"  [vis] saved: {save}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §8. Active Learner — 不確実性ドリブンQCサンプリング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ActiveLearner:
    """
    QC 計算予算 budget 内で bank を最効率に構築する。
    戦略: 不確実性が最大のフラグメントを優先的に QC 計算する。

    アルゴリズム:
    1. 全フラグメントを埋め込みで表現
    2. 初期 seed を等間隔サンプリング (多様性確保)
    3. 残りを ML予測し、σ 降順に QC を振り分ける
    4. QC 結果で ML モデルを再訓練 (アクティブ更新)
    """

    def __init__(self, budget=50, retrain_every=10):
        self.budget        = budget
        self.retrain_every = retrain_every

    def run(self, mols, qc_func, bank: MLBank, verbose=True):
        n = len(mols)
        print(f"\n[ActiveLearner] N={n} fragments, budget={self.budget} QC calls")

        # ── Phase 1: Farthest-point sampling (core-set) で初期 bank 構築 ──
        seed_n = min(max(self.budget // 5, 3), n)
        # 元素別記述子空間で最も多様な点を選ぶ (K-means centroid より確実)
        descs  = rdf_batch(mols)   # element_aware_descriptor を使用
        scl    = StandardScaler()
        X_scl  = scl.fit_transform(descs)
        seed_idx = farthest_point_sampling(X_scl, seed_n, seed=42)
        if not seed_idx:  # fallback
            seed_idx = list(np.random.choice(n, min(3, n), replace=False))

        if verbose:
            print(f"  Seeding with {len(seed_idx)} diverse fragments...")
        for idx in seed_idx:
            e = qc_func(mols[idx])
            bank.store(geom_key(mols[idx]), mols[idx], e)
        calls_used = len(seed_idx)

        # ── Phase 2: ML訓練 + 不確実性サンプリング ──
        if verbose:
            print(f"  Initial ML training...")
        bank.train_ml(verbose=False)

        remaining = list(set(range(n)) - set(seed_idx))
        history   = []

        while calls_used < self.budget and remaining:
            # 全残存フラグメントの不確実性を評価 (GP用 _desc を使用)
            uncertainties = []
            for idx in remaining:
                desc = bank._desc(mols[idx])
                _, sigma = bank.predictor.predict(desc, n_atoms=len(mols[idx]))
                uncertainties.append((sigma, idx))

            # 不確実性が最大のフラグメントを QC へ
            uncertainties.sort(reverse=True)
            n_batch = min(self.retrain_every, self.budget - calls_used,
                          len(remaining))
            batch_idx = [idx for _, idx in uncertainties[:n_batch]]

            for idx in batch_idx:
                e = qc_func(mols[idx])
                bank.store(geom_key(mols[idx]), mols[idx], e)
                remaining.remove(idx)
                calls_used += 1

            history.append({'calls': calls_used, 'bank_size': len(bank.data)})
            if verbose:
                max_sigma = uncertainties[0][0] if uncertainties else 0
                print(f"  [{calls_used:3d}/{self.budget} QC]  "
                      f"bank={len(bank.data)}  σ_max={max_sigma:.3e}")

            # 定期再訓練
            bank.train_ml(verbose=False)

        # 残り全フラグメントは ML 予測で処理 (GP用 _desc を使用)
        n_ml_pred = 0
        for idx in remaining:
            desc = bank._desc(mols[idx])
            e_pred, sigma = bank.predictor.predict(desc, n_atoms=len(mols[idx]))
            n_ml_pred += 1

        print(f"\n[ActiveLearner] Done.")
        print(f"  QC calls used : {calls_used} / {n}")
        print(f"  ML predictions: {n_ml_pred} / {n}")
        print(f"  QC savings    : {(n-calls_used)/n*100:.1f}%")
        return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# §9. デモ・CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _store_with_mol(bank, key, mol, energy):
    """mol を保存する拡張 store"""
    bank.data[key] = {
        'energy_Ha': energy,
        'dist_vec':  list(dist_vec(mol)),
        'hits':      0,
        'source':    'mock',
        'mol':       mol,
    }

def _build_mbe_bank(cif, supercell, mol_type, r_cut=R_CUT_DEF):
    """
    MBE (monomer+pair+trimer) でバンクを構築して mol も保存する。
    IZA理想構造はモノマーが全て同一なので、ペア・トリマーを含めて多様性を確保。
    """
    from motifbank_cli import cutoff_trimers
    mols, atypes, _ = from_cif(cif, supercell=supercell,
                                mol_type=mol_type, verbose=False)
    bank = MLBank(sigma_threshold=5e-3)
    qc   = qc_compute_mock

    # モノマー
    for mol in mols:
        k = geom_key(mol)
        if k not in bank.data:
            _store_with_mol(bank, k, mol, qc(mol))

    # ペア (R_cut 以内)
    coms = np.array([np.mean(np.vstack(m), axis=0) for m in mols])
    for i in range(len(mols)):
        for j in range(i+1, len(mols)):
            if np.linalg.norm(coms[i] - coms[j]) < r_cut:
                pair = mols[i] + mols[j]
                k = geom_key(pair)
                if k not in bank.data:
                    _store_with_mol(bank, k, pair, qc(pair))

    # トリマー (R_cut 以内、最大300件)
    trimer_idx = cutoff_trimers(mols, r_cut=r_cut, max_t=300)
    for i, j, k_idx in trimer_idx:
        tri = mols[i] + mols[j] + mols[k_idx]
        k   = geom_key(tri)
        if k not in bank.data:
            _store_with_mol(bank, k, tri, qc(tri))

    return bank, mols


def learning_curve_benchmark(bank_mols, bank_energies, n_train_range=None,
                              n_test=8, n_reps=5, verbose=True):
    """
    学習曲線ベンチマーク: GP (element-aware) vs MLP Ensemble (RDF)
    MAE (Ha) vs QCコール数
    Returns: (gp_curve, ens_curve) それぞれ [(n_train, mae), ...]
    """
    n_total = len(bank_mols)
    if n_total < 8:
        print("[LearningCurve] Not enough bank entries (need ≥8).")
        return [], []

    n_test = min(n_test, n_total - 4)
    max_tr = n_total - n_test
    if n_train_range is None:
        n_train_range = [n for n in [4, 6, 8, 10, 12, 15, 18, 22, max_tr]
                         if 4 <= n <= max_tr]

    descs_ea  = np.array([element_aware_descriptor(m) for m in bank_mols])
    descs_rdf = np.array([rdf_descriptor(m) for m in bank_mols])
    energies  = np.array(bank_energies)
    n_atoms   = np.array([len(m) for m in bank_mols], dtype=int)

    gp_curve, ens_curve = [], []

    if verbose:
        print("\n[Learning Curve]  GP(element-aware+PCA) vs Ensemble(RDF)")
        print(f"  {'n_train':>8}  {'GP MAE (Ha)':>14}  {'Ens MAE (Ha)':>14}")
        print("  " + "-" * 42)

    for n_tr in n_train_range:
        gp_reps, ens_reps = [], []
        for rep in range(n_reps):
            rng = np.random.RandomState(rep * 137)
            perm = rng.permutation(n_total)
            tr, te = perm[:n_tr], perm[n_tr:n_tr + n_test]

            # GP
            gp = GPEnergyPredictor()
            gp.fit(descs_ea[tr], energies[tr],
                   n_atoms_list=n_atoms[tr].tolist(), verbose=False)
            preds_gp = np.array([
                gp.predict(descs_ea[i], int(n_atoms[i]))[0] for i in te
            ])
            gp_reps.append(float(np.mean(np.abs(preds_gp - energies[te]))))

            # MLP Ensemble on RDF
            ens = EnsembleEnergyPredictor(embed_dim=RDF_BINS)
            ens.fit(descs_rdf[tr].tolist(), energies[tr].tolist(),
                    epochs=150, verbose=False)
            preds_ens = np.array([
                ens.predict(descs_rdf[i])[0] for i in te
            ])
            ens_reps.append(float(np.mean(np.abs(preds_ens - energies[te]))))

        gp_mae  = float(np.mean(gp_reps))
        ens_mae = float(np.mean(ens_reps))
        gp_curve.append((n_tr, gp_mae))
        ens_curve.append((n_tr, ens_mae))
        if verbose:
            print(f"  {n_tr:>8}  {gp_mae:>14.4f}  {ens_mae:>14.4f}")

    # 最良の改善率を表示
    if gp_curve and ens_curve:
        best_n = gp_curve[-1][0]
        best_gp  = gp_curve[-1][1]
        best_ens = ens_curve[-1][1]
        ratio = best_ens / max(best_gp, 1e-12)
        print(f"\n  [LC] @ n_train={best_n}: GP={best_gp:.4f} Ha, "
              f"Ens={best_ens:.4f} Ha → GP is {ratio:.1f}× better")

    return gp_curve, ens_curve


def demo_ml(cif='examples/MFI_iza.cif', supercell=(1,1,1), mol_type='si_oh4'):
    print("=" * 60)
    print("MotifBank ML Demo")
    print("=" * 60)

    # ── 1. MBE フラグメント全体でバンク構築 ──
    print(f"\n[1] Loading {cif} {supercell} (monomer+pair+trimer)...")
    bank, mols = _build_mbe_bank(cif, supercell, mol_type)
    print(f"    N_mol = {len(mols)},  bank size = {len(bank.data)} unique fragments")

    # ── 3. ML 学習 ──
    print(f"\n[2→3] Training ML models (bank={len(bank.data)} entries)...")
    bank.train_ml(verbose=True)

    # ── 4. Adaptive epsilon の効果 ──
    eps_95 = bank.adaptive_eps.safe_epsilon(0.95)
    eps_99 = bank.adaptive_eps.safe_epsilon(0.99)
    print(f"\n[4] Adaptive ε:")
    print(f"    95% safe: ε = {eps_95:.4f} Å  (fixed was 0.1000 Å)")
    print(f"    99% safe: ε = {eps_99:.4f} Å")

    # ── 5. ML クエリ精度検証 ──
    print("\n[5] ML query accuracy on new fragments...")
    _qc = qc_compute_mock
    test_mols, _, _ = from_cif(cif, supercell=(2,1,1),
                                mol_type=mol_type, verbose=False)
    n_test = min(100, len(test_mols))
    errors_pred, sources = [], []
    for mol in test_mols[:n_test]:
        e_true = _qc(mol)
        e_pred, src, sigma = bank.query_ml(mol, qc_func=None)
        if src != 'qc':
            errors_pred.append(abs(e_true - e_pred))
            sources.append(src.split('(')[0])

    if errors_pred:
        mae = np.mean(errors_pred)
        print(f"    MAE (non-QC queries): {mae:.6f} Ha  ({len(errors_pred)} samples)")
        from collections import Counter
        for s, cnt in Counter(sources).most_common():
            print(f"    {s:15s}: {cnt:4d} queries")

    # ── 6. 学習曲線ベンチマーク (GP vs Ensemble) ──
    print("\n[6] Learning curve benchmark: GP vs MLP Ensemble...")
    bank_mols  = [v['mol']       for v in bank.data.values() if 'mol' in v]
    bank_enrgs = [v['energy_Ha'] for v in bank.data.values() if 'mol' in v]
    gp_curve, ens_curve = learning_curve_benchmark(
        bank_mols, bank_enrgs, n_reps=3, verbose=True
    )

    # ── 7. UMAP 可視化 ──
    print("\n[7] UMAP visualization...")
    visualize_fragment_space(bank, title=f"MotifBank: {cif}", save="frag_space.png")

    # ── 8. Active learning デモ (farthest-point seeding) ──
    print("\n[8] Active learning (FPS seed) on unique MBE fragments (budget=15)...")
    unique_frags = [v['mol'] for v in bank.data.values() if 'mol' in v]
    print(f"    Pool size: {len(unique_frags)} unique fragments")
    al_bank3 = MLBank(sigma_threshold=5e-3)
    def store_with_mol3(k, mol, e):
        _store_with_mol(al_bank3, k, mol, e)
    al_bank3.store = store_with_mol3
    budget = min(15, len(unique_frags) - 2)
    al = ActiveLearner(budget=budget, retrain_every=5)
    history = al.run(unique_frags, qc_compute_mock, al_bank3, verbose=True)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    bank.print_stats()
    print(f"\nfrag_space.png: UMAP visualization saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo',   action='store_true')
    ap.add_argument('--train',  metavar='BANK_JSON')
    ap.add_argument('--active', metavar='CIF')
    ap.add_argument('--budget', type=int, default=50)
    ap.add_argument('--cif',    default='examples/LTA_iza.cif')
    ap.add_argument('--sc',     default='1,1,1')
    args = ap.parse_args()

    sc = tuple(int(x) for x in args.sc.split(','))

    if args.demo or (not args.train and not args.active):
        demo_ml(cif=args.cif, supercell=sc if args.sc != '1,1,1' else (2,2,1))

    elif args.train:
        bank = MLBank(path=args.train, sigma_threshold=1e-3)
        bank.train_ml(verbose=True)
        visualize_fragment_space(bank, save="frag_space.png")
        torch.save({
            'autoencoder': bank.autoencoder.state_dict(),
        }, args.train.replace('.json', '_ml.pt'))
        print(f"Saved: {args.train.replace('.json', '_ml.pt')}")

    elif args.active:
        mols, _, _ = from_cif(args.active, supercell=sc,
                               mol_type='si_oh4', verbose=False)
        bank = MLBank(sigma_threshold=1e-3)
        def store_with_mol(k, mol, e):
            _store_with_mol(bank, k, mol, e)
        bank.store = store_with_mol
        al = ActiveLearner(budget=args.budget)
        al.run(mols, qc_compute_mock, bank, verbose=True)
        bank.print_stats()


if __name__ == '__main__':
    main()
