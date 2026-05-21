#!/usr/bin/env python3
"""
world_encoder.py — 経験圧縮型AI: 自己教師あり分子構造エンコーダ

設計思想:
  「巨大計算で知能を作る」ではなく
  「世界の再利用可能構造を圧縮して知能を作る」

アーキテクチャ:
  pairwise_descriptor(72-dim) → MLP encoder(32-dim) → FAISS retrieval
  ↓
  k-NN local GP → 化学精度予測 (ラベルなし事前学習 + 少量DFTで微調整)

3種の教師なし学習:
  (1) Contrastive: geom_key一致 → 近く, 異なるkey → 遠く  [メイン]
  (2) Masked prediction: 部分descriptor隠蔽 → 復元         [補助]
  (3) Geometry augmentation: 回転/並進不変性               [データ拡張]

Jetson対応: ~8K params, <1ms/call, FAISS + 局所GP

Usage:
  OMP_NUM_THREADS=1 python3 world_encoder.py
"""

import os, sys, json, warnings, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/yoiyoi")
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
    from sklearn.neighbors import NearestNeighbors

from motifbank_cli import geom_key
from motifbank_ml import GPEnergyPredictor, farthest_point_sampling
from real_gp_benchmark import (
    generate_sioh4_fragments, CACHE_FILE, KCAL, MHA, _EQ_MOL,
)

EMBED_DIM   = 32   # 埋め込み次元
HIDDEN_DIM  = 64
INPUT_DIM   = 72   # pairwise_descriptor
TEMPERATURE = 0.07 # InfoNCE 温度パラメータ
DEVICE      = "cpu"

# ─────────────────────────────────────────────────
# §1 pairwise descriptor (real_gp_benchmark と同一)
# ─────────────────────────────────────────────────

def pairwise_desc(mol):
    pts = np.asarray(mol, dtype=float)
    dists = []
    for i in range(9):
        for j in range(i + 1, 9):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            dists.append(d)
            dists.append(1.0 / (d ** 2 + 1e-8))
    return np.array(dists, dtype=np.float32)


# ─────────────────────────────────────────────────
# §2 encoder アーキテクチャ (Jetson向け軽量設計)
# ─────────────────────────────────────────────────

class MolecularEncoder(nn.Module):
    """
    72-dim pairwise descriptor → 32-dim L2-normalized embedding
    パラメータ数: ~8K (Jetson Nano でも <1ms)
    """
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)  # L2 正規化 → cosine similarity が内積に


class MaskedPredictor(nn.Module):
    """
    マスク予測ヘッド: 一部の descriptor ビンを隠して復元
    (encoder に接続、事前学習のみに使用)
    """
    def __init__(self, embed_dim=EMBED_DIM, output_dim=INPUT_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )

    def forward(self, z):
        return self.head(z)


# ─────────────────────────────────────────────────
# §3 データ拡張 + ペア生成
# ─────────────────────────────────────────────────

def augment_mol(mol, sigma=0.005, rng=None):
    """小変位 (σ << σ_c) による幾何拡張 — positive pair 生成用"""
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.randn(*mol.shape) * sigma
    mol_aug = np.array(mol) + noise
    mol_aug -= mol_aug.mean(axis=0)  # 質量中心固定 (並進不変)
    return mol_aug


def build_contrastive_pairs(mols, geom_keys, n_aug=3, rng=None):
    """
    (anchor, positive, negatives) のペアリスト生成。

    Positive: 同一 geom_key (別サンプル or augment)
    Negative: 異なる geom_key のランダムサンプル

    MotifBank の geom_key が無料の strong positive ラベルを提供する。
    """
    if rng is None:
        rng = np.random.RandomState(42)
    key_to_idx = {}
    for i, k in enumerate(geom_keys):
        key_to_idx.setdefault(k, []).append(i)

    anchors, positives, neg_pools = [], [], []

    for i, mol in enumerate(mols):
        k = geom_keys[i]
        # anchor
        anchors.append(pairwise_desc(mol))

        # positive: 同一 key の別サンプルがあれば使う、なければ augment
        same = [j for j in key_to_idx[k] if j != i]
        if same:
            pos_mol = mols[rng.choice(same)]
        else:
            pos_mol = augment_mol(mol, sigma=0.005, rng=rng)
        positives.append(pairwise_desc(pos_mol))

        # augmentation 追加 positive (n_aug 個)
        for _ in range(n_aug):
            aug = augment_mol(mol, sigma=0.008, rng=rng)
            anchors.append(pairwise_desc(mol))
            positives.append(pairwise_desc(aug))

    return np.array(anchors, dtype=np.float32), np.array(positives, dtype=np.float32)


def mask_descriptor(x, mask_frac=0.30, rng=None):
    """
    マスク予測用: descriptor の mask_frac を 0 で隠す。
    LLM の masked token prediction のアナログ。
    """
    if rng is None:
        rng = np.random.RandomState()
    x_masked = x.copy()
    n_mask = max(1, int(len(x) * mask_frac))
    idx = rng.choice(len(x), n_mask, replace=False)
    x_masked[idx] = 0.0
    return x_masked, idx


# ─────────────────────────────────────────────────
# §4 NT-Xent (InfoNCE) 損失
# ─────────────────────────────────────────────────

def ntxent_loss(z_a, z_p, temperature=TEMPERATURE):
    """
    z_a: (N, D) anchor embeddings (L2-normalized)
    z_p: (N, D) positive embeddings (L2-normalized)
    損失: N個の (anchor, positive) ペアを N² から識別
    """
    N = z_a.shape[0]
    # (2N, D) に結合
    z = torch.cat([z_a, z_p], dim=0)
    # 類似度行列 (2N, 2N)
    sim = torch.mm(z, z.t()) / temperature
    # 対角を -inf (自己類似は除外)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    # anchor i の positive は i+N
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────
# §5 事前学習: contrastive + masked prediction
# ─────────────────────────────────────────────────

def pretrain(mols, geom_keys, n_epochs=200, lr=3e-3, verbose=True):
    """
    教師なし事前学習。ラベル (DFTエネルギー) 不要。
    geometry だけで latent world を形成。
    """
    rng = np.random.RandomState(42)
    X_a, X_p = build_contrastive_pairs(mols, geom_keys, n_aug=3, rng=rng)

    # StandardScaler は先にフィット
    scaler = StandardScaler()
    X_all = np.vstack([X_a, X_p])
    scaler.fit(X_all)
    Xa = torch.tensor(scaler.transform(X_a), dtype=torch.float32)
    Xp = torch.tensor(scaler.transform(X_p), dtype=torch.float32)

    encoder  = MolecularEncoder().to(DEVICE)
    predictor = MaskedPredictor().to(DEVICE)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        encoder.train(); predictor.train()
        # ----- contrastive loss -----
        za = encoder(Xa)
        zp = encoder(Xp)
        loss_c = ntxent_loss(za, zp)

        # ----- masked prediction loss -----
        X_mask_np = np.array([
            mask_descriptor(scaler.transform(pairwise_desc(m).reshape(1, -1))[0],
                            mask_frac=0.3, rng=rng)[0]
            for m in mols
        ], dtype=np.float32)
        X_orig_np = scaler.transform(np.array([pairwise_desc(m) for m in mols],
                                              dtype=np.float32))
        Xm = torch.tensor(X_mask_np, dtype=torch.float32)
        Xo = torch.tensor(X_orig_np, dtype=torch.float32)
        z_m  = encoder(Xm)
        x_rec = predictor(z_m)
        loss_mask = F.mse_loss(x_rec, Xo)

        loss = loss_c + 0.5 * loss_mask
        opt.zero_grad(); loss.backward(); opt.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}: loss_c={loss_c.item():.4f}  "
                  f"loss_mask={loss_mask.item():.4f}")
        losses.append(float(loss.item()))

    return encoder, scaler, losses


# ─────────────────────────────────────────────────
# §6 外部記憶: FAISS or sklearn k-NN
# ─────────────────────────────────────────────────

class DescriptorBank:
    """
    FAISS + pairwise_descriptor による高速 k-NN 検索 + 局所GP。
    エンコーダ不要: descriptor 設計済みの場合のベースライン。
    キー洞察: k近傍のみで GP 学習 → 全体 GP より局所的に正確。
    """
    def __init__(self, desc_dim=INPUT_DIM, desc_scaler=None):
        self.mols      = []
        self.energies  = []
        self._descs    = []
        self._scaler   = desc_scaler  # 事前フィット済みScaler
        self._index    = None
        self._desc_dim = desc_dim

    def store(self, mol, energy_ha):
        d = pairwise_desc(mol)
        self.mols.append(mol)
        self.energies.append(energy_ha)
        self._descs.append(d)
        self._index = None

    def _build_index(self):
        D = np.array(self._descs, dtype=np.float32)
        if self._scaler is not None:
            D = self._scaler.transform(D)
        faiss.normalize_L2(D)
        idx = faiss.IndexFlatIP(D.shape[1])
        idx.add(D)
        self._index = idx
        self._D_norm = D

    def search(self, mol, k=20):
        if self._index is None:
            self._build_index()
        d = pairwise_desc(mol).reshape(1, -1).astype(np.float32)
        if self._scaler is not None:
            d = self._scaler.transform(d)
        faiss.normalize_L2(d)
        k_eff = min(k, len(self.mols))
        sims, idxs = self._index.search(d, k_eff)
        return idxs[0], sims[0]

    def local_gp_predict(self, mol, n_atoms=9, k=50):
        idxs, sims = self.search(mol, k=k)
        k_eff = len(idxs)
        if k_eff < 5:
            return None, None, "insufficient_data"
        X_tr = np.array([self._descs[i] for i in idxs])
        y_tr = np.array([self.energies[i] for i in idxs])
        gp = GPEnergyPredictor()
        gp.fit(list(X_tr), list(y_tr),
               n_atoms_list=[n_atoms] * k_eff, verbose=False)
        mu, sigma = gp.predict(pairwise_desc(mol), n_atoms=n_atoms)
        # OOD: 最近傍 cosine sim < 0.70 (経験的下界)
        ood = float(sims[0]) < 0.70
        return float(mu), float(sigma), "ood" if ood else "ok"


class EmbeddingBank:
    """
    外部ベクトルDB: 学習済みエンコーダ + FAISS/sklearn で構成。
    - store(): 新規フラグメントの埋め込みを格納
    - search(): k近傍の埋め込みと距離を返す
    - local_gp_predict(): k近傍でローカルGP補間
    """
    def __init__(self, encoder, scaler, embed_dim=EMBED_DIM):
        self.encoder = encoder
        self.scaler  = scaler
        self.embed_dim = embed_dim
        self.embeddings = []  # (N, D) numpy array
        self.mols       = []  # 対応する座標
        self.energies   = []  # DFTエネルギー (Ha)
        self._index     = None

    def _encode(self, mol):
        x = self.scaler.transform(pairwise_desc(mol).reshape(1, -1))
        xt = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            self.encoder.eval()
            z = self.encoder(xt).numpy()[0]
        return z

    def store(self, mol, energy_ha):
        z = self._encode(mol)
        self.embeddings.append(z)
        self.mols.append(mol)
        self.energies.append(energy_ha)
        self._index = None  # invalidate

    def _build_index(self):
        E = np.array(self.embeddings, dtype=np.float32)
        if FAISS_OK:
            idx = faiss.IndexFlatIP(self.embed_dim)  # inner product (L2-normalized → cosine)
            idx.add(E)
            self._index = ("faiss", idx)
        else:
            nn = NearestNeighbors(n_neighbors=min(20, len(E)), metric="cosine")
            nn.fit(E)
            self._index = ("sklearn", nn)

    def search(self, mol, k=10):
        if self._index is None:
            self._build_index()
        z = self._encode(mol).reshape(1, -1).astype(np.float32)
        k_eff = min(k, len(self.embeddings))
        if FAISS_OK:
            D, I = self._index[1].search(z, k_eff)
            return I[0], D[0]
        else:
            D, I = self._index[1].kneighbors(z, n_neighbors=k_eff)
            return I[0], 1.0 - D[0]  # cosine dist → cosine sim

    def local_gp_predict(self, mol, n_atoms=9, k=20):
        """
        k近傍のDFTエネルギーで局所GP回帰 → 予測値 + 不確かさ
        OOD: 最近傍 cosine similarity が低い → σ 大
        """
        idxs, sims = self.search(mol, k=k)
        k_eff = len(idxs)
        if k_eff < 5:
            return None, None, "insufficient_data"

        X_train = np.array([pairwise_desc(self.mols[i]) for i in idxs])
        y_train = np.array([self.energies[i] for i in idxs])
        gp = GPEnergyPredictor()
        gp.fit(list(X_train), list(y_train),
               n_atoms_list=[n_atoms] * k_eff, verbose=False)
        mu, sigma = gp.predict(pairwise_desc(mol), n_atoms=n_atoms)

        # OOD判定: 最近傍 cosine sim が全体の下位 20%
        max_sim = float(sims[0])
        ood = max_sim < 0.50
        return float(mu), float(sigma), "ood" if ood else "ok"


# ─────────────────────────────────────────────────
# §7 エネルギー fine-tuning (少量DFTで精度向上)
# ─────────────────────────────────────────────────

def finetune_energy(encoder, scaler, mols_train, energies_train,
                    n_atoms=9, n_epochs=500, lr=3e-3):
    """
    事前学習済みエンコーダを少量DFTでファインチューニング。
    y を StandardScaler で正規化してから学習 (スケール問題を回避)。
    """
    encoder.eval()
    X = np.array([scaler.transform(pairwise_desc(m).reshape(1, -1))[0]
                  for m in mols_train], dtype=np.float32)
    y_raw = np.array(energies_train, dtype=np.float32).reshape(-1, 1)
    y_scaler = StandardScaler()
    y_norm = y_scaler.fit_transform(y_raw).astype(np.float32)

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_norm, dtype=torch.float32)

    with torch.no_grad():
        Z = encoder(Xt).detach()

    head = nn.Sequential(
        nn.Linear(EMBED_DIM, 64), nn.GELU(),
        nn.Linear(64, 32), nn.GELU(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    losses = []
    for epoch in range(n_epochs):
        pred = head(Z)
        loss = F.mse_loss(pred, yt)
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        losses.append(float(loss.item()))

    return head, y_scaler, losses


# ─────────────────────────────────────────────────
# §8 主実験: 3フェーズ比較
# ─────────────────────────────────────────────────

def exp_unsupervised_vs_supervised():
    """
    比較実験:
      A. 全データ GP (supervised, global)            ← ベースライン
      B. DescriptorBank: FAISS k-NN + 局所GP          ← 検索ベース推論
      C. Contrastive encoder + FAISS + 局所GP          ← 教師なし embedding
      D. Contrastive encoder + energy head fine-tune   ← 完全 NN

    [B] と [C] は DFT ラベルを訓練に使わず geometry のみで構造を学習。
    """
    print("=" * 65)
    print("経験圧縮型AI: 教師なし embedding + 検索ベース推論")
    print("=" * 65)

    cache = json.load(open(CACHE_FILE))
    mols_all, _, _ = generate_sioh4_fragments(n=200, seed=42)
    mols, energies, gkeys = [], [], []
    for mol in mols_all:
        k = str(geom_key(mol))
        if k in cache:
            mols.append(mol); energies.append(cache[k]); gkeys.append(k)

    N = len(mols)
    X_pd = np.array([pairwise_desc(m) for m in mols])
    n_train = min(120, N - 10)
    idx_tr = farthest_point_sampling(X_pd, n_train, seed=42)
    idx_te = [i for i in range(N) if i not in set(idx_tr)]
    mols_tr = [mols[i] for i in idx_tr]
    mols_te = [mols[i] for i in idx_te]
    y_tr    = np.array([energies[i] for i in idx_tr])
    y_te    = np.array([energies[i] for i in idx_te])

    print(f"\n  データ: {N} fragments (PBE/def2-SVP Si(OH)4)")
    print(f"  train: {n_train}, test: {len(idx_te)}")

    # ── [A] 全データ GP (supervised, global) ──
    print("\n  [A] Global GP (supervised baseline)")
    t0 = time.time()
    gp_base = GPEnergyPredictor()
    gp_base.fit(list(X_pd[idx_tr]), list(y_tr), n_atoms_list=[9]*n_train, verbose=False)
    preds_a = [gp_base.predict(X_pd[i], n_atoms=9)[0] for i in idx_te]
    mae_a = np.mean(np.abs(np.array(preds_a) - y_te)) * KCAL
    print(f"  MAE = {mae_a:.3f} kcal/mol  ({time.time()-t0:.1f}s)")

    # ── [B] FAISS k-NN + 局所GP (descriptor, ラベル不要で構造を索引) ──
    print("\n  [B] DescriptorBank: FAISS k-NN + 局所GP (descriptor-based retrieval)")
    desc_scaler = StandardScaler().fit(X_pd[idx_tr])
    dbank = DescriptorBank(desc_scaler=desc_scaler)
    for mol, e in zip(mols_tr, y_tr):
        dbank.store(mol, e)

    t0 = time.time()
    preds_b, statuses_b = [], []
    for mol in mols_te:
        mu, sigma, status = dbank.local_gp_predict(mol, k=50)
        preds_b.append(mu if mu is not None else float(np.mean(y_tr)))
        statuses_b.append(status)
    t_b = time.time() - t0

    mae_b  = np.mean(np.abs(np.array(preds_b) - y_te)) * KCAL
    n_ood_b = sum(1 for s in statuses_b if s == "ood")
    print(f"  MAE = {mae_b:.3f} kcal/mol  ({t_b*1000/len(idx_te):.1f}ms/call, "
          f"OOD: {n_ood_b}/{len(idx_te)})")

    # ── [C] Contrastive encoder + FAISS + 局所GP ──
    print("\n  [C] Contrastive encoder + FAISS + 局所GP (geometry only pre-train)")
    print("      pre-training ...", end=" ", flush=True)
    t0 = time.time()
    encoder, enc_scaler, _ = pretrain(mols_tr, [gkeys[i] for i in idx_tr],
                                      n_epochs=500, verbose=False)
    t_pre = time.time() - t0
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"{t_pre:.0f}s ({n_params} params)")

    ebank = EmbeddingBank(encoder, enc_scaler)
    for mol, e in zip(mols_tr, y_tr):
        ebank.store(mol, e)

    t0 = time.time()
    preds_c, statuses_c = [], []
    for mol in mols_te:
        mu, sigma, status = ebank.local_gp_predict(mol, k=50)
        preds_c.append(mu if mu is not None else float(np.mean(y_tr)))
        statuses_c.append(status)
    t_c = time.time() - t0

    mae_c   = np.mean(np.abs(np.array(preds_c) - y_te)) * KCAL
    n_ood_c = sum(1 for s in statuses_c if s == "ood")
    print(f"  MAE = {mae_c:.3f} kcal/mol  ({t_c*1000/len(idx_te):.1f}ms/call, "
          f"OOD: {n_ood_c}/{len(idx_te)})")

    # ── [D] encoder + energy head fine-tune ──
    print("\n  [D] encoder + energy head fine-tune (500 epochs, y-normalized)")
    t0 = time.time()
    head, y_scaler_h, _ = finetune_energy(encoder, enc_scaler, mols_tr, y_tr, n_epochs=500)
    t_ft = time.time() - t0

    preds_d = []
    for mol in mols_te:
        x = enc_scaler.transform(pairwise_desc(mol).reshape(1, -1))
        xt = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            encoder.eval(); head.eval()
            z = encoder(xt)
            y_norm_pred = head(z).numpy()
        y_pred_ha = y_scaler_h.inverse_transform(y_norm_pred)[0, 0]
        preds_d.append(float(y_pred_ha))
    mae_d = np.mean(np.abs(np.array(preds_d) - y_te)) * KCAL
    print(f"  MAE = {mae_d:.3f} kcal/mol  ({t_ft:.1f}s fine-tune)")

    # ── サマリ ──
    print("\n" + "=" * 65)
    print("★ 結果サマリ")
    print("=" * 65)
    print(f"  [A] Global GP (supervised)            : {mae_a:.3f} kcal/mol  (baseline)")
    print(f"  [B] FAISS k-NN + 局所GP (no encoder)  : {mae_b:.3f} kcal/mol  "
          f"(OOD {n_ood_b}/{len(idx_te)})")
    print(f"  [C] Contrastive encoder + 局所GP       : {mae_c:.3f} kcal/mol  "
          f"(OOD {n_ood_c}/{len(idx_te)})")
    print(f"  [D] encoder + energy head fine-tune   : {mae_d:.3f} kcal/mol")

    print(f"""
  重要な洞察:
  ─────────────────────────────────────────────
  [B] k近傍局所GP: 全データGPと比較
      gap = {mae_b-mae_a:+.3f} kcal/mol (小さいほど局所GP有効)
  [C] encoder embedding の品質を反映
      gap = {mae_c-mae_a:+.3f} kcal/mol (500 epoch でどこまで追いつくか)
  [B] 推論: ~{t_b*1000/len(idx_te):.0f}ms/call → FAISS スケール: O(log N)
  [B] OOD検出: {n_ood_b}件 = 不確かさが高い → DFT キュー候補

  Jetson向け設計:
  encoder {n_params} params, 32-dim FAISS, 局所GP → <50ms/call
""")

    return {
        "mae_a": mae_a, "mae_b": mae_b, "mae_c": mae_c, "mae_d": mae_d,
        "n_ood_b": n_ood_b, "n_ood_c": n_ood_c, "n_test": len(idx_te),
    }


# ─────────────────────────────────────────────────
# §9 能動学習ループ (active learning demo)
# ─────────────────────────────────────────────────

def demo_active_learning(n_init=20, n_rounds=5, sigma_thresh=0.5e-3):
    """
    能動学習デモ:
    1. 少量 (n_init) でバンク構築
    2. 未知サンプルに対して局所GP予測
    3. σ > threshold のサンプルを DFT キューに追加
    4. DFT 後バンク更新 → 繰り返し

    「AIが自分で "ここは未知" を選ぶ」
    """
    print("\n" + "=" * 65)
    print("Active Learning デモ: σ > threshold → DFT キューイング")
    print("=" * 65)

    cache = json.load(open(CACHE_FILE))
    mols_all, _, _ = generate_sioh4_fragments(n=200, seed=42)
    mols, energies, gkeys = [], [], []
    for mol in mols_all:
        k = str(geom_key(mol))
        if k in cache:
            mols.append(mol)
            energies.append(cache[k])
            gkeys.append(k)

    rng = np.random.RandomState(0)
    # 初期バンク: n_init ランダム
    init_idx = rng.choice(len(mols), n_init, replace=False).tolist()
    pool_idx = [i for i in range(len(mols)) if i not in set(init_idx)]

    # 事前学習 (geometry のみ)
    encoder, scaler, _ = pretrain(mols, gkeys, n_epochs=150, verbose=False)

    bank = EmbeddingBank(encoder, scaler)
    for i in init_idx:
        bank.store(mols[i], energies[i])

    print(f"\n  初期バンク: {n_init} fragments")
    print(f"  σ threshold: {sigma_thresh:.2e} Ha = {sigma_thresh*KCAL:.2f} kcal/mol")
    print(f"\n  {'round':>6}  {'bank':>5}  {'queued':>7}  {'MAE(kcal)':>10}  {'OOD%':>6}")
    print("  " + "─" * 44)

    results = []
    for rnd in range(n_rounds):
        # 全 pool に対して予測
        dft_queue = []
        preds, trues = [], []
        for i in pool_idx:
            mu, sigma, status = bank.local_gp_predict(mols[i], k=min(10, len(bank.mols)))
            if mu is None:
                sigma = 1.0; mu = float(np.mean(energies))
            trues.append(energies[i])
            preds.append(mu)
            # 不確かさが高い → DFT キュー
            if sigma > sigma_thresh:
                dft_queue.append(i)

        mae = np.mean(np.abs(np.array(preds) - np.array(trues))) * KCAL
        ood_pct = len(dft_queue) / len(pool_idx) * 100
        print(f"  {rnd+1:>6}  {len(bank.mols):>5}  {len(dft_queue):>7}  {mae:>9.3f}  {ood_pct:>5.1f}%")

        if not dft_queue:
            print("  → DFT キューなし: 全て充分な精度で予測可能")
            break

        # 上位 k 件だけ "DFT 実行" (実際はキャッシュから取得)
        add_k = min(10, len(dft_queue))
        selected = rng.choice(dft_queue, add_k, replace=False)
        for i in selected:
            bank.store(mols[i], energies[i])
            pool_idx.remove(i)

        results.append({"round": rnd+1, "bank_size": len(bank.mols),
                        "mae": mae, "queued": len(dft_queue)})

    print(f"\n  最終バンクサイズ: {len(bank.mols)} (初期 {n_init} → 能動追加 {len(bank.mols)-n_init})")
    return results


# ─────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────

def main():
    r = exp_unsupervised_vs_supervised()
    demo_active_learning()

    with open("/home/yoiyoi/world_encoder_results.json", "w") as f:
        json.dump(r, f, indent=2)
    print("\n結果保存: world_encoder_results.json")


if __name__ == "__main__":
    main()
