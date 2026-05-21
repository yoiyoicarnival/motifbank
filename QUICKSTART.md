# MotifBank クイックスタート

## 1. インストール (30秒)

```bash
cd /home/yoiyoi
pip install numpy pyscf ase fastapi uvicorn pydantic
```

## 2. 動作確認 (60秒)

```bash
# Phase 分類デモ (QC なし、即時)
OMP_NUM_THREADS=1 python3 motifbank_cli.py demo
```

期待出力:
```
  材料                 Phase   gamma   reuse  N_bank  ROI1st  戦略
  2D ice (4x4->6x6)      1  0.0833    87%      14   79.0%  DEPLOY
  Carpet Gen1->Gen2      0  0.0000   100%       8  100.0%  DEPLOY
  MOF 4->9 pores         1  0.1250    91%      16   81.0%  DEPLOY
```

## 3. テストスイート実行 (2〜5分)

```bash
OMP_NUM_THREADS=1 python3 test_motifbank.py
```

PySCF がインストール済みなら T4〜T6 で実 QC を確認できる。

## 4. CIF ファイルを使ったベンチマーク

```bash
# テスト用 CIF を生成
OMP_NUM_THREADS=1 python3 generate_cif.py

# mock QC でベンチマーク (数秒)
OMP_NUM_THREADS=1 python3 motifbank_cli.py benchmark examples/ice_Ih_3x3.cif

# 実 QC (HF/STO-3G, 1〜10分)
OMP_NUM_THREADS=1 python3 motifbank_cli.py benchmark examples/ice_Ih_3x3.cif --qc pyscf
```

## 5. 実在 CIF で試す

```bash
# 付属 CIF (ice_test.cif: 12 H2O)
OMP_NUM_THREADS=1 python3 motifbank_cli.py benchmark examples/ice_test.cif --qc pyscf

# 環境に /tmp/LTA_zeolite.cif がある場合 (Phase 分類のみ mock で確認)
OMP_NUM_THREADS=1 python3 motifbank_cli.py classify examples/zeolite_lta.json

# クリストバライト (SiO2, 2x2x2 supercell)
OMP_NUM_THREADS=1 python3 motifbank_cli.py classify examples/cristobalite.json

# 自前 CIF:
OMP_NUM_THREADS=1 python3 motifbank_cli.py benchmark your_material.cif --qc pyscf -o result.json
```

## 6. API サーバー起動

```bash
# 起動 (admin キーがコンソールに表示される)
OMP_NUM_THREADS=1 python3 api_server.py

# 別ターミナルでテスト
curl http://localhost:8000/v1/health
```

## 7. Python ライブラリとして使う

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'

from motifbank_cli import classify, run_mbe, MotifBank, from_cif, make_qc_func

# CIF 読み込み
mols, atypes, label = from_cif("examples/ice_Ih_3x3.cif")

# Phase 分類
r = classify(mols)
print(f"Phase {r['phase']}, ROI {r['roi_pct']:.0f}% → {r['strategy']}")

# MBE (PySCF HF/STO-3G)
bank = MotifBank("my_bank.json")
qc   = make_qc_func('pyscf', basis='sto-3g')
res  = run_mbe(mols, bank, qc_func=qc, atom_types_list=atypes)
print(f"E_total = {res['E_total_Ha']:.6f} Ha,  ROI = {res['roi_actual']*100:.0f}%")
bank.save("my_bank.json")
```

## 参考: 2週間スプリント進捗

| Day  | 内容                                    | 状態 |
|------|-----------------------------------------|------|
| 1-2  | PySCF バックエンド実装                  | ✅   |
| 3-4  | CIF 読み込み・API 更新                  | ✅   |
| 5    | test_motifbank.py + ドキュメント整備     | ✅   |
| 6-7  | 実 QC ベンチマーク数値確認・文献比較    | → ユーザーが `python3 test_motifbank.py` 実行待ち |
| 8-9  | ゼオライト/クリストバライト Phase 確認  | 予定 |
| 10-11| 大系メモリ最適化 (N>1000 pair_de2)     | 予定 |
| 12-13| API クライアント end-to-end テスト      | 予定 |
| 14   | 最終テスト・パッケージリリース準備      | 予定 |
