# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotifBank は MBE (Many-Body Expansion) を用いた量子化学計算の高速化ライブラリ。
結晶・ゼオライト等の周期構造を分子フラグメントに分解し、同一幾何構造（motif）を持つフラグメントのQCエネルギーを再利用することで大幅な高速化を実現する。

## Commands

### テスト実行

```bash
# 全テスト (必ず OMP_NUM_THREADS=1 を付ける)
OMP_NUM_THREADS=1 python3 test_motifbank.py       # 23テスト (コア機能)
OMP_NUM_THREADS=1 python3 test_mfi_accuracy.py    # 7テスト (MFI精度)
OMP_NUM_THREADS=1 python3 test_sioh4.py           # 7テスト (Si(OH)4フラグメント)
OMP_NUM_THREADS=1 python3 test_api.py             # 10テスト (API end-to-end)

# 単一テスト関数を実行したい場合はファイル内の test() 呼び出しを直接コメントアウト
```

### CLI 操作

```bash
OMP_NUM_THREADS=1 python3 motifbank_cli.py demo
OMP_NUM_THREADS=1 python3 motifbank_cli.py classify  INPUT.json
OMP_NUM_THREADS=1 python3 motifbank_cli.py build     INPUT.json
OMP_NUM_THREADS=1 python3 motifbank_cli.py mbe       INPUT.json
OMP_NUM_THREADS=1 python3 motifbank_cli.py benchmark INPUT.cif
OMP_NUM_THREADS=1 python3 motifbank_cli.py status    BANK.json
```

### API サーバー起動

```bash
OMP_NUM_THREADS=1 python3 api_server.py --bank motifbank_api.json --port 8000
# ドキュメント: http://localhost:8000/docs
```

### 依存パッケージインストール

```bash
pip install numpy ase "pyscf>=2.0" "fastapi>=0.100" "uvicorn>=0.23" "pydantic>=2.0"
# または: pip install -e ".[full]"
```

## Architecture

### コアモジュール構成

```
motifbank_cli.py     ← メインライブラリ (CLI + 全コア関数)
api_server.py        ← FastAPI REST サーバー (motifbank_cli をインポート)
motifbank_client.py  ← API クライアント
motifbank/           ← パッケージ版 (開発中)
  core/motif_db.py
  core/motif_analysis.py
  core/fractal_analyzer.py
```

**`motifbank_cli.py` が実質的な単一ソース**。api_server.py はここから全関数をインポートする。

### 主要関数・クラス (motifbank_cli.py)

| セクション | 内容 |
|-----------|------|
| `§1` `geom_key()` | 分子リスト → 距離タプル (元素非依存の幾何ハッシュキー) |
| `§1` `MotifBank` | バンク本体: `store()` / `query_exact()` / `query_soft()` / `save()` / `load()` |
| `§2` `cutoff_trimers()` | R_cut 以内のトリマー列挙 (近傍グリッド法、最大500k件) |
| `§3` `classify()` | Phase 0-3 分類 + γ (バンク成長率) 計算 |
| `§4` `run_mbe()` | MBE 計算本体 (1体+2体+3体、バンク使用) |
| `§5` `from_cif()` | CIF → フラグメント分解 (MIC周期境界補正済み) |
| `§6` `qc_compute_pyscf()` | PySCF QC計算バックエンド |

### Phase 分類

- `γ = d(log N_bank)/d(log N)` がバンク成長率
- Phase 0: γ ≈ 0 (完全飽和、結晶)
- Phase 1: 0 < γ < 0.48 (sub-linear、準周期)
- Phase 2/3: γ ≥ 0.48 (linear以上、非晶質) → MBE非推奨

### soft matching

`query_soft()` は `dist_vec` の RMSD が `eps=0.10Å` 以内ならキャッシュヒットとみなす。
exact match より先に soft match が呼ばれることはない (exact match は別メソッド)。

### API サーバー (api_server.py)

- `QCWorker`: バックグラウンドスレッドで PySCF を非同期実行
- `KeyStore`: APIキー管理 (JSON ファイル `api_keys.json`)
- 料金体系: bank hit=10円、mono=10円、pair=30円、trimer=100円、heavy=200円
- レート制限: free=60 req/min、paid=600 req/min

## Critical Constraints

**`OMP_NUM_THREADS=1` は必須**。これを外すと浮動小数点演算が非決定的になり (σ_software > 0)、同じフラグメントが異なるgeom_keyに分類されバンクが壊れる。すべてのスクリプト・テストで設定すること。

**SCF 収束閾値**: `conv_tol=1e-9` を標準とする (デフォルトの 1e-9 から変えない)。

**MIC (Minimum Image Convention)**: `from_cif()` 内の Si-O 距離計算には必ず周期境界補正を適用すること。未適用だと N_bank が過大になる (MFI: 644→282 の修正済み事例あり)。

## Key Numbers (確定値)

| 材料 | N_bank_sat | S_local | speedup (N=768) |
|------|-----------|---------|-----------------|
| ice Ih | 16 | 2.77 nats | - |
| α-cristobalite | 18 | 2.89 nats | - |
| LTA zeolite | 66 | 4.19 nats | - |
| MFI silicalite-1 | 282 | 5.64 nats | 52x |

PBE/def2-SVP: E(Si(OH)4) = -592.129475 Ha, T_QC ≈ 12.5s/call

## Input JSON フォーマット

```json
// 内蔵ビルダー
{"system": "ice2d", "nx": 6, "ny": 6}

// CIF ファイル
{"cif": "path/to/structure.cif", "supercell": [2,2,1], "mode": "si_oh4", "r_cut": 5.5}

// 座標直接指定
{"molecules": [[[x,y,z],...], ...], "atom_types": [["O","H","H"],...]}
```
