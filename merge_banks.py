#!/usr/bin/env python3
"""
merge_banks.py -- motif_bank.json のマージ

使い方:
  python3 merge_banks.py base.json new.json [--out merged.json] [--dry-run]

マージ優先順位:
  |ESA| > 1e-5  → genuine QC 計算値
  |ESA| <= 1e-5 → ATM/degenerate 近似値

  1. new にしか存在しない → 追加
  2. base=ATM, new=QC     → QCで上書き (upgrade)
  3. 両方 QC              → 絶対値の大きい方を採用 (OMP=1ならほぼ同値)
  4. new=ATM, base=QC     → base を維持
"""

import sys, json, argparse, math
from pathlib import Path

QC_THRESH = 1e-5


def load_bank(path: Path) -> dict:
    d = json.load(open(path))
    for k in ('mono', 'pair', 'trim'):
        d.setdefault(k, {})
    return d


def esa_val(entry) -> float:
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        v = entry.get('ESA_mean') or entry.get('ESA')
        return float(v) if v is not None else 0.0
    return 0.0


def merge(base: dict, new: dict) -> tuple[dict, dict]:
    out = {
        'schema_version': max(base.get('schema_version', 3),
                               new.get('schema_version', 3))
    }
    stats = {'added': 0, 'upgraded': 0, 'conflict_qc': 0, 'kept': 0, 'total': 0}

    for kind in ('mono', 'pair', 'trim'):
        merged = dict(base.get(kind, {}))
        for key, new_entry in new.get(kind, {}).items():
            stats['total'] += 1
            if key not in merged:
                merged[key] = new_entry
                stats['added'] += 1
                continue

            base_abs = abs(esa_val(merged[key]))
            new_abs  = abs(esa_val(new_entry))
            base_qc  = base_abs > QC_THRESH
            new_qc   = new_abs  > QC_THRESH

            if new_qc and not base_qc:
                merged[key] = new_entry
                stats['upgraded'] += 1
            elif new_qc and base_qc:
                if new_abs > base_abs:
                    merged[key] = new_entry
                stats['conflict_qc'] += 1
            else:
                stats['kept'] += 1

        out[kind] = merged

    return out, stats


def main():
    ap = argparse.ArgumentParser(description='motif_bank.json マージツール')
    ap.add_argument('base',       help='ベースbankファイル')
    ap.add_argument('new',        help='マージするbankファイル')
    ap.add_argument('--out',      default=None, help='出力先 (省略時: base を上書き)')
    ap.add_argument('--dry-run',  action='store_true')
    args = ap.parse_args()

    base_path = Path(args.base)
    new_path  = Path(args.new)
    out_path  = Path(args.out) if args.out else base_path

    print(f"Base : {base_path}  ({base_path.stat().st_size // 1024:,} KB)")
    print(f"New  : {new_path}   ({new_path.stat().st_size  // 1024:,} KB)")

    base = load_bank(base_path)
    new  = load_bank(new_path)

    def _counts(d):
        return {k: len(v) for k, v in d.items() if isinstance(v, dict)}

    print(f"\nBase entries : {_counts(base)}")
    print(f"New  entries : {_counts(new)}")

    merged, stats = merge(base, new)

    print(f"\nMerge 結果:")
    print(f"  追加       : {stats['added']:>6}  (new にのみ存在)")
    print(f"  QC昇格     : {stats['upgraded']:>6}  (ATM/degen → QCに置換)")
    print(f"  QC競合     : {stats['conflict_qc']:>6}  (両方QC、大きい方を採用)")
    print(f"  維持       : {stats['kept']:>6}  (既存を保持)")
    print(f"\nMerged entries: {_counts(merged)}")

    if args.dry_run:
        print("\n[DRY RUN] 書き込みをスキップ")
        return 0

    tmp = out_path.with_suffix('.merge_tmp')
    json.dump(merged, open(tmp, 'w'))
    tmp.rename(out_path)
    print(f"\n書き込み完了: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
