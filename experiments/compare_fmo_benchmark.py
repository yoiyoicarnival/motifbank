#!/usr/bin/env python3
"""
compare_fmo_benchmark.py
FMO / 周期 DFT ベンチマークデータと MotifBank MBE の比較

使い方:
  python3 compare_fmo_benchmark.py --ref ref_energies.csv --system MFI
  python3 compare_fmo_benchmark.py --ref ref_energies.json --system MFI

入力フォーマット (CSV):
  fragment_id, E_ref_Ha
  0, -578.4853...
  1, -578.4912...
  ...

入力フォーマット (JSON):
  {"fragment_energies": [E0, E1, ...], "pair_energies": [[i, j, Eij], ...], "total": E_total}

出力:
  - 断片ごとの ΔE (MotifBank - Reference)
  - MAE / RMSE / MaxAE (kcal/mol)
  - speedup 比較
"""
import os, sys, json, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motifbank_cli import from_cif, MotifBank, run_mbe, make_qc_func

HA2KCAL = 627.509  # Ha → kcal/mol
CIF = os.path.join(os.path.dirname(__file__), "examples", "MFI_iza.cif")


def load_ref_csv(path):
    """CSV: fragment_id, E_ref_Ha"""
    energies = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["fragment_id"])
            e   = float(row["E_ref_Ha"])
            energies[fid] = e
    return energies


def load_ref_json(path):
    """
    JSON: {"fragment_energies": [...], "pair_energies": [[i,j,E],...], "total": E}
    または FMO 出力の標準形式に合わせて拡張可能
    """
    with open(path) as f:
        data = json.load(f)
    return data


def compare_monomer_energies(ref_dict, mols, atypes, bank, qc_func, r_cut=5.5):
    """
    各断片の E_ref vs E_motifbank を比較
    ref_dict: {fragment_id: E_Ha}
    """
    from motifbank_cli import geom_key as gk_fn

    results = []
    for fid, e_ref in sorted(ref_dict.items()):
        if fid >= len(mols):
            continue
        mol = mols[fid]
        at  = atypes[fid]
        gk  = gk_fn([mol])

        # bank から取得 (なければ QC 計算)
        rec = bank.data.get(gk)
        if rec is None:
            e_mb = qc_func([mol], [at])
            bank.put(gk, e_mb, [mol])
        else:
            e_mb = rec["energy_Ha"]

        delta = (e_mb - e_ref) * HA2KCAL
        results.append({
            "fid": fid,
            "E_ref":   e_ref,
            "E_bank":  e_mb,
            "dE_kcal": delta,
        })

    return results


def print_report(results, label="monomer"):
    dEs = [r["dE_kcal"] for r in results]
    mae  = np.mean(np.abs(dEs))
    rmse = np.sqrt(np.mean(np.array(dEs)**2))
    maxae = np.max(np.abs(dEs))

    print(f"\n  {label} 比較 (N={len(results)}フラグメント):")
    print(f"    MAE  = {mae:.4f} kcal/mol")
    print(f"    RMSE = {rmse:.4f} kcal/mol")
    print(f"    MaxAE= {maxae:.4f} kcal/mol")
    target = 1.0  # kcal/mol / フラグメント
    ok = mae < target
    status = "PASS" if ok else "FAIL"
    print(f"    目標 < {target} kcal/mol:  {status}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",    required=True, help="参照エネルギーファイル (.csv/.json)")
    ap.add_argument("--system", default="MFI", help="系の名前 (ログ用)")
    ap.add_argument("--sc",     default="1,1,1", help="supercell (例: 1,1,1)")
    ap.add_argument("--mol-type", default="si_oh4")
    ap.add_argument("--basis",  default="def2-svp")
    ap.add_argument("--method", default="pbe")
    ap.add_argument("--bank",   default="motifbank_compare.json")
    args = ap.parse_args()

    sc = tuple(int(x) for x in args.sc.split(","))

    print(f"\nMotifBank vs {args.system} ベンチマーク比較")
    print("=" * 60)
    print(f"  参照: {args.ref}")
    print(f"  系:   {args.system}  supercell={sc}  mol_type={args.mol_type}")
    print(f"  QC:   {args.method}/{args.basis}")

    mols, atypes, _ = from_cif(CIF, supercell=sc,
                                mol_type=args.mol_type, verbose=False)
    bank = MotifBank(args.bank)
    qc_func = make_qc_func('pyscf', basis=args.basis, method=args.method,
                            charge=0, spin=0)

    ext = os.path.splitext(args.ref)[1].lower()
    if ext == ".csv":
        ref = load_ref_csv(args.ref)
        results = compare_monomer_energies(ref, mols, atypes, bank, qc_func)
        print_report(results, label="モノマー")

    elif ext == ".json":
        data = load_ref_json(args.ref)
        if "fragment_energies" in data:
            ref = {i: e for i, e in enumerate(data["fragment_energies"])}
            results = compare_monomer_energies(ref, mols, atypes, bank, qc_func)
            print_report(results, label="モノマー")
        if "total" in data:
            # 全エネルギー比較 (MBE vs 参照)
            E_ref_total = data["total"]
            E_mb_total  = run_mbe(mols, bank, r_cut=5.5,
                                  qc_func=qc_func, atom_types_list=atypes,
                                  charge_per_mol=0, verbose=False)
            dE_total = (E_mb_total - E_ref_total) * HA2KCAL
            n = len(mols)
            print(f"\n  全エネルギー比較:")
            print(f"    E_ref  = {E_ref_total:.6f} Ha")
            print(f"    E_bank = {E_mb_total:.6f} Ha")
            print(f"    ΔE     = {dE_total:.4f} kcal/mol  ({dE_total/n:.4f} kcal/mol/SiO4)")
            print(f"    目標 < 1 kcal/mol/SiO4:  {'PASS' if abs(dE_total/n) < 1 else 'FAIL'}")
    else:
        print(f"  未対応フォーマット: {ext}  (.csv または .json を指定)")
        sys.exit(1)

    bank.save()
    print(f"\n  bank 保存: {args.bank}  ({len(bank.data)} エントリ)")
    print("=" * 60)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
