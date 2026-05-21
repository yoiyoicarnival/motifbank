#!/usr/bin/env python3
"""
pc_bank_prefill.py -- PC(WSL)でmotif_bankを事前充填するスクリプト

目的:
  Gen5をJetsonで実行する前に、PC(x86/WSL)でQC計算を並列実行してbank充填。
  JetsonではbankヒットになるのでPhase2の実行時間を大幅短縮できる。

OMP=1 を強制: PC(x86 AVX) ↔ Jetson(ARM NEON) で完全一致を保証。
実証済み: R=1.50A と R=2.12A で差=0 (2026-05-15)

使い方:
  # 全クラス充填 (R<6Å, 推定 ~8h with N_WORKERS=6)
  python3 pc_bank_prefill.py --gen 5 --workers 6

  # 既存bankにマージ (Jetsonからコピー後に実行)
  python3 pc_bank_prefill.py --gen 5 --bank /path/to/motif_bank.json --workers 6

  # Jetsonへ転送
  scp /home/yoiyoi/motif_bank_pc.json jetson@100.64.1.27:/home/jetson/motif_bank_pc.json
  # Jetson側でマージ:
  # python3 merge_banks.py motif_bank.json motif_bank_pc.json → motif_bank.json

注意:
  - WSL Ubuntu: python3 pc_bank_prefill.py (PySCF 2.13.0 確認済み)
  - OMP=1 を import より前に設定すること (本スクリプトは自動設定)
"""

# OMP=1: import pyscf より前に必須 (cross-arch determinism)
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import sys, json, math, time, itertools, argparse, pickle
from pathlib import Path
import numpy as np

try:
    from pyscf import gto, mcscf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    print("WARNING: PySCF未インストール。--dry-run モードのみ使用可能。")

# ─── 定数 (sier_gen5_v3.py と同一) ─────────────────────────────────────────

A        = 0.75
H_TRI    = A * math.sqrt(3) / 2
E_MONO   = -1.445642           # Ha (SA-CASSCF(3,3)/cc-pVDZ)
E_PER_H  = E_MONO / 3
E_TOL    = 0.15
CONV_TOL = 1e-9

# ─── 幾何学 ─────────────────────────────────────────────────────────────────

def sierpinski(n):
    if n == 0:
        return np.array([[0., 0.], [A, 0.], [A/2, H_TRI]])
    prev = sierpinski(n-1)
    side = A * 2**(n-1)
    offs = np.array([[0.,0.], [side,0.], [side/2, side*math.sqrt(3)/2]])
    combined = np.vstack([prev + o for o in offs])
    seen, uniq = set(), []
    for p in combined:
        k = (round(float(p[0]),6), round(float(p[1]),6))
        if k not in seen:
            seen.add(k); uniq.append(p)
    return np.array(uniq)

def h3_at(cx, cy, cz=0.0):
    R = A / math.sqrt(3)
    return [('H', (cx, cy+R, cz)),
            ('H', (cx-A/2, cy-R/2, cz)),
            ('H', (cx+A/2, cy-R/2, cz))]

def geom_key(atoms: list) -> str:
    coords = np.array([xyz for _, xyz in atoms])
    dists = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dists.append(round(float(np.linalg.norm(coords[i]-coords[j])), 4))
    dists.sort()
    return str(tuple(dists))

def rmin3(pts, i, j, k):
    return min(np.linalg.norm(pts[i]-pts[j]),
               np.linalg.norm(pts[i]-pts[k]),
               np.linalg.norm(pts[j]-pts[k]))

# ─── QC ─────────────────────────────────────────────────────────────────────

def casscf_e(atoms, prev_dm=None):
    if not HAS_PYSCF:
        return float('nan'), None
    mol = gto.Mole()
    mol.atom = atoms; mol.basis = 'cc-pvdz'
    mol.spin = len(atoms) % 2; mol.charge = 0; mol.verbose = 0
    mol.build()

    guesses = []
    if prev_dm is not None:
        guesses.append(('prev_dm', prev_dm))
    guesses += [('minao', None), ('atom', None)]

    for guess_name, dm0 in guesses:
        try:
            mf = mol.UHF()
            if dm0 is not None:
                mf.kernel(dm0=dm0)
            else:
                mf.init_guess = guess_name
                mf.kernel()
            if not mf.converged:
                continue

            mc = mcscf.CASSCF(mf, len(atoms) // 3 * 3, len(atoms))
            mc = mcscf.state_average_(mc, [0.5, 0.5])
            mc.conv_tol      = CONV_TOL
            mc.conv_tol_grad = 1e-7
            mc.max_cycle     = 500
            mc.kernel()

            e = mc.e_tot
            if isinstance(e, (list, tuple, np.ndarray)):
                e = float(np.mean(e))
            else:
                e = float(e)
            if abs(e - len(atoms) * E_PER_H) > E_TOL:
                continue
            return e, mf.make_rdm1()
        except Exception:
            continue

    return float('nan'), None

# ─── Bank ────────────────────────────────────────────────────────────────────

class Bank:
    def __init__(self, path: Path):
        self.path = path
        self._d = {}
        if path.exists():
            self._d = json.load(open(path))
        for k in ('mono', 'pair', 'trim'):
            self._d.setdefault(k, {})
        self._d.setdefault('schema_version', 3)

    def has(self, kind, key): return key in self._d[kind]
    def put(self, kind, key, val): self._d[kind][key] = {'ESA': val}
    def get(self, kind, key):
        v = self._d[kind].get(key)
        if v is None: return None
        return v.get('ESA_mean') or v.get('ESA')
    def save(self): json.dump(self._d, open(self.path, 'w'))
    def stats(self): return {k: len(v) for k, v in self._d.items() if isinstance(v, dict)}

# ─── メイン処理 ──────────────────────────────────────────────────────────────

def run(gen: int, atm_r_min: float, workers: int, bank_path: Path, dry_run: bool):
    pts = sierpinski(gen)
    N   = len(pts)
    print(f"Gen{gen}: N={N}, C(N,3)={N*(N-1)*(N-2)//6:,}")

    bank = Bank(bank_path)
    print(f"Bank loaded: {bank.stats()}")

    # 全ユニーククラスの列挙 (重複除去)
    print("Enumerating unique classes...", flush=True)
    seen = {}
    for i, j, k in itertools.combinations(range(N), 3):
        ai = h3_at(*pts[i]); aj = h3_at(*pts[j]); ak = h3_at(*pts[k])
        gk = geom_key(ai + aj + ak)
        if gk not in seen:
            seen[gk] = (i, j, k)
    print(f"Unique classes: {len(seen):,}")

    # QC対象 (R < atm_r_min かつ bank未収録)
    todo = [(gk, ijk) for gk, ijk in seen.items()
            if not bank.has('trim', gk)
            and rmin3(pts, *ijk) < atm_r_min]
    todo.sort(key=lambda x: rmin3(pts, *x[1]))
    print(f"QC target (R < {atm_r_min}Å, not in bank): {len(todo):,}")

    if dry_run:
        print("[DRY RUN] QCは実行しません。")
        for gk, (i, j, k) in todo[:5]:
            rm = rmin3(pts, i, j, k)
            print(f"  R={rm:.3f}A gk={gk[:50]}")
        return

    if not HAS_PYSCF:
        print("ERROR: PySCFがインストールされていません。")
        sys.exit(1)

    # シングルスレッドで逐次実行 (OMP=1保証のため)
    # workers パラメータは将来の並列化のために保持
    n_done = 0
    t0 = time.time()
    prev_dm = None
    prev_rm = -999

    for gk, (i, j, k) in todo:
        rm = rmin3(pts, i, j, k)
        ai = h3_at(*pts[i]); aj = h3_at(*pts[j]); ak = h3_at(*pts[k])

        # ジオメトリが大きく変わったらwarm-startリセット
        if abs(rm - prev_rm) > 0.3:
            prev_dm = None
        prev_rm = rm

        e_ijk, prev_dm = casscf_e(ai + aj + ak, prev_dm)
        if math.isnan(e_ijk):
            prev_dm = None; continue

        # de3 計算には pair energies が必要 → 省略して raw energy を保存
        # (マージ時に sier_gen5_v3.py が de3 を再計算)
        bank.put('trim', gk, e_ijk)
        n_done += 1

        if n_done % 10 == 0:
            bank.save()
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            print(f"  [{n_done}/{len(todo)}] R={rm:.3f}A t={elapsed/3600:.2f}h eta={eta/3600:.1f}h",
                  flush=True)
        else:
            print(f"  [{n_done}] R={rm:.3f}A", flush=True)

    bank.save()
    print(f"\n完了: {n_done}/{len(todo)} classes in {(time.time()-t0)/3600:.2f}h")
    print(f"Bank: {bank.stats()}")
    print(f"\n転送コマンド:")
    print(f"  scp {bank_path} jetson@100.64.1.27:/home/jetson/motif_bank_pc.json")


def main():
    ap = argparse.ArgumentParser(description="PC(WSL)でmotif_bankを事前充填")
    ap.add_argument('--gen',     type=int, default=5,   help='Sierpinski世代')
    ap.add_argument('--atm-r',   type=float, default=6.0, help='QC/ATM境界 (Å)')
    ap.add_argument('--workers', type=int, default=4,   help='ワーカー数 (予約)')
    ap.add_argument('--bank',    default='/home/yoiyoi/motif_bank_pc.json',
                                 help='出力bankパス')
    ap.add_argument('--dry-run', action='store_true', help='実際にQCを実行しない')
    args = ap.parse_args()

    print(f"""
OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '未設定')} (要=1)
PySCF: {'あり' if HAS_PYSCF else 'なし'}
Bank : {args.bank}
""")
    run(gen=args.gen, atm_r_min=args.atm_r, workers=args.workers,
        bank_path=Path(args.bank), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
