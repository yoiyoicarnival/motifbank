#!/usr/bin/env python3
"""
motifbank_client.py — MotifBank API Python クライアント

使い方:
  from motifbank_client import MotifBankClient

  client = MotifBankClient("http://localhost:8000", api_key="YOUR_KEY")

  # Phase 分類
  r = client.classify(mols, T_K=300)
  print(r['phase'], r['roi_pct'])

  # フラグメントエネルギー照会
  atoms = [("O",0,0,0), ("H",0.957,0,0), ("H",-0.24,0.927,0)]
  r = client.fragment_energy(atoms, method="hf", basis="sto-3g")
  print(r['energy_Ha'], r['source'])  # "bank" or "qc_computed"

  # MBE 計算
  r = client.mbe(mols, atypes)
  print(r['E_total_Ha'])
"""

import json, time
from typing import List, Optional, Tuple
import numpy as np

try:
    import urllib.request as _req
    import urllib.error as _uerr
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


class MotifBankClient:
    """
    MotifBank API への Python クライアント

    urllib のみ使用 (requests 不要)。
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: str = "", timeout: int = 300):
        self.base    = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, data: dict) -> dict:
        url = self.base + path
        body = json.dumps(data).encode()
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key':    self.api_key,
        }
        req = _req.Request(url, data=body, headers=headers, method='POST')
        try:
            with _req.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except _uerr.HTTPError as e:
            msg = e.read().decode()
            raise RuntimeError(f"HTTP {e.code}: {msg}") from e

    def _get(self, path: str) -> dict:
        url = self.base + path
        headers = {'X-API-Key': self.api_key}
        req = _req.Request(url, headers=headers, method='GET')
        with _req.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        """サーバー状態確認"""
        return self._get("/v1/health")

    def bank_stats(self) -> dict:
        """バンク統計"""
        return self._get("/v1/bank/stats")

    def fragment_energy(self,
                         atoms: List[Tuple],
                         method: str = "hf",
                         basis:  str = "sto-3g",
                         async_: bool = False) -> dict:
        """
        フラグメントエネルギーを照会または計算する

        atoms: list of (symbol, x, y, z)
               例: [("O",0,0,0), ("H",0.957,0,0), ("H",-0.24,0.927,0)]

        Returns:
          {
            "energy_Ha": float or None,
            "source":    "bank" / "qc_computed" / "queued",
            "geom_key":  str,
            "cost_jpy":  int,
            ...
          }
        """
        atom_list = [{"symbol": s, "x": float(x), "y": float(y), "z": float(z)}
                     for s, x, y, z in atoms]
        return self._post("/v1/fragment/energy", {
            "atoms":  atom_list,
            "method": method,
            "basis":  basis,
            "async":  async_,
        })

    def classify(self,
                  mols:     List[np.ndarray],
                  atypes:   Optional[List[List[str]]] = None,
                  mols_2x:  Optional[List[np.ndarray]] = None,
                  r_cut:    float = 6.0,
                  T_K:      float = 300.0,
                  eps:      float = 0.10,
                  label:    str   = "query") -> dict:
        """
        Phase 分類を API サーバーに依頼する

        mols:   list of np.ndarray (N×3, Å)
        atypes: list of lists (省略可)
        """
        def _mol_to_dict(mol, at=None):
            coords = mol.tolist() if isinstance(mol, np.ndarray) else mol
            atom_types = at if at else ["X"] * len(coords)
            return {"coords": coords, "atom_types": atom_types}

        mol_list = [_mol_to_dict(m, a)
                    for m, a in zip(mols, atypes or [None]*len(mols))]
        mol2x_list = ([_mol_to_dict(m) for m in mols_2x]
                      if mols_2x else None)

        body = {
            "molecules": mol_list,
            "r_cut": r_cut, "T_K": T_K, "eps": eps, "label": label,
        }
        if mol2x_list:
            body["molecules_2x"] = mol2x_list

        return self._post("/v1/classify", body)

    def mbe(self,
             mols:   List[np.ndarray],
             atypes: List[List[str]],
             r_cut:  float = 6.0,
             eps:    float = 0.10,
             method: str   = "hf",
             basis:  str   = "sto-3g") -> dict:
        """
        MBE 計算を API サーバーに依頼する

        Returns:
          {
            "E_mono_Ha": float, "E_2body_Ha": float, "E_3body_Ha": float,
            "E_total_Ha": float, "roi_actual": float, "elapsed_s": float,
            "cost_jpy": int
          }
        """
        mol_list = [{"coords": m.tolist(), "atom_types": a}
                    for m, a in zip(mols, atypes)]
        return self._post("/v1/mbe", {
            "molecules": mol_list,
            "r_cut": r_cut, "eps": eps,
            "method": method, "basis": basis,
        })

    def issue_key(self, admin_key: str, note: str, tier: str = "free") -> str:
        """新しい API キーを発行する (管理者専用)"""
        r = self._post("/admin/keys", {
            "admin_key": admin_key, "note": note, "tier": tier
        })
        return r["api_key"]

    def usage(self, admin_key: str = "") -> dict:
        """使用量レポートを取得する (管理者専用)"""
        return self._get("/admin/usage")


# ── コマンドライン使用 ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, argparse

    ap = argparse.ArgumentParser(description="MotifBank API クライアント")
    ap.add_argument('--url',  default='http://localhost:8000')
    ap.add_argument('--key',  default='', help='API キー')
    ap.add_argument('cmd', choices=['health','stats','issue-key'],
                    nargs='?', default='health')
    ap.add_argument('--note',  default='test user')
    ap.add_argument('--tier',  default='free')
    ap.add_argument('--admin', default='', help='管理者キー (issue-key 用)')
    args = ap.parse_args()

    client = MotifBankClient(args.url, args.key)

    if args.cmd == 'health':
        r = client.health()
        print(json.dumps(r, indent=2))
    elif args.cmd == 'stats':
        r = client.bank_stats()
        print(json.dumps(r, indent=2))
    elif args.cmd == 'issue-key':
        k = client.issue_key(args.admin, args.note, args.tier)
        print(f"新しい API キー: {k}")
