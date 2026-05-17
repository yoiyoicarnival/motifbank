#!/usr/bin/env python3
"""
api_server.py -- MotifBank Fragment Energy API (v1.0)

起動方法:
  pip install fastapi uvicorn pydantic numpy pyscf ase
  OMP_NUM_THREADS=1 python3 api_server.py [--bank path] [--keys path]

エンドポイント:
  POST /v1/fragment/energy  -- フラグメントエネルギー取得 (bank hit / PySCF計算)
  POST /v1/classify         -- Phase分類 + ROI推定
  GET  /v1/bank/stats       -- bank統計
  GET  /v1/health           -- ヘルスチェック
  POST /admin/keys          -- APIキー発行 (admin only)
  GET  /admin/usage         -- 使用量レポート (admin only)

APIキー:
  ヘッダー: X-API-Key: <key>

料金目安:
  bank hit: 10円 (0.1秒)
  new QC:  10〜200円 (1秒〜30分 QC計算コストに応じて)
"""

import os, sys, json, time, secrets, logging, argparse, threading, queue
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import deque
import numpy as np

# FastAPI
try:
    from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# motifbank_cli をインポート (同じディレクトリ)
sys.path.insert(0, str(Path(__file__).parent))
from motifbank_cli import (
    MotifBank, geom_key as _geom_key, dist_vec, classify,
    qc_compute_pyscf, qc_compute_mock, make_qc_func,
    from_cif, run_mbe, R_CUT_DEF,
    build_ice2d, build_carpet, build_mof_pore,
)

# ─── パス設定 ────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
DEFAULT_BANK   = _BASE / 'motifbank_api.json'
DEFAULT_KEYS   = _BASE / 'api_keys.json'
DEFAULT_LOG    = _BASE / 'api_access.log'
BACKUP_DIR     = _BASE / 'backups'

# ─── 料金表 (円) ────────────────────────────────────────────────────────────
PRICE = {
    'bank_hit':      10,
    'qc_mono':       10,
    'qc_pair':       30,
    'qc_trim':      100,
    'qc_heavy':     200,
}

RATE_LIMITS = {'free': 60, 'paid': 600, 'admin': 99999}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('motifbank-api')


# ─── QC バックエンド判定 ─────────────────────────────────────────────────────

def _detect_qc_backend():
    try:
        import pyscf
        return 'pyscf'
    except ImportError:
        logger.warning("PySCF not found — using mock QC (for testing only)")
        return 'mock'

_QC_BACKEND = _detect_qc_backend()


# ─── 非同期 QC ワーカー ──────────────────────────────────────────────────────

class QCWorker:
    """
    バックグラウンドスレッドで QC 計算を処理するキュー
    bank miss 時: キューに追加 → 計算完了後に bank に格納
    """
    def __init__(self, bank: MotifBank, backend: str = 'mock'):
        self.bank      = bank
        self.backend   = backend
        self._queue    = queue.Queue()
        self._pending  = {}   # geom_key_str -> threading.Event
        self._lock     = threading.Lock()
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"QC worker started (backend={backend})")

    def submit(self, key_str: str, mol_list, atom_types_list,
               basis='sto-3g', method='hf', charge=0, spin=0):
        """計算キューに追加。同じキーが処理中なら Event を返す（待機可能）"""
        with self._lock:
            if key_str in self._pending:
                return self._pending[key_str]  # 既にキューイング済み
            evt = threading.Event()
            self._pending[key_str] = evt

        self._queue.put((key_str, mol_list, atom_types_list,
                         basis, method, charge, spin, evt))
        return evt

    def wait(self, key_str: str, timeout: float = 120.0) -> Optional[float]:
        """完了を待機して energy_Ha を返す (timeout 秒)"""
        with self._lock:
            evt = self._pending.get(key_str)
        if evt is None:
            # 既に計算済み → bank から取得
            key_tuple = tuple(json.loads(key_str)) if key_str.startswith('[') else None
            if key_tuple:
                return self.bank.data.get(key_tuple, {}).get("energy_Ha")
            return None
        if evt.wait(timeout=timeout):
            # 計算完了 → bank から取得
            key_tuple = None
            for k in self.bank.data:
                if json.dumps(list(k)) == key_str:
                    key_tuple = k
                    break
            if key_tuple:
                return self.bank.data[key_tuple]["energy_Ha"]
        return None

    def _run(self):
        while True:
            try:
                key_str, mol_list, atom_types_list, basis, method, charge, spin, evt = \
                    self._queue.get(timeout=5)
            except queue.Empty:
                continue

            try:
                t0 = time.perf_counter()
                if self.backend == 'pyscf':
                    e = qc_compute_pyscf(mol_list, atom_types_list,
                                         basis=basis, method=method,
                                         charge=charge, spin=spin)
                else:
                    e = qc_compute_mock(mol_list)
                dt = time.perf_counter() - t0

                key_tuple = _geom_key(mol_list)
                self.bank.store(key_tuple, mol_list, e,
                                source=f"{self.backend}/{method}/{basis}")
                # bank を自動保存
                try:
                    self.bank.save()
                except Exception:
                    pass

                logger.info(f"QC done: {key_str[:40]} e={e:.6f} Ha ({dt:.1f}s)")
            except Exception as ex:
                logger.error(f"QC failed: {key_str[:40]}: {ex}")
            finally:
                with self._lock:
                    self._pending.pop(key_str, None)
                evt.set()
                self._queue.task_done()


# ─── APIキー管理 ─────────────────────────────────────────────────────────────

class KeyStore:
    def __init__(self, path: Path):
        self.path = path
        self._keys = {}
        self.load()

    def load(self):
        if self.path.exists() and self.path.stat().st_size > 0:
            self._keys = json.load(open(self.path))
        else:
            admin_key = 'admin-' + secrets.token_hex(8)
            self._keys = {
                admin_key: {'tier': 'admin', 'note': 'auto-generated', 'created': time.time()}
            }
            self.save()
            logger.warning(f"Admin key: {admin_key}  (saved to {self.path})")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self._keys, open(self.path, 'w'), indent=2)

    def validate(self, key: str) -> Optional[dict]:
        return self._keys.get(key)

    def add(self, note: str, tier: str = 'free') -> str:
        k = 'mb-' + secrets.token_urlsafe(16)
        self._keys[k] = {'tier': tier, 'note': note, 'created': time.time()}
        self.save()
        return k

    def list_keys(self) -> list:
        return [{'key_prefix': k[:12]+'...', **v} for k, v in self._keys.items()]


# ─── レート制限 ──────────────────────────────────────────────────────────────

_rate_windows: Dict[str, deque] = {}
_usage: Dict[str, dict]         = {}
_access_log: list               = []

def _check_rate(api_key: str, tier: str) -> bool:
    now = time.time()
    win = _rate_windows.setdefault(api_key, deque())
    while win and now - win[0] > 60:
        win.popleft()
    limit = RATE_LIMITS.get(tier, 60)
    if len(win) >= limit:
        return False
    win.append(now)
    return True

def _track(api_key: str, source: str, cost: int, ms: int, gk: str):
    u = _usage.setdefault(api_key, {'queries': 0, 'hits': 0, 'cost_jpy': 0})
    u['queries'] += 1
    if source == 'bank':
        u['hits'] += 1
    u['cost_jpy'] += cost
    _access_log.append({'ts': time.time(), 'key': api_key[:8],
                        'src': source, 'cost': cost, 'ms': ms})
    if len(_access_log) % 100 == 0:
        try:
            json.dump(_access_log[-2000:], open(DEFAULT_LOG, 'w'))
        except Exception:
            pass


# ─── FastAPI アプリ ──────────────────────────────────────────────────────────

if HAS_FASTAPI:

    # Pydantic モデル
    class AtomIn(BaseModel):
        symbol: str
        x: float; y: float; z: float

    class FragmentRequest(BaseModel):
        atoms:   List[AtomIn] = Field(..., description="原子リスト (symbol + xyz, Å)")
        method:  str  = Field("hf",      description="hf / mp2 / ccsd")
        basis:   str  = Field("sto-3g",  description="sto-3g / 6-31g / cc-pvdz")
        async_:  bool = Field(False,     alias="async",
                              description="True=非同期 (即時 queued を返す)")

    class FragmentResponse(BaseModel):
        energy_Ha:           Optional[float]
        source:              str   # "bank" / "qc_computed" / "queued" / "pending"
        geom_key:            str
        computation_time_ms: int
        cost_jpy:            int
        message:             Optional[str]

    class MoleculeIn(BaseModel):
        coords:     List[List[float]]  # [[x,y,z], ...]
        atom_types: List[str]          # ["O","H","H"]

    class ClassifyRequest(BaseModel):
        molecules:    List[MoleculeIn]
        molecules_2x: Optional[List[MoleculeIn]] = None
        r_cut:  float = Field(R_CUT_DEF,  description="カットオフ Å")
        T_K:    float = Field(300.0,       description="温度 K")
        eps:    float = Field(0.10,        description="soft matching Å")
        label:  str   = Field("api_query", description="系の名前")

    class ClassifyResponse(BaseModel):
        phase:      int
        gamma:      float
        alpha_pred: float
        N_bank:     int
        compress:   float
        roi_pct:    float
        strategy:   str
        label:      str

    class MBERequest(BaseModel):
        molecules:   List[MoleculeIn]
        r_cut:  float = Field(R_CUT_DEF)
        eps:    float = Field(0.10)
        method: str   = Field("hf")
        basis:  str   = Field("sto-3g")

    class MBEResponse(BaseModel):
        E_mono_Ha:  float
        E_2body_Ha: float
        E_3body_Ha: float
        E_total_Ha: float
        n_mols:     int
        roi_actual: float
        elapsed_s:  float
        cost_jpy:   int

    class NewKeyRequest(BaseModel):
        note:      str
        tier:      str = "free"
        admin_key: str

    # ── グローバル状態 ──
    _bank:   Optional[MotifBank]  = None
    _keys:   Optional[KeyStore]   = None
    _worker: Optional[QCWorker]   = None
    _start_time = time.time()

    app = FastAPI(
        title="MotifBank Fragment Energy API",
        description=(
            "量子化学フラグメントエネルギーの再利用可能性情報交換所。\n"
            "bank ヒット: ~10円 (0.1秒)   新規 QC: 30〜200円 (数秒〜数分)"
        ),
        version="1.0.0",
    )
    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def _auth(x_api_key: Optional[str]) -> dict:
        if not x_api_key:
            raise HTTPException(401, "X-API-Key ヘッダーが必要です")
        info = _keys.validate(x_api_key)
        if info is None:
            raise HTTPException(401, "無効なAPIキー")
        if not _check_rate(x_api_key, info.get('tier', 'free')):
            raise HTTPException(429, f"レート制限超過")
        return info

    # ── ヘルスチェック ──
    @app.get("/v1/health")
    def health():
        return {
            "status":      "ok",
            "qc_backend":  _QC_BACKEND,
            "bank_entries": len(_bank.data) if _bank else 0,
            "uptime_s":    round(time.time() - _start_time, 1),
        }

    # ── バンク統計 ──
    @app.get("/v1/bank/stats")
    def bank_stats(x_api_key: Optional[str] = Header(None)):
        _auth(x_api_key)
        total_q = sum(u['queries'] for u in _usage.values())
        total_h = sum(u['hits']    for u in _usage.values())
        return {
            "n_entries":    len(_bank.data),
            "total_queries": total_q,
            "bank_hit_rate": round(total_h / max(total_q, 1), 4),
            "uptime_s":     round(time.time() - _start_time, 1),
        }

    # ── フラグメントエネルギー ──
    @app.post("/v1/fragment/energy", response_model=FragmentResponse)
    def fragment_energy(req: FragmentRequest,
                        x_api_key: Optional[str] = Header(None)):
        _auth(x_api_key)
        t0 = time.time()

        # 座標リスト化
        mol_list     = [np.array([[a.x, a.y, a.z]]) for a in req.atoms]
        atom_types   = [[a.symbol] for a in req.atoms]
        # 全原子を1つの分子として扱う (フラグメント = 1システム)
        all_coords   = np.array([[a.x, a.y, a.z] for a in req.atoms])
        all_types    = [a.symbol for a in req.atoms]
        mol_as_one   = [all_coords]
        types_as_one = [all_types]

        key     = _geom_key(mol_as_one)
        key_str = json.dumps(list(key))
        n_atoms = len(req.atoms)
        kind    = 'qc_mono' if n_atoms <= 3 else 'qc_pair' if n_atoms <= 6 \
                  else 'qc_trim' if n_atoms <= 9 else 'qc_heavy'

        # bank 照会
        e = _bank.query_soft(mol_as_one, eps=0.10)
        elapsed_ms = int((time.time() - t0) * 1000)

        if e is not None:
            _track(x_api_key, 'bank', PRICE['bank_hit'], elapsed_ms, key_str)
            return FragmentResponse(
                energy_Ha=round(e, 8), source='bank',
                geom_key=key_str, computation_time_ms=elapsed_ms,
                cost_jpy=PRICE['bank_hit'], message=None,
            )

        # bank miss
        cost = PRICE.get(kind, PRICE['qc_heavy'])

        if req.async_:
            # 非同期: キューに追加して即時返却
            _worker.submit(key_str, mol_as_one, types_as_one,
                           req.basis, req.method)
            _track(x_api_key, 'queued', cost, elapsed_ms, key_str)
            return FragmentResponse(
                energy_Ha=None, source='queued',
                geom_key=key_str, computation_time_ms=elapsed_ms,
                cost_jpy=cost,
                message="QC計算をキューに追加しました。30秒後に再クエリしてください。",
            )
        else:
            # 同期: PySCF 計算して返す
            try:
                if _QC_BACKEND == 'pyscf':
                    e = qc_compute_pyscf(mol_as_one, types_as_one,
                                         basis=req.basis, method=req.method)
                else:
                    e = qc_compute_mock(mol_as_one)
                _bank.store(key, mol_as_one, e,
                            source=f"{_QC_BACKEND}/{req.method}/{req.basis}")
                _bank.save()
            except Exception as ex:
                raise HTTPException(500, f"QC計算エラー: {ex}")
            elapsed_ms = int((time.time() - t0) * 1000)
            _track(x_api_key, 'qc_computed', cost, elapsed_ms, key_str)
            return FragmentResponse(
                energy_Ha=round(e, 8), source='qc_computed',
                geom_key=key_str, computation_time_ms=elapsed_ms,
                cost_jpy=cost, message=None,
            )

    # ── Phase 分類 ──
    @app.post("/v1/classify", response_model=ClassifyResponse)
    def classify_system(req: ClassifyRequest,
                         x_api_key: Optional[str] = Header(None)):
        _auth(x_api_key)
        mols    = [np.array(m.coords) for m in req.molecules]
        mols_2x = ([np.array(m.coords) for m in req.molecules_2x]
                   if req.molecules_2x else None)
        r = classify(mols, mols_2x,
                     r_cut=req.r_cut, T_K=req.T_K,
                     eps_match=req.eps, label=req.label,
                     verbose=False)
        return ClassifyResponse(
            phase=r['phase'], gamma=r['gamma'],
            alpha_pred=r['alpha_pred'], N_bank=r['N_bank'],
            compress=r['compress'], roi_pct=r['roi_pct'],
            strategy=r['strategy'], label=r['label'],
        )

    # ── MBE 計算 ──
    @app.post("/v1/mbe", response_model=MBEResponse)
    def mbe_calculation(req: MBERequest,
                         x_api_key: Optional[str] = Header(None)):
        _auth(x_api_key)
        mols       = [np.array(m.coords) for m in req.molecules]
        atom_types = [m.atom_types for m in req.molecules]
        qc_func    = make_qc_func(_QC_BACKEND, req.basis, req.method)

        r = run_mbe(mols, _bank,
                    r_cut=req.r_cut, eps_match=req.eps,
                    qc_func=qc_func, atom_types_list=atom_types,
                    verbose=False)
        _bank.save()

        # コスト推定
        n_qc = r['bank_stats']['n_miss']
        cost = n_qc * PRICE.get('qc_trim', 100)

        return MBEResponse(
            E_mono_Ha=r['E_mono_Ha'], E_2body_Ha=r['E_2body_Ha'],
            E_3body_Ha=r['E_3body_Ha'], E_total_Ha=r['E_total_Ha'],
            n_mols=r['n_mols'], roi_actual=r['roi_actual'],
            elapsed_s=r['elapsed_s'], cost_jpy=cost,
        )

    # ── 管理者: キー発行 ──
    @app.post("/admin/keys")
    def admin_add_key(req: NewKeyRequest):
        info = _keys.validate(req.admin_key)
        if info is None or info.get('tier') != 'admin':
            raise HTTPException(403, "管理者キーが必要です")
        k = _keys.add(note=req.note, tier=req.tier)
        return {"api_key": k, "tier": req.tier, "note": req.note}

    # ── 管理者: 使用量 ──
    @app.get("/admin/usage")
    def admin_usage(x_api_key: Optional[str] = Header(None)):
        info = _auth(x_api_key)
        if info.get('tier') != 'admin':
            raise HTTPException(403, "管理者専用")
        total_q = sum(u['queries'] for u in _usage.values())
        total_h = sum(u['hits']    for u in _usage.values())
        return {
            "per_key":       {k[:8]+'...': v for k, v in _usage.items()},
            "total_queries": total_q,
            "total_hits":    total_h,
            "hit_rate":      round(total_h / max(total_q, 1), 4),
            "uptime_s":      round(time.time() - _start_time, 1),
        }

    # ── 管理者: bank リロード ──
    @app.post("/admin/bank/reload")
    def admin_reload(x_api_key: Optional[str] = Header(None)):
        info = _auth(x_api_key)
        if info.get('tier') != 'admin':
            raise HTTPException(403, "管理者専用")
        _bank.load(_bank.path)
        return {"status": "reloaded", "n_entries": len(_bank.data)}

    # ── 起動 ──
    @app.on_event("startup")
    def on_startup():
        global _bank, _keys, _worker
        # コマンドライン引数をここで再パース
        bank_path = Path(sys.argv[sys.argv.index('--bank')+1]) \
            if '--bank' in sys.argv else DEFAULT_BANK
        keys_path = Path(sys.argv[sys.argv.index('--keys')+1]) \
            if '--keys' in sys.argv else DEFAULT_KEYS

        _bank   = MotifBank(str(bank_path) if bank_path.exists() else None)
        _bank.path = str(bank_path)
        _keys   = KeyStore(keys_path)
        _worker = QCWorker(_bank, backend=_QC_BACKEND)
        logger.info(f"MotifBank API v1.0  bank={bank_path}  QC={_QC_BACKEND}")


# ─── 起動エントリ ─────────────────────────────────────────────────────────────

def main():
    if not HAS_FASTAPI:
        print("pip install fastapi uvicorn pydantic が必要です")
        sys.exit(1)

    ap = argparse.ArgumentParser(description="MotifBank API v1.0")
    ap.add_argument('--bank',  default=str(DEFAULT_BANK))
    ap.add_argument('--keys',  default=str(DEFAULT_KEYS))
    ap.add_argument('--host',  default='127.0.0.1')
    ap.add_argument('--port',  default=8000, type=int)
    args = ap.parse_args()

    print(f"""
=======================================================
  MotifBank Fragment Energy API  v1.0
  QC backend : {_QC_BACKEND}
  Bank       : {args.bank}
  Keys       : {args.keys}
  URL        : http://{args.host}:{args.port}
  Docs       : http://{args.host}:{args.port}/docs
=======================================================
""")
    uvicorn.run("api_server:app",
                host=args.host, port=args.port,
                reload=False, log_level="info")


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
