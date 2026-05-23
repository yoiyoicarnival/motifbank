"""
motifbank/api/server.py  --  MotifBank REST API
FastAPI による geom_key → QCエネルギー マーケットプレイス API

起動: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import sys
sys.path.insert(0, '/home/yoiyoi/motifbank/core')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import time, json
from pathlib import Path

# Import core modules
from motif_db import MotifDB
from fractal_analyzer import analyze_fractal, FRACTAL_REGISTRY

DB_PATH = '/home/yoiyoi/motifbank/data/motif_db.sqlite'

app = FastAPI(
    title="MotifBank API",
    description="圧縮型計算科学 — 量子化学モチーフデータベース",
    version="0.1.0"
)

def get_db():
    return MotifDB(DB_PATH)

# ── Models ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    fractal: str          # 'sierpinski', 'vicsek', 'carpet'
    gen: int              # generation number
    A: float = 0.75       # lattice scale parameter

class QueryRequest(BaseModel):
    geom_key: str

class BatchQueryRequest(BaseModel):
    geom_keys: List[str]

class JobRequest(BaseModel):
    fractal: str
    gen: int
    A: float = 0.75

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "MotifBank",
        "version": "0.1.0",
        "description": "Quantum chemistry motif database for recursive systems",
        "endpoints": ["/analyze", "/query", "/stats", "/coverage"]
    }

@app.get("/stats")
def stats():
    """データベース統計を返す"""
    db = get_db()
    s = db.stats()
    db.close()
    return {
        "total_entries": s['total_entries'],
        "by_n_atoms": s['by_n_atoms'],
        "db_path": DB_PATH
    }

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    フラクタル系のユニーククラスを列挙し、DB重複率とコスト見積もりを返す。
    無料で利用可能。
    """
    if req.fractal not in FRACTAL_REGISTRY:
        raise HTTPException(400, f"Unknown fractal '{req.fractal}'. "
                                 f"Known: {list(FRACTAL_REGISTRY.keys())}")
    if req.gen < 1 or req.gen > 6:
        raise HTTPException(400, "gen must be 1-6")

    result = analyze_fractal(req.fractal, req.gen, req.A,
                             db_path=DB_PATH, verbose=False)

    response = {
        "fractal": req.fractal,
        "gen": req.gen,
        "A": req.A,
        "N_clusters": result['N'],
        "n_total_trimers": result['n_total_trimers'],
        "n_unique_pairs": result['n_unique_pairs'],
        "n_unique_trimers": result['n_unique_trimers'],
        "reuse_rate": round(result['reuse_rate'], 2),
    }

    if 'db_coverage' in result:
        cov = result['db_coverage']
        response.update({
            "db_coverage_pct": round(cov['coverage_pct'], 1),
            "cached_classes": cov['cached'],
            "new_classes": cov['new'],
            "est_compute_hours": round(cov['new'] * 62 / 3600, 2),
            "est_cost_usd": round(cov['new'] * 0.01, 2),
        })
    else:
        response.update({
            "db_coverage_pct": 0.0,
            "cached_classes": 0,
            "new_classes": result['n_unique_trimers'],
            "est_compute_hours": round(result['n_unique_trimers'] * 62 / 3600, 2),
            "est_cost_usd": round(result['n_unique_trimers'] * 0.01, 2),
        })

    return response

@app.get("/query")
def query_single(geom_key: str):
    """
    1つのgeom_keyのdelta_eを返す。
    DB内になければ 404。
    """
    db = get_db()
    result = db.query(geom_key)
    db.close()
    if result is None:
        raise HTTPException(404, f"geom_key not found in database")
    return result

@app.post("/query/batch")
def query_batch(req: BatchQueryRequest):
    """
    複数geom_keyを一括クエリ。
    DBにある分だけ返し、なければnull。
    """
    if len(req.geom_keys) > 10000:
        raise HTTPException(400, "Max 10000 keys per batch")
    db = get_db()
    results = db.query_batch(req.geom_keys)
    db.close()
    n_found = sum(1 for v in results.values() if v is not None)
    return {
        "total": len(req.geom_keys),
        "found": n_found,
        "coverage_pct": round(n_found / len(req.geom_keys) * 100, 1),
        "results": results
    }

@app.post("/coverage")
def coverage(req: AnalyzeRequest):
    """
    指定フラクタルのgeom_keysのうち、DB内にある割合を返す。
    クエリ前の重複チェックに使用。
    """
    result = analyze_fractal(req.fractal, req.gen, req.A,
                             db_path=DB_PATH, verbose=False)
    if 'db_coverage' in result:
        cov = result['db_coverage']
        return {
            "total": cov['total'],
            "cached": cov['cached'],
            "new": cov['new'],
            "coverage_pct": round(cov['coverage_pct'], 1),
        }
    return {"total": result['n_unique_trimers'], "cached": 0,
            "new": result['n_unique_trimers'], "coverage_pct": 0.0}

# ── Launch ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    print("Starting MotifBank API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
