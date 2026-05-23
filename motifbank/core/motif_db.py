"""
motif_db.py  --  Motif Database (Layer 1)
SQLiteベースのgeom_key → QCエネルギー データベース

使い方:
  from motif_db import MotifDB
  db = MotifDB('motif_db.sqlite')
  db.store(geom_key, n_atoms, delta_e, basis='cc-pvdz', method='CASSCF')
  e = db.query(geom_key)  # None if not found
"""
import sqlite3, json, time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS motifs (
    geom_key    TEXT PRIMARY KEY,
    n_atoms     INTEGER NOT NULL,
    qc_energy   REAL,
    delta_e     REAL,
    basis       TEXT DEFAULT 'cc-pvdz',
    method      TEXT DEFAULT 'CASSCF',
    computed_by TEXT DEFAULT 'unknown',
    computed_at TEXT,
    n_reuses    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_n_atoms ON motifs(n_atoms);
CREATE TABLE IF NOT EXISTS fractals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    gen         INTEGER,
    A_scale     REAL,
    molecule    TEXT,
    n_clusters  INTEGER,
    n_unique    INTEGER,
    n_total     INTEGER,
    reuse_rate  REAL,
    computed_at TEXT
);
"""

class MotifDB:
    def __init__(self, db_path: str = 'motif_db.sqlite'):
        self.path = Path(db_path)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def store(self, geom_key: str, n_atoms: int, delta_e: float,
              qc_energy: float = None, basis: str = 'cc-pvdz',
              method: str = 'CASSCF', computed_by: str = 'PC') -> bool:
        """クラスをDBに保存。既存なら上書き。"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO motifs
                (geom_key, n_atoms, qc_energy, delta_e, basis, method, computed_by, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (geom_key, n_atoms, qc_energy, delta_e, basis, method,
                  computed_by, time.strftime('%Y-%m-%dT%H:%M:%S')))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Store error: {e}")
            return False

    def query(self, geom_key: str) -> Optional[Dict]:
        """geom_keyに対応するエントリを返す。なければNone。"""
        cur = self.conn.execute(
            "SELECT geom_key, n_atoms, delta_e, qc_energy, basis, method FROM motifs WHERE geom_key=?",
            (geom_key,))
        row = cur.fetchone()
        if row is None:
            return None
        # n_reuses++
        self.conn.execute("UPDATE motifs SET n_reuses=n_reuses+1 WHERE geom_key=?", (geom_key,))
        self.conn.commit()
        return {'geom_key': row[0], 'n_atoms': row[1], 'delta_e': row[2],
                'qc_energy': row[3], 'basis': row[4], 'method': row[5]}

    def query_batch(self, geom_keys: List[str]) -> Dict[str, Optional[float]]:
        """複数クエリを一括処理。"""
        placeholders = ','.join('?' * len(geom_keys))
        cur = self.conn.execute(
            f"SELECT geom_key, delta_e FROM motifs WHERE geom_key IN ({placeholders})",
            geom_keys)
        result = {k: None for k in geom_keys}
        for row in cur.fetchall():
            result[row[0]] = row[1]
        return result

    def import_cache_json(self, cache_path: str, n_atoms: int,
                          computed_by: str = 'unknown') -> int:
        """既存のJSONキャッシュ (geom_key→delta_e) をDBにインポート。"""
        with open(cache_path) as f:
            cache = json.load(f)
        n = 0
        for gk, de in cache.items():
            self.store(gk, n_atoms, float(de), computed_by=computed_by)
            n += 1
        print(f"Imported {n} entries from {cache_path}")
        return n

    def stats(self) -> Dict:
        """DB統計を返す。"""
        cur = self.conn.execute(
            "SELECT n_atoms, COUNT(*), SUM(n_reuses) FROM motifs GROUP BY n_atoms")
        by_atoms = {}
        for row in cur.fetchall():
            by_atoms[row[0]] = {'count': row[1], 'total_reuses': row[2]}
        total = sum(v['count'] for v in by_atoms.values())
        return {'total_entries': total, 'by_n_atoms': by_atoms}

    def coverage(self, required_keys: List[str]) -> Dict:
        """指定されたgeom_keyリストのうち何%がDB内にあるかを返す。"""
        if not required_keys:
            return {'total': 0, 'cached': 0, 'new': 0, 'coverage_pct': 100.0}
        placeholders = ','.join('?' * len(required_keys))
        cur = self.conn.execute(
            f"SELECT geom_key FROM motifs WHERE geom_key IN ({placeholders})",
            required_keys)
        found = set(row[0] for row in cur.fetchall())
        new_keys = [k for k in required_keys if k not in found]
        return {
            'total': len(required_keys),
            'cached': len(found),
            'new': len(new_keys),
            'coverage_pct': len(found) / len(required_keys) * 100,
            'new_keys': new_keys
        }

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    # テスト
    db = MotifDB('/tmp/test_motif_db.sqlite')
    db.store('(0.75, 1.0, 1.5)', n_atoms=9, delta_e=-0.001234)
    result = db.query('(0.75, 1.0, 1.5)')
    print(f"Query result: {result}")
    print(f"DB stats: {db.stats()}")
    db.close()
    print("motif_db.py test OK")
