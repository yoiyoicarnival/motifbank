#!/usr/bin/env python3
"""
test_api.py -- MotifBank API end-to-end テスト

使い方:
  OMP_NUM_THREADS=1 python3 test_api.py [--port 8000]

動作:
  1. api_server.py をサブプロセスで起動
  2. /v1/health が応答するまで待機 (最大30秒)
  3. api_keys.json から admin キーを読み込む
  4. 全エンドポイントをテスト (10項目)
  5. サーバー停止
"""

import os, sys, json, time, subprocess, argparse
import urllib.request, urllib.error
from pathlib import Path

BASE      = Path(__file__).parent
KEYS_FILE = BASE / 'api_keys.json'


# ─── HTTP ヘルパー ───────────────────────────────────────────────────────────

def http(method, url, data=None, headers=None):
    body = json.dumps(data).encode() if data else None
    h    = {'Content-Type': 'application/json', **(headers or {})}
    req  = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def wait_for_server(url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ─── メイン ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MotifBank API end-to-end test")
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no-server', action='store_true',
                    help='サーバーを自前で起動しない (既に動いている場合)')
    args = ap.parse_args()

    base_url = f'http://127.0.0.1:{args.port}'
    proc     = None

    # ── テスト用一時ファイル ──
    import tempfile
    tmp_bank = tempfile.mktemp(suffix='_test_bank.json')
    tmp_keys = tempfile.mktemp(suffix='_test_keys.json')

    # ── サーバー起動 ──
    if not args.no_server:
        print("サーバー起動中...")
        env  = {**os.environ, 'OMP_NUM_THREADS': '1'}
        proc = subprocess.Popen(
            [sys.executable, str(BASE / 'api_server.py'),
             '--port', str(args.port),
             '--bank', tmp_bank,
             '--keys', tmp_keys],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not wait_for_server(f'{base_url}/v1/health', timeout=30):
            stderr = proc.stderr.read().decode() if proc.stderr else ''
            proc.terminate()
            print(f"FAIL: サーバーが起動しませんでした\n{stderr}")
            sys.exit(1)
        print("  サーバー起動 OK\n")

    # ── admin キー読み込み (一時キーファイル優先) ──
    admin_key = None
    keys_path = Path(tmp_keys) if Path(tmp_keys).exists() else KEYS_FILE
    if keys_path.exists():
        keys_data = json.load(open(keys_path))
        for k, v in keys_data.items():
            if v.get('tier') == 'admin':
                admin_key = k
                break

    if not admin_key:
        print("FAIL: admin キーが見つかりません (api_keys.json を確認してください)")
        if proc:
            proc.terminate()
        sys.exit(1)

    print(f"Admin key: {admin_key[:16]}...\n")

    # ── テスト実行 ──
    passed = 0
    failed = 0

    def check(name, ok, detail=''):
        nonlocal passed, failed
        if ok:
            print(f"  PASS  {name}")
            passed += 1
        else:
            short = str(detail)[:120] if detail else ''
            print(f"  FAIL  {name}  →  {short}")
            failed += 1

    auth = {'X-API-Key': admin_key}

    # T1: ヘルスチェック (認証不要)
    s, r = http('GET', f'{base_url}/v1/health')
    check('T1  GET  /v1/health',
          s == 200 and r.get('status') == 'ok',
          r)

    # T2: bank 統計
    s, r = http('GET', f'{base_url}/v1/bank/stats', headers=auth)
    check('T2  GET  /v1/bank/stats',
          s == 200 and 'n_entries' in r,
          r)

    # T3: フラグメントエネルギー (同期, H2O)
    water = [
        {'symbol': 'O', 'x': 0.000, 'y': 0.000, 'z': 0.000},
        {'symbol': 'H', 'x': 0.757, 'y': 0.586, 'z': 0.000},
        {'symbol': 'H', 'x':-0.757, 'y': 0.586, 'z': 0.000},
    ]
    s, r = http('POST', f'{base_url}/v1/fragment/energy',
                data={'atoms': water, 'method': 'hf', 'basis': 'sto-3g'},
                headers=auth)
    check('T3  POST /v1/fragment/energy  (sync,  bank miss → QC)',
          s == 200 and r.get('energy_Ha') is not None and r.get('source') == 'qc_computed',
          r)

    # T4: 2回目は bank ヒット
    s, r = http('POST', f'{base_url}/v1/fragment/energy',
                data={'atoms': water, 'method': 'hf', 'basis': 'sto-3g'},
                headers=auth)
    check('T4  POST /v1/fragment/energy  (sync,  bank hit)',
          s == 200 and r.get('source') == 'bank',
          r)

    # T5: 非同期モード (H2)
    h2 = [
        {'symbol': 'H', 'x': 0.00, 'y': 0.0, 'z': 0.0},
        {'symbol': 'H', 'x': 0.74, 'y': 0.0, 'z': 0.0},
    ]
    s, r = http('POST', f'{base_url}/v1/fragment/energy',
                data={'atoms': h2, 'method': 'hf', 'basis': 'sto-3g', 'async': True},
                headers=auth)
    check('T5  POST /v1/fragment/energy  (async queued/bank)',
          s == 200 and r.get('source') in ('bank', 'queued'),
          r)

    # T6: Phase 分類
    # 2Dグリッド (3.5Å間隔) → 対角線=4.95Å < R_cut=6Å → トリマー形成可
    def shift_mol(dx, dy=0):
        return {'coords': [[c[0]+dx, c[1]+dy, c[2]] for c in
                            [[0.0,0.0,0.0],[0.757,0.586,0.0],[-0.757,0.586,0.0]]],
                'atom_types': ['O','H','H']}

    mols4 = [shift_mol(i*3.5, j*3.5) for i in range(2) for j in range(2)]
    mols8 = [shift_mol(i*3.5, j*3.5) for i in range(2) for j in range(4)]

    s, r = http('POST', f'{base_url}/v1/classify',
                data={'molecules': mols4, 'molecules_2x': mols8, 'label': 'test_water4'},
                headers=auth)
    check('T6  POST /v1/classify',
          s == 200 and 'phase' in r and 'roi_pct' in r,
          r)

    # T7: MBE 計算 (2Dグリッド 4分子)
    s, r = http('POST', f'{base_url}/v1/mbe',
                data={'molecules': mols4, 'method': 'hf', 'basis': 'sto-3g'},
                headers=auth)
    check('T7  POST /v1/mbe  (4 H2O)',
          s == 200 and 'E_total_Ha' in r,
          r)

    # T8: /admin/keys — 新 API キー発行
    s, r = http('POST', f'{base_url}/admin/keys',
                data={'note': 'e2e_test', 'tier': 'free', 'admin_key': admin_key})
    check('T8  POST /admin/keys',
          s == 200 and 'api_key' in r,
          r)
    new_key = r.get('api_key', '')

    # T9: 新キーで bank/stats
    s, r = http('GET', f'{base_url}/v1/bank/stats',
                headers={'X-API-Key': new_key})
    check('T9  GET  /v1/bank/stats  (new free key)',
          s == 200 and 'n_entries' in r,
          r)

    # T10: 不正キーで 401
    s, r = http('GET', f'{base_url}/v1/bank/stats',
                headers={'X-API-Key': 'totally-invalid-key'})
    check('T10 GET  /v1/bank/stats  (invalid key → 401)',
          s == 401,
          r)

    # ── 結果 ──
    print(f"\n{'='*50}")
    print(f"  結果: {passed}/{passed+failed} PASS")
    print(f"{'='*50}")

    if proc:
        proc.terminate()
        proc.wait()

    # 一時ファイル削除
    for f in (tmp_bank, tmp_keys):
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
