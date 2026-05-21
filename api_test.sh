#!/bin/bash
# api_test.sh -- MotifBank API v0.3 エンドツーエンドテスト
#
# 使い方:
#   bash api_test.sh [API_URL] [ADMIN_KEY]
#   bash api_test.sh http://100.64.1.27:8000   # Jetson Tailscale 経由
#   bash api_test.sh http://localhost:8000      # ローカル

set -e
BASE="${1:-http://localhost:8000}"
ADMIN_KEY="${2:-}"

PASS=0; FAIL=0

ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
sep()  { echo; echo "── $1 ──────────────────────────────"; }

# ── adminキー取得 ────────────────────────────────────────────────────────────
if [ -z "$ADMIN_KEY" ]; then
    if [ -f /home/jetson/api_keys.json ]; then
        ADMIN_KEY=$(python3 -c "
import json
ks = json.load(open('/home/jetson/api_keys.json'))
print([k for k,v in ks.items() if v.get('tier')=='admin'][0])
" 2>/dev/null)
    fi
fi
if [ -z "$ADMIN_KEY" ]; then
    echo "ADMIN_KEY が不明です。引数2で指定してください。"
    exit 1
fi
echo "Admin key: ${ADMIN_KEY:0:16}..."

# ── 1. ヘルスチェック ────────────────────────────────────────────────────────
sep "1. Health check"
R=$(curl -sf "$BASE/v1/health") && echo "  $R"
echo "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='ok'" \
    && ok "health" || fail "health"

# ── 2. 認証エラー ────────────────────────────────────────────────────────────
sep "2. Auth (no key → 401)"
HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE/v1/fragment/energy" \
    -H "Content-Type: application/json" \
    -d '{"atoms":[{"symbol":"H","x":0,"y":0,"z":0}]}')
[ "$HTTP" = "401" ] && ok "401 without key" || fail "expected 401, got $HTTP"

# ── 3. 無効キー ──────────────────────────────────────────────────────────────
sep "3. Auth (bad key → 401)"
HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE/v1/fragment/energy" \
    -H "X-API-Key: bad-key-xyz" \
    -H "Content-Type: application/json" \
    -d '{"atoms":[{"symbol":"H","x":0,"y":0,"z":0}]}')
[ "$HTTP" = "401" ] && ok "401 bad key" || fail "expected 401, got $HTTP"

# ── 4. 新規キー発行 ──────────────────────────────────────────────────────────
sep "4. Issue new key (admin)"
R=$(curl -sf -X POST "$BASE/admin/keys" \
    -H "Content-Type: application/json" \
    -d "{\"note\":\"test key\",\"tier\":\"free\",\"admin_key\":\"$ADMIN_KEY\"}")
echo "  $R"
FREE_KEY=$(echo "$R" | python3 -c "import json,sys; print(json.load(sys.stdin)['api_key'])")
echo "  Free key: ${FREE_KEY:0:16}..."
[ -n "$FREE_KEY" ] && ok "key issued" || fail "key issue failed"

# ── 5. bankクエリ (H3トリマー) ───────────────────────────────────────────────
sep "5. Fragment energy query"
R=$(curl -sf -X POST "$BASE/v1/fragment/energy" \
    -H "X-API-Key: $FREE_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "atoms": [
            {"symbol":"H","x":0.0,  "y":0.4330, "z":0.0},
            {"symbol":"H","x":-0.375,"y":-0.2165,"z":0.0},
            {"symbol":"H","x":0.375, "y":-0.2165,"z":0.0},
            {"symbol":"H","x":0.75,  "y":0.4330, "z":0.0},
            {"symbol":"H","x":0.375, "y":-0.2165,"z":0.0},
            {"symbol":"H","x":1.125, "y":-0.2165,"z":0.0},
            {"symbol":"H","x":0.375, "y":0.8660, "z":0.0},
            {"symbol":"H","x":0.0,   "y":0.2165, "z":0.0},
            {"symbol":"H","x":0.75,  "y":0.2165, "z":0.0}
        ]
    }')
echo "  $R" | python3 -c "import json,sys; d=json.load(sys.stdin); print('  source:', d['source'], '| cost:', d['cost_jpy'], '円')"
[ -n "$R" ] && ok "query returned" || fail "query failed"

# ── 6. bank stats ────────────────────────────────────────────────────────────
sep "6. Bank stats"
R=$(curl -sf "$BASE/v1/bank/stats" -H "X-API-Key: $FREE_KEY")
echo "  $R"
[ -n "$R" ] && ok "stats returned" || fail "stats failed"

# ── 7. admin usage ───────────────────────────────────────────────────────────
sep "7. Admin usage"
R=$(curl -sf "$BASE/admin/usage" -H "X-API-Key: $ADMIN_KEY")
echo "  $R" | python3 -c "import json,sys; d=json.load(sys.stdin); print('  queries:', d['total_queries'])"
[ -n "$R" ] && ok "usage returned" || fail "usage failed"

# ── 8. bank hot-reload ───────────────────────────────────────────────────────
sep "8. Admin bank reload"
R=$(curl -sf -X POST "$BASE/admin/bank/reload" -H "X-API-Key: $ADMIN_KEY")
echo "  $R"
echo "$R" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='reloaded'" \
    && ok "reload" || fail "reload"

# ── 結果 ─────────────────────────────────────────────────────────────────────
echo
echo "══════════════════════════════════════"
echo "  PASS: $PASS   FAIL: $FAIL"
[ "$FAIL" = "0" ] && echo "  ALL PASS ✓" || echo "  SOME FAILURES ✗"
echo "══════════════════════════════════════"
[ "$FAIL" = "0" ]
