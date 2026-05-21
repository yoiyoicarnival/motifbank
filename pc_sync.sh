#!/bin/bash
# pc_sync.sh -- PC(WSL) motif_bank_pc.json → Jetson マージ & API hot-reload
#
# 使い方 (WSL Ubuntu から):
#   bash /home/yoiyoi/pc_sync.sh
#
# 前提:
#   /home/yoiyoi/motif_bank_pc.json が存在すること
#   ssh jetson@100.64.1.27 がパスフレーズなしで通ること
#   Jetson に merge_banks.py が配置済みであること

set -e
JETSON="jetson@100.64.1.27"
PC_BANK="/home/yoiyoi/motif_bank_pc.json"
JETSON_BANK="/home/jetson/motif_bank.json"
MERGE_SCRIPT="/home/jetson/merge_banks.py"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') PC→Jetson bank sync ==="

if [ ! -f "$PC_BANK" ]; then
    echo "ERROR: $PC_BANK が見つかりません。pc_bank_prefill.py を先に実行してください。" >&2
    exit 1
fi

PC_ENTRIES=$(python3 -c "
import json; b=json.load(open('$PC_BANK'))
print(sum(len(v) for v in b.values() if isinstance(v,dict)))
" 2>/dev/null || echo "?")
echo "PC bank: $PC_ENTRIES エントリ"

# 1. PC bank を Jetson へ転送
echo "[1/3] 転送中..."
scp "$PC_BANK" "$JETSON:/home/jetson/motif_bank_pc.json"
echo "      転送完了"

# 2. Jetson でマージ
echo "[2/3] マージ中..."
ssh "$JETSON" "python3 $MERGE_SCRIPT $JETSON_BANK /home/jetson/motif_bank_pc.json"

# 3. API hot-reload (Jetson で api_server.py が動いていれば)
echo "[3/3] API hot-reload..."
API_KEY=$(ssh "$JETSON" "python3 -c \"
import json, pathlib
p = pathlib.Path('/home/jetson/api_keys.json')
if p.exists():
    ks = json.load(open(p))
    admin = [k for k, v in ks.items() if v.get('tier') == 'admin']
    print(admin[0] if admin else '')
\" 2>/dev/null" || echo "")

if [ -n "$API_KEY" ]; then
    RESULT=$(ssh "$JETSON" "curl -sf -X POST http://localhost:8000/admin/bank/reload \
        -H 'X-API-Key: $API_KEY'" 2>/dev/null || echo "")
    if [ -n "$RESULT" ]; then
        echo "      API reload 成功: $RESULT"
    else
        echo "      API は起動していないか、reloadエラー (問題なし)"
    fi
else
    echo "      APIキーが取得できなかった (API未起動?)"
fi

echo "=== 完了 ==="
