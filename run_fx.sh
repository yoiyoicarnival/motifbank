#!/bin/bash

cd ~/fxpy || exit
source venv/bin/activate

echo "===== $(date '+%Y-%m-%d %H:%M') ====="
python main.py
echo ""
