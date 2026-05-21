#!/bin/bash
cd /home/yoiyoi
python3 carpet_gen2_pc.py > carpet_gen2_pc.out 2>&1
echo "Done at $(date)" >> carpet_gen2_pc.out
