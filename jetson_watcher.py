#!/usr/bin/env python3
"""
jetson_watcher.py — Jetsonキャッシュが300に達したらgp_free実行、次に500目標DFT開始
Usage: python3 jetson_watcher.py
"""
import json, os, subprocess, time, sys

CACHE = "/home/jetson/real_gp_cache.json"
LOG   = "/tmp/gp_free_300.log"

print("Watcher started. Checking cache every 60s...", flush=True)
while True:
    try:
        n = len(json.load(open(CACHE)))
    except Exception as e:
        print(f"Error reading cache: {e}", flush=True)
        n = 0

    print(f"[{time.strftime('%H:%M:%S')}] cache N={n}", flush=True)

    if n >= 300:
        print("Cache ≥ 300! Running gp_free_predictor.py...", flush=True)
        os.environ["OMP_NUM_THREADS"] = "1"
        ret = subprocess.run(
            [sys.executable, "/home/jetson/gp_free_predictor.py"],
            capture_output=True, text=True, cwd="/home/jetson",
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        with open(LOG, "w") as f:
            f.write(ret.stdout + ret.stderr)
        print("gp_free done. Output:", flush=True)
        print(ret.stdout[-1500:] if len(ret.stdout) > 1500 else ret.stdout, flush=True)

        print("Starting DFT expansion to 500...", flush=True)
        subprocess.Popen(
            [sys.executable, "/home/jetson/real_gp_benchmark.py", "--n", "500"],
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            stdout=open("/tmp/cache_expand_500.log", "w"),
            stderr=subprocess.STDOUT,
        )
        print("DFT 500 started.", flush=True)
        break

    time.sleep(60)

print("Watcher done.", flush=True)
