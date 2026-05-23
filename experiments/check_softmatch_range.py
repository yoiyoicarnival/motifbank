#!/usr/bin/env python3
"""geom_key RMSD vs 原子変位 sigma の関係を調べる"""
import os, sys, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from motifbank_cli import dist_vec
from measure_lipschitz import ref_sioh4, geometry_distance

coords0, types = ref_sioh4()
rng = np.random.default_rng(42)

print('geom_key RMSD vs 原子変位 sigma')
print('(MotifBank soft match 条件: geom_RMSD < 0.10 A)')
print(f"  {'sigma(A)':>9}  {'mean_RMSD':>10}  {'pass%':>7}  {'max_RMSD':>10}")
print('  ' + '-'*46)

for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    rmsd_vals = []
    for _ in range(30):
        delta = rng.normal(0, sigma, coords0.shape)
        coords1 = coords0 + delta
        d = geometry_distance(coords0, coords1)
        rmsd_vals.append(d)
    mean_r = np.mean(rmsd_vals)
    max_r  = np.max(rmsd_vals)
    pct    = np.mean([v < 0.10 for v in rmsd_vals]) * 100
    print(f'  {sigma:>9.2f}  {mean_r:>10.4f}    {pct:>6.0f}%  {max_r:>10.4f}')

print()
print('-> soft-match が 100% 通るのは sigma ~ 0.02A 以下')
print('-> sigma=0.10A では大半が bank miss になる')
