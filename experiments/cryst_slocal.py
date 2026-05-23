#!/usr/bin/env python3
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/yoiyoi')
from motifbank_cli import from_cif, cutoff_trimers, geom_key, com

R_CUT = 5.5

def count_bank(mols):
    n = len(mols)
    coms = np.array([com(m) for m in mols])
    pairs = [(i,j) for i,j in itertools.combinations(range(n),2)
             if np.linalg.norm(coms[i]-coms[j]) < R_CUT]
    trims = cutoff_trimers(mols, R_CUT)
    um = len({geom_key([mols[i]]) for i in range(n)})
    up = len({geom_key([mols[i],mols[j]]) for i,j in pairs})
    ut = len({geom_key([mols[i],mols[j],mols[k]]) for i,j,k in trims})
    return n, um, up, ut, um+up+ut

cif = '/home/yoiyoi/examples/cristobalite_alpha.cif'
print("alpha-cristobalite (sio4):")
print("  sc       N   um   up    ut   N_bank   S_local")
sizes = [((1,1,1),'1x'),((2,1,1),'2x'),((2,2,1),'4x'),
         ((2,2,2),'8x'),((4,2,2),'16x'),((4,4,2),'32x')]
prev_nb = None
for sc, label in sizes:
    try:
        mols, _, _ = from_cif(cif, supercell=sc, mol_type='sio4', verbose=False)
        n, um, up, ut, nb = count_bank(mols)
        s = np.log(max(nb, 1))
        delta = "" if prev_nb is None else " (SAT)" if nb == prev_nb else ""
        print("  %5s  %3d  %3d  %4d  %5d  %6d   %.2f%s" % (label, n, um, up, ut, nb, s, delta))
        prev_nb = nb
    except Exception as e:
        print("  %5s: ERROR %s" % (label, str(e)[:60]))
