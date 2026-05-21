#!/usr/bin/env python3
"""
carpet_gen2_pc.py  --  PC (WSL) 版 Sierpinski Carpet Gen2 MBE
Intel i7-1165G7 (8コア, 14GB RAM) で実行
"""
import numpy as np, json, itertools, time, sys
from pathlib import Path
from pyscf import gto, mcscf

A_H3    = 0.75
D_CARP  = 2.0 * A_H3

OFFSETS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

# PC (WSL) 用パス
HOME           = Path('/home/yoiyoi')
OUT_JSON       = HOME / 'carpet_gen2_pc_result.json'
CACHE_FILE     = HOME / 'carpet_gen2_qc_cache.json'
CKPT_FILE      = HOME / 'carpet_gen2_pc_ckpt.json'
VICSEK_CACHE   = HOME / 'vicsek_gen2_qc_cache.json'  # Vicsek完了後に転送

def h3_at(cx, cy, cz=0.0):
    R = A_H3 / np.sqrt(3)
    return [('H',(cx,cy+R,cz)),('H',(cx-A_H3/2,cy-R/2,cz)),('H',(cx+A_H3/2,cy-R/2,cz))]

def geom_key(atoms):
    c = np.array([a[1] for a in atoms])
    return str(tuple(sorted(round(np.linalg.norm(c[i]-c[j]),4)
                            for i in range(len(c)) for j in range(i+1,len(c)))))

def carpet_centers(gen, d=D_CARP):
    if gen == 1: return [(d*ox, d*oy) for (ox,oy) in OFFSETS]
    elif gen == 2:
        D3 = 3*d
        return [(D3*ox+gx, D3*oy+gy)
                for (ox,oy) in OFFSETS
                for (gx,gy) in carpet_centers(1, d)]
    raise ValueError

def sub_ids_gen2():
    ids = []
    for s, (ox,oy) in enumerate(OFFSETS):
        for _ in carpet_centers(1, D_CARP):
            ids.append(s)
    return ids

def casscf_e_atoms(atoms):
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = 'cc-pvdz'
    mol.spin = len(atoms) % 2
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    mf = mol.UHF().run()
    ncas = len(atoms)
    mc = mcscf.CASSCF(mf, ncas, ncas)
    mc.fcisolver.nroots = 2
    mc = mcscf.state_average_(mc, [0.5, 0.5])
    mc.kernel()
    return float(np.atleast_1d(mc.e_tot)[0])

def load_disk_cache():
    cache = {}
    if VICSEK_CACHE.exists():
        try:
            vc = json.load(open(VICSEK_CACHE))
            cache.update(vc)
            print(f"  Loaded Vicsek cache: {len(vc)} entries", flush=True)
        except Exception as e:
            print(f"  Vicsek cache load failed: {e}", flush=True)
    if CACHE_FILE.exists():
        try:
            dc = json.load(open(CACHE_FILE))
            n_before = len(cache)
            cache.update(dc)
            print(f"  Loaded Carpet cache: {len(dc)} entries (+{len(cache)-n_before} new)", flush=True)
        except Exception as e:
            print(f"  Carpet cache load failed: {e}", flush=True)
    return cache

def save_disk_cache(cache):
    tmp = CACHE_FILE.with_suffix('.tmp')
    json.dump(cache, open(tmp, 'w'))
    tmp.rename(CACHE_FILE)

def enumerate_all_classes(centers):
    N = len(centers)
    mono_keys = set()
    pair_keys = {}
    trim_keys = {}

    for i in range(N):
        k = geom_key(h3_at(*centers[i]))
        mono_keys.add(k)

    for i, j in itertools.combinations(range(N), 2):
        k = geom_key(h3_at(*centers[i]) + h3_at(*centers[j]))
        if k not in pair_keys:
            pair_keys[k] = (i, j)

    for i, j, k_ in itertools.combinations(range(N), 3):
        k = geom_key(h3_at(*centers[i]) + h3_at(*centers[j]) + h3_at(*centers[k_]))
        if k not in trim_keys:
            trim_keys[k] = (i, j, k_)

    print(f"Unique classes: mono={len(mono_keys)}, pairs={len(pair_keys)}, trimers={len(trim_keys)}", flush=True)
    return list(mono_keys), pair_keys, trim_keys

def compute_missing_classes(centers, mono_keys, pair_keys, trim_keys, disk_cache, t0):
    N = len(centers)
    mono_vals = {}

    print("\n--- Phase 2a: Monomer energies ---", flush=True)
    for mk in mono_keys:
        if mk not in disk_cache:
            for i in range(N):
                if geom_key(h3_at(*centers[i])) == mk:
                    disk_cache[mk] = casscf_e_atoms(h3_at(*centers[i]))
                    save_disk_cache(disk_cache)
                    break
        mono_vals[mk] = disk_cache[mk]

    print(f"\n--- Phase 2b: Pair de2 ({len(pair_keys)} unique classes) ---", flush=True)
    pair_done = sum(1 for k in pair_keys if k in disk_cache)
    print(f"  Already cached: {pair_done}/{len(pair_keys)}", flush=True)

    for idx, (pk, (i, j)) in enumerate(pair_keys.items()):
        if pk not in disk_cache:
            ai = h3_at(*centers[i])
            aj = h3_at(*centers[j])
            e_pair = casscf_e_atoms(ai + aj)
            mk_i = geom_key(ai); mk_j = geom_key(aj)
            disk_cache[pk] = e_pair - mono_vals[mk_i] - mono_vals[mk_j]
            if idx % 10 == 0:
                save_disk_cache(disk_cache)
                print(f"  pair {idx+1}/{len(pair_keys)} t={time.time()-t0:.0f}s", flush=True)
    save_disk_cache(disk_cache)

    print(f"\n--- Phase 2c: Trimer de3 ({len(trim_keys)} unique classes) ---", flush=True)
    trim_done = sum(1 for k in trim_keys if k in disk_cache)
    print(f"  Already cached: {trim_done}/{len(trim_keys)}", flush=True)
    n_computed = 0

    for idx, (tk, (i, j, k_)) in enumerate(trim_keys.items()):
        if tk in disk_cache:
            continue
        ai = h3_at(*centers[i]); aj = h3_at(*centers[j]); ak = h3_at(*centers[k_])
        mk_i = geom_key(ai); mk_j = geom_key(aj); mk_k = geom_key(ak)
        pk_ij = geom_key(ai+aj); pk_ik = geom_key(ai+ak); pk_jk = geom_key(aj+ak)

        e_ijk = casscf_e_atoms(ai + aj + ak)
        e_ij = disk_cache.get(pk_ij, casscf_e_atoms(ai+aj) - mono_vals[mk_i] - mono_vals[mk_j])
        e_ik = disk_cache.get(pk_ik, casscf_e_atoms(ai+ak) - mono_vals[mk_i] - mono_vals[mk_k])
        e_jk = disk_cache.get(pk_jk, casscf_e_atoms(aj+ak) - mono_vals[mk_j] - mono_vals[mk_k])

        disk_cache[tk] = (e_ijk
                          - (e_ij + mono_vals[mk_i] + mono_vals[mk_j])
                          - (e_ik + mono_vals[mk_i] + mono_vals[mk_k])
                          - (e_jk + mono_vals[mk_j] + mono_vals[mk_k])
                          + mono_vals[mk_i] + mono_vals[mk_j] + mono_vals[mk_k])
        n_computed += 1
        if n_computed % 20 == 0:
            save_disk_cache(disk_cache)
            eta_h = (time.time()-t0)/n_computed * (len(trim_keys)-trim_done-n_computed) / 3600
            print(f"  trimer {trim_done+n_computed}/{len(trim_keys)} computed "
                  f"t={time.time()-t0:.0f}s eta={eta_h:.1f}h", flush=True)

    save_disk_cache(disk_cache)
    print(f"Phase 2 complete. All {len(trim_keys)} trimer classes cached.", flush=True)
    return mono_vals

def accumulate(centers, sub_ids, disk_cache, mono_vals):
    N = len(centers)
    all_t = list(itertools.combinations(range(N), 3))
    n_total = len(all_t)

    start = 0
    de3_intra = de3_inter = de2_sum = 0.0
    n_intra = n_inter = n_unk = 0
    if CKPT_FILE.exists():
        try:
            ck = json.load(open(CKPT_FILE))
            if ck.get('phase') == 'accum':
                start = ck['done']
                de3_intra = ck['de3_intra']; de3_inter = ck['de3_inter']
                de2_sum = ck['de2_sum']
                n_intra = ck['n_intra']; n_inter = ck['n_inter']
                n_unk = ck['n_unk']
                print(f"Resume accumulation from {start}/{n_total}", flush=True)
        except Exception: pass

    print(f"\n--- Phase 3: Accumulation ({n_total} trimers, resuming from {start}) ---", flush=True)
    t0 = time.time()

    if de2_sum == 0.0:
        for i, j in itertools.combinations(range(N), 2):
            ai = h3_at(*centers[i]); aj = h3_at(*centers[j])
            pk = geom_key(ai + aj)
            e_pair = disk_cache.get(pk, 0.0)
            de2_sum += e_pair
        print(f"de2_sum = {de2_sum*1000:.4f} mHa", flush=True)

    for cnt, (i, j, k_) in enumerate(all_t[start:], start):
        ai = h3_at(*centers[i]); aj = h3_at(*centers[j]); ak = h3_at(*centers[k_])
        tk = geom_key(ai + aj + ak)
        de3_v = disk_cache.get(tk, None)
        is_intra = (sub_ids[i] == sub_ids[j] == sub_ids[k_])
        if de3_v is None:
            n_unk += 1
        elif is_intra:
            de3_intra += de3_v; n_intra += 1
        else:
            de3_inter += de3_v; n_inter += 1

        if cnt % 5000 == 0 or cnt == n_total-1:
            elapsed = time.time()-t0
            rate = (cnt-start+1)/elapsed if elapsed > 0 else 0
            eta = (n_total-cnt-1)/rate/60 if rate > 0 else 0
            ck_data = {'phase': 'accum', 'done': cnt+1,
                       'de3_intra': de3_intra, 'de3_inter': de3_inter,
                       'de2_sum': de2_sum, 'n_intra': n_intra,
                       'n_inter': n_inter, 'n_unk': n_unk}
            json.dump(ck_data, open(CKPT_FILE, 'w'))
            print(f"  {cnt+1}/{n_total} intra={n_intra} inter={n_inter} unk={n_unk} "
                  f"rate={rate:.0f}/s eta={eta:.1f}min", flush=True)

    return de2_sum, de3_intra, de3_inter, n_intra, n_inter, n_unk

def main():
    t0 = time.time()
    print("=== Carpet Gen2 PC (i7-1165G7) MBE ===", flush=True)
    print(f"A_H3={A_H3}, D_CARP={D_CARP}", flush=True)

    centers = carpet_centers(2)
    sub_ids = sub_ids_gen2()
    N = len(centers)
    print(f"N={N} clusters, C(N,3)={N*(N-1)*(N-2)//6} trimers", flush=True)

    disk_cache = load_disk_cache()
    print(f"Initial cache size: {len(disk_cache)} entries", flush=True)

    mono_keys, pair_keys, trim_keys = enumerate_all_classes(centers)
    n_new_pairs = sum(1 for k in pair_keys if k not in disk_cache)
    n_new_trims = sum(1 for k in trim_keys if k not in disk_cache)
    print(f"New QC needed: {n_new_pairs} pairs, {n_new_trims} trimers", flush=True)
    print(f"ETA: {(n_new_pairs*1 + n_new_trims*3)/60:.1f} h", flush=True)

    mono_vals = compute_missing_classes(centers, mono_keys, pair_keys, trim_keys, disk_cache, t0)
    de2, de3i, de3e, ni, ne, nunk = accumulate(centers, sub_ids, disk_cache, mono_vals)

    rs_intra = abs(de3i)/abs(de2)
    rs_inter = abs(de3e)/abs(de2)
    rs = rs_intra + rs_inter
    result = {
        'GEN': 2, 'N': N,
        'de2_sum': de2, 'de3_intra': de3i, 'de3_inter': de3e,
        'rs_intra': rs_intra, 'rs_inter': rs_inter, 'rs': rs,
        'n_intra': ni, 'n_inter': ne, 'n_unk': nunk,
        'elapsed_h': (time.time()-t0)/3600
    }
    json.dump(result, open(OUT_JSON, 'w'), indent=2)

    print("\n=== RESULT ===", flush=True)
    print(f"de2_sum={de2*1000:.4f} mHa", flush=True)
    print(f"de3_intra={de3i*1000:.4f} mHa  de3_inter={de3e*1000:.4f} mHa", flush=True)
    print(f"rs_intra={rs_intra:.6f}  rs_inter={rs_inter:.6f}  rs={rs:.6f}", flush=True)
    print(f"n_unk={nunk}", flush=True)
    print(f"elapsed={((time.time()-t0)/3600):.1f}h", flush=True)
    print(f"Saved {OUT_JSON}", flush=True)

if __name__ == '__main__':
    main()
