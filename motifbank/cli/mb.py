#!/usr/bin/env python3
"""
mb.py  --  MotifBank CLI
使い方:
  python3 mb.py analyze sierpinski 4
  python3 mb.py analyze carpet 2
  python3 mb.py query "(0.75, 1.0, 1.5)"
  python3 mb.py import /home/yoiyoi/carpet_gen2_qc_cache.json --n-atoms 9
  python3 mb.py stats
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
from motif_db import MotifDB
from fractal_analyzer import analyze_fractal, print_transferability_matrix, FRACTAL_REGISTRY

DB_PATH = str(Path(__file__).parent.parent / 'data' / 'motif_db.sqlite')

def cmd_analyze(args):
    r = analyze_fractal(args.fractal, args.gen, args.A,
                        db_path=DB_PATH, verbose=True)
    if args.json:
        r.pop('pair_keys', None)
        r.pop('trimer_keys', None)
        print(json.dumps(r, indent=2))

def cmd_query(args):
    db = MotifDB(DB_PATH)
    result = db.query(args.geom_key)
    db.close()
    if result:
        print(f"Found: delta_e = {result['delta_e']:.8f} Ha")
        print(json.dumps(result, indent=2))
    else:
        print(f"Not found: {args.geom_key}")
        sys.exit(1)

def cmd_import(args):
    db = MotifDB(DB_PATH)
    n = db.import_cache_json(args.cache_file, args.n_atoms,
                              computed_by=args.computed_by)
    print(f"Imported {n} entries. DB now has {db.stats()['total_entries']} total.")
    db.close()

def cmd_stats(args):
    db = MotifDB(DB_PATH)
    s = db.stats()
    db.close()
    print(f"MotifBank DB: {DB_PATH}")
    print(f"  Total entries: {s['total_entries']:,}")
    for n_atoms, info in sorted(s['by_n_atoms'].items()):
        label = {3: 'monomer', 6: 'pair', 9: 'trimer'}.get(n_atoms, f'{n_atoms}-atom')
        print(f"  {label:10s} ({n_atoms} atoms): {info['count']:>6,} classes, "
              f"{info['total_reuses']:>6,} reuses")

def cmd_matrix(args):
    specs = []
    for s in args.specs:
        parts = s.split(':')
        specs.append((parts[0], int(parts[1])))
    print_transferability_matrix(specs, args.A)

def main():
    parser = argparse.ArgumentParser(description='MotifBank CLI')
    sub = parser.add_subparsers(dest='command')

    # analyze
    p_analyze = sub.add_parser('analyze', help='Analyze fractal motif classes')
    p_analyze.add_argument('fractal', choices=list(FRACTAL_REGISTRY.keys()))
    p_analyze.add_argument('gen', type=int)
    p_analyze.add_argument('--A', type=float, default=0.75)
    p_analyze.add_argument('--json', action='store_true')

    # query
    p_query = sub.add_parser('query', help='Query a geom_key')
    p_query.add_argument('geom_key')

    # import
    p_import = sub.add_parser('import', help='Import JSON cache into DB')
    p_import.add_argument('cache_file')
    p_import.add_argument('--n-atoms', type=int, default=9)
    p_import.add_argument('--computed-by', default='unknown')

    # stats
    p_stats = sub.add_parser('stats', help='Show DB statistics')

    # matrix
    p_matrix = sub.add_parser('matrix', help='Show transferability matrix')
    p_matrix.add_argument('specs', nargs='+', help='fractal:gen e.g. sierpinski:3')
    p_matrix.add_argument('--A', type=float, default=0.75)

    args = parser.parse_args()
    if args.command == 'analyze': cmd_analyze(args)
    elif args.command == 'query': cmd_query(args)
    elif args.command == 'import': cmd_import(args)
    elif args.command == 'stats': cmd_stats(args)
    elif args.command == 'matrix': cmd_matrix(args)
    else: parser.print_help()

if __name__ == '__main__':
    main()
