"""Offline LiDAR preparation commands: python -m deploy.lidar.cli --help."""
import argparse
import json
from pathlib import Path
from .grid_map import SavedGrid
from .recording import replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('replay',help='Read-only saved-map matching and command analysis')
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--map',type=Path,required=True)
    p.add_argument('--search-center',type=float,nargs=2,required=True,metavar=('X','Y'))
    p.add_argument('--search-radius',type=float,default=1.)
    p.add_argument('--evaluate-goal',type=float,nargs=2,help='Analyze commands only; sends nothing to a robot')
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=replay(args.recording,SavedGrid.load(args.map),args.search_center,args.search_radius,args.evaluate_goal)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
