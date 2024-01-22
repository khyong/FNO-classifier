import torch

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', type=str)
args = parser.parse_args()

"""
cnt = 0
for filepath in sorted(Path(args.dirpath).glob('**/**/models/best_model.pth')):
    cnt += 1
    if cnt == 1:
        print("=" * 80)
        print(filepath)
    if cnt % 5 == 0:
        cnt = 0
        ckpt = torch.load(filepath, map_location='cpu')
        print("{:5.2f}".format(ckpt['best_result'] * 100))
        print()
        continue
    ckpt = torch.load(filepath, map_location='cpu')
    print("{:5.2f}".format(ckpt['best_result'] * 100), end=',')
"""
for filepath in sorted(Path(args.dirpath).glob('**/**/models/best_model.pth')):
    print("=" * 80)
    print(filepath)
    ckpt = torch.load(filepath, map_location='cpu')
    print("{:5.2f}".format(ckpt['best_result'] * 100))
