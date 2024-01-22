import torch

import math
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', type=str)
args = parser.parse_args()

ts_accs = []
num_seeds = 20
cnt = 0
for filepath in sorted(Path(args.dirpath).glob('**/models/best_model.pth')):
    cnt += 1
    if cnt == 1:
        print("=" * 80)
        print(filepath)
    if cnt % num_seeds == 0:
        cnt = 0
        ckpt = torch.load(filepath, map_location='cpu')
        print("{:5.2f}".format(ckpt['best_result'] * 100))
        ts_accs.append(ckpt['best_result'] * 100)
        print()
        continue
    ckpt = torch.load(filepath, map_location='cpu')
    print("{:5.2f}".format(ckpt['best_result'] * 100), end=',')
    ts_accs.append(ckpt['best_result'] * 100)

print("max: {:.2f} / min: {:.2f}".format(np.max(ts_accs), np.min(ts_accs)))
print("mean (std): {:.2f} ({:.2f})".format(np.mean(ts_accs), np.std(ts_accs)))

