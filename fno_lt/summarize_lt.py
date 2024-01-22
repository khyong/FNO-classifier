import torch

import math
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', type=str)
args = parser.parse_args()

ts_accs = [[], [], []]
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
        accs = ckpt['best_lt_result']
        print("{:5.2f} {:5.2f} {:5.2f}".format(accs[0]*100, accs[1]*100, accs[2]*100))
        many, med, few = accs[0] * 100, accs[1] * 100, accs[2] * 100
        ts_accs[0].append(many)
        ts_accs[1].append(med)
        ts_accs[2].append(few)
        print()
        continue
    ckpt = torch.load(filepath, map_location='cpu')
    accs = ckpt['best_lt_result']
    print("{:5.2f} {:5.2f} {:5.2f}".format(accs[0]*100, accs[1]*100, accs[2]*100))
    many, med, few = accs[0] * 100, accs[1] * 100, accs[2] * 100
    ts_accs[0].append(many)
    ts_accs[1].append(med)
    ts_accs[2].append(few)

for i in range(3):
    tag = 'many' if i == 0 else 'med' if i == 1 else 'few'
    print("[{}] max: {:.2f} / min: {:.2f}".format(tag, np.max(ts_accs[i]), np.min(ts_accs[i])))
    print("[{}] mean (std): {:.2f} ({:.2f})".format(tag, np.mean(ts_accs[i]), np.std(ts_accs[i])))

