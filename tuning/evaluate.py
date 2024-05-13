#! /usr/bin/env python3

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+')
args = parser.parse_args()
inputs = args.input

def load_npy(path):
  with open(path, 'rb') as f:
    return np.load(f)

loaded = list(load_npy(i) for i in inputs)
first = loaded[0]
M, N = first.shape

max_diff = np.max([np.max(np.abs(x - first)) for x in loaded[1:]])
print(f'Max error: {max_diff}')

diffs_printed = 0
for r in range(M):
  for c in range(N):
    elems = list(x[r, c] for x in loaded)
    diff = np.max(elems) - np.min(elems)
    if diff > max_diff / 2:
      print(f'Difference: {diff}, at element [{r}, {c}]: {elems}')
      diffs_printed += 1
      if diffs_printed >= 32:
        print('...')
        exit(1)
