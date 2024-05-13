#! /usr/bin/env python3
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('shape', type=str, help='MxNxK')
args = parser.parse_args()

shape = str(args.shape).split('x')
assert len(shape) == 3

M, N, K = int(shape[0]), int(shape[1]), int(shape[2])

np.random.seed(0)

a = np.random.randint(0, 11, size=(M, K)) / (2 * 10.0)
b = np.random.randint(0, 11, size=(N, K)) / (2 * 10.0)
a = np.float16(a)
b = np.float16(b)
c = np.matmul(a, b.T, dtype=np.float32)
d = np.random.randint(0, 11, size=(N,)) / (2 * 10.0)
d = np.float32(d)
e = np.random.randint(0, 11, size=(M, N)) / (2 * 10.0)
e = np.float16(e)

res = np.zeros((M, N))
for row in range(M):
    for col in range(N):
        res[row, col] = c[row, col] + d[col] + e[row, col]
res = np.float16(res)

# Write matrices
with open(f'matrix_a.npy', 'wb') as f:
    np.save(f, a)
with open(f'matrix_b.npy', 'wb') as f:
    np.save(f, b)
with open(f'matrix_c.npy', 'wb') as f:
    np.save(f, c)
with open(f'matrix_d.npy', 'wb') as f:
    np.save(f, d)
with open(f'matrix_e.npy', 'wb') as f:
    np.save(f, e)
with open(f'matrix_res.npy', 'wb') as f:
    np.save(f, res)
