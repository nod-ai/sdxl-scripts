#!/usr/bin/env python
import glob
import shutil
import re
import os

original_mlir = "base_ir/punet_07_18.mlir"
base_mlir = 'tmp/punet_flow.mlir'
new_mlir = 'tmp/punet_tk.mlir'
kernels = glob.glob("tk_kernels/bs1/*")

# BUG: Dialect resources not copied when compiling from flow. So need to add manually for now.
print("Capturing dialect resources ...")
with open(f'{original_mlir}', 'r') as f:
    original_ir = f.readlines()
dialect_resources = ['\n']
capture = False
for line in original_ir:
    if '{-#' in line:
        capture = True
    if capture:
        dialect_resources.append(line)

# Replace all calls to old kernel with new kernel
print("Inserting kernels and updating calls to kernels...")
kernel_name = {}
for kernel in kernels:
    kernel_name[kernel] = kernel.split('/')[-1].split('.')[0]
kernel_map = {}
prefix_map = {}
with open(base_mlir, 'r') as f:
    base = f.readlines()
new_base = []
for line in base:
    for kernel in kernels:
        suffix = kernel.split('.')[0].split('_')[-1]
        # Uncomment/rework when a kernel with bias comes in
        # bias_explicit = False
        # if 'bias' in suffix:
        #     bias_explicit = True
        #     kernel_args = 3 + int(suffix[4:])
        #     suffix = kernel.split('.')[0].split('_')[-2]
        B, M, N, K = suffix.split('x')
        old_kernel = f'matmul_like_{B}x{M}x{N}x{K}'
        if not old_kernel in line:
            continue
        if old_kernel in line and 'func.func' in line:
            num_args = line.count('arg')
            with open(kernel, 'r') as f:
                data = f.readlines()
            idx_with_kernel_args = [idx for idx, s in enumerate(data) if 'func.func' in s][0]
            kernel_args = data[idx_with_kernel_args].count('arg')
            if num_args != kernel_args:
                continue
            kernel_map[kernel] = line.strip().split(' ')[1][1:-7]
            prefix_map[kernel] = kernel_map[kernel].split(old_kernel)[0][:-1]
        if old_kernel in line and 'flow.dispatch' in line and not 'func.func' in line:
            line = line.replace(kernel_map[kernel], kernel_name[kernel])
            line = line.replace(prefix_map[kernel], kernel_name[kernel])
    new_base.append(line)

# Insert kernels in appropriate locations
final_ir = []
for line in new_base:
    for kernel in kernels:
        if prefix_map[kernel] + ' {' in line and 'flow.executable' in line and 'private' in line:
            with open(kernel, 'r') as f:
                data = f.readlines()
            translation_info = data[0].split('#translation = ')[1].strip()
            extract = ''.join(data[2:-2])
            extract = extract.replace('#translation', translation_info)
            final_ir += extract
    final_ir.append(line)

with open(new_mlir, 'w') as f:
    f.writelines(final_ir + dialect_resources)
