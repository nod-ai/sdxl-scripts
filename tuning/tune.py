#!/usr/bin/env python3

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import argparse
import logging
import re
import z3
from dataclasses import asdict, dataclass
from os import mkdir, path
from textwrap import indent

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Configuration:
    subgroup_size : int
    workgroup_size : list[int]
    intrinsic : str
    tile_sizes : list[int]
    subgroup_m_count : int
    subgroup_n_count : int
    waves_per_eu : int
    no_workgroup_reorder : int

def read_input_mlir(filename):
    template = f''
    with open(filename, 'r') as f:
        template = f.readlines()
    return template

def is_mmt(lines) -> bool:
    return any('func.func' in line and 'matmul_transpose_b' in line for line in lines)

def is_conv(lines) -> bool:
    return any('func.func' in line and 'conv_2d_nhwc_hwcf' in line for line in lines)

def is_contract(lines) -> bool:
    return any('func.func' in line and 'matmul_like' in line for line in lines)

def is_batch_matmul(lines) -> bool:
    return any('func.func' in line and 'batch_matmul' in line for line in lines)

def get_mmt_tile_sizes(configuration: Configuration):
    return configuration.tile_sizes

def get_conv_tile_sizes(configuration: Configuration):
    m, n, k = configuration.tile_sizes
    batch = 1
    fh = 1
    fw = 1

    oh = 1

    oc = n
    ow = m
    ic = k
    return batch, oh, ow, oc, fh, fw, ic

def get_contract_tile_sizes(configuration: Configuration, tile_dims):
    m, n, k = configuration.tile_sizes
    tile_size = [1] * len(tile_dims)
    for idx, dim in enumerate(tile_dims):
        if dim == 'm':
            tile_size[idx] = m
        if dim == 'n':
            tile_size[idx] = n
        if dim == 'k':
            tile_size[idx] = k
    return tile_size

def get_batch_matmul_tile_sizes(configuration: Configuration):
    m, n, k = configuration.tile_sizes
    return 1, m, n, k

def get_pipeline_config(configuration : Configuration) -> str:
    extra_config = ''
    if configuration.no_workgroup_reorder == 1:
        extra_config += ', no_reorder_workgroups'
    if configuration.waves_per_eu != 2:
        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
    return extra_config

def get_transform_function_mmt(filename: str, configuration: Configuration):
    tile_sizes = ", ".join(map(str, get_mmt_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f'''
transform.named_sequence @{filename}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<{M}x{K}xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<{N}x{K}xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
    translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
      workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
      {{mma_schedule = #iree_gpu.mma_schedule<
         intrinsic = {configuration.intrinsic},
         subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
       {extra_config}}}>
    > -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}}
'''

# int64_t n = outputShape[0];
# int64_t oh = outputShape[1];
# int64_t ow = outputShape[2];
# int64_t oc = outputShape[3];
# int64_t fh = filterShape[0];
# int64_t fw = filterShape[1];
# int64_t ic = filterShape[2];
def get_transform_function_conv(N, OH, OW, OC, FH, FW, IC, configuration: Configuration):
    wg_x, wg_y, wg_z = configuration.workgroup_size

    input = f"tensor<{N}x?x?x{IC}xf16>"
    filter = f"tensor<{FH}x{FW}x{IC}x{OC}xf16>"
    output = f"tensor<{N}x{OH}x{OW}x{OC}xf32>"

    tile_sizes = ", ".join(map(str, get_conv_tile_sizes(configuration)))

    return f'''
transform.named_sequence @match_conv_2d_nhwc_hwcf_{N}x{OH}x{OW}x{OC}x{FH}x{FW}x{IC}(%conv: !transform.any_op {{transform.readonly}})
  -> (!transform.any_op, !transform.any_param) {{
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {{
  ^bb0(%lhs: {input}, %rhs: {filter}, %out: {output}):
    %13 = linalg.conv_2d_nhwc_hwcf {{ dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }}
      ins(%lhs, %rhs : {input}, {filter})
      outs(%out : {output}) -> {output}
  }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
  translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute,
    {{mma_schedule = #iree_gpu.mma_schedule<
      intrinsic = {configuration.intrinsic},
      subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count},
      subgroup_m_tile_count = {configuration.subgroup_m_tile_count},
      subgroup_n_tile_count = {configuration.subgroup_n_tile_count},
      subgroup_k_tile_count = {configuration.subgroup_k_tile_count}>}}>,
    workgroup_size = [{wg_x}, {wg_y}, {wg_z}], subgroup_size = {configuration.subgroup_size}> -> !transform.any_param
  transform.yield %conv, %config : !transform.any_op, !transform.any_param
}}
     '''

def apply_params_mmt(M, N, K, template, configuration: Configuration):
    print(configuration)
    extra_config = get_pipeline_config(configuration)
    expr0 = re.compile(r'<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>')
    expr1 = re.compile(r'LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),')
    expr2 = re.compile(r'tile_sizes = \[\[([0-9]+), ([0-9]+), ([0-9]+)\]\]')
    expr3 = re.compile(r', waves_per_eu = ([0-9]) : i64')
    repl0 = f'<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>{extra_config}'
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, get_mmt_tile_sizes(configuration)))}]]'
    repl3 = f', waves_per_eu = {config.waves_per_eu} : i64'

    modified = indent(get_transform_function_mmt(f'match_mmt_{M}x{N}x{K}', configuration), '//   ')
    for line in template:
        if 'intrinsic =' in line:
            line = re.sub(expr0, repl0, line)
        if 'LLVMGPUVectorDistribute ' in line:
            line = re.sub(expr1, repl1, line)
        if 'tile_sizes' in line:
            line = re.sub(expr2, repl2, line)
        if 'waves_per_eu' in line:
            line = re.sub(expr3, repl3, line)
        modified += line


    embeddable = indent(get_transform_function_mmt(f'match_mmt', configuration), '  ')
    return modified, embeddable

def apply_params_conv(N, OH, OW, OC, FH, FW, IC, template, configuration):
    params = asdict(configuration)
    keys = list(params.keys())
    expr = re.compile(r'intrinsic = #iree_gpu.mfma_layout<([0-9A-Za-z_]+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+), subgroup_m_tile_count = ([0-9]+), subgroup_n_tile_count = ([0-9]+), subgroup_k_tile_count = ([0-9]+)>}>, workgroup_size = \[([0-9]+) : index, ([0-9]+) : index, ([0-9]+) : index\]')
    expr2 = re.compile(r'tile_sizes = \[\[[0-9,\s]+\]\]')
    repl = ''
    for i, key in enumerate(keys):
        if 'workgroup_size' in key or 'tile_sizes' in key or 'subgroup_size' in key:
            continue
        if len(repl) != 0:
            repl += ', '
        repl += f'{key} = {params[key]}'

    repl += '>}>, workgroup_size = ['
    for i, val in enumerate(params['workgroup_size']):
        repl += f'{val} : index'
        if i != len(params['workgroup_size']) - 1:
            repl += ', '
    repl += ']'

    repl2 = f'tile_sizes = [[{", ".join(map(str, get_conv_tile_sizes(configuration)))}]]'

    #print("Repl: ", repl)
    #print("Repl2: ", repl2)
    modified = indent(get_transform_function_conv(N, OH, OW, OC, FH, FW, IC, configuration), '//   ')
    for line in template:
        if 'intrinsic' in line:
            line = re.sub(expr, repl, line)
        if 'tile_sizes' in line:
            line = re.sub(expr2, repl2, line)
        modified += line

    return modified

def apply_params_contract(LHS, RHS, RES, tile_dims, template, configuration):
    print(configuration)
    extra_config = get_pipeline_config(configuration)
    expr0 = re.compile(r'<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>')
    expr1 = re.compile(r'LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),')
    expr2 = re.compile(r'tile_sizes = \[\[(([0-9]+), )+([0-9]+)\]\]')
    expr3 = re.compile(r', waves_per_eu = ([0-9]) : i64')
    repl0 = f'<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>{extra_config}'
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, get_contract_tile_sizes(configuration, tile_dims)))}]]'
    repl3 = f', waves_per_eu = {config.waves_per_eu} : i64'

    modified = ''  # indent(get_transform_function_mmt(f'match_mmt_{M}x{N}x{K}', configuration), '//   ')
    for line in template:
        if 'intrinsic =' in line:
            line = re.sub(expr0, repl0, line)
        if 'LLVMGPUVectorDistribute ' in line:
            line = re.sub(expr1, repl1, line)
        if 'tile_sizes' in line:
            line = re.sub(expr2, repl2, line)
        if 'waves_per_eu' in line:
            line = re.sub(expr3, repl3, line)
        modified += line

    return modified

def apply_params_batch_matmul(B, M, N, K, template, configuration):
    params = asdict(configuration)
    keys = list(params.keys())
    expr = re.compile(r'intrinsic = #iree_gpu.mfma_layout<([0-9A-Za-z_]+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+), subgroup_m_tile_count = ([0-9]+), subgroup_n_tile_count = ([0-9]+), subgroup_k_tile_count = ([0-9]+)>}>, workgroup_size = \[([0-9]+) : index, ([0-9]+) : index, ([0-9]+) : index\]')
    expr2 = re.compile(r'tile_sizes = \[\[[0-9,\s]+\]\]')
    repl = ''
    for i, key in enumerate(keys):
        if 'workgroup_size' in key or 'tile_sizes' in key or 'subgroup_size' in key:
            continue
        if len(repl) != 0:
            repl += ', '
        repl += f'{key} = {params[key]}'

    repl += '>}>, workgroup_size = ['
    for i, val in enumerate(params['workgroup_size']):
        repl += f'{val} : index'
        if i != len(params['workgroup_size']) - 1:
            repl += ', '
    repl += ']'

    repl2 = f'tile_sizes = [[{", ".join(map(str, get_batch_matmul_tile_sizes(configuration)))}]]'

    #print("Repl: ", repl)
    #print("Repl2: ", repl2)
    #modified = indent(get_transform_function_mmt(M, N, K, configuration), '//   ')
    modified = f'// Configuration: {configuration}\n'
    for line in template:
        if 'intrinsic' in line:
            line = re.sub(expr, repl, line)
        if 'tile_sizes' in line:
            line = re.sub(expr2, repl2, line)
        modified += line

    return modified

def get_shape_dims(shape_str):
    return [int(x) for x in shape_str.split('x')[:-1]]

def get_shapes_mmt(template):
    for line in template:
        if 'linalg.generic' not in line:
            continue
        if r'iterator_types = ["parallel", "parallel", "reduction"]' not in line:
            continue
        # ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>)
        mmt_re = r'ins\(.+tensor<([0-9]+)x([0-9]+)xf16>, tensor<([0-9]+)x([0-9]+)xf16>\).+outs'
        shape = re.search(mmt_re, line)
        if shape is None:
            continue

        assert len(shape.groups()) == 4
        M, K0, N, K1 = shape.groups()
        assert K0 == K1
        return int(M), int(N), int(K0)

    assert False, "Shape not found"

def get_shapes_conv(template):
    for line in template:
        if 'linalg.conv_2d_nhwc_hwcf' not in line:
            continue
        # ins(%19, %20 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs (%27 : tensor<2x32x32x1280xf32>)
        ins_re = r'ins\(.+:\s*tensor<([0-9xf]+)>,\s*tensor<([0-9xf]+)>\)'
        ins_shape = re.search(ins_re, line)
        if ins_shape is None:
            continue
        logging.debug(f'ins: {ins_shape.groups()}')

        outs_re = r'outs\(.+:\s*tensor<([0-9xf]+)>\)'
        outs_shape = re.search(outs_re, line)
        assert outs_shape is not None
        logging.debug(f'outs: {outs_shape.groups()}')

        assert len(ins_shape.groups()) == 2
        assert len(outs_shape.groups()) == 1

        in0_dims = get_shape_dims(ins_shape.groups()[0])
        in1_dims = get_shape_dims(ins_shape.groups()[1])
        out_dims = get_shape_dims(outs_shape.groups()[0])

        assert len(in0_dims) == 4
        assert len(in1_dims) == 4
        assert len(out_dims) == 4

        # int64_t n = outputShape[0];
        # int64_t oh = outputShape[1];
        # int64_t ow = outputShape[2];
        # int64_t oc = outputShape[3];
        # int64_t fh = filterShape[0];
        # int64_t fw = filterShape[1];
        # int64_t ic = filterShape[2];
        n, oh, ow, oc = out_dims
        fh, fw, ic, _ = in1_dims
        return n, oh, ow, oc, fh, fw, ic

    assert False, "Shape not found"

def get_shapes_contract(template):
    for line in template:
        if 'linalg.generic' not in line:
            continue
        if 'lowering_config =' not in line:
            continue
        if '"reduction"' not in line:
            continue

        # ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
        tensor_re = r'tensor<([0-9xf]+)>'
        ins_re = rf'ins\(.+:\s*{tensor_re},\s*{tensor_re}\)'
        ins_shape = re.search(ins_re, line)
        if ins_shape is None:
            continue
        logging.debug(f'ins: {ins_shape.groups()}')

        # outs(%11 : tensor<2x20x1024x64xf32>)
        outs_re = rf'outs\(.+:\s*{tensor_re}\)'
        outs_shape = re.search(outs_re, line)
        assert outs_shape is not None
        logging.debug(f'outs: {outs_shape.groups()}')

        assert len(ins_shape.groups()) == 2
        assert len(outs_shape.groups()) == 1

        in0_dims = get_shape_dims(ins_shape.groups()[0])
        in1_dims = get_shape_dims(ins_shape.groups()[1])
        out_dims = get_shape_dims(outs_shape.groups()[0])

        return in0_dims, in1_dims, out_dims

    assert False, "Shape not found"

def get_shapes_batch_matmul(template):
    for line in template:
        if 'linalg.batch_matmul' not in line:
            continue
        # ins(%9, %10 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>)
        # outs(%12 : tensor<64x72x1280xf32>)
        tensor_re = r'tensor<([0-9xf]+)>'
        ins_re = rf'ins\(.+:\s*{tensor_re},\s*{tensor_re}\)'
        ins_shape = re.search(ins_re, line)
        if ins_shape is None:
            continue
        logging.debug(f'ins: {ins_shape.groups()}')

        # outs(%11 : tensor<2x20x1024x64xf32>)
        outs_re = rf'outs\(.+:\s*{tensor_re}\)'
        outs_shape = re.search(outs_re, line)
        assert outs_shape is not None
        logging.debug(f'outs: {outs_shape.groups()}')

        assert len(ins_shape.groups()) == 2
        assert len(outs_shape.groups()) == 1

        in0_dims = get_shape_dims(ins_shape.groups()[0])
        in1_dims = get_shape_dims(ins_shape.groups()[1])
        out_dims = get_shape_dims(outs_shape.groups()[0])

        assert len(in0_dims) == 3
        assert len(in1_dims) == 3
        assert len(out_dims) == 3
        assert in0_dims[0] == in1_dims[0] == out_dims[0]
        assert in0_dims[1] ==  out_dims[1]
        assert in1_dims[1] ==  out_dims[2]
        assert in0_dims[2] ==  in1_dims[2]

        return in0_dims[0], in0_dims[1], in1_dims[1], in0_dims[2]

    assert False, "Shape not found"

def is_pow2(x, min, max):
    return z3.Or(list(x == 2 ** i for i in range(min, max + 1)))

def is_not_pow2(x, min, max):
    return z3.And(list(x != 2 ** i for i in range(min, max + 1)))

def generate_constraints(problem_size, tile_sizes, subgroup_size, intrinsic_size, workgroup_size,
                         subgroup_m_count, subgroup_n_count,
                         waves_per_eu, no_workgroup_reorder):
    M, N, K = problem_size
    m, n, k = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    constraints = [subgroup_size == 64, wg_x * wg_y * wg_z <= 1024]
    constraints += [z3.Or(z3.And(intrinsic_mn == 16, intrinsic_k == 16),
                          z3.And(intrinsic_mn == 32, intrinsic_k == 8))]
    subgroup_k_count = 1
    constraints += [m >= intrinsic_mn, m <= 512, m <= M,] # , M == m * z3.FreshInt()]
    constraints += [n >= intrinsic_mn, n <= 512, n <= N, N == n * z3.FreshInt()]
    constraints += [k >= intrinsic_k, k <= 512, k <= K, K == k * z3.FreshInt()]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    subgroup_m_tile_count = z3.Int('sg_m_tcnt')
    subgroup_n_tile_count = z3.Int('sg_n_tcnt')
    subgroup_k_tile_count = z3.Int('sg_k_tcnt')
    for x in (subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
        constraints += [x >= 1, x <= 32]

    constraints += [m == subgroup_m_count * subgroup_m_tile_count * intrinsic_mn]
    constraints += [n == subgroup_n_count * subgroup_n_tile_count * intrinsic_mn]
    constraints += [k == subgroup_k_count * subgroup_k_tile_count * intrinsic_k]
    constraints += [wg_x == subgroup_size * subgroup_n_count]
    constraints += [wg_y == subgroup_m_count]
    constraints += [wg_z == subgroup_k_count]
    constraints += [wg_x <= m, wg_x <= n]
    constraints += [k == intrinsic_mn * z3.FreshInt()]
    constraints += [k * n % (wg_x * wg_y * wg_z) == 0]
    constraints += [k * m % (wg_x * wg_y * wg_z) == 0]
    constraints += [subgroup_m_count * subgroup_n_count == 4]

    constraints += [z3.Or(waves_per_eu == 1, waves_per_eu == 2, waves_per_eu == 4)]
    constraints += [no_workgroup_reorder >= 0, no_workgroup_reorder <= 1]

    return constraints

def generate_solutions(M, N, K):
    print(M, N, K)
    m, n, k = z3.Int('m'), z3.Int('n'), z3.Int('k')
    subgroup_size = z3.Int('subgroup_size')
    intrinsic_mn = z3.Int('intrinsic_mn')
    intrinsic_k = z3.Int('intrinsic_k')
    wg_x, wg_y, wg_z = z3.Int('wg_x'), z3.Int('wg_y'), z3.Int('wg_z')
    sg_m_cnt = z3.Int('sg_m_cnt')
    sg_n_cnt = z3.Int('sg_n_cnt')
    waves_per_eu = z3.Int('waves_per_eu')
    no_workgroup_reorder = z3.Int('no_workgroup_reorder')
    all_vars = [m, n, k, subgroup_size, intrinsic_mn, intrinsic_k, wg_x, wg_y, wg_z,
                sg_m_cnt, sg_n_cnt, waves_per_eu, no_workgroup_reorder]

    solver = z3.Solver()
    constraints = generate_constraints([M, N, K], [m, n, k], subgroup_size, [intrinsic_mn, intrinsic_k],
                                       [wg_x, wg_y, wg_z], sg_m_cnt, sg_n_cnt,
                                       waves_per_eu, no_workgroup_reorder)
    solver.add(z3.simplify(z3.And(constraints)))
    logging.debug(f'Initial constraints: {solver}')
    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()

        config = Configuration(lookup(subgroup_size), [lookup(wg_x), lookup(wg_y), lookup(wg_z)],
                               f'#iree_gpu.mma_layout<MFMA_F16_{lookup(intrinsic_mn)}x{lookup(intrinsic_mn)}x{lookup(intrinsic_k)}_F32>',
                               [lookup(m), lookup(n), lookup(k)], lookup(sg_m_cnt), lookup(sg_n_cnt),
                               lookup(waves_per_eu), lookup(no_workgroup_reorder))
        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1
        yield config

def product(vals):
    res = 1
    for val in vals:
        res *= val
    return res

def get_default_output_dir():
    from datetime import datetime
    return 'tuning_' + datetime.now().strftime('%Y_%m_%d_%H_%M')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input mlir file', type=str)
    parser.add_argument('-o', '--output', help='Output dir', type=str, default=get_default_output_dir())
    parser.add_argument('-l', '--limit', help='Max number of candidates generated', type=int, default=4096)
    parser.add_argument('--lhs-dims', help='Map of LHS matmul dims', type=str, default="mk")
    parser.add_argument('--rhs-dims', help='Map of RHS matmul dims', type=str, default="nk")
    parser.add_argument('--tile-dims', help='Map of tile size matmul dims', type=str, default="mnk")

    args = parser.parse_args()

    input_file = str(args.input)
    mkdir(str(args.output))
    logging.debug(f'Output directory {args.output}')
    logging.debug(f'Processing {input_file}')
    mlir_template = read_input_mlir(input_file)

    detected_mmt = is_mmt(mlir_template)
    detected_conv = is_conv(mlir_template)
    detected_contract = is_contract(mlir_template)
    detected_batch_matmul = is_batch_matmul(mlir_template)
    assert [detected_mmt, detected_conv, detected_contract, detected_batch_matmul].count(True) == 1

    # Save the input file as the first candidate.
    with open(path.join(args.output, f'0.mlir') , 'w') as f:
        f.write(''.join(mlir_template))

    if detected_mmt:
        M, N, K = get_shapes_mmt(mlir_template)
        logging.debug(f'Matmul shape: [{M}, {N}, {K}]')

        for i, config in enumerate(generate_solutions(M, N, K)):
            if i >= args.limit:
                break
            print(f'Solution #{i+1}: {config}')
            new_mlir, embeddable_tuning = apply_params_mmt(M, N, K, mlir_template, config)

            with open(path.join(args.output, f'{i+1}.mlir') , 'w') as f:
                f.write(new_mlir)
            with open(path.join(args.output, f'{i+1}_config.mlir') , 'w') as f:
                f.write(embeddable_tuning)
    elif detected_conv:
        N, OH, OW, OC, FH, FW, IC = get_shapes_conv(mlir_template)
        logging.debug(f'Conv shape: [n{N}, oh{OH}, oc{OC}, fh{FH}, fw{FW}, ic{IC}]')
        M = OH * OW
        N = OC
        K = FH * FW * IC
        logging.debug(f'Equivalent matmul shape: [{M}, {N}, {K}]')

        for i, config in enumerate(generate_solutions(M, N, K)):
            if i >= args.limit:
                break
            #print(f'Solution #{i+1}: {config}')
            new_mlir = apply_params_conv(N, OH, OW, OC, FH, FW, IC, mlir_template, config)

            with open(path.join(args.output, f'{i+1}.mlir') , 'w') as f:
                f.write(new_mlir)
    elif detected_contract:
        LHS, RHS, RES = get_shapes_contract(mlir_template)
        logging.debug(f'Contract shape: ({LHS}, {RHS}) -> {RES}')
        lhs_dims = args.lhs_dims
        rhs_dims = args.rhs_dims
        tile_dims = args.tile_dims
        assert len(LHS) == len(lhs_dims)
        assert len(RHS) == len(rhs_dims)
        M = product(val if dim == 'm' else 1 for dim, val in zip(lhs_dims, LHS))
        N = product(val if dim == 'n' else 1 for dim, val in zip(rhs_dims, RHS))
        K0 = product(val if dim == 'k' else 1 for dim, val in zip(lhs_dims, LHS))
        K1 = product(val if dim == 'k' else 1 for dim, val in zip(rhs_dims, RHS))
        assert K0 == K1
        logging.debug(f'Equivalent matmul shape: [{M}, {N}, {K0}]')
        for i, config in enumerate(generate_solutions(M, N, K0)):
            if i >= args.limit:
                break
            #print(f'Solution #{i+1}: {config}')
            new_mlir = apply_params_contract(LHS, RHS, RES, tile_dims, mlir_template, config)

            with open(path.join(args.output, f'{i+1}.mlir') , 'w') as f:
                f.write(new_mlir)
    elif detected_batch_matmul:
        B, M, N, K = get_shapes_batch_matmul(mlir_template)
        logging.debug(f'Batch Matmul shape: {B}x[{M}, {N}, {K}]')

        for i, config in enumerate(generate_solutions(M, N, K)):
            if i >= args.limit:
                break
            #print(f'Solution #{i+1}: {config}')
            new_mlir = apply_params_batch_matmul(B, M, N, K, mlir_template, config)

            with open(path.join(args.output, f'{i+1}.mlir') , 'w') as f:
                f.write(new_mlir)
    else:
        assert False
