#!/usr/bin/env python3

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import argparse
import logging
import re
import z3
from dataclasses import asdict, dataclass
from os import mkdir, path, makedirs
from textwrap import indent
from iree.compiler import ir
import iree.compiler as ireec
from iree.compiler.dialects import _linalg_ops_gen, _util_ops_gen
from enum import Enum

"""
Usage: ./tune.py 121.mlir -o "tuning/candidates" -l 1024 --lhs-dims=mk --rhs-dims=nk --tile-dims=mnk
"""

tune_logger = logging.getLogger("tune")


@dataclass
class Configuration:
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: str
    tile_sizes: list[int]
    subgroup_m_count: int
    subgroup_n_count: int
    waves_per_eu: int


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    contraction = 3
    batch_matmul = 4


@dataclass
class OpWalkResult:
    was_interrupted: bool = False
    dispatch_kind: DispatchKind = None


@dataclass
class ProblemSize:
    M: int
    N: int
    K: int
    lhs_bw: int
    rhs_bw: int
    out_bw: int
    dispatch_kind: DispatchKind


def read_input_mlir(filename):
    template = f""
    with open(filename, "r") as f:
        template = f.readlines()
    return template


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
        if dim == "m":
            tile_size[idx] = m
        if dim == "n":
            tile_size[idx] = n
        if dim == "k":
            tile_size[idx] = k
    return tile_size


def get_pipeline_config(configuration: Configuration) -> str:
    extra_config = ""
    if configuration.waves_per_eu != 2:
        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
    return extra_config


def get_transform_function_mmt(
    M, N, K, functionName: str, configuration: Configuration
):
    tile_sizes = ", ".join(map(str, get_mmt_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
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
"""


# int64_t n = outputShape[0];
# int64_t oh = outputShape[1];
# int64_t ow = outputShape[2];
# int64_t oc = outputShape[3];
# int64_t fh = filterShape[0];
# int64_t fw = filterShape[1];
# int64_t ic = filterShape[2];
def get_transform_function_conv(
    N, OH, OW, OC, FH, FW, IC, functionName: str, configuration: Configuration
):
    input = f"tensor<{N}x?x?x{IC}xf16>"
    filter = f"tensor<{FH}x{FW}x{IC}x{OC}xf16>"
    output = f"tensor<{N}x{OH}x{OW}x{OC}xf32>"

    tile_sizes = ", ".join(map(str, get_conv_tile_sizes(configuration)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%conv: !transform.any_op {{transform.readonly}})
  -> (!transform.any_op, !transform.any_param) {{
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {{
  ^bb0(%lhs: {input}, %rhs: {filter}, %out: {output}):
    %13 = linalg.conv_2d_nhwc_hwcf {{ dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64> }}
      ins(%lhs, %rhs : {input}, {filter})
      outs(%out : {output}) -> {output}
  }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
       workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
        {{mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = {configuration.intrinsic},
            subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
        {extra_config}}}>
    > -> !transform.any_param
  transform.yield %conv, %config : !transform.any_op, !transform.any_param
}}
"""


def get_transform_function_batch_matmul(
    LHS, RHS, RES, tile_dims, functionName: str, configuration: Configuration
):
    input0 = f"tensor<{'x'.join(map(str, LHS))}xf16>"
    input1 = f"tensor<{'x'.join(map(str, RHS))}xf16>"
    output = f"tensor<{'x'.join(map(str, RES))}xf32>"

    tile_sizes = ", ".join(map(str, get_contract_tile_sizes(configuration, tile_dims)))

    wg_x, wg_y, wg_z = configuration.workgroup_size
    extra_config = get_pipeline_config(configuration)

    return f"""
transform.named_sequence @{functionName}(%batch_matmul: !transform.any_op {{transform.readonly}})
  -> (!transform.any_op, !transform.any_param) {{
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {{
  ^bb0(%lhs: {input0}, %rhs: {input1}, %out: {output}):
    %13 = linalg.batch_matmul
      ins(%lhs, %rhs : {input0}, {input1})
      outs(%out : {output}) -> {output}
  }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %config = transform.param.constant #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
      translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
       workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
        {{mma_schedule = #iree_gpu.mma_schedule<
            intrinsic = {configuration.intrinsic},
            subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>
        {extra_config}}}>
    > -> !transform.any_param
  transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
}}
"""


def apply_params_mmt(M, N, K, template, configuration: Configuration):
    tune_logger.info(f"{configuration}")
    extra_config = get_pipeline_config(configuration)
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"tile_sizes = \[\[([0-9]+), ([0-9]+), ([0-9]+)\]\]")
    expr3 = re.compile(r", waves_per_eu = ([0-9]) : i64")
    repl0 = f"<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>{extra_config}"
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, get_mmt_tile_sizes(configuration)))}]]'
    repl3 = f", waves_per_eu = {configuration.waves_per_eu} : i64"

    modified = indent(
        get_transform_function_mmt(M, N, K, f"match_mmt_{M}x{N}x{K}", configuration),
        "//   ",
    )
    for line in template:
        if "intrinsic =" in line:
            line = re.sub(expr0, repl0, line)
        if "LLVMGPUVectorDistribute " in line:
            line = re.sub(expr1, repl1, line)
        if "tile_sizes" in line:
            line = re.sub(expr2, repl2, line)
        if "waves_per_eu" in line:
            line = re.sub(expr3, repl3, line)
        modified += line

    embeddable = indent(
        get_transform_function_mmt(M, N, K, f"match_op", configuration), "  "
    )
    return modified, embeddable


def apply_params_conv(
    N, OH, OW, OC, FH, FW, IC, template, configuration: Configuration
):
    tune_logger.info(f"{configuration}")
    extra_config = get_pipeline_config(configuration)
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")
    expr3 = re.compile(r", waves_per_eu = ([0-9]) : i64")
    repl0 = f"<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>{extra_config}"
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = (
        f'tile_sizes = [[{", ".join(map(str, get_conv_tile_sizes(configuration)))}]]'
    )
    repl3 = f", waves_per_eu = {configuration.waves_per_eu} : i64"

    modified = indent(
        get_transform_function_conv(
            N,
            OH,
            OW,
            OC,
            FH,
            FW,
            IC,
            f"match_conv_2d_nhwc_hwcf_{N}x{OH}x{OW}x{OC}x{FH}x{FW}x{IC}",
            configuration,
        ),
        "//   ",
    )
    for line in template:
        if "intrinsic =" in line:
            line = re.sub(expr0, repl0, line)
        if "LLVMGPUVectorDistribute " in line:
            line = re.sub(expr1, repl1, line)
        if "tile_sizes" in line:
            line = re.sub(expr2, repl2, line)
        if "waves_per_eu" in line:
            line = re.sub(expr3, repl3, line)
        modified += line

    embeddable = indent(
        get_transform_function_conv(
            N, OH, OW, OC, FH, FW, IC, f"match_op", configuration
        ),
        "  ",
    )
    return modified, embeddable


def apply_params_contract(
    LHS, RHS, RES, tile_dims, template, configuration: Configuration
):
    tune_logger.info(f"{configuration}")
    extra_config = get_pipeline_config(configuration)
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"tile_sizes = \[\[(([0-9]+), )+([0-9]+)\]\]")
    expr3 = re.compile(r", waves_per_eu = ([0-9]) : i64")
    repl0 = f"<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>{extra_config}"
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, get_contract_tile_sizes(configuration, tile_dims)))}]]'
    repl3 = f", waves_per_eu = {configuration.waves_per_eu} : i64"

    modified = ""  # indent(get_transform_function_mmt(f'match_mmt_{M}x{N}x{K}', configuration), '//   ', M, N, K)
    for line in template:
        if "intrinsic =" in line:
            line = re.sub(expr0, repl0, line)
        if "LLVMGPUVectorDistribute " in line:
            line = re.sub(expr1, repl1, line)
        if "tile_sizes" in line:
            line = re.sub(expr2, repl2, line)
        if "waves_per_eu" in line:
            line = re.sub(expr3, repl3, line)
        modified += line

    return modified


def apply_params_batch_matmul(
    LHS, RHS, RES, B, M, N, K, tile_dims, template, configuration: Configuration
):
    tune_logger.info(f"{configuration}")
    extra_config = get_pipeline_config(configuration)
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUPandAndVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"tile_sizes = \[\[(([0-9]+), )+([0-9]+)\]\]")
    expr3 = re.compile(r", waves_per_eu = ([0-9]) : i64")
    repl0 = f"<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>{extra_config}"
    repl1 = f'LLVMGPUPadAndVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
    repl2 = f'tile_sizes = [[{", ".join(map(str, get_contract_tile_sizes(configuration, tile_dims)))}]]'
    repl3 = f", waves_per_eu = {configuration.waves_per_eu} : i64"

    modified = indent(
        get_transform_function_batch_matmul(
            LHS,
            RHS,
            RES,
            tile_dims,
            f"match_batch_matmul_{B}x{M}x{N}x{K}",
            configuration,
        ),
        "//   ",
    )
    for line in template:
        if "intrinsic =" in line:
            line = re.sub(expr0, repl0, line)
        if "LLVMGPUPadAndVectorDistribute " in line:
            line = re.sub(expr1, repl1, line)
        if "tile_sizes" in line:
            line = re.sub(expr2, repl2, line)
        if "waves_per_eu" in line:
            line = re.sub(expr3, repl3, line)
        modified += line

    embeddable = indent(
        get_transform_function_batch_matmul(
            LHS, RHS, RES, tile_dims, f"match_op", configuration
        ),
        "  ",
    )
    return modified, embeddable


def get_shape_dims(shape_str):
    return [int(x) for x in shape_str.split("x")[:-1]]


def get_bit_width() -> tuple[int, int, int]:
    lhs_type_bit_width = -1
    rhs_type_bit_width = 16
    output_type_bit_width = -1
    return (lhs_type_bit_width, rhs_type_bit_width, output_type_bit_width)


def get_shapes_mmt(template):
    for line in template:
        if "linalg.generic" not in line:
            continue
        if r'iterator_types = ["parallel", "parallel", "reduction"]' not in line:
            continue
        # ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>)
        mmt_re = r"ins\(.+tensor<([0-9]+)x([0-9]+)xf16>, tensor<([0-9]+)x([0-9]+)xf16>\).+outs"
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
        if "linalg.conv_2d_nhwc_hwcf" not in line:
            continue
        # ins(%19, %20 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs (%27 : tensor<2x32x32x1280xf32>)
        ins_re = r"ins\(.+:\s*tensor<([0-9xf]+)>,\s*tensor<([0-9xf]+)>\)"
        ins_shape = re.search(ins_re, line)
        if ins_shape is None:
            continue
        tune_logger.debug(f"ins: {ins_shape.groups()}")

        outs_re = r"outs\(.+:\s*tensor<([0-9xf]+)>\)"
        outs_shape = re.search(outs_re, line)
        assert outs_shape is not None
        tune_logger.debug(f"outs: {outs_shape.groups()}")

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
        if "linalg.generic" not in line:
            continue
        if "lowering_config =" not in line:
            continue
        if '"reduction"' not in line:
            continue

        # ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
        tensor_re = r"tensor<([0-9xf]+)>"
        ins_re = rf"ins\(.+:\s*{tensor_re},\s*{tensor_re}\)"
        ins_shape = re.search(ins_re, line)
        if ins_shape is None:
            continue
        tune_logger.debug(f"ins: {ins_shape.groups()}")

        # outs(%11 : tensor<2x20x1024x64xf32>)
        outs_re = rf"outs\(.+:\s*{tensor_re}\)"
        outs_shape = re.search(outs_re, line)
        assert outs_shape is not None
        tune_logger.debug(f"outs: {outs_shape.groups()}")

        assert len(ins_shape.groups()) == 2
        assert len(outs_shape.groups()) == 1

        in0_dims = get_shape_dims(ins_shape.groups()[0])
        in1_dims = get_shape_dims(ins_shape.groups()[1])
        out_dims = get_shape_dims(outs_shape.groups()[0])

        return in0_dims, in1_dims, out_dims

    assert False, "Shape not found"


def get_shapes_batch_matmul(template):
    for line in template:
        if "linalg.batch_matmul" not in line:
            continue
        # ins(%9, %10 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>)
        # outs(%12 : tensor<64x72x1280xf32>)
        tensor_re = r"tensor<([0-9xf]+)>"
        ins_re = rf"ins\(.+:\s*{tensor_re},\s*{tensor_re}\)"
        ins_shape = re.search(ins_re, line)
        if ins_shape is None:
            continue
        tune_logger.debug(f"ins: {ins_shape.groups()}")

        # outs(%11 : tensor<2x20x1024x64xf32>)
        outs_re = rf"outs\(.+:\s*{tensor_re}\)"
        outs_shape = re.search(outs_re, line)
        assert outs_shape is not None
        tune_logger.debug(f"outs: {outs_shape.groups()}")

        assert len(ins_shape.groups()) == 2
        assert len(outs_shape.groups()) == 1

        in0_dims = get_shape_dims(ins_shape.groups()[0])
        in1_dims = get_shape_dims(ins_shape.groups()[1])
        out_dims = get_shape_dims(outs_shape.groups()[0])

        assert len(in0_dims) == len(in1_dims)
        assert len(in0_dims) == len(out_dims)
        return in0_dims, in1_dims, out_dims

    assert False, "Shape not found"


def is_pow2(x, min, max):
    return z3.Or(list(x == 2 ** i for i in range(min, max + 1)))


def is_not_pow2(x, min, max):
    return z3.And(list(x != 2 ** i for i in range(min, max + 1)))


def generate_constraints(
    problem_size: ProblemSize,
    tile_sizes,
    subgroup_size,
    intrinsic_size,
    workgroup_size,
    subgroup_m_count,
    subgroup_n_count,
    waves_per_eu,
):
    M, N, K = problem_size.M, problem_size.N, problem_size.K
    m, n, k = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    constraints = [subgroup_size == 64, wg_x * wg_y * wg_z <= 1024]
    constraints += [
        z3.Or(
            z3.And(intrinsic_mn == 16, intrinsic_k == 16),
            z3.And(intrinsic_mn == 32, intrinsic_k == 8),
        )
    ]
    subgroup_k_count = 1
    constraints += [
        m >= intrinsic_mn,
        m <= 512,
        m <= M,
    ]  # , M == m * z3.FreshInt()]
    constraints += [n >= intrinsic_mn, n <= 512, n <= N, N == n * z3.FreshInt()]
    constraints += [k >= intrinsic_k, k <= 512, k <= K, K == k * z3.FreshInt()]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    subgroup_m_tile_count = z3.Int("sg_m_tcnt")
    subgroup_n_tile_count = z3.Int("sg_n_tcnt")
    subgroup_k_tile_count = z3.Int("sg_k_tcnt")
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

    wg_n = z3.Int("wg_n")
    wg_k = z3.Int("wg_k")
    inner_lhs_dim_size = z3.Int("inner_lhs_dim_size")
    inner_rhs_dim_size = z3.Int("inner_rhs_dim_size")
    kMaxVectorLoadBitWidth = z3.Int("kMaxVectorLoadBitWidth")
    elems_per_thread = z3.Int("elems_per_thread")
    wg_threads = z3.Int("wg_threads")
    constraints += [wg_n == intrinsic_mn * subgroup_n_tile_count * subgroup_n_count]
    constraints += [wg_k == intrinsic_k * subgroup_k_tile_count]
    constraints += [inner_lhs_dim_size == wg_k]
    if problem_size.dispatch_kind == DispatchKind.mmt:
        constraints += [inner_rhs_dim_size == wg_k]
    else:
        constraints += [inner_rhs_dim_size == wg_n]
    constraints += [kMaxVectorLoadBitWidth == 128]
    constraints += [elems_per_thread == kMaxVectorLoadBitWidth / problem_size.rhs_bw]
    constraints += [wg_threads == subgroup_m_count * subgroup_n_count * subgroup_size]
    constraints += [
        z3.And(
            z3.Or(
                (inner_lhs_dim_size / elems_per_thread) % wg_threads == 0,
                wg_threads % (inner_lhs_dim_size / elems_per_thread) == 0,
            ),
            z3.Or(
                (inner_rhs_dim_size / elems_per_thread) % wg_threads == 0,
                wg_threads % (inner_rhs_dim_size / elems_per_thread) == 0,
            ),
        )
    ]

    return constraints


def generate_solutions(problem_size: ProblemSize):
    M, N, K = problem_size.M, problem_size.N, problem_size.K
    tune_logger.info(f"{M},{N},{K}")
    m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    waves_per_eu = z3.Int("waves_per_eu")
    all_vars = [
        m,
        n,
        k,
        subgroup_size,
        intrinsic_mn,
        intrinsic_k,
        wg_x,
        wg_y,
        wg_z,
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    ]

    solver = z3.Solver()
    constraints = generate_constraints(
        problem_size,
        [m, n, k],
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )
    solver.add(z3.simplify(z3.And(constraints)))
    tune_logger.debug(f"Initial constraints: {solver}")
    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()

        config = Configuration(
            lookup(subgroup_size),
            [lookup(wg_x), lookup(wg_y), lookup(wg_z)],
            f"#iree_gpu.mma_layout<MFMA_F16_{lookup(intrinsic_mn)}x{lookup(intrinsic_mn)}x{lookup(intrinsic_k)}_F32>",
            [lookup(m), lookup(n), lookup(k)],
            lookup(sg_m_cnt),
            lookup(sg_n_cnt),
            lookup(waves_per_eu),
        )
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

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


def parse_mlir(mlir_text: str) -> ir.Module:
    mlir_module = None
    with ireec.ir.Context() as context:
        try:
            mlir_module = ireec.ir.Module.parse(mlir_text)
            tune_logger.info("MLIR parsing successful!")
        except ireec.ir.MLIRError as e:
            tune_logger.error(f"Error parsing MLIR: {e}")
            raise RuntimeError(f"Error parsing MLIR: {e}")

    return mlir_module


def walk_callback_detect_type(
    op: ir.Operation, walk_result: OpWalkResult
) -> ir.WalkResult:
    if isinstance(op.opview, ireec.dialects._linalg_ops_gen.Conv2DNhwcHwcfOp):
        walk_result.was_interrupted = True
        walk_result.dispatch_kind = DispatchKind.conv

        return ir.WalkResult.INTERRUPT

    if isinstance(op.opview, ireec.dialects._util_ops_gen.FuncOp):
        if "matmul_transpose_b" in str(op.opview.sym_name):
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.mmt
            return ir.WalkResult.INTERRUPT
        if "matmul_like" in str(op.opview.sym_name):
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.contraction
            return ir.WalkResult.INTERRUPT
        if "batch_matmul" in str(op.opview.sym_name):
            walk_result.was_interrupted = True
            walk_result.dispatch_kind = DispatchKind.batch_matmul
            return ir.WalkResult.INTERRUPT

    return ir.WalkResult.ADVANCE


def walk_mlir_op(mlir_module: str) -> OpWalkResult:
    walk_result = OpWalkResult()
    for op in mlir_module.body.operations:
        op.walk(
            lambda op: walk_callback_detect_type(op, walk_result),
            ir.WalkOrder.POST_ORDER,
        )
        if walk_result.was_interrupted:
            break

    return walk_result


def tune(
    input: str,
    output: str = None,
    limit: int = 4096,
    lhs_dims: str = "mk",
    rhs_dims: str = "nk",
    tile_dims: str = "mnk",
):
    input_file = str(input)

    if output is None:
        output = get_default_output_dir()

    # Create the directory if it does not exist
    makedirs(str(output), exist_ok=True)

    tune_logger.debug(f"Output directory {output}")
    tune_logger.debug(f"Processing {input_file}")
    mlir_template = read_input_mlir(input_file)
    mlir_text = "".join(mlir_template)

    mlir_module = parse_mlir(mlir_text)
    walk_result = walk_mlir_op(mlir_module)
    walk_result.dispatch_kind = DispatchKind.mmt
    assert walk_result.dispatch_kind != None

    # Save the input file as the first candidate.
    with open(path.join(output, f"0.mlir"), "w") as f:
        f.write(mlir_text)

    if walk_result.dispatch_kind == DispatchKind.conv:
        n, oh, ow, oc, fh, fw, ic = get_shapes_conv(mlir_template)
        tune_logger.debug(f"Conv shape: [n{n}, oh{oh}, oc{oc}, fh{fh}, fw{fw}, ic{ic}]")
        M = oh * ow
        N = oc
        K = fh * fw * ic
        tune_logger.debug(f"Equivalent matmul shape: [{M}, {N}, {K}]")
        problem_size = ProblemSize(
            M, N, K, *get_bit_width(), dispatch_kind=walk_result.dispatch_kind
        )

        for i, config in enumerate(generate_solutions(problem_size)):
            if i >= limit:
                break
            tune_logger.info(f"Solution #{i+1}: {config}")
            new_mlir, embeddable_tuning = apply_params_conv(
                n, oh, ow, oc, fh, fw, ic, mlir_template, config
            )

            with open(path.join(output, f"{i+1}.mlir"), "w") as f:
                f.write(new_mlir)
            with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
                f.write(embeddable_tuning)
    elif walk_result.dispatch_kind == DispatchKind.mmt:
        M, N, K = get_shapes_mmt(mlir_template)
        tune_logger.debug(f"Matmul shape: [{M}, {N}, {K}]")
        problem_size = ProblemSize(
            M, N, K, *get_bit_width(), dispatch_kind=walk_result.dispatch_kind
        )

        for i, config in enumerate(generate_solutions(problem_size)):
            if i >= limit:
                break
            tune_logger.info(f"Solution #{i+1}: {config}")
            new_mlir, embeddable_tuning = apply_params_mmt(
                M, N, K, mlir_template, config
            )

            with open(path.join(output, f"{i+1}.mlir"), "w") as f:
                f.write(new_mlir)
            with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
                f.write(embeddable_tuning)
    elif walk_result.dispatch_kind == DispatchKind.contraction:
        LHS, RHS, RES = get_shapes_contract(mlir_template)
        tune_logger.debug(f"Contract shape: ({LHS}, {RHS}) -> {RES}")
        assert len(LHS) == len(lhs_dims)
        assert len(RHS) == len(rhs_dims)
        M = product(val if dim == "m" else 1 for dim, val in zip(lhs_dims, LHS))
        N = product(val if dim == "n" else 1 for dim, val in zip(rhs_dims, RHS))
        K0 = product(val if dim == "k" else 1 for dim, val in zip(lhs_dims, LHS))
        K1 = product(val if dim == "k" else 1 for dim, val in zip(rhs_dims, RHS))
        assert K0 == K1
        tune_logger.debug(f"Equivalent matmul shape: [{M}, {N}, {K0}]")
        problem_size = ProblemSize(
            M, N, K0, *get_bit_width(), dispatch_kind=walk_result.dispatch_kind
        )

        for i, config in enumerate(generate_solutions(problem_size)):
            if i >= limit:
                break
            new_mlir = apply_params_contract(
                LHS, RHS, RES, tile_dims, mlir_template, config
            )

            with open(path.join(output, f"{i+1}.mlir"), "w") as f:
                f.write(new_mlir)
    elif walk_result.dispatch_kind == DispatchKind.batch_matmul:
        LHS, RHS, RES = get_shapes_batch_matmul(mlir_template)
        assert len(LHS) == len(lhs_dims)
        assert len(RHS) == len(rhs_dims)
        B = product(val if dim == "b" else 1 for dim, val in zip(lhs_dims, LHS))
        B0 = product(val if dim == "b" else 1 for dim, val in zip(lhs_dims, RHS))
        B1 = product(val if dim == "b" else 1 for dim, val in zip(lhs_dims, RES))
        M = product(val if dim == "m" else 1 for dim, val in zip(lhs_dims, LHS))
        N = product(val if dim == "n" else 1 for dim, val in zip(rhs_dims, RHS))
        K0 = product(val if dim == "k" else 1 for dim, val in zip(lhs_dims, LHS))
        K1 = product(val if dim == "k" else 1 for dim, val in zip(rhs_dims, RHS))
        assert B == B0 and B == B1
        assert K0 == K1
        tune_logger.debug(f"Batch matmul shape: {B}x[{M}, {N}, {K0}]")
        problem_size = ProblemSize(
            M, N, K0, *get_bit_width(), dispatch_kind=walk_result.dispatch_kind
        )

        for i, config in enumerate(generate_solutions(problem_size)):
            if i >= limit:
                break
            new_mlir, embeddable_tuning = apply_params_batch_matmul(
                LHS, RHS, RES, B, M, N, K0, tile_dims, mlir_template, config
            )

            with open(path.join(output, f"{i+1}.mlir"), "w") as f:
                f.write(new_mlir)
            with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
                f.write(embeddable_tuning)
    else:
        assert False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input mlir file", type=str)
    parser.add_argument(
        "-o", "--output", help="Output dir", type=str, default=get_default_output_dir()
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Max number of candidates generated",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--lhs-dims", help="Map of LHS matmul dims", type=str, default="mk"
    )
    parser.add_argument(
        "--rhs-dims", help="Map of RHS matmul dims", type=str, default="nk"
    )
    parser.add_argument(
        "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
    )

    args = parser.parse_args()

    tune(
        args.input,
        args.output,
        args.limit,
        args.lhs_dims,
        args.rhs_dims,
        args.tile_dims,
    )

    tune_logger.setLevel(logging.INFO)

    # Create printing formatter for logging info
    formatter = logging.Formatter("%(message)s")

    # Create a handler to print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    tune_logger.addHandler(console_handler)

    # # Optionally, add a file handler to log to a file
    # file_handler = logging.FileHandler("tune.log")
    # file_handler.setFormatter(formatter)
    # tune_logger.addHandler(file_handler)

    tune(
        args.input,
        args.output,
        args.limit,
        args.lhs_dims,
        args.rhs_dims,
        args.tile_dims,
    )


if __name__ == "__main__":
    args = main()
