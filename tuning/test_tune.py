import pytest
import tune

"""
Usage: python -m pytest test_tune.py
"""


def test_get_mmt_tile_sizes():
    config = tune.Configuration(
        subgroup_size=0,
        workgroup_size=[],
        intrinsic="",
        tile_sizes=[128, 320, 32],
        subgroup_m_count=0,
        subgroup_n_count=0,
        waves_per_eu=0
    )
    assert tune.get_mmt_tile_sizes(config) == [128, 320, 32]


def test_get_conv_tile_sizes():
    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic="#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>",
        tile_sizes=[464, 320, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=1
    )
    assert tune.get_conv_tile_sizes(config) == (1, 1, 464, 320, 1, 1, 16)


def test_get_contract_tile_sizes():
    config = tune.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=2
    )
    assert tune.get_contract_tile_sizes(config, ["m", "n", "k"]) == [4, 8, 16]
    assert tune.get_contract_tile_sizes(config, ["n", "m", "k"]) == [8, 4, 16]
    assert tune.get_contract_tile_sizes(config, ["k", "n", "m"]) == [16, 8, 4]
    assert tune.get_contract_tile_sizes(config, ["k", "k", "k"]) == [16, 16, 16]


def test_get_pipeline_config():
    config1 = tune.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=2
    )
    config2 = tune.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=4
    )
    assert tune.get_pipeline_config(config1) == ""
    assert (
        tune.get_pipeline_config(config2)
        == ', llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}'
    )


def test_get_shape_dims():
    assert tune.get_shape_dims("2048x1280xf16") == [2048, 1280]
    assert tune.get_shape_dims("64x32x128xf32") == [64, 32, 128]


def test_get_shapes_mmt():
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert tune.get_shapes_mmt(template) == (2048, 1280, 1280)


def test_get_shapes_conv():
    template = [
        r"%7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"%8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>",
    ]
    assert tune.get_shapes_conv(template) == (1, 1, 32, 256, 3, 3, 1280)


def test_get_shapes_contract():
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert tune.get_shapes_contract(template) == (
        [2048, 1280],
        [1280, 1280],
        [2048, 1280],
    )


def test_get_shapes_batch_matmul():
    template = [
        "%10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "%11 = linalg.batch_matmul ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>",
    ]
    assert tune.get_shapes_batch_matmul(template) == (
        [1, 32, 1024],
        [1, 1024, 32],
        [1, 32, 32],
    )


def test_generate_solutions():
    M, N, K = 2048, 3840, 1280
    configs = None
    configs = tune.generate_solutions(M, N, K)
    assert configs is not None


def test_generate_constraints_valid_input():
    # Define input parameters as z3 Ints
    M, N, K = 256, 384, 32
    m, n, k = tune.z3.Int("m"), tune.z3.Int("n"), tune.z3.Int("k")
    subgroup_size = tune.z3.Int("subgroup_size")
    intrinsic_mn = tune.z3.Int("intrinsic_mn")
    intrinsic_k = tune.z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = tune.z3.Int("wg_x"), tune.z3.Int("wg_y"), tune.z3.Int("wg_z")
    sg_m_cnt = tune.z3.Int("sg_m_cnt")
    sg_n_cnt = tune.z3.Int("sg_n_cnt")
    waves_per_eu = tune.z3.Int("waves_per_eu")

    constraints = tune.generate_constraints(
        [M, N, K],
        [m, n, k],
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu
    )

    solver = tune.z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable
    assert solver.check() == tune.z3.sat


def test_generate_constraints_invalid_input():
    # Define input parameters that should lead to unsatisfiable constraints
    M, N, K = 256, 384, 32
    m, n, k = tune.z3.Int("m"), tune.z3.Int("n"), tune.z3.Int("k")
    subgroup_size = tune.z3.Int("subgroup_size")
    intrinsic_mn = tune.z3.Int("intrinsic_mn")
    intrinsic_k = tune.z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = tune.z3.Int("wg_x"), tune.z3.Int("wg_y"), tune.z3.Int("wg_z")
    sg_m_cnt = tune.z3.Int("sg_m_cnt")
    sg_n_cnt = tune.z3.Int("sg_n_cnt")
    waves_per_eu = tune.z3.Int("waves_per_eu")

    constraints = tune.generate_constraints(
        [M, N, K],
        [m, n, k],
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu
    )
    constraints.append(m > 1000)  # Adding an additional unsatisfiable constraint

    solver = tune.z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == tune.z3.unsat


def test_apply_params_mmt():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<16x16x16_F32>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [16, 16] subgroup_size = 16,",
        "<tile_sizes = [[8, 8, 8]]>",
        ", waves_per_eu = 8 : i64",
    ]

    M, N, K = 2048, 1280, 1280

    config = tune.Configuration(
        subgroup_size=16,
        workgroup_size=[16, 16, 1],
        intrinsic="16x16x16_F32",
        tile_sizes=[8, 8, 8],
        subgroup_m_count=16,
        subgroup_n_count=16,
        waves_per_eu=8
    )

    modified, embeddable = tune.apply_params_mmt(M, N, K, mlir_template, config)

    assert modified is not None
    assert embeddable is not None
    assert (
        "intrinsic = 16x16x16_F32, subgroup_m_count = 16, subgroup_n_count = 16"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [16, 16, 1] subgroup_size = 16"
        in modified
    )
    assert "tile_sizes = [[8, 8, 8]]" in modified
    assert "waves_per_eu = 8 : i64" in modified


def test_apply_params_conv():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<16x16x16_F32>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 64, 128, 1, 1, 32]]>",
        ", waves_per_eu = 2 : i64",
    ]

    n, oh, ow, oc, fh, fw, ic = 2, 64, 64, 640, 3, 3, 640

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic="#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>",
        tile_sizes=[464, 320, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=1
    )

    modified, embeddable = tune.apply_params_conv(
        n, oh, ow, oc, fh, fw, ic, mlir_template, config
    )

    assert modified is not None
    assert embeddable is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 1, subgroup_n_count = 4"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 1, 464, 320, 1, 1, 16]]" in modified
    assert "waves_per_eu = 1 : i64" in modified


def test_apply_params_contract():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 1, 64, 64, 128]]>",
        ", waves_per_eu = 2 : i64",
    ]

    LHS, RHS, RES = ([2, 1024, 1280], [3, 20, 64, 1280], [3, 2, 20, 1024, 64])
    tile_dims = "*mnk"
    M, N, K0, K1 = (2048, 3840, 1280, 1280)

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic="#iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>",
        tile_sizes=[480, 384, 32],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=2
    )

    new_mlir = tune.apply_params_contract(
        LHS, RHS, RES, tile_dims, mlir_template, config
    )

    assert new_mlir is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 1, subgroup_n_count = 4"
        in new_mlir
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in new_mlir
    )
    assert "tile_sizes = [[1, 480, 384, 32]]" in new_mlir
    assert ", waves_per_eu = 2 : i64" in new_mlir


def test_apply_params_batch_matmul():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 64, 64]]>",
        ", waves_per_eu = 2 : i64",
    ]

    LHS, RHS, RES = ([64, 968, 640], [64, 640, 320], [64, 968, 320])
    lhs_dims, rhs_dims, tile_dims = "bmk", "bkn", "*mnk"
    B, B0, B1, M, N, K0, K1 = (64, 64, 64, 968, 320, 640, 640)

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 1, 1],
        intrinsic="#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>",
        tile_sizes=[416, 320, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=1
    )

    modified, embeddable = tune.apply_params_batch_matmul(
        LHS, RHS, RES, B, M, N, K0, tile_dims, mlir_template, config
    )

    assert modified is not None
    assert embeddable is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 416, 320, 128]]" in modified
    assert "waves_per_eu = 1 : i64" in modified

def test_parse_mlir():
    mlir_str = r"""
    builtin.module  {
      func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
  """
    mlir_module = tune.parse_mlir(mlir_str)
    assert mlir_module != None
    assert isinstance(mlir_module, tune.ireec._mlir_libs._mlir.ir.Module) == True
    assert isinstance(mlir_module.body.operations[0], tune.ireec.dialects.func.FuncOp) == True

mlir_str_conv = r"""
module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @run_forward$async_dispatch_1269 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = hal.interface.constant.load[3] : i32
          %4 = hal.interface.constant.load[4] : i32
          %5 = hal.interface.constant.load[5] : i32
          %6 = arith.index_castui %0 : i32 to index
          %7 = arith.index_castui %1 : i32 to index
          %8 = arith.index_castui %2 : i32 to index
          %9 = arith.index_castui %3 : i32 to index
          %10 = arith.index_castui %4 : i32 to index
          %11 = arith.index_castui %5 : i32 to index
          %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x130x130x640xf16>>
          %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x640x320xf16>>
          %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
          %15 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x320xf32>>
          %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
          %17 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%11) : !flow.dispatch.tensor<writeonly:tensor<24x320x128x128xf16>>
          %18 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [24, 130, 130, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x130x130x640xf16>> -> tensor<24x130x130x640xf16>
          %19 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0, 0], sizes = [3, 3, 640, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x640x320xf16>> -> tensor<3x3x640x320xf16>
          %20 = flow.dispatch.tensor.load %14, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
          %21 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [24, 320], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24x320xf32>> -> tensor<24x320xf32>
          %22 = flow.dispatch.tensor.load %16, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
          %23 = tensor.empty() : tensor<24x320x128x128xf16>
          %24 = tensor.empty() : tensor<24x128x128x320xf32>
          %25 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} ins(%cst : f32) outs(%24 : tensor<24x128x128x320xf32>) -> tensor<24x128x128x320xf32>
          %26 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%18, %19 : tensor<24x130x130x640xf16>, tensor<3x3x640x320xf16>) outs(%25 : tensor<24x128x128x320xf32>) -> tensor<24x128x128x320xf32>
          %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %20, %21, %22 : tensor<24x128x128x320xf32>, tensor<320xf32>, tensor<24x320xf32>, tensor<320xf32>) outs(%23 : tensor<24x320x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
            %28 = arith.addf %in_1, %in_2 : f32
            %29 = arith.addf %in, %in_0 : f32
            %30 = arith.truncf %28 : f32 to f16
            %31 = arith.truncf %29 : f32 to f16
            %32 = arith.addf %31, %30 : f16
            linalg.yield %32 : f16
          } -> tensor<24x320x128x128xf16>
          flow.dispatch.tensor.store %27, %17, offsets = [0, 0, 0, 0], sizes = [24, 320, 128, 128], strides = [1, 1, 1, 1] : tensor<24x320x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<24x320x128x128xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c11641601536 = arith.constant 11641601536 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c11641601536}
    util.global.store %buffer, @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c6804067584 = arith.constant 6804067584 : index
    %c2 = arith.constant 2 : index
    %c1966533568 = arith.constant 1966533568 : index
    %c4837533952 = arith.constant 4837533952 : index
    %c1 = arith.constant 1 : index
    %c4837533760 = arith.constant 4837533760 : index
    %c1022576704_i32 = arith.constant 1022576704 : i32
    %c1964184704_i32 = arith.constant 1964184704 : i32
    %c1960497024_i32 = arith.constant 1960497024 : i32
    %c1960498304_i32 = arith.constant 1960498304 : i32
    %c1022545984_i32 = arith.constant 1022545984 : i32
    %c503377984_i32 = arith.constant 503377984 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c503377984_i32, %c1022545984_i32, %c1960498304_i32, %c1960497024_i32, %c1964184704_i32, %c1022576704_i32]) : i32, i32, i32, i32, i32, i32
    %run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer = util.global.load @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer)[%c0, %c4837533760], 
      %c1 = (%run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer)[%c4837533952, %c1966533568], 
      %c2 = (%run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer)[%c6804067584, %c4837533760]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@run_forward$async_dispatch_1269::@rocm_hsaco_fb::@run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@run_forward$async_dispatch_1269) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@run_forward$async_dispatch_1269::@rocm_hsaco_fb::@run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32) : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z])
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device_0 : !hal.device> affinity(%c-1_i64) wait(%1) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
"""

mlir_str_mmt = r"""
module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @run_forward$async_dispatch_142 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24576x1280xf16>>
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240xf32>>
          %9 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<24576x10240xf16>>
          %10 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [24576, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24576x1280xf16>> -> tensor<24576x1280xf16>
          %11 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
          %12 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [10240], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10240xf32>> -> tensor<10240xf32>
          %13 = tensor.empty() : tensor<24576x10240xf16>
          %14 = tensor.empty() : tensor<24576x10240xf32>
          %15 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%14 : tensor<24576x10240xf32>) -> tensor<24576x10240xf32>
          %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %11 : tensor<24576x1280xf16>, tensor<10240x1280xf16>) outs(%15 : tensor<24576x10240xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
          ^bb0(%in: f16, %in_0: f16, %out: f32):
            %18 = arith.extf %in : f16 to f32
            %19 = arith.extf %in_0 : f16 to f32
            %20 = arith.mulf %18, %19 : f32
            %21 = arith.addf %out, %20 : f32
            linalg.yield %21 : f32
          } -> tensor<24576x10240xf32>
          %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %12 : tensor<24576x10240xf32>, tensor<10240xf32>) outs(%13 : tensor<24576x10240xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
          ^bb0(%in: f32, %in_0: f32, %out: f16):
            %18 = arith.addf %in, %in_0 : f32
            %19 = arith.truncf %18 : f32 to f16
            linalg.yield %19 : f16
          } -> tensor<24576x10240xf16>
          flow.dispatch.tensor.store %17, %9, offsets = [0, 0], sizes = [24576, 10240], strides = [1, 1] : tensor<24576x10240xf16> -> !flow.dispatch.tensor<writeonly:tensor<24576x10240xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c11667815936 = arith.constant 11667815936 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c11667815936}
    util.global.store %buffer, @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c6830281984 = arith.constant 6830281984 : index
    %c3 = arith.constant 3 : index
    %c1966533568 = arith.constant 1966533568 : index
    %c4863748352 = arith.constant 4863748352 : index
    %c2 = arith.constant 2 : index
    %c26214400 = arith.constant 26214400 : index
    %c4837533952 = arith.constant 4837533952 : index
    %c1 = arith.constant 1 : index
    %c4837533760 = arith.constant 4837533760 : index
    %c1274689600_i32 = arith.constant 1274689600 : i32
    %c1787843776_i32 = arith.constant 1787843776 : i32
    %c1211775040_i32 = arith.constant 1211775040 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c1211775040_i32, %c1787843776_i32, %c1274689600_i32]) : i32, i32, i32
    %run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer = util.global.load @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c0, %c4837533760], 
      %c1 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c4837533952, %c26214400], 
      %c2 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c4863748352, %c1966533568], 
      %c3 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c6830281984, %c4837533760]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@run_forward$async_dispatch_142::@rocm_hsaco_fb::@run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@run_forward$async_dispatch_142) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@run_forward$async_dispatch_142::@rocm_hsaco_fb::@run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32) : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z])
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device_0 : !hal.device> affinity(%c-1_i64) wait(%1) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
"""

mlir_str_contra = r"""
module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @run_forward$async_dispatch_131 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x1024x1280xf16>>
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x20x64x1280xf16>>
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<3x24x20x1024x64xf16>>
          %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [24, 1024, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x1024x1280xf16>> -> tensor<24x1024x1280xf16>
          %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [3, 20, 64, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x20x64x1280xf16>> -> tensor<3x20x64x1280xf16>
          %11 = tensor.empty() : tensor<3x24x20x1024x64xf16>
          %12 = tensor.empty() : tensor<3x24x20x1024x64xf32>
          %13 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 128]]>} ins(%cst : f32) outs(%12 : tensor<3x24x20x1024x64xf32>) -> tensor<3x24x20x1024x64xf32>
          %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : tensor<24x1024x1280xf16>, tensor<3x20x64x1280xf16>) outs(%13 : tensor<3x24x20x1024x64xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 128]]>} {
          ^bb0(%in: f16, %in_0: f16, %out: f32):
            %16 = arith.extf %in : f16 to f32
            %17 = arith.extf %in_0 : f16 to f32
            %18 = arith.mulf %16, %17 : f32
            %19 = arith.addf %out, %18 : f32
            linalg.yield %19 : f32
          } -> tensor<3x24x20x1024x64xf32>
          %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<3x24x20x1024x64xf32>) outs(%11 : tensor<3x24x20x1024x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 128]]>} {
          ^bb0(%in: f32, %out: f16):
            %16 = arith.truncf %in : f32 to f16
            linalg.yield %16 : f16
          } -> tensor<3x24x20x1024x64xf16>
          flow.dispatch.tensor.store %15, %8, offsets = [0, 0, 0, 0, 0], sizes = [3, 24, 20, 1024, 64], strides = [1, 1, 1, 1, 1] : tensor<3x24x20x1024x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<3x24x20x1024x64xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c11641601536 = arith.constant 11641601536 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c11641601536}
    util.global.store %buffer, @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c6804067584 = arith.constant 6804067584 : index
    %c2 = arith.constant 2 : index
    %c1966533568 = arith.constant 1966533568 : index
    %c4837533952 = arith.constant 4837533952 : index
    %c1 = arith.constant 1 : index
    %c4837533760 = arith.constant 4837533760 : index
    %c1274689600_i32 = arith.constant 1274689600 : i32
    %c1767517376_i32 = arith.constant 1767517376 : i32
    %c1211775040_i32 = arith.constant 1211775040 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c1211775040_i32, %c1767517376_i32, %c1274689600_i32]) : i32, i32, i32
    %run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer = util.global.load @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer)[%c0, %c4837533760], 
      %c1 = (%run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer)[%c4837533952, %c1966533568], 
      %c2 = (%run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer)[%c6804067584, %c4837533760]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@run_forward$async_dispatch_131::@rocm_hsaco_fb::@run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@run_forward$async_dispatch_131) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@run_forward$async_dispatch_131::@rocm_hsaco_fb::@run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32) : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z])
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device_0 : !hal.device> affinity(%c-1_i64) wait(%1) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
"""

mlir_str_batch_mmt = r""

def test_walk_mlir_op():
    mlir_module = tune.parse_mlir(mlir_str_conv)
    walk_result = tune.walk_mlir_op(mlir_module)
    assert walk_result.wasInterrupted == True
    assert walk_result.isConv == True

    mlir_module = tune.parse_mlir(mlir_str_mmt)
    walk_result = tune.walk_mlir_op(mlir_module)
    assert walk_result.wasInterrupted == True
    assert walk_result.isMatmul == True

    mlir_module = tune.parse_mlir(mlir_str_contra)
    walk_result = tune.walk_mlir_op(mlir_module)
    assert walk_result.wasInterrupted == True
    assert walk_result.isContraction == True

    #TODO: add batch_mmt mlir
    # mlir_module = tune.parse_mlir(mlir_str_batch_mmt)
    # walk_result = tune.walk_mlir_op(mlir_module)
    # assert walk_result.wasInterrupted == True
    # assert walk_result.isBatchMatmul == True