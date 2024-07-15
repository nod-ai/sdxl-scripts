import pytest
import tune

"""
Usage: python -m pytest test_tune.py
"""


def test_get_shaped_type_element_bitwidth():
    assert tune.ShapedType([1024, 2048], tune.ElementType.i8).bitwidth == 8
    assert tune.ShapedType([2048], tune.ElementType.i32).bitwidth == 32
    assert tune.ShapedType([2048, 512, 384], tune.ElementType.f8).bitwidth == 8
    assert tune.ShapedType([1, 1], tune.ElementType.f16).bitwidth == 16


def test_get_shaped_type_to_str():
    assert str(tune.ShapedType([1024, 2048], tune.ElementType.i8)) == "1024x2048xi8"
    assert str(tune.ShapedType([1024], tune.ElementType.f32)) == "1024xf32"
    assert str(tune.ShapedType([1, 2, 3], tune.ElementType.f16)) == "1x2x3xf16"


def test_parse_tensor_type():
    assert tune.parse_tensor_type("tensor<1x2x3xf32>") == tune.ShapedType(
        [1, 2, 3], tune.ElementType.f32
    )
    assert tune.parse_tensor_type("tensor<123xi8>") == tune.ShapedType(
        [123], tune.ElementType.i8
    )


def test_get_mmt_tile_sizes():
    config = tune.Configuration(
        subgroup_size=0,
        workgroup_size=[],
        intrinsic="",
        tile_sizes=[128, 320, 32],
        subgroup_m_count=0,
        subgroup_n_count=0,
        waves_per_eu=0,
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
        waves_per_eu=1,
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
        waves_per_eu=2,
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
        waves_per_eu=2,
    )
    config2 = tune.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=4,
    )
    assert tune.get_pipeline_config(config1) == ", prefetch_shared_memory"
    assert (
        tune.get_pipeline_config(config2)
        == ', prefetch_shared_memory, llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}'
    )


def test_get_shapes_mmt():
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert tune.get_shapes_mmt(template) == tune.ProblemSize(
        tune.MatmulSize(2048, 1280, 1280),
        tune.ShapedType([2048, 1280], tune.ElementType.f16),
        tune.ShapedType([1280, 1280], tune.ElementType.f16),
        tune.ShapedType([2048, 1280], tune.ElementType.f32),
        tune.DispatchKind.mmt,
    )


def test_get_shapes_conv():
    template = [
        r"%7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"%8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>",
    ]
    assert tune.get_shapes_conv(template) == tune.ProblemSize(
        tune.MatmulSize(32, 256, 11520),
        tune.ShapedType([1, 3, 34, 1280], tune.ElementType.f16),
        tune.ShapedType([3, 3, 1280, 256], tune.ElementType.f16),
        tune.ShapedType([1, 1, 32, 256], tune.ElementType.f32),
        tune.DispatchKind.conv,
    )


def test_get_shapes_contract():
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert tune.get_shapes_contract(template, "mk", "nk") == tune.ProblemSize(
        tune.MatmulSize(2048, 1280, 1280),
        tune.ShapedType([2048, 1280], tune.ElementType.f16),
        tune.ShapedType([1280, 1280], tune.ElementType.f16),
        tune.ShapedType([2048, 1280], tune.ElementType.f32),
        tune.DispatchKind.contraction,
    )


def test_get_shapes_batch_matmul():
    template = [
        "%10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "%11 = linalg.batch_matmul ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>",
    ]
    assert tune.get_shapes_batch_matmul(template, "bmk", "bkn") == tune.ProblemSize(
        tune.MatmulSize(32, 32, 1024, 1),
        tune.ShapedType([1, 32, 1024], tune.ElementType.f32),
        tune.ShapedType([1, 1024, 32], tune.ElementType.f32),
        tune.ShapedType([1, 32, 32], tune.ElementType.f32),
        tune.DispatchKind.batch_matmul,
    )


def test_get_shapes_batch_mmt():
    template = [
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
        r"flow.dispatch.tensor.store %21, %10, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : tensor<2x4096x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x4096x640xf16>>",
    ]
    assert tune.get_shapes_batch_mmt(template) == tune.ProblemSize(
        tune.MatmulSize(4096, 640, 640, 2),
        tune.ShapedType([2, 4096, 640], tune.ElementType.i8),
        tune.ShapedType([2, 640, 640], tune.ElementType.i8),
        tune.ShapedType([2, 4096, 640], tune.ElementType.i32),
        tune.DispatchKind.batch_mmt,
    )


def test_mfma_intrinsic_to_str():
    assert str(tune.MfmaIntrinsic.mfma_f16_16x16x16_f32()) == "MFMA_F16_16x16x16_F32"
    assert str(tune.MfmaIntrinsic.mfma_i8_32x32x16_i32()) == "MFMA_I8_32x32x16_I32"


def test_get_compatible_mfma_intrinsics():
    assert tune.get_compatible_mfma_intrinsics(
        tune.ProblemSize(
            tune.MatmulSize(2048, 1280, 1280),
            tune.ShapedType([2048, 1280], tune.ElementType.f16),
            tune.ShapedType([1280, 1280], tune.ElementType.f16),
            tune.ShapedType([2048, 1280], tune.ElementType.f32),
            tune.DispatchKind.mmt,
        )
    ) == [
        tune.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tune.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
    ]

    assert tune.get_compatible_mfma_intrinsics(
        tune.ProblemSize(
            tune.MatmulSize(2048, 1280, 1280),
            tune.ShapedType([2048, 1280], tune.ElementType.i8),
            tune.ShapedType([1280, 1280], tune.ElementType.i8),
            tune.ShapedType([2048, 1280], tune.ElementType.i32),
            tune.DispatchKind.mmt,
        )
    ) == [
        tune.MfmaIntrinsic.mfma_i8_16x16x32_i32(),
        tune.MfmaIntrinsic.mfma_i8_32x32x16_i32(),
    ]

    assert tune.get_compatible_mfma_intrinsics(
        tune.ProblemSize(
            tune.MatmulSize(968, 320, 640, 64),
            tune.ShapedType([64, 968, 640], tune.ElementType.f32),
            tune.ShapedType([64, 640, 320], tune.ElementType.f32),
            tune.ShapedType([64, 968, 320], tune.ElementType.f32),
            tune.DispatchKind.batch_matmul,
        )
    ) == [
        tune.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tune.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
    ]


def test_generate_solutions():
    matmul_size = tune.MatmulSize(2048, 3840, 1280)
    lhs_type = tune.ShapedType([2048, 1280], tune.ElementType.f16)
    rhs_type = tune.ShapedType([3840, 1280], tune.ElementType.f16)
    res_type = tune.ShapedType([2048, 3840], tune.ElementType.f32)
    problem_size = tune.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, tune.DispatchKind.mmt
    )
    configs = tune.generate_solutions(problem_size)
    assert configs is not None


def test_generate_constraints_valid_input():
    matmul_size = tune.MatmulSize(1024, 1024, 1024)
    lhs_type = tune.ShapedType([1024, 1024], tune.ElementType.f16)
    rhs_type = tune.ShapedType([1024, 1024], tune.ElementType.f16)
    res_type = tune.ShapedType([1024, 1024], tune.ElementType.f32)
    problem_size = tune.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, tune.DispatchKind.mmt
    )
    # Define input parameters as z3 Ints
    m, n, k = tune.z3.Int("m"), tune.z3.Int("n"), tune.z3.Int("k")
    subgroup_size = tune.z3.Int("subgroup_size")
    intrinsic_mn = tune.z3.Int("intrinsic_mn")
    intrinsic_k = tune.z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = tune.z3.Int("wg_x"), tune.z3.Int("wg_y"), tune.z3.Int("wg_z")
    sg_m_cnt = tune.z3.Int("sg_m_cnt")
    sg_n_cnt = tune.z3.Int("sg_n_cnt")
    waves_per_eu = tune.z3.Int("waves_per_eu")

    constraints = tune.generate_constraints(
        problem_size,
        [m, n, k],
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )

    solver = tune.z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable
    assert solver.check() == tune.z3.sat


def test_generate_constraints_invalid_input():
    # Define input parameters that should lead to unsatisfiable constraints
    matmul_size = tune.MatmulSize(1024, 1024, 1024)
    lhs_type = tune.ShapedType([1024, 1024], tune.ElementType.f16)
    rhs_type = tune.ShapedType([1024, 1024], tune.ElementType.f16)
    res_type = tune.ShapedType([1024, 1024], tune.ElementType.f32)
    problem_size = tune.ProblemSize(
        matmul_size, lhs_type, rhs_type, res_type, tune.DispatchKind.mmt
    )
    m, n, k = tune.z3.Int("m"), tune.z3.Int("n"), tune.z3.Int("k")
    subgroup_size = tune.z3.Int("subgroup_size")
    intrinsic_mn = tune.z3.Int("intrinsic_mn")
    intrinsic_k = tune.z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = tune.z3.Int("wg_x"), tune.z3.Int("wg_y"), tune.z3.Int("wg_z")
    sg_m_cnt = tune.z3.Int("sg_m_cnt")
    sg_n_cnt = tune.z3.Int("sg_n_cnt")
    waves_per_eu = tune.z3.Int("waves_per_eu")

    constraints = tune.generate_constraints(
        problem_size,
        [m, n, k],
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        waves_per_eu,
    )
    constraints.append(m > 1000)  # Adding an additional unsatisfiable constraint

    solver = tune.z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == tune.z3.unsat


def test_apply_params_mmt():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [16, 16] subgroup_size = 16,",
        "<tile_sizes = [[8, 8, 8]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    M, N, K = 2048, 1280, 1280

    config = tune.Configuration(
        subgroup_size=16,
        workgroup_size=[16, 16, 1],
        intrinsic=tune.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tile_sizes=[8, 8, 8],
        subgroup_m_count=16,
        subgroup_n_count=16,
        waves_per_eu=8,
    )

    problem_size = tune.ProblemSize(
        tune.MatmulSize(M, N, K),
        tune.ShapedType([M, K], tune.ElementType.f16),
        tune.ShapedType([N, K], tune.ElementType.f16),
        tune.ShapedType([M, N], tune.ElementType.f32),
        tune.DispatchKind.mmt,
    )
    modified, embeddable = tune.apply_params_mmt(problem_size, mlir_template, config)

    assert modified is not None
    assert embeddable is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 16, subgroup_n_count = 16"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [16, 16, 1] subgroup_size = 16"
        in modified
    )
    assert "tile_sizes = [[8, 8, 8]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "8"}' in modified


def test_apply_params_conv():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 64, 128, 1, 1, 32]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    n, oh, ow, oc, fh, fw, ic = 2, 64, 64, 640, 3, 3, 640

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic=tune.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tile_sizes=[464, 320, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=2,
    )

    problem_size = tune.ProblemSize(
        tune.MatmulSize(oh * ow, oc, fh * fw * ic),
        tune.ShapedType([n, oh + 2, ow + 2, oc], tune.ElementType.f16),
        tune.ShapedType([fh, fw, ic, oc], tune.ElementType.f16),
        tune.ShapedType([n, oh, ow, oc], tune.ElementType.f32),
        tune.DispatchKind.conv,
    )
    modified, embeddable = tune.apply_params_conv(problem_size, mlir_template, config)

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
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_contract():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 1, 1, 64, 64, 128]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "*mnk"
    problem_size = tune.ProblemSize(
        tune.MatmulSize(2048, 3840, 1280),
        tune.ShapedType([2, 1024, 1280], tune.ElementType.f16),
        tune.ShapedType([3, 20, 64, 1280], tune.ElementType.f16),
        tune.ShapedType([3, 2, 20, 1024, 64], tune.ElementType.f32),
        tune.DispatchKind.contraction,
    )

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[256, 1, 1],
        intrinsic=tune.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
        tile_sizes=[480, 384, 32],
        subgroup_m_count=1,
        subgroup_n_count=4,
        waves_per_eu=2,
    )

    new_mlir, _embeddable = tune.apply_params_contract(
        problem_size, tile_dims, mlir_template, config
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
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in new_mlir


def test_apply_params_batch_matmul():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 64, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "bmnk"
    problem_size = tune.ProblemSize(
        tune.MatmulSize(968, 320, 640, 64),
        tune.ShapedType([64, 968, 640], tune.ElementType.f16),
        tune.ShapedType([64, 640, 320], tune.ElementType.f16),
        tune.ShapedType([64, 968, 320], tune.ElementType.f32),
        tune.DispatchKind.batch_matmul,
    )

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=tune.MfmaIntrinsic.mfma_f16_32x32x8_f32(),
        tile_sizes=[416, 320, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=2,
    )

    modified, embeddable = tune.apply_params_batch_matmul(
        problem_size, tile_dims, mlir_template, config
    )

    assert modified is not None
    assert embeddable is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 416, 320, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_float():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = tune.ProblemSize(
        tune.MatmulSize(4096, 640, 640, 2),
        tune.ShapedType([2, 4096, 640], tune.ElementType.f16),
        tune.ShapedType([2, 640, 640], tune.ElementType.f16),
        tune.ShapedType([2, 4096, 640], tune.ElementType.f32),
        tune.DispatchKind.batch_mmt,
    )

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=tune.MfmaIntrinsic.mfma_f16_16x16x16_f32(),
        tile_sizes=[128, 64, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=2,
    )

    modified, _embeddable = tune.apply_params_batch_mmt(
        problem_size, mlir_template, config
    )

    assert modified is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_int():
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<tile_sizes = [[1, 128, 128, 64]]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = tune.ProblemSize(
        tune.MatmulSize(4096, 640, 640, 2),
        tune.ShapedType([2, 4096, 640], tune.ElementType.i8),
        tune.ShapedType([2, 640, 640], tune.ElementType.i8),
        tune.ShapedType([2, 4096, 640], tune.ElementType.i32),
        tune.DispatchKind.batch_mmt,
    )

    config = tune.Configuration(
        subgroup_size=64,
        workgroup_size=[128, 2, 1],
        intrinsic=tune.MfmaIntrinsic.mfma_i8_32x32x16_i32(),
        tile_sizes=[128, 64, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
        waves_per_eu=2,
    )

    modified, _embeddable = tune.apply_params_batch_mmt(
        problem_size, mlir_template, config
    )

    assert modified is not None
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I8_32x32x16_I32>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "tile_sizes = [[1, 128, 64, 128]]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


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
    assert isinstance(mlir_module, tune.ireec._mlir_libs._mlir.ir.Module)
    assert isinstance(mlir_module.body.operations[0], tune.ireec.dialects.func.FuncOp)


def test_walk_mlir_op():
    def detect_mlir_type(mlir_file_path):
        with open(mlir_file_path, "r") as f:
            mlir_text = f.read()
        mlir_module = tune.parse_mlir(mlir_text)
        walk_result = tune.walk_mlir_op(mlir_module)

        return walk_result

    walk_result = detect_mlir_type("./test-data/convolution.mlir")
    assert walk_result.was_interrupted
    assert walk_result.dispatch_kind == tune.DispatchKind.conv

    walk_result = detect_mlir_type("./test-data/matmul.mlir")
    assert walk_result.was_interrupted
    assert walk_result.dispatch_kind == tune.DispatchKind.mmt

    walk_result = detect_mlir_type("./test-data/contraction.mlir")
    assert walk_result.was_interrupted
    assert walk_result.dispatch_kind == tune.DispatchKind.contraction

    walk_result = detect_mlir_type("./test-data/batch_matmul.mlir")
    assert walk_result.was_interrupted
    assert walk_result.dispatch_kind == tune.DispatchKind.batch_matmul

    walk_result = detect_mlir_type("./test-data/batch_mmt.mlir")
    assert walk_result.was_interrupted
    assert walk_result.dispatch_kind == tune.DispatchKind.batch_mmt
