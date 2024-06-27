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
    assert isinstance(mlir_module, tune.ireec._mlir_libs._mlir.ir.Module)
    assert isinstance(mlir_module.body.operations[0], tune.ireec.dialects.func.FuncOp)

def test_walk_mlir_op():
    def detect_mlir_type(mlir_file_path):
        with open(mlir_file_path, 'r') as f:
            mlir_text = f.read()
        mlir_module = tune.parse_mlir(mlir_text)
        walk_result = tune.walk_mlir_op(mlir_module)

        return walk_result

    walk_result = detect_mlir_type("./test-data/convolution.mlir")
    assert walk_result.wasInterrupted == True
    assert walk_result.isConv == True

    walk_result = detect_mlir_type("./test-data/matmul.mlir")
    assert walk_result.wasInterrupted == True
    assert walk_result.isMatmul == True

    walk_result = detect_mlir_type("./test-data/contraction.mlir")
    assert walk_result.wasInterrupted == True
    assert walk_result.isContraction == True

    walk_result = detect_mlir_type("./test-data/batch_matmul.mlir")
    assert walk_result.wasInterrupted == True
    assert walk_result.isBatchMatmul == True
