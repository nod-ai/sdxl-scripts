import pytest
import tune

'''
Usage: python -m pytest test_tune.py
'''

def test_get_contract_tile_sizes():
    config = tune.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=2,
        no_workgroup_reorder=0
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
        no_workgroup_reorder=0
    )
    config2 = tune.Configuration(
        subgroup_size=32,
        workgroup_size=[16, 16, 1],
        intrinsic="",
        tile_sizes=[4, 8, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
        waves_per_eu=4,
        no_workgroup_reorder=1
    )
    assert tune.get_pipeline_config(config1) == ""
    assert tune.get_pipeline_config(config2) == ', no_reorder_workgroups, llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}'

def test_get_shape_dims():
    assert tune.get_shape_dims("2048x1280xf16") == [2048, 1280]
    assert tune.get_shape_dims("64x32x128xf32") == [64, 32, 128]

def test_get_shapes_mmt():
    template = [
        r'%18 = tensor.empty() : tensor<2048x1280xf32>',
        r'%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>',
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r'^bb0(%in: f16, %in_0: f16, %out: f32):'
    ]
    assert tune.get_shapes_mmt(template) == (2048, 1280, 1280)

def test_get_shapes_conv():
    template = [
        r'%7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>',
        r'%8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>',
        r'flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>'
    ]
    assert tune.get_shapes_conv(template) == (1, 1, 32, 256, 3, 3, 1280)

def test_get_shapes_contract():
    template = [
        r'%18 = tensor.empty() : tensor<2048x1280xf32>',
        r'%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>',
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r'^bb0(%in: f16, %in_0: f16, %out: f32):'
    ]
    assert tune.get_shapes_contract(template) == ([2048, 1280], [1280, 1280], [2048, 1280])

def test_get_shapes_batch_matmul():
    template = [
        '%10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>',
        '%11 = linalg.batch_matmul ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>',
        'flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>'
    ]
    assert tune.get_shapes_batch_matmul(template) == ([1, 32, 1024], [1, 1024, 32], [1, 32, 32])

def test_generate_solutions():
    pass

def test_generate_constraints():
    pass

def test_apply_params_mmt():
    pass

def test_apply_params_conv():
    pass

def test_apply_params_contract():
    pass

def test_apply_params_batch_matmul():
    pass