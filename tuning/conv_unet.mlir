func.func @main_0(%arg0: tensor<2x34x34x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>) -> tensor<2x32x32x1280xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<2x32x32x1280xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
  %28 = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : vector<2xi64>,
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>,
    strides = dense<1> : vector<2xi64>
  } ins(%arg0, %arg1 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>)
    outs(%6 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
  return %28 : tensor<2x32x32x1280xf32>
}
