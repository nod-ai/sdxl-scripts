func.func @main_0(%arg0: tensor<64x72x1280xf16>, %arg1: tensor<64x1280x1280xf16>) -> tensor<64x72x1280xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<64x72x1280xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x72x1280xf32>) -> tensor<64x72x1280xf32>
  %7 = linalg.batch_matmul
    ins(%arg0, %arg1 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>)
    outs(%6 : tensor<64x72x1280xf32>) -> tensor<64x72x1280xf32>
  return %7 : tensor<64x72x1280xf32>
}
