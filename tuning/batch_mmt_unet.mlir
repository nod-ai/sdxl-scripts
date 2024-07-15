!matA_0 = tensor<2x2048x1280xf16>
!matB_0 = tensor<2x10240x1280xf16>
!matC_0 = tensor<2x2048x10240xf32>

func.func @main_0(%arg0: !matA_0, %arg1: !matB_0) -> !matC_0 {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matC_0
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matC_0) -> !matC_0
  %8 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : !matA_0, !matB_0) outs(%6 : !matC_0) -> !matC_0
  return %8 : !matC_0
}

!matA_1 = tensor<2x4096x640xi8>
!matB_1 = tensor<2x640x640xi8>
!matC_1 = tensor<2x4096x640xi32>

func.func @main_1(%arg0: !matA_1, %arg1: !matB_1) -> !matC_1 {
  %cst = arith.constant 0 : i32
  %5 = tensor.empty() : !matC_1
  %6 = linalg.fill ins(%cst : i32) outs(%5 : !matC_1) -> !matC_1
  %8 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : !matA_1, !matB_1) outs(%6 : !matC_1) -> !matC_1
  return %8 : !matC_1
}
