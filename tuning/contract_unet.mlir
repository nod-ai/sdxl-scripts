func.func @main_0(%arg0: tensor<2x1024x1280xf16>, %arg1: tensor<3x20x64x1280xf16>) -> tensor<3x2x20x1024x64xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<3x2x20x1024x64xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<3x2x20x1024x64xf32>) -> tensor<3x2x20x1024x64xf32>
  %14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<2x1024x1280xf16>, tensor<3x20x64x1280xf16>) outs(%6 : tensor<3x2x20x1024x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_0 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    linalg.yield %19 : f32
  } -> tensor<3x2x20x1024x64xf32>
  return %14 : tensor<3x2x20x1024x64xf32>
}

func.func @main_1(%arg0: tensor<2x64x2048xf16>, %arg1: tensor<20x64x2048xf16>) -> tensor<2x20x64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<2x20x64x64xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x20x64x64xf32>) -> tensor<2x20x64x64xf32>
  %14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<2x64x2048xf16>, tensor<20x64x2048xf16>) outs(%6 : tensor<2x20x64x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_0 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    linalg.yield %19 : f32
  } -> tensor<2x20x64x64xf32>
  return %14 : tensor<2x20x64x64xf32>
}

func.func @main_2(%arg0: tensor<2x1024x1280xf16>, %arg1: tensor<20x64x1280xf16>) -> tensor<2x20x1024x64xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<2x20x1024x64xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x20x1024x64xf32>) -> tensor<2x20x1024x64xf32>
  %14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>) outs(%6 : tensor<2x20x1024x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_0 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    linalg.yield %19 : f32
  } -> tensor<2x20x1024x64xf32>
  return %14 : tensor<2x20x1024x64xf32>
}

func.func @main_3(%arg0: tensor<2x4096x640xf16>, %arg1: tensor<3x10x64x640xf16>) -> tensor<3x2x10x4096x64xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<3x2x10x4096x64xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<3x2x10x4096x64xf32>) -> tensor<3x2x10x4096x64xf32>
  %14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<2x4096x640xf16>, tensor<3x10x64x640xf16>) outs(%6 : tensor<3x2x10x4096x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_0 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    linalg.yield %19 : f32
  } -> tensor<3x2x10x4096x64xf32>
  return %14 : tensor<3x2x10x4096x64xf32>
}

func.func @main_4(%arg0: tensor<2x64x2048xf16>, %arg1: tensor<10x64x2048xf16>) -> tensor<2x10x64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : tensor<2x10x64x64xf32>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2x10x64x64xf32>) -> tensor<2x10x64x64xf32>
  %14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<2x64x2048xf16>, tensor<10x64x2048xf16>) outs(%6 : tensor<2x10x64x64xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %16 = arith.extf %in : f16 to f32
    %17 = arith.extf %in_0 : f16 to f32
    %18 = arith.mulf %16, %17 : f32
    %19 = arith.addf %out, %18 : f32
    linalg.yield %19 : f32
  } -> tensor<2x10x64x64xf32>
  return %14 : tensor<2x10x64x64xf32>
}
