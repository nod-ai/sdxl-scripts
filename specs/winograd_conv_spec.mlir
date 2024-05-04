
module attributes { transform.with_named_sequence } {

  //===------------------------------------------------------===
  // Convolution
  //===------------------------------------------------------===
  
  transform.named_sequence @match_conv2x4x128x128x3x3x320(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 320 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 4 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x320x128x128x3x3x320(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 320 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 320 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x320x64x64x3x3x640(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 320 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x640x64x64x3x3x640(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x640x32x32x3x3x1280(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x1280x32x32x3x3x1280(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x2560x32x32x3x3x1280(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 2560 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x1920x32x32x3x3x1280(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 1920 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 32 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x1280x64x64x3x3x1280(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x1920x64x64x3x3x640(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 1920 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x1280x64x64x3x3x640(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 1280 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x960x64x64x3x3x640(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 960 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 64 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x640x128x128x3x3x640(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x960x128x128x3x3x320(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 320 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 960 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @match_conv2x640x128x128x3x3x320(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["linalg.conv_2d_nchw_fchw"] : !transform.any_op
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c7 = transform.param.constant 7 : i64 -> !transform.param<i64>
      %cN = transform.param.constant 2 : i64 -> !transform.param<i64>
      %cF = transform.param.constant 320 : i64 -> !transform.param<i64>
      %cC = transform.param.constant 640 : i64 -> !transform.param<i64>
      %cH = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cW = transform.param.constant 128 : i64 -> !transform.param<i64>
      %cP = transform.param.constant 3 : i64 -> !transform.param<i64>
      %cQ = transform.param.constant 3 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c7 : !transform.param<i64>

      %target_sizes = transform.merge_handles %cN, %cF, %cH, %cW, %cC, %cP, %cQ : !transform.param<i64>
      %dims = transform.match.structured.dim %arg1[all] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %target_sizes, %dims : !transform.param<i64>

      transform.match.structured.dim %arg1[0, 1, 2, 3] { parallel } : !transform.any_op
      transform.match.structured.dim %arg1[-3, -2, -1] { reduction } : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @placeholder(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %arg0 ["fail"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  //===------------------------------------------------------===
  // Annotation and Application
  //===------------------------------------------------------===

  transform.named_sequence @annotate_op(%target: !transform.any_op {transform.readonly}) {
    transform.annotate %target "winograd_conv" : !transform.any_op
    transform.yield
  }


  transform.named_sequence @__annotate_winograd_convs(%func: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %func
        // @match_conv2x4x128x128x3x3x320 -> @annotate_op,
        // @match_conv2x320x128x128x3x3x4 -> @annotate_op,
        // @match_conv2x320x128x128x3x3x320 -> @annotate_op,
        // @match_conv2x320x64x64x3x3x640 -> @annotate_op,
        // @match_conv2x640x64x64x3x3x640 -> @annotate_op,
        // @match_conv2x640x32x32x3x3x1280 -> @annotate_op,
        // @match_conv2x1280x32x32x3x3x1280 -> @annotate_op,
        // @match_conv2x1920x64x64x3x3x640 -> @annotate_op,
        // @match_conv2x1280x64x64x3x3x640 -> @annotate_op,
        // @match_conv2x960x64x64x3x3x640 -> @annotate_op,
        // @match_conv2x640x128x128x3x3x640 -> @annotate_op,
        // @match_conv2x960x128x128x3x3x320 -> @annotate_op,
        // @match_conv2x640x128x128x3x3x320 -> @annotate_op,
        @match_conv2x2560x32x32x3x3x1280 -> @annotate_op,
        @match_conv2x1920x32x32x3x3x1280 -> @annotate_op,
        @match_conv2x1280x64x64x3x3x1280 -> @annotate_op,
        @placeholder -> @annotate_op
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
