module attributes {transform.with_named_sequence} {
  
//===----------------------------------------------------------------------===//
// Tuning infra
//===----------------------------------------------------------------------===//

transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
  transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
  transform.yield 
}
transform.named_sequence @apply_attn_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}, %arg2: !transform.any_param {transform.readonly}) {
  transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
  transform.annotate %arg0 "decomposition_config" = %arg2 : !transform.any_op, !transform.any_param
  transform.yield 
}

//===----------------------------------------------------------------------===//
// Attention tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_attention_f16(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
  transform.match.operation_name %arg0 ["iree_linalg_ext.attention"] : !transform.any_op
  %0 = transform.get_operand %arg0[0] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %0 = tensor<?x?x?x?xf16> : !transform.any_value
  %1 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{promote_operands = [1, 2], reduction = [0, 0, 0, 0, 0, 64], workgroup = [1, 1, 128, 0, 0, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 4] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>> -> !transform.any_param
  %2 = transform.param.constant {pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, promote_operands = [1], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64}>}, qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>, promote_operands = [1], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64}>}} -> !transform.any_param
  transform.yield %arg0, %1, %2 : !transform.any_op, !transform.any_param, !transform.any_param
}
transform.named_sequence @match_attention_f8(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
  transform.match.operation_name %arg0 ["iree_linalg_ext.attention"] : !transform.any_op
  %0 = transform.get_operand %arg0[0] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %0 = tensor<?x?x?x?xf8E4M3FNUZ> : !transform.any_value
  %1 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{promote_operands = [1, 2], reduction = [0, 0, 0, 0, 0, 64], workgroup = [1, 1, 64, 0, 0, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 4] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>> -> !transform.any_param
  %2 = transform.param.constant {pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_16x16x32_F8E4M3FNUZ>, promote_operands = [1], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64}>}, qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>, promote_operands = [1], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64}>}} -> !transform.any_param
  transform.yield %arg0, %1, %2 : !transform.any_op, !transform.any_param, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Convolution FHWC Filter Layout Tuning for UNET Batch Size 14
//===----------------------------------------------------------------------===//

transform.named_sequence @match_generic_28x64x64_640_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x1280xi8>, %arg2: tensor<640x3x3x1280xi8>, %arg3: tensor<28x64x64x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x66x66x1280xi8>, tensor<640x3x3x1280xi8>) outs(%arg3 : tensor<28x64x64x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x64x64_640_17280_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x1920xi8>, %arg2: tensor<640x3x3x1920xi8>, %arg3: tensor<28x64x64x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x66x66x1920xi8>, tensor<640x3x3x1920xi8>) outs(%arg3 : tensor<28x64x64x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 8, 1, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 8, 16, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x128x128_320_2880_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x320xi8>, %arg2: tensor<320x3x3x320xi8>, %arg3: tensor<28x128x128x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x130x130x320xi8>, tensor<320x3x3x320xi8>) outs(%arg3 : tensor<28x128x128x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 2, 2, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 2, 32, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x128x128_320_8640_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x960xi8>, %arg2: tensor<320x3x3x960xi8>, %arg3: tensor<28x128x128x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x130x130x960xi8>, tensor<320x3x3x960xi8>) outs(%arg3 : tensor<28x128x128x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 16, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x128x128_640_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x640xi8>, %arg2: tensor<640x3x3x640xi8>, %arg3: tensor<28x128x128x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x130x130x640xi8>, tensor<640x3x3x640xi8>) outs(%arg3 : tensor<28x128x128x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x128x128x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 1, 2, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 1, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x64x64_640_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x640xi8>, %arg2: tensor<640x3x3x640xi8>, %arg3: tensor<28x64x64x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x66x66x640xi8>, tensor<640x3x3x640xi8>) outs(%arg3 : tensor<28x64x64x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 2, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 4, 16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x32x32_1280_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x34x34x1280xi8>, %arg2: tensor<1280x3x3x1280xi8>, %arg3: tensor<28x32x32x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x34x34x1280xi8>, tensor<1280x3x3x1280xi8>) outs(%arg3 : tensor<28x32x32x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 4, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x32x32_1280_23040_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x34x34x2560xi8>, %arg2: tensor<1280x3x3x2560xi8>, %arg3: tensor<28x32x32x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x34x34x2560xi8>, tensor<1280x3x3x2560xi8>) outs(%arg3 : tensor<28x32x32x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 2, 1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 8, 16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x64x64_1280_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x1280xi8>, %arg2: tensor<1280x3x3x1280xi8>, %arg3: tensor<28x64x64x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x66x66x1280xi8>, tensor<1280x3x3x1280xi8>) outs(%arg3 : tensor<28x64x64x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x64x64x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 3], subgroup = [4, 2, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [4, 4, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_28x128x128_320_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x640xi8>, %arg2: tensor<320x3x3x640xi8>, %arg3: tensor<28x128x128x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<28x130x130x640xi8>, tensor<320x3x3x640xi8>) outs(%arg3 : tensor<28x128x128x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 2, 2, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 2, 32, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Convolution HWCF Filter Layout Tuning for UNET Batch Size 14
//===----------------------------------------------------------------------===//

transform.named_sequence @match_conv_2d_nhwc_hwcf_28x32x32_1280_17280_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x34x34x1920xi8>, %arg2: tensor<3x3x1920x1280xi8>, %arg3: tensor<28x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x34x34x1920xi8>, tensor<3x3x1920x1280xi8>) outs(%arg3 : tensor<28x32x32x1280xi32>) -> tensor<28x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [2, 4, 16, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x64x64_640_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x1280xi8>, %arg2: tensor<3x3x1280x640xi8>, %arg3: tensor<28x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x66x66x1280xi8>, tensor<3x3x1280x640xi8>) outs(%arg3 : tensor<28x64x64x640xi32>) -> tensor<28x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x64x64_640_17280_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x1920xi8>, %arg2: tensor<3x3x1920x640xi8>, %arg3: tensor<28x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x66x66x1920xi8>, tensor<3x3x1920x640xi8>) outs(%arg3 : tensor<28x64x64x640xi32>) -> tensor<28x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 4, 4, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 4, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x128x128_320_2880_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x320xi8>, %arg2: tensor<3x3x320x320xi8>, %arg3: tensor<28x128x128x320xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x130x130x320xi8>, tensor<3x3x320x320xi8>) outs(%arg3 : tensor<28x128x128x320xi32>) -> tensor<28x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 8, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x128x128_320_8640_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x960xi8>, %arg2: tensor<3x3x960x320xi8>, %arg3: tensor<28x128x128x320xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x130x130x960xi8>, tensor<3x3x960x320xi8>) outs(%arg3 : tensor<28x128x128x320xi32>) -> tensor<28x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 16, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x128x128_640_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x640xi8>, %arg2: tensor<3x3x640x640xi8>, %arg3: tensor<28x128x128x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x130x130x640xi8>, tensor<3x3x640x640xi8>) outs(%arg3 : tensor<28x128x128x640xi32>) -> tensor<28x128x128x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 4, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [4, 1, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x64x64_640_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x640xi8>, %arg2: tensor<3x3x640x640xi8>, %arg3: tensor<28x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x66x66x640xi8>, tensor<3x3x640x640xi8>) outs(%arg3 : tensor<28x64x64x640xi32>) -> tensor<28x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 2, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 4, 16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x32x32_1280_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x34x34x1280xi8>, %arg2: tensor<3x3x1280x1280xi8>, %arg3: tensor<28x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x34x34x1280xi8>, tensor<3x3x1280x1280xi8>) outs(%arg3 : tensor<28x32x32x1280xi32>) -> tensor<28x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 4, 16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x32x32_1280_23040_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x34x34x2560xi8>, %arg2: tensor<3x3x2560x1280xi8>, %arg3: tensor<28x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x34x34x2560xi8>, tensor<3x3x2560x1280xi8>) outs(%arg3 : tensor<28x32x32x1280xi32>) -> tensor<28x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 2, 4, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [4, 1, 32, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x64x64_1280_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x66x66x1280xi8>, %arg2: tensor<3x3x1280x1280xi8>, %arg3: tensor<28x64x64x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x66x66x1280xi8>, tensor<3x3x1280x1280xi8>) outs(%arg3 : tensor<28x64x64x1280xi32>) -> tensor<28x64x64x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 4, 2, 4, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 4, 32, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_28x128x128_320_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x130x130x640xi8>, %arg2: tensor<3x3x640x320xi8>, %arg3: tensor<28x128x128x320xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<28x130x130x640xi8>, tensor<3x3x640x320xi8>) outs(%arg3 : tensor<28x128x128x320xi32>) -> tensor<28x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 2, 2, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 2, 32, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Contraction Tuning for UNET Batch Size 14
//===----------------------------------------------------------------------===//

transform.named_sequence @match_contraction_28672x10240x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28672x1280xi8>, %arg2: tensor<10240x1280xi8>, %arg3: tensor<28672x10240xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28672x1280xi8>, tensor<10240x1280xi8>) outs(%arg3 : tensor<28672x10240xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28672x10240xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x20x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x1024x1280xi8>, %arg2: tensor<20x64x1280xi8>, %arg3: tensor<28x20x1024x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x1024x1280xi8>, tensor<20x64x1280xi8>) outs(%arg3 : tensor<28x20x1024x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x20x1024x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 4, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 4, 64, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28672x1280x5120_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28672x5120xi8>, %arg2: tensor<1280x5120xi8>, %arg3: tensor<28672x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28672x5120xi8>, tensor<1280x5120xi8>) outs(%arg3 : tensor<28672x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28672x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 8, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28672x1280x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28672x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<28672x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28672x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<28672x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28672x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 8, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x1280x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x1024x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<28x1280x1024xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x1024x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<28x1280x1024xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x1280x1024xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [2, 2, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 256, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_114688x5120x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<114688x640xi8>, %arg2: tensor<5120x640xi8>, %arg3: tensor<114688x5120xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<114688x640xi8>, tensor<5120x640xi8>) outs(%arg3 : tensor<114688x5120xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<114688x5120xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 8, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x10x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x4096x640xi8>, %arg2: tensor<10x64x640xi8>, %arg3: tensor<28x10x4096x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x4096x640xi8>, tensor<10x64x640xi8>) outs(%arg3 : tensor<28x10x4096x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x10x4096x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [4, 1, 1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 2, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_114688x640x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<114688x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<114688x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<114688x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<114688x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<114688x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_114688x640x2560_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<114688x2560xi8>, %arg2: tensor<640x2560xi8>, %arg3: tensor<114688x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<114688x2560xi8>, tensor<640x2560xi8>) outs(%arg3 : tensor<114688x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<114688x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x640x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x4096x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<28x640x4096xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x4096x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<28x640x4096xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x640x4096xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28672x1280x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28672x640xi8>, %arg2: tensor<1280x640xi8>, %arg3: tensor<28672x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28672x640xi8>, tensor<1280x640xi8>) outs(%arg3 : tensor<28672x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28672x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_114688x640x320_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<114688x320xi8>, %arg2: tensor<640x320xi8>, %arg3: tensor<114688x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<114688x320xi8>, tensor<640x320xi8>) outs(%arg3 : tensor<114688x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<114688x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x1024x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x1024x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<28x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x1024x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<28x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 8, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x4096x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x4096x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<28x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x4096x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<28x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x1024x1920_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x1920x1024xi8>, %arg2: tensor<1280x1920xi8>, %arg3: tensor<28x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x1920x1024xi8>, tensor<1280x1920xi8>) outs(%arg3 : tensor<28x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x4096x960_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x960x4096xi8>, %arg2: tensor<640x960xi8>, %arg3: tensor<28x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x960x4096xi8>, tensor<640x960xi8>) outs(%arg3 : tensor<28x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x4096x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x1280x4096xi8>, %arg2: tensor<640x1280xi8>, %arg3: tensor<28x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x1280x4096xi8>, tensor<640x1280xi8>) outs(%arg3 : tensor<28x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x4096x1920_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x1920x4096xi8>, %arg2: tensor<640x1920xi8>, %arg3: tensor<28x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x1920x4096xi8>, tensor<640x1920xi8>) outs(%arg3 : tensor<28x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x1024x2560_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x2560x1024xi8>, %arg2: tensor<1280x2560xi8>, %arg3: tensor<28x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x2560x1024xi8>, tensor<1280x2560xi8>) outs(%arg3 : tensor<28x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [2, 2, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x16384x960_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x960x16384xi8>, %arg2: tensor<320x960xi8>, %arg3: tensor<28x16384x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x960x16384xi8>, tensor<320x960xi8>) outs(%arg3 : tensor<28x16384x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x16384x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 1], subgroup = [1, 4, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 512, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_28x16384x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<28x640x16384xi8>, %arg2: tensor<320x640xi8>, %arg3: tensor<28x16384x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<28x640x16384xi8>, tensor<320x640xi8>) outs(%arg3 : tensor<28x16384x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<28x16384x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 1], subgroup = [1, 4, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 512, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Contraction Tuning for UNET Batch Size 8
//===----------------------------------------------------------------------===//

transform.named_sequence @match_contraction_16x16384x960_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x960x16384xi8>, %arg2: tensor<320x960xi8>, %arg3: tensor<16x16384x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x960x16384xi8>, tensor<320x960xi8>) outs(%arg3 : tensor<16x16384x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x16384x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 2, 20, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x16384x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x640x16384xi8>, %arg2: tensor<320x640xi8>, %arg3: tensor<16x16384x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x640x16384xi8>, tensor<320x640xi8>) outs(%arg3 : tensor<16x16384x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x16384x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 2, 20, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x20x64_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x64x2048xi8>, %arg2: tensor<20x64x2048xi8>, %arg3: tensor<16x20x64x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x64x2048xi8>, tensor<20x64x2048xi8>) outs(%arg3 : tensor<16x20x64x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x20x64x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 8], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 2, 32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x1280x64_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x64x2048xi8>, %arg2: tensor<1280x2048xi8>, %arg3: tensor<16x1280x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x64x2048xi8>, tensor<1280x2048xi8>) outs(%arg3 : tensor<16x1280x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x1280x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [2, 5, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 160, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x640x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x4096x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<16x640x4096xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x4096x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<16x640x4096xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x640x4096xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 2, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [4, 128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_65536x640x2560_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<65536x2560xi8>, %arg2: tensor<640x2560xi8>, %arg3: tensor<65536x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<65536x2560xi8>, tensor<640x2560xi8>) outs(%arg3 : tensor<65536x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<65536x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_65536x640x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<65536x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<65536x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<65536x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<65536x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<65536x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x10x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x4096x640xi8>, %arg2: tensor<10x64x640xi8>, %arg3: tensor<16x10x4096x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x4096x640xi8>, tensor<10x64x640xi8>) outs(%arg3 : tensor<16x10x4096x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x10x4096x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 2, 2, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 2, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_65536x5120x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<65536x640xi8>, %arg2: tensor<5120x640xi8>, %arg3: tensor<65536x5120xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<65536x640xi8>, tensor<5120x640xi8>) outs(%arg3 : tensor<65536x5120xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<65536x5120xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x1280x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x1024x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<16x1280x1024xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x1024x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<16x1280x1024xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x1280x1024xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [2, 2, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [8, 128, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16384x1280x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<16384x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<16384x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16384x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16384x1280x5120_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x5120xi8>, %arg2: tensor<1280x5120xi8>, %arg3: tensor<16384x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x5120xi8>, tensor<1280x5120xi8>) outs(%arg3 : tensor<16384x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16384x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x20x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x1024x1280xi8>, %arg2: tensor<20x64x1280xi8>, %arg3: tensor<16x20x1024x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x1024x1280xi8>, tensor<20x64x1280xi8>) outs(%arg3 : tensor<16x20x1024x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x20x1024x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 10, 2, 1, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 10, 64, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16384x10240x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x1280xi8>, %arg2: tensor<10240x1280xi8>, %arg3: tensor<16384x10240xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x1280xi8>, tensor<10240x1280xi8>) outs(%arg3 : tensor<16384x10240xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16384x10240xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16x1024x2560_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16x2560x1024xi8>, %arg2: tensor<1280x2560xi8>, %arg3: tensor<16x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16x2560x1024xi8>, tensor<1280x2560xi8>) outs(%arg3 : tensor<16x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<16x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 2, 20, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated_root = transform.foreach_match in %arg0 
        // Attention.
        @match_attention_f16 -> @apply_attn_op_config, 
        @match_attention_f8 -> @apply_attn_op_config, 

        // Unet Batch Size 14 Convolution FHWC Filter Layout.
        @match_generic_28x64x64_640_11520_ -> @apply_op_config, 
        @match_generic_28x64x64_640_17280_ -> @apply_op_config, 
        @match_generic_28x128x128_320_2880_ -> @apply_op_config, 
        @match_generic_28x128x128_320_8640_ -> @apply_op_config, 
        @match_generic_28x128x128_640_5760_ -> @apply_op_config, 
        @match_generic_28x64x64_640_5760_ -> @apply_op_config, 
        @match_generic_28x32x32_1280_11520_ -> @apply_op_config, 
        @match_generic_28x32x32_1280_23040_ -> @apply_op_config, 
        @match_generic_28x64x64_1280_11520_ -> @apply_op_config, 
        @match_generic_28x128x128_320_5760_ -> @apply_op_config, 

        // Unet Batch Size 14 Convolution HWCF Filter Layout.
        @match_conv_2d_nhwc_hwcf_28x32x32_1280_17280_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x64x64_640_11520_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x64x64_640_17280_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x128x128_320_2880_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x128x128_320_8640_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x128x128_640_5760_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x64x64_640_5760_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x32x32_1280_11520_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x32x32_1280_23040_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x64x64_1280_11520_ -> @apply_op_config, 
        @match_conv_2d_nhwc_hwcf_28x128x128_320_5760_ -> @apply_op_config, 

        // Unet Batch Size 14 Contraction.
        @match_contraction_28672x10240x1280_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x20x1024_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28672x1280x5120_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28672x1280x1280_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x1280x1024_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_114688x5120x640_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x10x4096_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_114688x640x640_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_114688x640x2560_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x640x4096_i8xi8xi32 -> @apply_op_config,
        @match_contraction_28672x1280x640_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_114688x640x320_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x1024x1024_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x4096x4096_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x1024x1920_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x4096x960_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x4096x1280_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x4096x1920_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x1024x2560_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x16384x960_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_28x16384x640_i8xi8xi32 -> @apply_op_config, 

        // Unet Batch Size 8 Contractions.
        @match_contraction_16x16384x960_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x16384x640_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x20x64_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x1280x64_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x640x4096_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_65536x640x2560_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_65536x640x640_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x10x4096_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_65536x5120x640_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x1280x1024_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16384x1280x1280_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16384x1280x5120_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x20x1024_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16384x10240x1280_i8xi8xi32 -> @apply_op_config, 
        @match_contraction_16x1024x2560_i8xi8xi32 -> @apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated_root : !transform.any_op
  }
}
