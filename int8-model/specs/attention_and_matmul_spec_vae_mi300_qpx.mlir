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
// Convolution Tuning for VAE Batch Size 1
//===----------------------------------------------------------------------===//

transform.named_sequence @match_generic_1024x1024_16_1152_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<1026x1026x128xf16>, %arg2: tensor<16x3x3x128xf16>, %arg3: tensor<1024x1024x16xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<1026x1026x128xf16>, tensor<16x3x3x128xf16>) outs(%arg3 : tensor<1024x1024x16xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1024x1024x16xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 2, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 32, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_1024x1024_128_1152_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<1026x1026x128xf16>, %arg2: tensor<128x3x3x128xf16>, %arg3: tensor<1024x1024x128xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<1026x1026x128xf16>, tensor<128x3x3x128xf16>) outs(%arg3 : tensor<1024x1024x128xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1024x1024x128xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [2, 2, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_1024x1024_128_2304_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<1026x1026x256xf16>, %arg2: tensor<128x3x3x256xf16>, %arg3: tensor<1024x1024x128xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<1026x1026x256xf16>, tensor<128x3x3x256xf16>) outs(%arg3 : tensor<1024x1024x128xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1024x1024x128xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_1024x1024_256_2304_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<1026x1026x256xf16>, %arg2: tensor<256x3x3x256xf16>, %arg3: tensor<1024x1024x256xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<1026x1026x256xf16>, tensor<256x3x3x256xf16>) outs(%arg3 : tensor<1024x1024x256xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1024x1024x256xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_512x512_256_2304_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<514x514x256xf16>, %arg2: tensor<256x3x3x256xf16>, %arg3: tensor<512x512x256xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<514x514x256xf16>, tensor<256x3x3x256xf16>) outs(%arg3 : tensor<512x512x256xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<512x512x256xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [2, 2, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_512x512_256_4608_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<514x514x512xf16>, %arg2: tensor<256x3x3x512xf16>, %arg3: tensor<512x512x256xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<514x514x512xf16>, tensor<256x3x3x512xf16>) outs(%arg3 : tensor<512x512x256xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<512x512x256xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_512x512_512_4608_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<514x514x512xf16>, %arg2: tensor<512x3x3x512xf16>, %arg3: tensor<512x512x512xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<514x514x512xf16>, tensor<512x3x3x512xf16>) outs(%arg3 : tensor<512x512x512xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<512x512x512xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_256x256_512_4608_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<258x258x512xf16>, %arg2: tensor<512x3x3x512xf16>, %arg3: tensor<256x256x512xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<258x258x512xf16>, tensor<512x3x3x512xf16>) outs(%arg3 : tensor<256x256x512xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<256x256x512xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_128x128_512_4608_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<130x130x512xf16>, %arg2: tensor<512x3x3x512xf16>, %arg3: tensor<128x128x512xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<130x130x512xf16>, tensor<512x3x3x512xf16>) outs(%arg3 : tensor<128x128x512xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128x512xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_generic_128x128_512_144_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<130x130x16xf16>, %arg2: tensor<512x3x3x16xf16>, %arg3: tensor<128x128x512xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg2 : tensor<130x130x16xf16>, tensor<512x3x3x16xf16>) outs(%arg3 : tensor<128x128x512xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128x512xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 1], subgroup = [1, 4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Contraction Tuning for VAE Batch Size 1
//===----------------------------------------------------------------------===//

transform.named_sequence @match_contraction_1048576x128x1048576_f16xf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<256x1048576xf16>, %arg2: tensor<128x256xf16>, %arg3: tensor<1048576x128xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<256x1048576xf16>, tensor<128x256xf16>) outs(%arg3 : tensor<1048576x128xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1048576x128xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, promote_operands = [0, 1], reduction = [0, 0, 8], subgroup = [4, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_262144x256x262144_f16xf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<512x262144xf16>, %arg2: tensor<256x512xf16>, %arg3: tensor<262144x256xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<512x262144xf16>, tensor<256x512xf16>) outs(%arg3 : tensor<262144x256xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<262144x256xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16384x512x16384_f16xf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x16384xf16>, %arg2: tensor<16384x512xf16>, %arg3: tensor<16384x512xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x16384xf16>, tensor<16384x512xf16>) outs(%arg3 : tensor<16384x512xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16384x512xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [4, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16384x16384x512_f16xf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x512xf16>, %arg2: tensor<512x16384xf16>, %arg3: tensor<16384x16384xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x512xf16>, tensor<512x16384xf16>) outs(%arg3 : tensor<16384x16384xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16384x16384xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_512x16384x512_f16xf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x512xf16>, %arg2: tensor<512x512xf16>, %arg3: tensor<512x16384xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x512xf16>, tensor<512x512xf16>) outs(%arg3 : tensor<512x16384xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<512x16384xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_16384x512x512_f16xf16xf32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<16384x512xf16>, %arg2: tensor<512x512xf16>, %arg3: tensor<16384x512xf32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<16384x512xf16>, tensor<512x512xf16>) outs(%arg3 : tensor<16384x512xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %in_0 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16384x512xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
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

        // VAE Batch Size 1 Convolutions.
        @match_generic_1024x1024_16_1152_ -> @apply_op_config, 
        @match_generic_1024x1024_128_1152_ -> @apply_op_config, 
        @match_generic_1024x1024_128_2304_ -> @apply_op_config, 
        @match_generic_1024x1024_256_2304_ -> @apply_op_config, 
        @match_generic_512x512_256_2304_ -> @apply_op_config, 
        @match_generic_512x512_256_4608_ -> @apply_op_config, 
        @match_generic_512x512_512_4608_ -> @apply_op_config, 
        @match_generic_256x256_512_4608_ -> @apply_op_config, 
        @match_generic_128x128_512_4608_ -> @apply_op_config, 
        @match_generic_128x128_512_144_ -> @apply_op_config,
        
        // VAE Batch Size 1 Contractions.
        @match_contraction_1048576x128x1048576_f16xf16xf32 -> @apply_op_config, 
        @match_contraction_262144x256x262144_f16xf16xf32 -> @apply_op_config, 
        @match_contraction_16384x512x16384_f16xf16xf32 -> @apply_op_config, 
        @match_contraction_16384x16384x512_f16xf16xf32 -> @apply_op_config, 
        @match_contraction_512x16384x512_f16xf16xf32 -> @apply_op_config, 
        @match_contraction_16384x512x512_f16xf16xf32 -> @apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated_root : !transform.any_op
  }
}
