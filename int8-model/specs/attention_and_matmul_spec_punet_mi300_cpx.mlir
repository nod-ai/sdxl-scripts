module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Tuning infra
//===----------------------------------------------------------------------===//

  transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
                                                 %config: !transform.any_param {transform.readonly},
                                                 %decomposition_config: !transform.any_param {transform.readonly}) {
    transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
    // transform.print %attention {name = "Applied attention config"} : !transform.any_op
    transform.yield
  }
  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield 
  }

//===----------------------------------------------------------------------===//
// Attention tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

    %decomposition_config = transform.param.constant {
      qk_attrs = {attention_qk_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

transform.named_sequence @match_attention_f8(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf8E4M3FNUZ> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

    %decomposition_config = transform.param.constant {
      qk_attrs = {attention_qk_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

transform.named_sequence @match_attention_f8_m_1024(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x1024x?xf8E4M3FNUZ> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

    %decomposition_config = transform.param.constant {
      qk_attrs = {attention_qk_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_16x16x32_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

transform.named_sequence @match_attention_f8_m_4096(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x4096x?xf8E4M3FNUZ> : !transform.any_value

    %config = transform.param.constant #iree_codegen.compilation_info<
            lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 128, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
            translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                                              workgroup_size = [64, 4]
                                                              subgroup_size = 64 ,
              {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
    -> !transform.any_param

    %decomposition_config = transform.param.constant {
      qk_attrs = {attention_qk_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x16_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
      pv_attrs = {attention_pv_matmul,
                  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F8E4M3FNUZ>,
                                                               subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
    } -> !transform.any_param

    transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
  }

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Contraction Tuning for UNET Batch Size 1
//===----------------------------------------------------------------------===//

transform.named_sequence @match_contraction_2048x1280x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2048x640xi8>, %arg2: tensor<1280x640xi8>, %arg3: tensor<2048x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2048x640xi8>, tensor<1280x640xi8>) outs(%arg3 : tensor<2048x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2048x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_8192x640x320_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<8192x320xi8>, %arg2: tensor<640x320xi8>, %arg3: tensor<8192x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<8192x320xi8>, tensor<640x320xi8>) outs(%arg3 : tensor<8192x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<8192x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x1024x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1024x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<2x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x1024x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<2x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [2, 2, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x4096x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x4096x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<2x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x4096x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<2x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x4096x960_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x960x4096xi8>, %arg2: tensor<640x960xi8>, %arg3: tensor<2x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x960x4096xi8>, tensor<640x960xi8>) outs(%arg3 : tensor<2x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 1], subgroup = [1, 4, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x1024x1920_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1920x1024xi8>, %arg2: tensor<1280x1920xi8>, %arg3: tensor<2x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x1920x1024xi8>, tensor<1280x1920xi8>) outs(%arg3 : tensor<2x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 2, 20, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x4096x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1280x4096xi8>, %arg2: tensor<640x1280xi8>, %arg3: tensor<2x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x1280x4096xi8>, tensor<640x1280xi8>) outs(%arg3 : tensor<2x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 1, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x10x64_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x64x2048xi8>, %arg2: tensor<10x64x2048xi8>, %arg3: tensor<2x10x64x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x64x2048xi8>, tensor<10x64x2048xi8>) outs(%arg3 : tensor<2x10x64x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x10x64x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 8], subgroup = [1, 2, 2, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 2, 32, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x4096x1920_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1920x4096xi8>, %arg2: tensor<640x1920xi8>, %arg3: tensor<2x4096x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x1920x4096xi8>, tensor<640x1920xi8>) outs(%arg3 : tensor<2x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x4096x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 1], subgroup = [1, 2, 8, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x640x64_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x64x2048xi8>, %arg2: tensor<640x2048xi8>, %arg3: tensor<2x640x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x64x2048xi8>, tensor<640x2048xi8>) outs(%arg3 : tensor<2x640x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x640x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [2, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [2, 128, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x1024x2560_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x2560x1024xi8>, %arg2: tensor<1280x2560xi8>, %arg3: tensor<2x1024x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x2560x1024xi8>, tensor<1280x2560xi8>) outs(%arg3 : tensor<2x1024x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x1024x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [1, 2, 20, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 64, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x16384x960_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x960x16384xi8>, %arg2: tensor<320x960xi8>, %arg3: tensor<2x16384x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x960x16384xi8>, tensor<320x960xi8>) outs(%arg3 : tensor<2x16384x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x16384x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 10, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x16384x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x640x16384xi8>, %arg2: tensor<320x640xi8>, %arg3: tensor<2x16384x320xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x640x16384xi8>, tensor<320x640xi8>) outs(%arg3 : tensor<2x16384x320xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x16384x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 1, 10, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x640x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x4096x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<2x640x4096xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x4096x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<2x640x4096xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x640x4096xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 4, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_8192x640x2560_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<8192x2560xi8>, %arg2: tensor<640x2560xi8>, %arg3: tensor<8192x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<8192x2560xi8>, tensor<640x2560xi8>) outs(%arg3 : tensor<8192x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<8192x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [2, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x20x64_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x64x2048xi8>, %arg2: tensor<20x64x2048xi8>, %arg3: tensor<2x20x64x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x64x2048xi8>, tensor<20x64x2048xi8>) outs(%arg3 : tensor<2x20x64x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x20x64x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [1, 1, 1, 4, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [2, 1, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_8192x640x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<8192x640xi8>, %arg2: tensor<640x640xi8>, %arg3: tensor<8192x640xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<8192x640xi8>, tensor<640x640xi8>) outs(%arg3 : tensor<8192x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<8192x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 4], subgroup = [1, 2, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x1280x64_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x64x2048xi8>, %arg2: tensor<1280x2048xi8>, %arg3: tensor<2x1280x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x64x2048xi8>, tensor<1280x2048xi8>) outs(%arg3 : tensor<2x1280x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x1280x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 16], subgroup = [1, 1, 1, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 128, 32, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x10x4096_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x4096x640xi8>, %arg2: tensor<10x64x640xi8>, %arg3: tensor<2x10x4096x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x4096x640xi8>, tensor<10x64x640xi8>) outs(%arg3 : tensor<2x10x4096x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x10x4096x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 2, 4, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 2, 128, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_8192x5120x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<8192x640xi8>, %arg2: tensor<5120x640xi8>, %arg3: tensor<8192x5120xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<8192x640xi8>, tensor<5120x640xi8>) outs(%arg3 : tensor<8192x5120xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<8192x5120xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 4, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 256, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x1280x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1024x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<2x1280x1024xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x1024x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<2x1280x1024xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x1280x1024xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 4], subgroup = [1, 2, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2048x1280x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2048x1280xi8>, %arg2: tensor<1280x1280xi8>, %arg3: tensor<2048x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2048x1280xi8>, tensor<1280x1280xi8>) outs(%arg3 : tensor<2048x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2048x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [8, 5, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [256, 160, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2048x1280x5120_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2048x5120xi8>, %arg2: tensor<1280x5120xi8>, %arg3: tensor<2048x1280xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2048x5120xi8>, tensor<1280x5120xi8>) outs(%arg3 : tensor<2048x1280xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2048x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [4, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [128, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2x20x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1024x1280xi8>, %arg2: tensor<20x64x1280xi8>, %arg3: tensor<2x20x1024x64xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2x1024x1280xi8>, tensor<20x64x1280xi8>) outs(%arg3 : tensor<2x20x1024x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2x20x1024x64xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [1, 2, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 2, 64, 64, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_contraction_2048x10240x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2048x1280xi8>, %arg2: tensor<10240x1280xi8>, %arg3: tensor<2048x10240xi32>):
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<2048x1280xi8>, tensor<10240x1280xi8>) outs(%arg3 : tensor<2048x10240xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      linalg.yield %5 : i32
    } -> tensor<2048x10240xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 2], subgroup = [2, 8, 0], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Convolution Tuning for UNET Batch Size 1
//===----------------------------------------------------------------------===//

transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128_16_2880_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x130x130x320xf16>, %arg2: tensor<3x3x320x16xf16>, %arg3: tensor<2x128x128x16xf32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg1, %arg2 : tensor<2x130x130x320xf16>, tensor<3x3x320x16xf16>) outs(%arg3 : tensor<2x128x128x16xf32>) -> tensor<2x128x128x16xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [1, 1, 1, 1, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 32, 16, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64_640_2880_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x66x66x320xi8>, %arg2: tensor<3x3x320x640xi8>, %arg3: tensor<2x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x320xi8>, tensor<3x3x320x640xi8>) outs(%arg3 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 1], subgroup = [1, 4, 2, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 4, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32_1280_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x34x34x640xi8>, %arg2: tensor<3x3x640x1280xi8>, %arg3: tensor<2x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x34x34x640xi8>, tensor<3x3x640x1280xi8>) outs(%arg3 : tensor<2x32x32x1280xi32>) -> tensor<2x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 1], subgroup = [2, 4, 1, 5, 0], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 16, 16, 160, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64_640_8640_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x66x66x960xi8>, %arg2: tensor<3x3x960x640xi8>, %arg3: tensor<2x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x960xi8>, tensor<3x3x960x640xi8>) outs(%arg3 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 8, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 16, 16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64_640_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x66x66x1280xi8>, %arg2: tensor<3x3x1280x640xi8>, %arg3: tensor<2x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x1280xi8>, tensor<3x3x1280x640xi8>) outs(%arg3 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 2, 4, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 2, 64, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32_1280_17280_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x34x34x1920xi8>, %arg2: tensor<3x3x1920x1280xi8>, %arg3: tensor<2x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x34x34x1920xi8>, tensor<3x3x1920x1280xi8>) outs(%arg3 : tensor<2x32x32x1280xi32>) -> tensor<2x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 4], subgroup = [1, 8, 1, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 10 : i64, workgroup = [1, 8, 32, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [640, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128_320_2880_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x130x130x320xi8>, %arg2: tensor<3x3x320x320xi8>, %arg3: tensor<2x128x128x320xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x130x130x320xi8>, tensor<3x3x320x320xi8>) outs(%arg3 : tensor<2x128x128x320xi32>) -> tensor<2x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 8, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64_640_17280_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x66x66x1920xi8>, %arg2: tensor<3x3x1920x640xi8>, %arg3: tensor<2x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x1920xi8>, tensor<3x3x1920x640xi8>) outs(%arg3 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 2, 1, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 8 : i64, workgroup = [2, 4, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128_640_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x130x130x640xi8>, %arg2: tensor<3x3x640x640xi8>, %arg3: tensor<2x128x128x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x130x130x640xi8>, tensor<3x3x640x640xi8>) outs(%arg3 : tensor<2x128x128x640xi32>) -> tensor<2x128x128x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128_320_8640_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x130x130x960xi8>, %arg2: tensor<3x3x960x320xi8>, %arg3: tensor<2x128x128x320xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x130x130x960xi8>, tensor<3x3x960x320xi8>) outs(%arg3 : tensor<2x128x128x320xi32>) -> tensor<2x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 8, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64_640_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x66x66x640xi8>, %arg2: tensor<3x3x640x640xi8>, %arg3: tensor<2x64x64x640xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x640xi8>, tensor<3x3x640x640xi8>) outs(%arg3 : tensor<2x64x64x640xi32>) -> tensor<2x64x64x640xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [2, 4, 1, 2, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [2, 4, 16, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x64x64_1280_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x66x66x1280xi8>, %arg2: tensor<3x3x1280x1280xi8>, %arg3: tensor<2x64x64x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x66x66x1280xi8>, tensor<3x3x1280x1280xi8>) outs(%arg3 : tensor<2x64x64x1280xi32>) -> tensor<2x64x64x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 8, 1, 2, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 8, 32, 128, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x128x128_320_5760_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x130x130x640xi8>, %arg2: tensor<3x3x640x320xi8>, %arg3: tensor<2x128x128x320xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x130x130x640xi8>, tensor<3x3x640x320xi8>) outs(%arg3 : tensor<2x128x128x320xi32>) -> tensor<2x128x128x320xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 1, 8, 5, 0], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 1, 128, 320, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32_1280_23040_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x34x34x2560xi8>, %arg2: tensor<3x3x2560x1280xi8>, %arg3: tensor<2x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x34x34x2560xi8>, tensor<3x3x2560x1280xi8>) outs(%arg3 : tensor<2x32x32x1280xi32>) -> tensor<2x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 4, 2, 5, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 4, 32, 160, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_conv_2d_nhwc_hwcf_2x32x32_1280_11520_(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x34x34x1280xi8>, %arg2: tensor<3x3x1280x1280xi8>, %arg3: tensor<2x32x32x1280xi32>):
    %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg1, %arg2 : tensor<2x34x34x1280xi8>, tensor<3x3x1280x1280xi8>) outs(%arg3 : tensor<2x32x32x1280xi32>) -> tensor<2x32x32x1280xi32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 2], subgroup = [1, 4, 2, 5, 0], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [2, 4, 32, 160, 0]}>, translation_info = <pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Horizontally Fused Contraction Tuning for UNET Batch Size 1
//===----------------------------------------------------------------------===//

transform.named_sequence @match_horizontal_multi_contraction_2x64x20x64x2048_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x64x2048xi8>, %arg2: tensor<20x64x2048xi8>, %arg3: tensor<20x64x2048xi8>, %arg4: tensor<2x20x64x64xi32>):
    %1:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2, %arg3 : tensor<2x64x2048xi8>, tensor<20x64x2048xi8>, tensor<20x64x2048xi8>) outs(%arg4, %arg4 : tensor<2x20x64x64xi32>, tensor<2x20x64x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i32, %out_2: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      %6 = arith.extsi %in_1 : i8 to i32
      %7 = arith.muli %2, %6 : i32
      %8 = arith.addi %out_2, %7 : i32
      linalg.yield %5, %8 : i32, i32
    } -> (tensor<2x20x64x64xi32>, tensor<2x20x64x64xi32>)
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 128], subgroup_m_count = 1 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 4, 64, 16, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_horizontal_multi_contraction_2x1024x20x64x1280_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x1024x1280xi8>, %arg2: tensor<20x64x1280xi8>, %arg3: tensor<20x64x1280xi8>, %arg4: tensor<20x64x1280xi8>, %arg5: tensor<2x20x1024x64xi32>, %arg6: tensor<2x20x64x1024xi32>):
    %1:3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2, %arg3, %arg4 : tensor<2x1024x1280xi8>, tensor<20x64x1280xi8>, tensor<20x64x1280xi8>, tensor<20x64x1280xi8>) outs(%arg5, %arg5, %arg6 : tensor<2x20x1024x64xi32>, tensor<2x20x1024x64xi32>, tensor<2x20x64x1024xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %in_2: i8, %out: i32, %out_3: i32, %out_4: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      %6 = arith.extsi %in_1 : i8 to i32
      %7 = arith.muli %2, %6 : i32
      %8 = arith.addi %out_3, %7 : i32
      %9 = arith.extsi %in_2 : i8 to i32
      %10 = arith.muli %2, %9 : i32
      %11 = arith.addi %out_4, %10 : i32
      linalg.yield %5, %8, %11 : i32, i32, i32
    } -> (tensor<2x20x1024x64xi32>, tensor<2x20x1024x64xi32>, tensor<2x20x64x1024xi32>)
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1, 2, 3], reduction = [0, 0, 0, 0, 64], subgroup_m_count = 2 : i64, subgroup_n_count = 4 : i64, workgroup = [1, 1, 128, 64, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_horizontal_multi_contraction_2x64x10x64x2048_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x64x2048xi8>, %arg2: tensor<10x64x2048xi8>, %arg3: tensor<10x64x2048xi8>, %arg4: tensor<2x10x64x64xi32>):
    %1:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2, %arg3 : tensor<2x64x2048xi8>, tensor<10x64x2048xi8>, tensor<10x64x2048xi8>) outs(%arg4, %arg4 : tensor<2x10x64x64xi32>, tensor<2x10x64x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i32, %out_2: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      %6 = arith.extsi %in_1 : i8 to i32
      %7 = arith.muli %2, %6 : i32
      %8 = arith.addi %out_2, %7 : i32
      linalg.yield %5, %8 : i32, i32
    } -> (tensor<2x10x64x64xi32>, tensor<2x10x64x64xi32>)
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1, 2], reduction = [0, 0, 0, 0, 128], subgroup_m_count = 4 : i64, subgroup_n_count = 2 : i64, workgroup = [1, 1, 64, 32, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}
transform.named_sequence @match_horizontal_multi_contraction_2x4096x10x64x640_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
  ^bb0(%arg1: tensor<2x4096x640xi8>, %arg2: tensor<10x64x640xi8>, %arg3: tensor<10x64x640xi8>, %arg4: tensor<10x64x640xi8>, %arg5: tensor<2x10x4096x64xi32>, %arg6: tensor<2x10x64x4096xi32>):
    %1:3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg1, %arg2, %arg3, %arg4 : tensor<2x4096x640xi8>, tensor<10x64x640xi8>, tensor<10x64x640xi8>, tensor<10x64x640xi8>) outs(%arg5, %arg5, %arg6 : tensor<2x10x4096x64xi32>, tensor<2x10x4096x64xi32>, tensor<2x10x64x4096xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %in_2: i8, %out: i32, %out_3: i32, %out_4: i32):
      %2 = arith.extsi %in : i8 to i32
      %3 = arith.extsi %in_0 : i8 to i32
      %4 = arith.muli %2, %3 : i32
      %5 = arith.addi %out, %4 : i32
      %6 = arith.extsi %in_1 : i8 to i32
      %7 = arith.muli %2, %6 : i32
      %8 = arith.addi %out_3, %7 : i32
      %9 = arith.extsi %in_2 : i8 to i32
      %10 = arith.muli %2, %9 : i32
      %11 = arith.addi %out_4, %10 : i32
      linalg.yield %5, %8, %11 : i32, i32, i32
    } -> (tensor<2x10x4096x64xi32>, tensor<2x10x4096x64xi32>, tensor<2x10x64x4096xi32>)
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, promote_operands = [0, 1, 2, 3], reduction = [0, 0, 0, 0, 64], subgroup_m_count = 8 : i64, subgroup_n_count = 1 : i64, workgroup = [1, 2, 128, 16, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
  transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
}

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Attention.
        @match_attention_f16 -> @apply_attn_op_config
        , @match_attention_f8_m_1024 -> @apply_attn_op_config
        , @match_attention_f8_m_4096 -> @apply_attn_op_config
        , @match_attention_f8 -> @apply_attn_op_config

        // UNET Batch Size 1 Contractions.
        , @match_contraction_2048x1280x640_i8xi8xi32 -> @apply_op_config
        , @match_contraction_8192x640x320_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x1024x1024_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x4096x4096_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x4096x960_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x1024x1920_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x4096x1280_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x10x64_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x4096x1920_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x640x64_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x1024x2560_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x16384x960_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x16384x640_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x640x4096_i8xi8xi32 -> @apply_op_config
        , @match_contraction_8192x640x2560_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x20x64_i8xi8xi32 -> @apply_op_config
        , @match_contraction_8192x640x640_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x1280x64_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x10x4096_i8xi8xi32 -> @apply_op_config
        , @match_contraction_8192x5120x640_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x1280x1024_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2048x1280x1280_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2048x1280x5120_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2x20x1024_i8xi8xi32 -> @apply_op_config
        , @match_contraction_2048x10240x1280_i8xi8xi32 -> @apply_op_config

        // UNET Batch Size 1 Convolutions.
        , @match_conv_2d_nhwc_hwcf_2x128x128_16_2880_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64_640_2880_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32_1280_5760_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64_640_8640_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64_640_11520_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32_1280_17280_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x128x128_320_2880_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64_640_17280_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x128x128_640_5760_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x128x128_320_8640_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64_640_5760_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x64x64_1280_11520_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x128x128_320_5760_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32_1280_23040_ -> @apply_op_config
        , @match_conv_2d_nhwc_hwcf_2x32x32_1280_11520_ -> @apply_op_config

        // UNET Batch Size 1 Horizontally Fused Contractions.
        , @match_horizontal_multi_contraction_2x64x20x64x2048_i8xi8xi32 -> @apply_op_config
        , @match_horizontal_multi_contraction_2x1024x20x64x1280_i8xi8xi32 -> @apply_op_config
        , @match_horizontal_multi_contraction_2x64x10x64x2048_i8xi8xi32 -> @apply_op_config
        , @match_horizontal_multi_contraction_2x4096x10x64x640_i8xi8xi32 -> @apply_op_config

        // TUNING_MATCH_BEGIN DO NOT REMOVE

        // TUNING_MATCH_END DO NOT REMOVE
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
