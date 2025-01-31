module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Tuning infra
//===----------------------------------------------------------------------===//

  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                            %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    // transform.print %op {name = "Applied"} : !transform.any_op
    transform.yield
  }

  transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
                                                 %config: !transform.any_param {transform.readonly},
                                                 %decomposition_config: !transform.any_param {transform.readonly}) {
    transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
    // transform.print %attention {name = "Applied attention config"} : !transform.any_op
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

// TUNING_SPEC_BEGIN DO NOT REMOVE

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

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Attention.
        @match_attention_f16 -> @apply_attn_op_config
        , @match_attention_f8 -> @apply_attn_op_config
        , @match_contraction_28672x10240x1280_i8xi8xi32 -> @apply_op_config

        // TUNING_MATCH_BEGIN DO NOT REMOVE

        // TUNING_MATCH_END DO NOT REMOVE
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
