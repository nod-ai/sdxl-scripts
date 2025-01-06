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

  transform.named_sequence @match_broadcast_rhs_mmt_i8_i8_i32(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
    // transform.print %root {name = "Generic"} : !transform.any_op
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?x?xi8>, %rhs: tensor<?x?xi8>, %out: tensor<?x?x?xi32>):
      %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                            iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%lhs, %rhs : tensor<?x?x?xi8>, tensor<?x?xi8>) outs(%out : tensor<?x?x?xi32>) {
        ^bb0(%in: i8, %in_0: i8, %acc: i32):
          %22 = arith.extsi %in : i8 to i32
          %23 = arith.extsi %in_0 : i8 to i32
          %24 = arith.muli %22, %23 : i32
          %25 = arith.addi %acc, %24 : i32
          linalg.yield %25 : i32
        } -> tensor<?x?x?xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %root : !transform.any_op
  }

// TUNING_SPEC_BEGIN DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Matmul tuning
//===----------------------------------------------------------------------===//

transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
  // transform.print %root {name = "Generic"} : !transform.any_op
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
    ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf32>):
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf32>) {
      ^bb0(%in: f16, %in_0: f16, %acc: f32):
        %18 = arith.extf %in : f16 to f32
        %19 = arith.extf %in_0 : f16 to f32
        %20 = arith.mulf %18, %19 : f32
        %21 = arith.addf %acc, %20 : f32
        linalg.yield %21 : f32
      } -> tensor<?x?xf32>
  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
  transform.yield %root : !transform.any_op
}

transform.named_sequence @match_mmt_2048x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
  transform.iree.match.cast_compatible_type %lhs = tensor<2048x1280xf16> : !transform.any_value
  transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xf16> : !transform.any_value
  %config = transform.param.constant #iree_codegen.compilation_info<
  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                                subgroup_m_count = 1, subgroup_n_count = 2,
                                                reduction = [0, 0, 32],
                                                workgroup = [128, 320, 0]}>,
  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
    workgroup_size = [128, 4, 1] subgroup_size = 64,
    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>
    }>> -> !transform.any_param
  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
}

//===----------------------------------------------------------------------===//
// Convolution tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Batch matmul tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Broadcast rhs mmt tuning
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Contraction tuning
//===----------------------------------------------------------------------===//

// TUNING_SPEC_END DO NOT REMOVE

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.readonly}) {
    // transform.foreach_match in %variant_op
    //     // TUNING_MATCH_BEGIN DO NOT REMOVE
    //     // TUNING_MATCH_END DO NOT REMOVE
    //   : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
