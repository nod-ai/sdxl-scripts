// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942", ukernels = "none"}>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

module attributes {transform.with_named_sequence} {

  util.func private @conv_entry_point_k1(%arg0: tensor<2x34x34x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>)
                                          -> tensor<2x32x32x1280xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<1280x3x3x1280xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<1280x3x3x1280xf16>) -> tensor<1280x3x3x1280xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x1280x1280xf16>) outs(%weight_splat : tensor<1280x3x3x1280xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x32x32x1280xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x32x32x1280xf16>) -> tensor<2x32x32x1280xf16>
    
    %hi = arith.constant 34 : i32
    %wi = arith.constant 34 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 1280 : i32
    %c = arith.constant 1280 : i32
    %ho = arith.constant 32 : i32
    %wo = arith.constant 32 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 2576980378 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 2576980378 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 0 : i32
    %shift_pack_0 = arith.constant 117770756 : i32
    %shift_pack_1 = arith.constant 0 : i32
    %ks = arith.constant 3 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x34x34x1280xf16>, tensor<1280x3x3x1280xf16>, tensor<2x32x32x1280xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_640 = arith.constant 640 : index
        hal.return %c_640, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x32x32x1280xf16> to tensor<2x32x32x1280xf32>
    util.return %6 : tensor<2x32x32x1280xf32>
  }

  transform.named_sequence @cast_and_call_dag_k1(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k1 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k1(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x34x34x1280xf16>, %rhs: tensor<3x3x1280x1280xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x32x32x1280xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%87 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_real_k2(%arg0: tensor<2x34x34x2560xf16>, %arg1: tensor<3x3x2560x1280xf16>)
                                          -> tensor<2x32x32x1280xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<1280x3x3x2560xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<1280x3x3x2560xf16>) -> tensor<1280x3x3x2560xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x2560x1280xf16>) outs(%weight_splat : tensor<1280x3x3x2560xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x32x32x1280xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x32x32x1280xf16>) -> tensor<2x32x32x1280xf16>

    %hi = arith.constant 34 : i32
    %wi = arith.constant 34 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 1280 : i32
    %c = arith.constant 2560 : i32
    %ho = arith.constant 32 : i32
    %wo = arith.constant 32 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 2576980378 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 2576980378 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 0 : i32
    %shift_pack_0 = arith.constant 117770756 : i32
    %shift_pack_1 = arith.constant 0 : i32
    %ks = arith.constant 4 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x34x34x2560xf16>, tensor<1280x3x3x2560xf16>, tensor<2x32x32x1280xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_1280 = arith.constant 1280 : index
        hal.return %c_1280, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x32x32x1280xf16> to tensor<2x32x32x1280xf32>
    util.return %6 : tensor<2x32x32x1280xf32>
  }

  transform.named_sequence @cast_and_call_dag_real_k2(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_real_k2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_real_k2(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x34x34x2560xf16>, %rhs: tensor<3x3x2560x1280xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x32x32x1280xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x34x34x2560xf16>, tensor<3x3x2560x1280xf16>) outs(%87 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_real_k3(%arg0: tensor<2x130x130x640xf16>, %arg1: tensor<3x3x640x320xf16>)
                                          -> tensor<2x128x128x320xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<320x3x3x640xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<320x3x3x640xf16>) -> tensor<320x3x3x640xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x640x320xf16>) outs(%weight_splat : tensor<320x3x3x640xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x128x128x320xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>

    %hi = arith.constant 130 : i32
    %wi = arith.constant 130 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 320 : i32
    %c = arith.constant 640 : i32
    %ho = arith.constant 128 : i32
    %wo = arith.constant 128 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 1431655766 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 1431655766 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 2 : i32
    %shift_pack_0 = arith.constant 151457282 : i32
    %shift_pack_1 = arith.constant 4996 : i32
    %ks = arith.constant 1 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x130x130x640xf16>, tensor<320x3x3x640xf16>, tensor<2x128x128x320xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_768 = arith.constant 768 : index
        hal.return %c_768, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x128x128x320xf16> to tensor<2x128x128x320xf32>
    util.return %6 : tensor<2x128x128x320xf32>
  }

  transform.named_sequence @cast_and_call_dag_real_k3(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_real_k3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_real_k3(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x130x130x640xf16>, %rhs: tensor<3x3x640x320xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x128x128x320xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x130x130x640xf16>, tensor<3x3x640x320xf16>) outs(%87 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_real_k6(%arg0: tensor<2x130x130x960xf16>, %arg1: tensor<3x3x960x320xf16>)
                                          -> tensor<2x128x128x320xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<320x3x3x960xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<320x3x3x960xf16>) -> tensor<320x3x3x960xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x960x320xf16>) outs(%weight_splat : tensor<320x3x3x960xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x128x128x320xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>

    %hi = arith.constant 130 : i32
    %wi = arith.constant 130 : i32
    %n = arith.constant 2 : i32
    %k = arith.constant 320 : i32
    %c = arith.constant 960 : i32
    %ho = arith.constant 128 : i32
    %wo = arith.constant 128 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 1431655766 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 1431655766 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 2 : i32
    %shift_pack_0 = arith.constant 151457282 : i32
    %shift_pack_1 = arith.constant 4996 : i32
    %ks = arith.constant 1 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x130x130x960xf16>, tensor<320x3x3x960xf16>, tensor<2x128x128x320xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_768 = arith.constant 768 : index
        hal.return %c_768, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x128x128x320xf16> to tensor<2x128x128x320xf32>
    util.return %6 : tensor<2x128x128x320xf32>
  }

  transform.named_sequence @cast_and_call_dag_real_k6(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_real_k6 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_real_k6(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x130x130x960xf16>, %rhs: tensor<3x3x960x320xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x128x128x320xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x130x130x960xf16>, tensor<3x3x960x320xf16>) outs(%87 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_k2(%arg0: tensor<2x34x34x1280xf16>, %arg1: tensor<3x3x1280x2560xf16>) 
                                          -> tensor<2x32x32x2560xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<2560x3x3x1280xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<2560x3x3x1280xf16>) -> tensor<2560x3x3x1280xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x1280x2560xf16>) outs(%weight_splat : tensor<2560x3x3x1280xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x32x32x2560xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x32x32x2560xf16>) -> tensor<2x32x32x2560xf16>

    %hi = arith.constant 34 : i32
    %wi = arith.constant 34 : i32
    %n = arith.constant 2 : i32
    %c = arith.constant 1280 : i32
    %k = arith.constant 2560 : i32 // number of filters
    %ho = arith.constant 32 : i32
    %wo = arith.constant 32 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 2576980378 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 2576980378 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 0 : i32
    %shift_pack_0 = arith.constant 134547973 : i32
    %shift_pack_1 = arith.constant 113711 : i32
    %ks = arith.constant 3 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x34x34x1280xf16>, tensor<2560x3x3x1280xf16>, tensor<2x32x32x2560xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_1280 = arith.constant 1280 : index
        hal.return %c_1280, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x32x32x2560xf16> to tensor<2x32x32x2560xf32>
    util.return %6 : tensor<2x32x32x2560xf32>
  }

  transform.named_sequence @cast_and_call_dag_k2(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k2 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k2(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x34x34x1280xf16>, %rhs: tensor<3x3x1280x2560xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x32x32x2560xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x32x32x2560xf32>) -> tensor<2x32x32x2560xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x34x34x1280xf16>, tensor<3x3x1280x2560xf16>) outs(%87 : tensor<2x32x32x2560xf32>) -> tensor<2x32x32x2560xf32>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

    util.func private @conv_entry_point_k3(%arg0: tensor<2x66x66x640xf16>, %arg1: tensor<3x3x640x640xf16>) 
                                          -> tensor<2x64x64x640xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<640x3x3x640xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<640x3x3x640xf16>) -> tensor<640x3x3x640xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x640x640xf16>) outs(%weight_splat : tensor<640x3x3x640xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x64x64x640xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x64x64x640xf16>) -> tensor<2x64x64x640xf16>

    %hi = arith.constant 66 : i32
    %wi = arith.constant 66 : i32
    %n = arith.constant 2 : i32
    %c = arith.constant 640 : i32
    %k = arith.constant 640 : i32 // number of filters
    %ho = arith.constant 64 : i32
    %wo = arith.constant 64 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 1431655766 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 1431655766 : i32
    %magic_4 = arith.constant 286331154 : i32
    %magic_5 = arith.constant 2576980378 : i32
    %shift_pack_0 = arith.constant 134614018 : i32
    %shift_pack_1 = arith.constant 2571 : i32
    %ks = arith.constant 0 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt128x256x32_wt32x32x8_ws1x2_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x4x1_1x4x1x64"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x66x66x640xf16>, tensor<640x3x3x640xf16>, tensor<2x64x64x640xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_192 = arith.constant 192 : index
        hal.return %c_192, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x64x64x640xf16> to tensor<2x64x64x640xf32>
    util.return %6 : tensor<2x64x64x640xf32>
  }

  transform.named_sequence @cast_and_call_dag_k3(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k3 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k3(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x66x66x640xf16>, %rhs: tensor<3x3x640x640xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x64x64x640xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x64x64x640xf32>) -> tensor<2x64x64x640xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x66x66x640xf16>, tensor<3x3x640x640xf16>) outs(%87 : tensor<2x64x64x640xf32>) -> tensor<2x64x64x640xf32>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_k4(%arg0: tensor<2x34x34x640xf16>, %arg1: tensor<3x3x640x640xf16>) 
                                          -> tensor<2x32x32x640xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<640x3x3x640xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<640x3x3x640xf16>) -> tensor<640x3x3x640xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x640x640xf16>) outs(%weight_splat : tensor<640x3x3x640xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x32x32x640xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x32x32x640xf16>) -> tensor<2x32x32x640xf16>

    %hi = arith.constant 34 : i32
    %wi = arith.constant 34 : i32
    %n = arith.constant 2 : i32
    %c = arith.constant 640 : i32
    %k = arith.constant 640 : i32 // number of filters
    %ho = arith.constant 32 : i32
    %wo = arith.constant 32 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 1431655766 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 1431655766 : i32
    %magic_4 = arith.constant 286331154 : i32
    %magic_5 = arith.constant 2576980378 : i32
    %shift_pack_0 = arith.constant 100993538 : i32
    %shift_pack_1 = arith.constant 2571 : i32
    %ks = arith.constant 2 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt128x256x32_wt32x32x8_ws1x2_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x4x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x34x34x640xf16>, tensor<640x3x3x640xf16>, tensor<2x32x32x640xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_192 = arith.constant 192 : index
        hal.return %c_192, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x32x32x640xf16> to tensor<2x32x32x640xf32>
    util.return %6 : tensor<2x32x32x640xf32>
  }

  transform.named_sequence @cast_and_call_dag_k4(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k4 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k4(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x34x34x640xf16>, %rhs: tensor<3x3x640x640xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x32x32x640xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x32x32x640xf32>) -> tensor<2x32x32x640xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x34x34x640xf16>, tensor<3x3x640x640xf16>) outs(%87 : tensor<2x32x32x640xf32>) -> tensor<2x32x32x640xf32>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_k5(%arg0: tensor<2x66x66x1280xf16>, %arg1: tensor<3x3x1280x1280xf16>) 
                                          -> tensor<2x64x64x1280xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<1280x3x3x1280xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<1280x3x3x1280xf16>) -> tensor<1280x3x3x1280xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x1280x1280xf16>) outs(%weight_splat : tensor<1280x3x3x1280xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x64x64x1280xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x64x64x1280xf16>) -> tensor<2x64x64x1280xf16>

    %hi = arith.constant 66 : i32
    %wi = arith.constant 66 : i32
    %n = arith.constant 2 : i32
    %c = arith.constant 1280 : i32
    %k = arith.constant 1280 : i32 // number of filters
    %ho = arith.constant 64 : i32
    %wo = arith.constant 64 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 2576980378 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 2576980378 : i32
    %magic_4 = arith.constant 286331154 : i32
    %magic_5 = arith.constant 2576980378 : i32
    %shift_pack_0 = arith.constant 168168452 : i32
    %shift_pack_1 = arith.constant 2828 : i32
    %ks = arith.constant 3 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x66x66x1280xf16>, tensor<1280x3x3x1280xf16>, tensor<2x64x64x1280xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_5120 = arith.constant 5120 : index
        hal.return %c_5120, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x64x64x1280xf16> to tensor<2x64x64x1280xf32>
    util.return %6 : tensor<2x64x64x1280xf32>
  }

  transform.named_sequence @cast_and_call_dag_k5(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k5 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k5(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x66x66x1280xf16>, %rhs: tensor<3x3x1280x1280xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x64x64x1280xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x66x66x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%87 : tensor<2x64x64x1280xf32>) -> tensor<2x64x64x1280xf32>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  util.func private @conv_entry_point_k6(%arg0: tensor<2x130x130x320xf16>, %arg1: tensor<3x3x320x320xf16>) 
                                          -> tensor<2x128x128x320xf32> {
    %c_0 = arith.constant 0.000000e+00 : f16
    %weight_empty = tensor.empty() : tensor<320x3x3x320xf16>
    %weight_splat = linalg.fill ins(%c_0 : f16) outs(%weight_empty : tensor<320x3x3x320xf16>) -> tensor<320x3x3x320xf16>
    %weight_nhwc = linalg.transpose ins(%arg1: tensor<3x3x320x320xf16>) outs(%weight_splat : tensor<320x3x3x320xf16>) permutation = [3, 0, 1, 2]
    %out_empty = tensor.empty() : tensor<2x128x128x320xf16>
    %out_splat = linalg.fill ins(%c_0 : f16) outs(%out_empty : tensor<2x128x128x320xf16>) -> tensor<2x128x128x320xf16>

    %hi = arith.constant 130 : i32
    %wi = arith.constant 130 : i32
    %n = arith.constant 2 : i32
    %c = arith.constant 320 : i32
    %k = arith.constant 320 : i32 // number of filters
    %ho = arith.constant 128 : i32
    %wo = arith.constant 128 : i32
    %stride_h = arith.constant 1 : i32
    %stride_w = arith.constant 1 : i32
    %dilation_h = arith.constant 1 : i32
    %dilation_w = arith.constant 1 : i32
    %pad_h = arith.constant 0 : i32
    %pad_w = arith.constant 0 : i32
    %y = arith.constant 3 : i32
    %x = arith.constant 3 : i32
    %group = arith.constant 1 : i32
    %magic_0 = arith.constant 1431655766 : i32
    %magic_1 = arith.constant 1 : i32
    %magic_2 = arith.constant 1 : i32
    %magic_3 = arith.constant 1431655766 : i32
    %magic_4 = arith.constant 0 : i32
    %magic_5 = arith.constant 0 : i32
    %shift_pack_0 = arith.constant 151457282 : i32
    %shift_pack_1 = arith.constant 113711 : i32
    %ks = arith.constant 1 : i32
    %__pack_0 = arith.constant 0 : i32

    %5 = hal.dispatch.extern "igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"(
        %hi, %wi, %n, %k, %c, %ho, %wo, %stride_h, %stride_w, %dilation_h, %dilation_w, %pad_h, %pad_w, %y, %x, %group,
        %magic_0, %magic_1, %magic_2, %magic_3, %magic_4, %magic_5, %shift_pack_0, %shift_pack_1, %ks, %__pack_0, %arg0, %weight_nhwc, %out_splat) :
        (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
        tensor<2x130x130x320xf16>, tensor<320x3x3x320xf16>, tensor<2x128x128x320xf16>) -> %out_splat
      count(%device: !hal.device) -> (index, index, index) {
        %c_1 = arith.constant 1 : index
        %c_768 = arith.constant 768 : index
        hal.return %c_768, %c_1, %c_1 : index, index, index
      }
      layout(#hal.pipeline.layout<push_constants = 26, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer, ReadOnly>,
            <2, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ])
      objects({
        #rocm_target ordinal(0) = [
          #hal.executable.object<{
            path = "/data/home/perf/nithin/sdxl-scripts/specs/igemm_fwd_gtc_gfx942_nhwc_fp16.hsaco"
          }>
        ]
      })
      attributes {subgroupSize = 64, workgroup_size = [256 : index, 1 : index, 1 : index]}
    %6 = arith.extf %5 : tensor<2x128x128x320xf16> to tensor<2x128x128x320xf32>
    util.return %6 : tensor<2x128x128x320xf32>
  }

  transform.named_sequence @cast_and_call_dag_k6(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @conv_entry_point_k6 into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      //  transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_conv_k6(
    %root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<2x130x130x320xf16>, %rhs: tensor<3x3x320x320xf16>):
        %c_0 = arith.constant 0.000000e+00 : f32
        %84 = tensor.empty() : tensor<2x128x128x320xf32>
        %87 = linalg.fill {"match.operation_name_only"} ins(%c_0 : f32) outs(%84 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
        %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%lhs, %rhs : tensor<2x130x130x320xf16>, tensor<3x3x320x320xf16>) outs(%87 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            @match_conv_k1 -> @cast_and_call_dag_k1,
            @match_conv_k2 -> @cast_and_call_dag_k2,
            @match_conv_k3 -> @cast_and_call_dag_k3,
            @match_conv_k4 -> @cast_and_call_dag_k4,
            @match_conv_k5 -> @cast_and_call_dag_k5,
            @match_conv_k6 -> @cast_and_call_dag_k6,
            @match_conv_real_k2 -> @cast_and_call_dag_real_k2,
            @match_conv_real_k3 -> @cast_and_call_dag_real_k3,
            @match_conv_real_k6 -> @cast_and_call_dag_real_k6
          : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

