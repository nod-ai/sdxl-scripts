module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @run_forward$async_dispatch_1269 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = hal.interface.constant.load[3] : i32
          %4 = hal.interface.constant.load[4] : i32
          %5 = hal.interface.constant.load[5] : i32
          %6 = arith.index_castui %0 : i32 to index
          %7 = arith.index_castui %1 : i32 to index
          %8 = arith.index_castui %2 : i32 to index
          %9 = arith.index_castui %3 : i32 to index
          %10 = arith.index_castui %4 : i32 to index
          %11 = arith.index_castui %5 : i32 to index
          %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x130x130x640xf16>>
          %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x640x320xf16>>
          %14 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%9) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
          %15 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x320xf32>>
          %16 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%10) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf32>>
          %17 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%11) : !flow.dispatch.tensor<writeonly:tensor<24x320x128x128xf16>>
          %18 = flow.dispatch.tensor.load %12, offsets = [0, 0, 0, 0], sizes = [24, 130, 130, 640], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x130x130x640xf16>> -> tensor<24x130x130x640xf16>
          %19 = flow.dispatch.tensor.load %13, offsets = [0, 0, 0, 0], sizes = [3, 3, 640, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x640x320xf16>> -> tensor<3x3x640x320xf16>
          %20 = flow.dispatch.tensor.load %14, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
          %21 = flow.dispatch.tensor.load %15, offsets = [0, 0], sizes = [24, 320], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24x320xf32>> -> tensor<24x320xf32>
          %22 = flow.dispatch.tensor.load %16, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf32>> -> tensor<320xf32>
          %23 = tensor.empty() : tensor<24x320x128x128xf16>
          %24 = tensor.empty() : tensor<24x128x128x320xf32>
          %25 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} ins(%cst : f32) outs(%24 : tensor<24x128x128x320xf32>) -> tensor<24x128x128x320xf32>
          %26 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%18, %19 : tensor<24x130x130x640xf16>, tensor<3x3x640x320xf16>) outs(%25 : tensor<24x128x128x320xf32>) -> tensor<24x128x128x320xf32>
          %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %20, %21, %22 : tensor<24x128x128x320xf32>, tensor<320xf32>, tensor<24x320xf32>, tensor<320xf32>) outs(%23 : tensor<24x320x128x128xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 64, 64, 1, 1, 32]]>} {
          ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f16):
            %28 = arith.addf %in_1, %in_2 : f32
            %29 = arith.addf %in, %in_0 : f32
            %30 = arith.truncf %28 : f32 to f16
            %31 = arith.truncf %29 : f32 to f16
            %32 = arith.addf %31, %30 : f16
            linalg.yield %32 : f16
          } -> tensor<24x320x128x128xf16>
          flow.dispatch.tensor.store %27, %17, offsets = [0, 0, 0, 0], sizes = [24, 320, 128, 128], strides = [1, 1, 1, 1] : tensor<24x320x128x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<24x320x128x128xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c11641601536 = arith.constant 11641601536 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c11641601536}
    util.global.store %buffer, @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c6804067584 = arith.constant 6804067584 : index
    %c2 = arith.constant 2 : index
    %c1966533568 = arith.constant 1966533568 : index
    %c4837533952 = arith.constant 4837533952 : index
    %c1 = arith.constant 1 : index
    %c4837533760 = arith.constant 4837533760 : index
    %c1022576704_i32 = arith.constant 1022576704 : i32
    %c1964184704_i32 = arith.constant 1964184704 : i32
    %c1960497024_i32 = arith.constant 1960497024 : i32
    %c1960498304_i32 = arith.constant 1960498304 : i32
    %c1022545984_i32 = arith.constant 1022545984 : i32
    %c503377984_i32 = arith.constant 503377984 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 6, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c503377984_i32, %c1022545984_i32, %c1960498304_i32, %c1960497024_i32, %c1964184704_i32, %c1022576704_i32]) : i32, i32, i32, i32, i32, i32
    %run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer = util.global.load @run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer)[%c0, %c4837533760],
      %c1 = (%run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer)[%c4837533952, %c1966533568],
      %c2 = (%run_forward$async_dispatch_1269_rocm_hsaco_fb_run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32_buffer : !hal.buffer)[%c6804067584, %c4837533760]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@run_forward$async_dispatch_1269::@rocm_hsaco_fb::@run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@run_forward$async_dispatch_1269) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@run_forward$async_dispatch_1269::@rocm_hsaco_fb::@run_forward$async_dispatch_1269_conv_2d_nhwc_hwcf_24x128x128x320x3x3x640_f16xf16xf32) : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z])
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device_0 : !hal.device> affinity(%c-1_i64) wait(%1) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
