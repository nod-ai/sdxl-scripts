module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @run_forward$async_dispatch_131 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24x1024x1280xf16>>
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x20x64x1280xf16>>
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<3x24x20x1024x64xf16>>
          %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [24, 1024, 1280], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x1024x1280xf16>> -> tensor<24x1024x1280xf16>
          %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0, 0], sizes = [3, 20, 64, 1280], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x20x64x1280xf16>> -> tensor<3x20x64x1280xf16>
          %11 = tensor.empty() : tensor<3x24x20x1024x64xf16>
          %12 = tensor.empty() : tensor<3x24x20x1024x64xf32>
          %13 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 128]]>} ins(%cst : f32) outs(%12 : tensor<3x24x20x1024x64xf32>) -> tensor<3x24x20x1024x64xf32>
          %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%9, %10 : tensor<24x1024x1280xf16>, tensor<3x20x64x1280xf16>) outs(%13 : tensor<3x24x20x1024x64xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 128]]>} {
          ^bb0(%in: f16, %in_0: f16, %out: f32):
            %16 = arith.extf %in : f16 to f32
            %17 = arith.extf %in_0 : f16 to f32
            %18 = arith.mulf %16, %17 : f32
            %19 = arith.addf %out, %18 : f32
            linalg.yield %19 : f32
          } -> tensor<3x24x20x1024x64xf32>
          %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<3x24x20x1024x64xf32>) outs(%11 : tensor<3x24x20x1024x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64, 64, 128]]>} {
          ^bb0(%in: f32, %out: f16):
            %16 = arith.truncf %in : f32 to f16
            linalg.yield %16 : f16
          } -> tensor<3x24x20x1024x64xf16>
          flow.dispatch.tensor.store %15, %8, offsets = [0, 0, 0, 0, 0], sizes = [3, 24, 20, 1024, 64], strides = [1, 1, 1, 1, 1] : tensor<3x24x20x1024x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<3x24x20x1024x64xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c11641601536 = arith.constant 11641601536 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c11641601536}
    util.global.store %buffer, @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c6804067584 = arith.constant 6804067584 : index
    %c2 = arith.constant 2 : index
    %c1966533568 = arith.constant 1966533568 : index
    %c4837533952 = arith.constant 4837533952 : index
    %c1 = arith.constant 1 : index
    %c4837533760 = arith.constant 4837533760 : index
    %c1274689600_i32 = arith.constant 1274689600 : i32
    %c1767517376_i32 = arith.constant 1767517376 : i32
    %c1211775040_i32 = arith.constant 1211775040 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c1211775040_i32, %c1767517376_i32, %c1274689600_i32]) : i32, i32, i32
    %run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer = util.global.load @run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer)[%c0, %c4837533760],
      %c1 = (%run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer)[%c4837533952, %c1966533568],
      %c2 = (%run_forward$async_dispatch_131_rocm_hsaco_fb_run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32_buffer : !hal.buffer)[%c6804067584, %c4837533760]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@run_forward$async_dispatch_131::@rocm_hsaco_fb::@run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@run_forward$async_dispatch_131) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@run_forward$async_dispatch_131::@rocm_hsaco_fb::@run_forward$async_dispatch_131_matmul_like_3x24x20x1024x64x1280_f16xf16xf32) : index
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
