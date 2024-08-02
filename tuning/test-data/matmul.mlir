module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @run_forward$async_dispatch_142 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = hal.interface.constant.load[2] : i32
          %3 = arith.index_castui %0 : i32 to index
          %4 = arith.index_castui %1 : i32 to index
          %5 = arith.index_castui %2 : i32 to index
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<24576x1280xf16>>
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%4) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240xf32>>
          %9 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<24576x10240xf16>>
          %10 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [24576, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<24576x1280xf16>> -> tensor<24576x1280xf16>
          %11 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
          %12 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [10240], strides = [1] : !flow.dispatch.tensor<readonly:tensor<10240xf32>> -> tensor<10240xf32>
          %13 = tensor.empty() : tensor<24576x10240xf16>
          %14 = tensor.empty() : tensor<24576x10240xf32>
          %15 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%14 : tensor<24576x10240xf32>) -> tensor<24576x10240xf32>
          %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%10, %11 : tensor<24576x1280xf16>, tensor<10240x1280xf16>) outs(%15 : tensor<24576x10240xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
          ^bb0(%in: f16, %in_0: f16, %out: f32):
            %18 = arith.extf %in : f16 to f32
            %19 = arith.extf %in_0 : f16 to f32
            %20 = arith.mulf %18, %19 : f32
            %21 = arith.addf %out, %20 : f32
            linalg.yield %21 : f32
          } -> tensor<24576x10240xf32>
          %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %12 : tensor<24576x10240xf32>, tensor<10240xf32>) outs(%13 : tensor<24576x10240xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
          ^bb0(%in: f32, %in_0: f32, %out: f16):
            %18 = arith.addf %in, %in_0 : f32
            %19 = arith.truncf %18 : f32 to f16
            linalg.yield %19 : f16
          } -> tensor<24576x10240xf16>
          flow.dispatch.tensor.store %17, %9, offsets = [0, 0], sizes = [24576, 10240], strides = [1, 1] : tensor<24576x10240xf16> -> !flow.dispatch.tensor<writeonly:tensor<24576x10240xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c11667815936 = arith.constant 11667815936 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c11667815936}
    util.global.store %buffer, @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c6830281984 = arith.constant 6830281984 : index
    %c3 = arith.constant 3 : index
    %c1966533568 = arith.constant 1966533568 : index
    %c4863748352 = arith.constant 4863748352 : index
    %c2 = arith.constant 2 : index
    %c26214400 = arith.constant 26214400 : index
    %c4837533952 = arith.constant 4837533952 : index
    %c1 = arith.constant 1 : index
    %c4837533760 = arith.constant 4837533760 : index
    %c1274689600_i32 = arith.constant 1274689600 : i32
    %c1787843776_i32 = arith.constant 1787843776 : i32
    %c1211775040_i32 = arith.constant 1211775040 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 3, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c1211775040_i32, %c1787843776_i32, %c1274689600_i32]) : i32, i32, i32
    %run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer = util.global.load @run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c0, %c4837533760],
      %c1 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c4837533952, %c26214400],
      %c2 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c4863748352, %c1966533568],
      %c3 = (%run_forward$async_dispatch_142_rocm_hsaco_fb_run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32_buffer : !hal.buffer)[%c6830281984, %c4837533760]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@run_forward$async_dispatch_142::@rocm_hsaco_fb::@run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@run_forward$async_dispatch_142) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@run_forward$async_dispatch_142::@rocm_hsaco_fb::@run_forward$async_dispatch_142_matmul_transpose_b_24576x10240x1280_f16xf16xf32) : index
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
