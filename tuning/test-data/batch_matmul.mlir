module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none", waves_per_eu = 2 : i64}>]>]} {
  hal.executable private @main$async_dispatch_1120 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {mma_intrinsics = [#iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>], target_arch = "gfx942", ukernels = "none", waves_per_eu = 2 : i64}>) {
      hal.executable.export public @main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %c177043968 = arith.constant 177043968 : index
          %c1997620928 = arith.constant 1997620928 : index
          %c90495488 = arith.constant 90495488 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c177043968) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x242x1920xf16>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c1997620928) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x1920x640xf16>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c90495488) : !flow.dispatch.tensor<writeonly:tensor<64x242x640xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [64, 242, 1920], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x242x1920xf16>> -> tensor<64x242x1920xf16>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [64, 1920, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<64x1920x640xf16>> -> tensor<64x1920x640xf16>
          %5 = tensor.empty() : tensor<64x242x640xf32>
          %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64, 128]]>} ins(%cst : f32) outs(%5 : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
          %7 = linalg.batch_matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64, 128]]>} ins(%3, %4 : tensor<64x242x1920xf16>, tensor<64x1920x640xf16>) outs(%6 : tensor<64x242x640xf32>) -> tensor<64x242x640xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [64, 242, 640], strides = [1, 1, 1] : tensor<64x242x640xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x242x640xf32>>
          return
        }
      }
    }
  }
  util.global private mutable @main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c3377769472 = arith.constant 3377769472 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c3377769472}
    util.global.store %buffer, @main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c2970637824 = arith.constant 2970637824 : index
    %c2 = arith.constant 2 : index
    %c2563506176 = arith.constant 2563506176 : index
    %c1 = arith.constant 1 : index
    %c407131648 = arith.constant 407131648 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    %main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer = util.global.load @main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer : !hal.buffer)[%c0, %c407131648],
      %c1 = (%main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer : !hal.buffer)[%c407131648, %c2563506176],
      %c2 = (%main$async_dispatch_1120_rocm_hsaco_fb_main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32_buffer : !hal.buffer)[%c2970637824, %c407131648]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@main$async_dispatch_1120::@rocm_hsaco_fb::@main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@main$async_dispatch_1120) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@main$async_dispatch_1120::@rocm_hsaco_fb::@main$async_dispatch_1120_batch_matmul_64x242x640x1920_f16xf16xf32) : index
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
