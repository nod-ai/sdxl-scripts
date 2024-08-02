module attributes {hal.device.targets = [#hal.device.target<"rocm", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>, <MFMA_I8_16x16x32_I32>, <MFMA_I8_32x32x16_I32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none"}>]>]} {
  hal.executable private @main$async_dispatch_104 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>, <MFMA_I8_16x16x32_I32>, <MFMA_I8_32x32x16_I32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none"}>) {
      hal.executable.export public @main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>, #hal.interface.binding<0, 4>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_I8_16x16x32_I32>, subgroup_m_count = 2, subgroup_n_count = 2>, prefetch_shared_memory}>} {
          %c0_i32 = arith.constant 0 : i32
          %c140352 = arith.constant 140352 : index
          %c959552 = arith.constant 959552 : index
          %c0 = arith.constant 0 : index
          %0 = hal.interface.constant.load[0] : i32
          %1 = hal.interface.constant.load[1] : i32
          %2 = arith.index_castui %0 : i32 to index
          %3 = arith.index_castui %1 : i32 to index
          %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096x640xi8>>
          %5 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c140352) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x640x640xi8>>
          %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c959552) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4096xi32>>
          %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xi8>>
          %8 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xi32>>
          %9 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf32>>
          %10 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<writeonly:tensor<2x4096x640xf16>>
          %11 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096x640xi8>> -> tensor<2x4096x640xi8>
          %12 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [2, 640, 640], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x640x640xi8>> -> tensor<2x640x640xi8>
          %13 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4096xi32>> -> tensor<2x4096xi32>
          %14 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xi8>> -> tensor<640xi8>
          %15 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xi32>> -> tensor<640xi32>
          %16 = flow.dispatch.tensor.load %9, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf32>> -> tensor<640xf32>
          %17 = tensor.empty() : tensor<2x4096x640xf16>
          %18 = tensor.empty() : tensor<2x4096x640xi32>
          %19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
          %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {
          ^bb0(%in: i8, %in_0: i8, %out: i32):
            %22 = arith.extsi %in : i8 to i32
            %23 = arith.extsi %in_0 : i8 to i32
            %24 = arith.muli %22, %23 : i32
            %25 = arith.addi %out, %24 : i32
            linalg.yield %25 : i32
          } -> tensor<2x4096x640xi32>
          %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %13, %14, %15, %16 : tensor<2x4096x640xi32>, tensor<2x4096xi32>, tensor<640xi8>, tensor<640xi32>, tensor<640xf32>) outs(%17 : tensor<2x4096x640xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {
          ^bb0(%in: i32, %in_0: i32, %in_1: i8, %in_2: i32, %in_3: f32, %out: f16):
            %22 = arith.extsi %in_1 : i8 to i32
            %23 = arith.muli %in_0, %22 : i32
            %24 = arith.subi %in, %23 : i32
            %25 = arith.addi %24, %in_2 : i32
            %26 = arith.sitofp %25 : i32 to f32
            %27 = arith.mulf %26, %in_3 : f32
            %28 = arith.truncf %27 : f32 to f16
            linalg.yield %28 : f16
          } -> tensor<2x4096x640xf16>
          flow.dispatch.tensor.store %21, %10, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : tensor<2x4096x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x4096x640xf16>>
          return
        }
      }
    }
  }
  util.global private mutable @main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer
  util.initializer {
    %c654546688 = arith.constant 654546688 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c654546688}
    util.global.store %buffer, @main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer
    util.return
  }
  util.func public @main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c327276288 = arith.constant 327276288 : index
    %c4 = arith.constant 4 : index
    %c327273728 = arith.constant 327273728 : index
    %c3 = arith.constant 3 : index
    %c2560 = arith.constant 2560 : index
    %c327271168 = arith.constant 327271168 : index
    %c2 = arith.constant 2 : index
    %c640 = arith.constant 640 : index
    %c327270400 = arith.constant 327270400 : index
    %c1 = arith.constant 1 : index
    %c327270272 = arith.constant 327270272 : index
    %c85437312_i32 = arith.constant 85437312 : i32
    %c22061440_i32 = arith.constant 22061440 : i32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 2, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer, ReadOnly>, <4, storage_buffer>]>]>) : !hal.pipeline_layout
    hal.command_buffer.push_constants<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout) offset(0) values([%c22061440_i32, %c85437312_i32]) : i32, i32
    %main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer = util.global.load @main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer)[%c0, %c327270272],
      %c1 = (%main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer)[%c327270400, %c640],
      %c2 = (%main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer)[%c327271168, %c2560],
      %c3 = (%main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer)[%c327273728, %c2560],
      %c4 = (%main$async_dispatch_104_rocm_hsaco_fb_main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32_buffer : !hal.buffer)[%c327276288, %c327270272]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@main$async_dispatch_104::@rocm_hsaco_fb::@main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32) : index, index, index
    %exe = hal.executable.lookup device(%device_0 : !hal.device) executable(@main$async_dispatch_104) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@main$async_dispatch_104::@rocm_hsaco_fb::@main$async_dispatch_104_batch_matmul_transpose_b_2x4096x640x640_i8xi8xi32) : index
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
