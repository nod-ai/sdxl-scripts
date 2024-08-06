module {
  util.global private @__device_0 = #hal.device.target<"hip", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>, <MFMA_F8E4M3FNUZ_16x16x32_F32>, <MFMA_I8_16x16x32_I32>, <MFMA_I8_32x32x16_I32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none"}>]> : !hal.device
  hal.executable private @main_4_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>, <MFMA_F8E4M3FNUZ_16x16x32_F32>, <MFMA_I8_16x16x32_I32>, <MFMA_I8_32x32x16_I32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none"}>) {
      hal.executable.export public @main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>, prefetch_shared_memory}>} {
          %cst = arith.constant 0.000000e+00 : f16
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
          %1 = hal.interface.binding.subspan layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<5120x640xf16>>
          %2 = hal.interface.binding.subspan layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8192x5120xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5120, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<5120x640xf16>> -> tensor<5120x640xf16>
          %5 = tensor.empty() : tensor<8192x5120xf32>
          %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f16) outs(%5 : tensor<8192x5120xf32>) -> tensor<8192x5120xf32>
          %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<8192x640xf16>, tensor<5120x640xf16>) outs(%6 : tensor<8192x5120xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {
          ^bb0(%in: f16, %in_0: f16, %out: f32):
            %8 = arith.extf %in : f16 to f32
            %9 = arith.extf %in_0 : f16 to f32
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %out, %10 : f32
            linalg.yield %11 : f32
          } -> tensor<8192x5120xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8192, 5120], strides = [1, 1] : tensor<8192x5120xf32> -> !flow.dispatch.tensor<writeonly:tensor<8192x5120xf32>>
          return
        }
      }
    }
  }
  util.global private mutable @main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer : !hal.buffer
  util.initializer {
    %c184811520 = arith.constant 184811520 : index
    %device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c184811520}
    util.global.store %buffer, @main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %0 = util.null : !hal.fence
    %c167772160 = arith.constant 167772160 : index
    %c17039360 = arith.constant 17039360 : index
    %c2 = arith.constant 2 : index
    %c6553600 = arith.constant 6553600 : index
    %c1 = arith.constant 1 : index
    %c10485760 = arith.constant 10485760 : index
    %c0 = arith.constant 0 : index
    %1 = arith.index_cast %arg0 : i32 to index
    %device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device : !hal.device) layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    %main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer = util.global.load @main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer : !hal.buffer)[%c0, %c10485760], 
      %c1 = (%main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer : !hal.buffer)[%c10485760, %c6553600], 
      %c2 = (%main_4_dispatch_0_rocm_hsaco_fb_main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32_buffer : !hal.buffer)[%c17039360, %c167772160]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@main_4_dispatch_0::@rocm_hsaco_fb::@main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@main_4_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@main_4_dispatch_0::@rocm_hsaco_fb::@main_4_dispatch_0_matmul_transpose_b_8192x5120x640_f16xf16xf32) : index
    scf.for %arg1 = %c0 to %1 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) flags("None")
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%0) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
