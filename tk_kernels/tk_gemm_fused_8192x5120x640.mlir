#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}> 
module {
  flow.executable private @tk_gemm_fused_8192x5120x640 {
    flow.executable.export public @tk_gemm_fused_8192x5120x640 workgroups() -> (index, index, index) {
      %c64 = arith.constant 64 : index
      %c40 = arith.constant 40 : index
      %c1 = arith.constant 1 : index
      flow.return %c64, %c40, %c1 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_8192x5120x640(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c35 = arith.constant 35 : index
        %c34 = arith.constant 34 : index
        %c33 = arith.constant 33 : index
        %c51 = arith.constant 51 : index
        %c50 = arith.constant 50 : index
        %c49 = arith.constant 49 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c512_i32 = arith.constant 512 : i32
        %c256_i32 = arith.constant 256 : i32
        %c8_i32 = arith.constant 8 : i32
        %c1_i32 = arith.constant 1 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c20 = arith.constant 20 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c48 = arith.constant 48 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<8192x640xf16, strided<[640, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c32 : index
        %2 = arith.muli %thread_id_z, %c64 : index
        %3 = arith.muli %workgroup_id_0, %c128 : index
        %4 = arith.divsi %thread_id_x, %c4 : index
        %5 = arith.addi %4, %3 : index
        %6 = arith.addi %5, %2 : index
        %7 = arith.addi %6, %1 : index
        %8 = arith.remsi %thread_id_x, %c4 : index
        %9 = arith.muli %8, %c8 : index
        %10 = vector.load %0[%7, %9] : memref<8192x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
        %11 = arith.addi %7, %c64 : index
        %12 = vector.load %0[%11, %9] : memref<8192x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
        %13 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120x640xf16, strided<[640, 1], offset: ?>>
        %14 = arith.muli %workgroup_id_1, %c128 : index
        %15 = arith.addi %4, %14 : index
        %16 = arith.addi %15, %2 : index
        %17 = arith.addi %16, %1 : index
        %18 = vector.load %13[%17, %9] : memref<5120x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
        %19 = arith.addi %17, %c64 : index
        %20 = vector.load %13[%19, %9] : memref<5120x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
        %21 = arith.addi %4, %2 : index
        %22 = arith.addi %21, %1 : index
        vector.store %10, %alloc_0[%22, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %23 = arith.addi %22, %c64 : index
        vector.store %12, %alloc_0[%23, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %18, %alloc[%22, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %20, %alloc[%23, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        amdgpu.lds_barrier
        %24 = arith.divsi %thread_id_x, %c64 : index
        %25 = arith.muli %24, %c64 : index
        %26 = arith.remsi %thread_id_x, %c16 : index
        %27 = arith.addi %26, %25 : index
        %28 = arith.addi %27, %c48 : index
        %29 = arith.remsi %thread_id_x, %c64 : index
        %30 = arith.divsi %29, %c16 : index
        %31 = arith.muli %30, %c4 : index
        %32 = arith.addi %31, %c16 : index
        %33 = vector.load %alloc_0[%28, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %34 = arith.muli %thread_id_y, %c64 : index
        %35 = arith.addi %26, %34 : index
        %36 = arith.addi %35, %c48 : index
        %37 = vector.load %alloc[%36, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %38 = vector.load %alloc_0[%28, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %39 = vector.load %alloc[%36, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %40 = amdgpu.mfma %38 * %39 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %41 = arith.addi %35, %c32 : index
        %42 = vector.load %alloc[%41, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %43 = vector.load %alloc[%41, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %44 = amdgpu.mfma %33 * %37 + %40 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %45 = amdgpu.mfma %38 * %43 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %46 = arith.addi %35, %c16 : index
        %47 = vector.load %alloc[%46, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %48 = vector.load %alloc[%46, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %49 = amdgpu.mfma %33 * %42 + %45 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %50 = amdgpu.mfma %38 * %48 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %51 = vector.load %alloc[%35, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %52 = vector.load %alloc[%35, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %53 = amdgpu.mfma %33 * %47 + %50 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %54 = amdgpu.mfma %38 * %52 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %55 = arith.addi %27, %c32 : index
        %56 = vector.load %alloc_0[%55, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %57 = vector.load %alloc_0[%55, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %58:27 = scf.for %arg4 = %c1 to %c20 step %c1 iter_args(%arg5 = %54, %arg6 = %52, %arg7 = %51, %arg8 = %48, %arg9 = %47, %arg10 = %43, %arg11 = %42, %arg12 = %39, %arg13 = %37, %arg14 = %57, %arg15 = %56, %arg16 = %33, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %cst, %arg28 = %cst, %arg29 = %53, %arg30 = %49, %arg31 = %44) -> (vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %321 = arith.muli %arg4, %c32 : index
          %322 = arith.addi %321, %9 : index
          %323 = vector.load %0[%7, %322] : memref<8192x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
          %324 = vector.load %0[%11, %322] : memref<8192x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
          %325 = amdgpu.mfma %arg16 * %arg7 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %326 = amdgpu.mfma %arg14 * %arg12 + %arg28 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %327 = amdgpu.mfma %arg14 * %arg10 + %arg27 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %328 = amdgpu.mfma %arg14 * %arg8 + %arg26 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %329 = arith.addi %27, %c16 : index
          %330 = vector.load %alloc_0[%329, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %331 = vector.load %alloc_0[%329, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %332 = vector.load %13[%17, %322] : memref<5120x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
          %333 = vector.load %13[%19, %322] : memref<5120x640xf16, strided<[640, 1], offset: ?>>, vector<8xf16>
          %334 = amdgpu.mfma %arg15 * %arg13 + %326 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %335 = amdgpu.mfma %arg15 * %arg11 + %327 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %336 = amdgpu.mfma %arg15 * %arg9 + %328 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %337 = amdgpu.mfma %arg14 * %arg6 + %arg25 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %338 = vector.load %alloc_0[%27, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %339 = vector.load %alloc_0[%27, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          vector.store %323, %alloc_0[%22, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %324, %alloc_0[%23, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %340 = amdgpu.mfma %arg15 * %arg7 + %337 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %341 = amdgpu.mfma %331 * %arg12 + %arg24 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %342 = amdgpu.mfma %331 * %arg10 + %arg23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %343 = amdgpu.mfma %331 * %arg8 + %arg22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %332, %alloc[%22, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %333, %alloc[%23, %9] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %344 = amdgpu.mfma %330 * %arg13 + %341 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %345 = amdgpu.mfma %330 * %arg11 + %342 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %346 = amdgpu.mfma %330 * %arg9 + %343 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %347 = amdgpu.mfma %331 * %arg6 + %arg21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %348 = vector.load %alloc_0[%28, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %349 = vector.load %alloc[%36, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %350 = amdgpu.mfma %330 * %arg7 + %347 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %351 = amdgpu.mfma %339 * %arg12 + %arg20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %352 = amdgpu.mfma %339 * %arg10 + %arg19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %353 = amdgpu.mfma %339 * %arg8 + %arg18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %354 = vector.load %alloc_0[%28, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %355 = vector.load %alloc[%36, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %356 = amdgpu.mfma %338 * %arg13 + %351 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %357 = amdgpu.mfma %338 * %arg11 + %352 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %358 = amdgpu.mfma %338 * %arg9 + %353 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %359 = amdgpu.mfma %339 * %arg6 + %arg17 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %360 = amdgpu.mfma %354 * %355 + %arg31 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %361 = vector.load %alloc[%41, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %362 = vector.load %alloc[%41, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %363 = amdgpu.mfma %338 * %arg7 + %359 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %364 = amdgpu.mfma %348 * %349 + %360 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %365 = amdgpu.mfma %354 * %362 + %arg30 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %366 = vector.load %alloc[%46, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %367 = vector.load %alloc[%46, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %368 = amdgpu.mfma %348 * %361 + %365 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %369 = amdgpu.mfma %354 * %367 + %arg29 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %370 = vector.load %alloc[%35, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %371 = vector.load %alloc[%35, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %372 = amdgpu.mfma %348 * %366 + %369 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %373 = amdgpu.mfma %354 * %371 + %325 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %374 = vector.load %alloc_0[%55, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %375 = vector.load %alloc_0[%55, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          scf.yield %373, %371, %370, %367, %366, %362, %361, %355, %349, %375, %374, %348, %363, %358, %357, %356, %350, %346, %345, %344, %340, %336, %335, %334, %372, %368, %364 : vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %59 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<8192x5120xf16, strided<[5120, 1], offset: ?>>
        %60 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<5120xf32, strided<[1], offset: ?>>
        %61 = arith.addi %3, %25 : index
        %62 = arith.addi %61, %31 : index
        %63 = arith.addi %62, %c48 : index
        %64 = arith.addi %26, %14 : index
        %65 = arith.addi %64, %34 : index
        %66 = arith.addi %65, %c16 : index
        %67 = vector.extract_strided_slice %58#24 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %68 = vector.load %60[%66] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %69 = arith.addf %67, %68 : vector<1xf32>
        %70 = arith.truncf %69 : vector<1xf32> to vector<1xf16>
        vector.store %70, %59[%63, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %71 = vector.extract_strided_slice %58#24 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %72 = arith.addi %62, %c49 : index
        %73 = arith.addf %71, %68 : vector<1xf32>
        %74 = arith.truncf %73 : vector<1xf32> to vector<1xf16>
        vector.store %74, %59[%72, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %75 = vector.extract_strided_slice %58#24 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %76 = arith.addi %62, %c50 : index
        %77 = arith.addf %75, %68 : vector<1xf32>
        %78 = arith.truncf %77 : vector<1xf32> to vector<1xf16>
        vector.store %78, %59[%76, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %79 = vector.extract_strided_slice %58#24 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %80 = arith.addi %62, %c51 : index
        %81 = arith.addf %79, %68 : vector<1xf32>
        %82 = arith.truncf %81 : vector<1xf32> to vector<1xf16>
        vector.store %82, %59[%80, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %83 = arith.addi %65, %c32 : index
        %84 = vector.extract_strided_slice %58#25 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %85 = vector.load %60[%83] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %86 = arith.addf %84, %85 : vector<1xf32>
        %87 = arith.truncf %86 : vector<1xf32> to vector<1xf16>
        vector.store %87, %59[%63, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %88 = vector.extract_strided_slice %58#25 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %89 = arith.addf %88, %85 : vector<1xf32>
        %90 = arith.truncf %89 : vector<1xf32> to vector<1xf16>
        vector.store %90, %59[%72, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %91 = vector.extract_strided_slice %58#25 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %92 = arith.addf %91, %85 : vector<1xf32>
        %93 = arith.truncf %92 : vector<1xf32> to vector<1xf16>
        vector.store %93, %59[%76, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %94 = vector.extract_strided_slice %58#25 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %95 = arith.addf %94, %85 : vector<1xf32>
        %96 = arith.truncf %95 : vector<1xf32> to vector<1xf16>
        vector.store %96, %59[%80, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %97 = arith.addi %65, %c48 : index
        %98 = vector.extract_strided_slice %58#26 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %99 = vector.load %60[%97] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %100 = arith.addf %98, %99 : vector<1xf32>
        %101 = arith.truncf %100 : vector<1xf32> to vector<1xf16>
        vector.store %101, %59[%63, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %102 = vector.extract_strided_slice %58#26 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %103 = arith.addf %102, %99 : vector<1xf32>
        %104 = arith.truncf %103 : vector<1xf32> to vector<1xf16>
        vector.store %104, %59[%72, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %105 = vector.extract_strided_slice %58#26 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %106 = arith.addf %105, %99 : vector<1xf32>
        %107 = arith.truncf %106 : vector<1xf32> to vector<1xf16>
        vector.store %107, %59[%76, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %108 = vector.extract_strided_slice %58#26 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %109 = arith.addf %108, %99 : vector<1xf32>
        %110 = arith.truncf %109 : vector<1xf32> to vector<1xf16>
        vector.store %110, %59[%80, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %111 = amdgpu.mfma %58#11 * %58#2 + %58#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %112 = vector.extract_strided_slice %111 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %113 = vector.load %60[%65] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %114 = arith.addf %112, %113 : vector<1xf32>
        %115 = arith.truncf %114 : vector<1xf32> to vector<1xf16>
        vector.store %115, %59[%63, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %116 = vector.extract_strided_slice %111 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %117 = arith.addf %116, %113 : vector<1xf32>
        %118 = arith.truncf %117 : vector<1xf32> to vector<1xf16>
        vector.store %118, %59[%72, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %119 = vector.extract_strided_slice %111 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %120 = arith.addf %119, %113 : vector<1xf32>
        %121 = arith.truncf %120 : vector<1xf32> to vector<1xf16>
        vector.store %121, %59[%76, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %122 = vector.extract_strided_slice %111 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %123 = arith.addf %122, %113 : vector<1xf32>
        %124 = arith.truncf %123 : vector<1xf32> to vector<1xf16>
        vector.store %124, %59[%80, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %125 = amdgpu.mfma %58#9 * %58#7 + %58#23 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %126 = amdgpu.mfma %58#9 * %58#5 + %58#22 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %127 = amdgpu.mfma %58#9 * %58#3 + %58#21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %128 = arith.addi %27, %c16 : index
        %129 = vector.load %alloc_0[%128, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %130 = vector.load %alloc_0[%128, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %131 = amdgpu.mfma %58#10 * %58#8 + %125 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %132 = arith.addi %62, %c32 : index
        %133 = vector.extract_strided_slice %131 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %134 = vector.load %60[%97] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %135 = arith.addf %133, %134 : vector<1xf32>
        %136 = arith.truncf %135 : vector<1xf32> to vector<1xf16>
        vector.store %136, %59[%132, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %137 = vector.extract_strided_slice %131 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %138 = arith.addi %62, %c33 : index
        %139 = arith.addf %137, %134 : vector<1xf32>
        %140 = arith.truncf %139 : vector<1xf32> to vector<1xf16>
        vector.store %140, %59[%138, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %141 = vector.extract_strided_slice %131 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %142 = arith.addi %62, %c34 : index
        %143 = arith.addf %141, %134 : vector<1xf32>
        %144 = arith.truncf %143 : vector<1xf32> to vector<1xf16>
        vector.store %144, %59[%142, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %145 = vector.extract_strided_slice %131 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %146 = arith.addi %62, %c35 : index
        %147 = arith.addf %145, %134 : vector<1xf32>
        %148 = arith.truncf %147 : vector<1xf32> to vector<1xf16>
        vector.store %148, %59[%146, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %149 = amdgpu.mfma %58#10 * %58#6 + %126 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %150 = vector.extract_strided_slice %149 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %151 = vector.load %60[%83] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %152 = arith.addf %150, %151 : vector<1xf32>
        %153 = arith.truncf %152 : vector<1xf32> to vector<1xf16>
        vector.store %153, %59[%132, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %154 = vector.extract_strided_slice %149 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %155 = arith.addf %154, %151 : vector<1xf32>
        %156 = arith.truncf %155 : vector<1xf32> to vector<1xf16>
        vector.store %156, %59[%138, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %157 = vector.extract_strided_slice %149 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %158 = arith.addf %157, %151 : vector<1xf32>
        %159 = arith.truncf %158 : vector<1xf32> to vector<1xf16>
        vector.store %159, %59[%142, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %160 = vector.extract_strided_slice %149 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %161 = arith.addf %160, %151 : vector<1xf32>
        %162 = arith.truncf %161 : vector<1xf32> to vector<1xf16>
        vector.store %162, %59[%146, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %163 = amdgpu.mfma %58#10 * %58#4 + %127 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %164 = vector.extract_strided_slice %163 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %165 = vector.load %60[%66] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %166 = arith.addf %164, %165 : vector<1xf32>
        %167 = arith.truncf %166 : vector<1xf32> to vector<1xf16>
        vector.store %167, %59[%132, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %168 = vector.extract_strided_slice %163 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %169 = arith.addf %168, %165 : vector<1xf32>
        %170 = arith.truncf %169 : vector<1xf32> to vector<1xf16>
        vector.store %170, %59[%138, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %171 = vector.extract_strided_slice %163 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %172 = arith.addf %171, %165 : vector<1xf32>
        %173 = arith.truncf %172 : vector<1xf32> to vector<1xf16>
        vector.store %173, %59[%142, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %174 = vector.extract_strided_slice %163 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %175 = arith.addf %174, %165 : vector<1xf32>
        %176 = arith.truncf %175 : vector<1xf32> to vector<1xf16>
        vector.store %176, %59[%146, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %177 = amdgpu.mfma %58#9 * %58#1 + %58#20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %178 = vector.load %alloc_0[%27, %32] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %179 = vector.load %alloc_0[%27, %31] : memref<128x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %180 = amdgpu.mfma %58#10 * %58#2 + %177 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %181 = vector.extract_strided_slice %180 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %182 = vector.load %60[%65] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %183 = arith.addf %181, %182 : vector<1xf32>
        %184 = arith.truncf %183 : vector<1xf32> to vector<1xf16>
        vector.store %184, %59[%132, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %185 = vector.extract_strided_slice %180 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %186 = arith.addf %185, %182 : vector<1xf32>
        %187 = arith.truncf %186 : vector<1xf32> to vector<1xf16>
        vector.store %187, %59[%138, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %188 = vector.extract_strided_slice %180 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %189 = arith.addf %188, %182 : vector<1xf32>
        %190 = arith.truncf %189 : vector<1xf32> to vector<1xf16>
        vector.store %190, %59[%142, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %191 = vector.extract_strided_slice %180 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %192 = arith.addf %191, %182 : vector<1xf32>
        %193 = arith.truncf %192 : vector<1xf32> to vector<1xf16>
        vector.store %193, %59[%146, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %194 = amdgpu.mfma %130 * %58#7 + %58#19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %195 = amdgpu.mfma %130 * %58#5 + %58#18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %196 = amdgpu.mfma %130 * %58#3 + %58#17 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %197 = amdgpu.mfma %129 * %58#8 + %194 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %198 = arith.addi %62, %c16 : index
        %199 = vector.extract_strided_slice %197 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %200 = vector.load %60[%97] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %201 = arith.addf %199, %200 : vector<1xf32>
        %202 = arith.truncf %201 : vector<1xf32> to vector<1xf16>
        vector.store %202, %59[%198, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %203 = vector.extract_strided_slice %197 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %204 = arith.addi %62, %c17 : index
        %205 = arith.addf %203, %200 : vector<1xf32>
        %206 = arith.truncf %205 : vector<1xf32> to vector<1xf16>
        vector.store %206, %59[%204, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %207 = vector.extract_strided_slice %197 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %208 = arith.addi %62, %c18 : index
        %209 = arith.addf %207, %200 : vector<1xf32>
        %210 = arith.truncf %209 : vector<1xf32> to vector<1xf16>
        vector.store %210, %59[%208, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %211 = vector.extract_strided_slice %197 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %212 = arith.addi %62, %c19 : index
        %213 = arith.addf %211, %200 : vector<1xf32>
        %214 = arith.truncf %213 : vector<1xf32> to vector<1xf16>
        vector.store %214, %59[%212, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %215 = amdgpu.mfma %129 * %58#6 + %195 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %216 = vector.extract_strided_slice %215 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %217 = vector.load %60[%83] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %218 = arith.addf %216, %217 : vector<1xf32>
        %219 = arith.truncf %218 : vector<1xf32> to vector<1xf16>
        vector.store %219, %59[%198, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %220 = vector.extract_strided_slice %215 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %221 = arith.addf %220, %217 : vector<1xf32>
        %222 = arith.truncf %221 : vector<1xf32> to vector<1xf16>
        vector.store %222, %59[%204, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %223 = vector.extract_strided_slice %215 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %224 = arith.addf %223, %217 : vector<1xf32>
        %225 = arith.truncf %224 : vector<1xf32> to vector<1xf16>
        vector.store %225, %59[%208, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %226 = vector.extract_strided_slice %215 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %227 = arith.addf %226, %217 : vector<1xf32>
        %228 = arith.truncf %227 : vector<1xf32> to vector<1xf16>
        vector.store %228, %59[%212, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %229 = amdgpu.mfma %129 * %58#4 + %196 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %230 = vector.extract_strided_slice %229 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %231 = vector.load %60[%66] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %232 = arith.addf %230, %231 : vector<1xf32>
        %233 = arith.truncf %232 : vector<1xf32> to vector<1xf16>
        vector.store %233, %59[%198, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %234 = vector.extract_strided_slice %229 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %235 = arith.addf %234, %231 : vector<1xf32>
        %236 = arith.truncf %235 : vector<1xf32> to vector<1xf16>
        vector.store %236, %59[%204, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %237 = vector.extract_strided_slice %229 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %238 = arith.addf %237, %231 : vector<1xf32>
        %239 = arith.truncf %238 : vector<1xf32> to vector<1xf16>
        vector.store %239, %59[%208, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %240 = vector.extract_strided_slice %229 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %241 = arith.addf %240, %231 : vector<1xf32>
        %242 = arith.truncf %241 : vector<1xf32> to vector<1xf16>
        vector.store %242, %59[%212, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %243 = amdgpu.mfma %130 * %58#1 + %58#16 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %244 = amdgpu.mfma %129 * %58#2 + %243 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %245 = vector.extract_strided_slice %244 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %246 = vector.load %60[%65] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %247 = arith.addf %245, %246 : vector<1xf32>
        %248 = arith.truncf %247 : vector<1xf32> to vector<1xf16>
        vector.store %248, %59[%198, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %249 = vector.extract_strided_slice %244 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %250 = arith.addf %249, %246 : vector<1xf32>
        %251 = arith.truncf %250 : vector<1xf32> to vector<1xf16>
        vector.store %251, %59[%204, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %252 = vector.extract_strided_slice %244 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %253 = arith.addf %252, %246 : vector<1xf32>
        %254 = arith.truncf %253 : vector<1xf32> to vector<1xf16>
        vector.store %254, %59[%208, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %255 = vector.extract_strided_slice %244 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %256 = arith.addf %255, %246 : vector<1xf32>
        %257 = arith.truncf %256 : vector<1xf32> to vector<1xf16>
        vector.store %257, %59[%212, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %258 = amdgpu.mfma %179 * %58#7 + %58#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %259 = amdgpu.mfma %179 * %58#5 + %58#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %260 = amdgpu.mfma %179 * %58#3 + %58#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %261 = amdgpu.mfma %178 * %58#8 + %258 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %262 = vector.extract_strided_slice %261 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %263 = vector.load %60[%97] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %264 = arith.addf %262, %263 : vector<1xf32>
        %265 = arith.truncf %264 : vector<1xf32> to vector<1xf16>
        vector.store %265, %59[%62, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %266 = vector.extract_strided_slice %261 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %267 = arith.addi %62, %c1 : index
        %268 = arith.addf %266, %263 : vector<1xf32>
        %269 = arith.truncf %268 : vector<1xf32> to vector<1xf16>
        vector.store %269, %59[%267, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %270 = vector.extract_strided_slice %261 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %271 = arith.addi %62, %c2 : index
        %272 = arith.addf %270, %263 : vector<1xf32>
        %273 = arith.truncf %272 : vector<1xf32> to vector<1xf16>
        vector.store %273, %59[%271, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %274 = vector.extract_strided_slice %261 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %275 = arith.addi %62, %c3 : index
        %276 = arith.addf %274, %263 : vector<1xf32>
        %277 = arith.truncf %276 : vector<1xf32> to vector<1xf16>
        vector.store %277, %59[%275, %97] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %278 = amdgpu.mfma %178 * %58#6 + %259 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %279 = vector.extract_strided_slice %278 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %280 = vector.load %60[%83] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %281 = arith.addf %279, %280 : vector<1xf32>
        %282 = arith.truncf %281 : vector<1xf32> to vector<1xf16>
        vector.store %282, %59[%62, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %283 = vector.extract_strided_slice %278 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %284 = arith.addf %283, %280 : vector<1xf32>
        %285 = arith.truncf %284 : vector<1xf32> to vector<1xf16>
        vector.store %285, %59[%267, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %286 = vector.extract_strided_slice %278 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %287 = arith.addf %286, %280 : vector<1xf32>
        %288 = arith.truncf %287 : vector<1xf32> to vector<1xf16>
        vector.store %288, %59[%271, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %289 = vector.extract_strided_slice %278 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %290 = arith.addf %289, %280 : vector<1xf32>
        %291 = arith.truncf %290 : vector<1xf32> to vector<1xf16>
        vector.store %291, %59[%275, %83] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %292 = amdgpu.mfma %178 * %58#4 + %260 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %293 = vector.extract_strided_slice %292 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %294 = vector.load %60[%66] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %295 = arith.addf %293, %294 : vector<1xf32>
        %296 = arith.truncf %295 : vector<1xf32> to vector<1xf16>
        vector.store %296, %59[%62, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %297 = vector.extract_strided_slice %292 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %298 = arith.addf %297, %294 : vector<1xf32>
        %299 = arith.truncf %298 : vector<1xf32> to vector<1xf16>
        vector.store %299, %59[%267, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %300 = vector.extract_strided_slice %292 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %301 = arith.addf %300, %294 : vector<1xf32>
        %302 = arith.truncf %301 : vector<1xf32> to vector<1xf16>
        vector.store %302, %59[%271, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %303 = vector.extract_strided_slice %292 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %304 = arith.addf %303, %294 : vector<1xf32>
        %305 = arith.truncf %304 : vector<1xf32> to vector<1xf16>
        vector.store %305, %59[%275, %66] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %306 = amdgpu.mfma %179 * %58#1 + %58#12 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %307 = amdgpu.mfma %178 * %58#2 + %306 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %308 = vector.extract_strided_slice %307 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %309 = vector.load %60[%65] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %310 = arith.addf %308, %309 : vector<1xf32>
        %311 = arith.truncf %310 : vector<1xf32> to vector<1xf16>
        vector.store %311, %59[%62, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %312 = vector.extract_strided_slice %307 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %313 = arith.addf %312, %309 : vector<1xf32>
        %314 = arith.truncf %313 : vector<1xf32> to vector<1xf16>
        vector.store %314, %59[%267, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %315 = vector.extract_strided_slice %307 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %316 = arith.addf %315, %309 : vector<1xf32>
        %317 = arith.truncf %316 : vector<1xf32> to vector<1xf16>
        vector.store %317, %59[%271, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        %318 = vector.extract_strided_slice %307 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %319 = arith.addf %318, %309 : vector<1xf32>
        %320 = arith.truncf %319 : vector<1xf32> to vector<1xf16>
        vector.store %320, %59[%275, %65] : memref<8192x5120xf16, strided<[5120, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

