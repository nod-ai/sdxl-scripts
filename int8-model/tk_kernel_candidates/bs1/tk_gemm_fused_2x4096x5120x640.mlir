#translation = #iree_codegen.translation_info<None workgroup_size = [128, 4, 1] subgroup_size = 64>
module {
  flow.executable private @tk_gemm_fused_2x4096x5120x640 {
    flow.executable.export public @tk_gemm_fused_2x4096x5120x640 workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      flow.return %c32, %c16, %c2 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2x4096x5120x640(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
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
        %c4 = arith.constant 4 : index
        %c1_i32 = arith.constant 1 : i32
        %c7_i32 = arith.constant 7 : i32
        %c512_i32 = arith.constant 512 : i32
        %c2_i32 = arith.constant 2 : i32
        %c3_i32 = arith.constant 3 : i32
        %c5_i32 = arith.constant 5 : i32
        %c256_i32 = arith.constant 256 : i32
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c5 = arith.constant 5 : index
        %c1 = arith.constant 1 : index
        %c80 = arith.constant 80 : index
        %c96 = arith.constant 96 : index
        %c48 = arith.constant 48 : index
        %c256 = arith.constant 256 : index
        %c192 = arith.constant 192 : index
        %c320 = arith.constant 320 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0> : vector<4xi32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<320x136xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<128x136xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2x4096x640xi8, strided<[2621440, 640, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c16 : index
        %2 = arith.muli %thread_id_z, %c64 : index
        %3 = arith.muli %workgroup_id_1, %c32 : index
        %4 = arith.addi %3, %workgroup_id_0 : index
        %5 = arith.divsi %4, %c16 : index
        %6 = arith.muli %5, %c128 : index
        %7 = arith.divsi %thread_id_x, %c8 : index
        %8 = arith.addi %7, %6 : index
        %9 = arith.addi %8, %2 : index
        %10 = arith.addi %9, %1 : index
        %11 = arith.remsi %thread_id_x, %c8 : index
        %12 = arith.muli %11, %c16 : index
        %13 = vector.load %0[%workgroup_id_2, %10, %12] : memref<2x4096x640xi8, strided<[2621440, 640, 1], offset: ?>>, vector<16xi8>
        %14 = arith.addi %10, %c64 : index
        %15 = vector.load %0[%workgroup_id_2, %14, %12] : memref<2x4096x640xi8, strided<[2621440, 640, 1], offset: ?>>, vector<16xi8>
        %16 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120x640xi8, strided<[640, 1], offset: ?>>
        %17 = arith.remsi %4, %c16 : index
        %18 = arith.muli %17, %c320 : index
        %19 = arith.addi %7, %18 : index
        %20 = arith.addi %19, %2 : index
        %21 = arith.addi %20, %1 : index
        %22 = vector.load %16[%21, %12] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
        %23 = arith.addi %21, %c64 : index
        %24 = vector.load %16[%23, %12] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
        %25 = arith.addi %21, %c128 : index
        %26 = vector.load %16[%25, %12] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
        %27 = arith.addi %21, %c192 : index
        %28 = vector.load %16[%27, %12] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
        %29 = arith.addi %21, %c256 : index
        %30 = vector.load %16[%29, %12] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
        %31 = arith.addi %7, %2 : index
        %32 = arith.addi %31, %1 : index
        vector.store %13, %alloc_0[%32, %12] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %33 = arith.addi %32, %c64 : index
        vector.store %15, %alloc_0[%33, %12] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %22, %alloc[%32, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %24, %alloc[%33, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %34 = arith.addi %32, %c128 : index
        vector.store %26, %alloc[%34, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %35 = arith.addi %32, %c192 : index
        vector.store %28, %alloc[%35, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %36 = arith.addi %32, %c256 : index
        vector.store %30, %alloc[%36, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        amdgpu.lds_barrier
        amdgpu.lds_barrier
        %37 = arith.divsi %thread_id_x, %c64 : index
        %38 = arith.muli %37, %c64 : index
        %39 = arith.remsi %thread_id_x, %c16 : index
        %40 = arith.addi %39, %38 : index
        %41 = arith.addi %40, %c48 : index
        %42 = arith.remsi %thread_id_x, %c64 : index
        %43 = arith.divsi %42, %c16 : index
        %44 = arith.muli %43, %c8 : index
        %45 = arith.addi %44, %c96 : index
        %46 = vector.load %alloc_0[%41, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %47 = arith.muli %thread_id_y, %c80 : index
        %48 = arith.addi %39, %47 : index
        %49 = arith.addi %48, %c64 : index
        %50 = vector.load %alloc[%49, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %51 = arith.addi %44, %c64 : index
        %52 = vector.load %alloc_0[%41, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %53 = vector.load %alloc[%49, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %54 = arith.addi %44, %c32 : index
        %55 = vector.load %alloc_0[%41, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %56 = vector.load %alloc[%49, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %57 = vector.load %alloc_0[%41, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %58 = vector.load %alloc[%49, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %59 = arith.addi %48, %c48 : index
        %60 = vector.load %alloc[%59, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %61 = vector.load %alloc[%59, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %62 = vector.load %alloc[%59, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %63 = amdgpu.mfma %57 * %58 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %64 = vector.load %alloc[%59, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %65 = amdgpu.mfma %55 * %56 + %63 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %66 = arith.addi %48, %c32 : index
        %67 = vector.load %alloc[%66, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %68 = amdgpu.mfma %52 * %53 + %65 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %69 = vector.load %alloc[%66, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %70 = amdgpu.mfma %46 * %50 + %68 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %71 = vector.load %alloc[%66, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %72 = amdgpu.mfma %57 * %64 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %73 = vector.load %alloc[%66, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %74 = amdgpu.mfma %55 * %62 + %72 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %75 = arith.addi %48, %c16 : index
        %76 = vector.load %alloc[%75, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %77 = amdgpu.mfma %52 * %61 + %74 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %78 = vector.load %alloc[%75, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %79 = amdgpu.mfma %46 * %60 + %77 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %80 = vector.load %alloc[%75, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %81 = amdgpu.mfma %57 * %73 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %82 = vector.load %alloc[%75, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %83 = amdgpu.mfma %55 * %71 + %81 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %84 = vector.load %alloc[%48, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %85 = amdgpu.mfma %52 * %69 + %83 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %86 = vector.load %alloc[%48, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %87 = amdgpu.mfma %46 * %67 + %85 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %88 = vector.load %alloc[%48, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %89 = amdgpu.mfma %57 * %82 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %90 = vector.load %alloc[%48, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %91 = amdgpu.mfma %55 * %80 + %89 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %92 = arith.addi %40, %c32 : index
        %93 = vector.load %alloc_0[%92, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %94 = amdgpu.mfma %52 * %78 + %91 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %95 = vector.load %alloc_0[%92, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %96 = amdgpu.mfma %46 * %76 + %94 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %97 = vector.load %alloc_0[%92, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %98 = amdgpu.mfma %57 * %90 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %99 = vector.load %alloc_0[%92, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %100 = amdgpu.mfma %55 * %88 + %98 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %101 = arith.addi %40, %c16 : index
        %102 = vector.load %alloc_0[%101, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %103 = amdgpu.mfma %52 * %86 + %100 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %104 = vector.load %alloc_0[%101, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %105 = amdgpu.mfma %46 * %84 + %103 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %106 = vector.load %alloc_0[%101, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %107:47 = scf.for %arg5 = %c1 to %c5 step %c1 iter_args(%arg6 = %90, %arg7 = %88, %arg8 = %86, %arg9 = %84, %arg10 = %106, %arg11 = %82, %arg12 = %80, %arg13 = %78, %arg14 = %76, %arg15 = %104, %arg16 = %73, %arg17 = %71, %arg18 = %69, %arg19 = %67, %arg20 = %102, %arg21 = %64, %arg22 = %62, %arg23 = %61, %arg24 = %60, %arg25 = %58, %arg26 = %56, %arg27 = %53, %arg28 = %50, %arg29 = %99, %arg30 = %97, %arg31 = %95, %arg32 = %93, %arg33 = %cst, %arg34 = %cst, %arg35 = %cst, %arg36 = %cst, %arg37 = %cst, %arg38 = %cst, %arg39 = %cst, %arg40 = %cst, %arg41 = %cst, %arg42 = %cst, %arg43 = %cst, %arg44 = %cst, %arg45 = %cst, %arg46 = %cst, %arg47 = %cst, %arg48 = %105, %arg49 = %96, %arg50 = %87, %arg51 = %79, %arg52 = %70) -> (vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
          %638 = arith.muli %arg5, %c128 : index
          %639 = arith.addi %638, %12 : index
          %640 = vector.load %0[%workgroup_id_2, %10, %639] : memref<2x4096x640xi8, strided<[2621440, 640, 1], offset: ?>>, vector<16xi8>
          %641 = vector.load %0[%workgroup_id_2, %14, %639] : memref<2x4096x640xi8, strided<[2621440, 640, 1], offset: ?>>, vector<16xi8>
          %642 = vector.load %16[%21, %639] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
          %643 = vector.load %16[%23, %639] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
          %644 = amdgpu.mfma %arg29 * %arg25 + %arg47 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %645 = amdgpu.mfma %arg29 * %arg21 + %arg46 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %646 = amdgpu.mfma %arg29 * %arg16 + %arg45 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %647 = amdgpu.mfma %arg29 * %arg11 + %arg44 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %648 = vector.load %alloc_0[%101, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %649 = vector.load %alloc_0[%40, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %650 = vector.load %alloc_0[%40, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %651 = vector.load %alloc_0[%40, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %652 = vector.load %alloc_0[%40, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c5_i32, %c0_i32) : (i32, i32, i32) -> ()
          %653 = vector.load %16[%25, %639] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
          %654 = vector.load %16[%27, %639] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
          %655 = vector.load %16[%29, %639] : memref<5120x640xi8, strided<[640, 1], offset: ?>>, vector<16xi8>
          %656 = amdgpu.mfma %arg30 * %arg26 + %644 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %657 = amdgpu.mfma %arg30 * %arg22 + %645 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %658 = amdgpu.mfma %arg30 * %arg17 + %646 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %659 = amdgpu.mfma %arg30 * %arg12 + %647 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %660 = amdgpu.mfma %arg31 * %arg27 + %656 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %661 = amdgpu.mfma %arg31 * %arg23 + %657 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %662 = amdgpu.mfma %arg31 * %arg18 + %658 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %663 = amdgpu.mfma %arg31 * %arg13 + %659 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %664 = amdgpu.mfma %arg32 * %arg28 + %660 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %665 = amdgpu.mfma %arg32 * %arg24 + %661 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %666 = amdgpu.mfma %arg32 * %arg19 + %662 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %667 = amdgpu.mfma %arg32 * %arg14 + %663 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %668 = amdgpu.mfma %arg29 * %arg6 + %arg43 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %669 = amdgpu.mfma %648 * %arg25 + %arg42 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %670 = amdgpu.mfma %648 * %arg21 + %arg41 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %671 = amdgpu.mfma %648 * %arg16 + %arg40 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %672 = amdgpu.mfma %arg30 * %arg7 + %668 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %673 = amdgpu.mfma %arg10 * %arg26 + %669 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %674 = amdgpu.mfma %arg10 * %arg22 + %670 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %675 = amdgpu.mfma %arg10 * %arg17 + %671 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %676 = amdgpu.mfma %arg31 * %arg8 + %672 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %677 = amdgpu.mfma %arg15 * %arg27 + %673 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %678 = amdgpu.mfma %arg15 * %arg23 + %674 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %679 = amdgpu.mfma %arg15 * %arg18 + %675 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %680 = amdgpu.mfma %arg32 * %arg9 + %676 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %681 = amdgpu.mfma %arg20 * %arg28 + %677 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %682 = amdgpu.mfma %arg20 * %arg24 + %678 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %683 = amdgpu.mfma %arg20 * %arg19 + %679 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %684 = amdgpu.mfma %648 * %arg11 + %arg39 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %685 = amdgpu.mfma %648 * %arg6 + %arg38 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %686 = amdgpu.mfma %arg10 * %arg12 + %684 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %687 = amdgpu.mfma %arg10 * %arg7 + %685 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %688 = amdgpu.mfma %arg15 * %arg13 + %686 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %689 = amdgpu.mfma %arg15 * %arg8 + %687 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          vector.store %640, %alloc_0[%32, %12] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %641, %alloc_0[%33, %12] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %642, %alloc[%32, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %643, %alloc[%33, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %653, %alloc[%34, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %654, %alloc[%35, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %655, %alloc[%36, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %690 = amdgpu.mfma %arg20 * %arg14 + %688 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %691 = amdgpu.mfma %arg20 * %arg9 + %689 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c7_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %692 = vector.load %alloc_0[%41, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %693 = vector.load %alloc[%49, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %694 = vector.load %alloc_0[%41, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %695 = vector.load %alloc[%49, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %696 = amdgpu.mfma %652 * %arg25 + %arg37 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %697 = amdgpu.mfma %652 * %arg21 + %arg36 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %698 = amdgpu.mfma %652 * %arg16 + %arg35 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %699 = amdgpu.mfma %652 * %arg11 + %arg34 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %700 = vector.load %alloc_0[%41, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %701 = amdgpu.mfma %651 * %arg26 + %696 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %702 = amdgpu.mfma %651 * %arg22 + %697 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %703 = amdgpu.mfma %651 * %arg17 + %698 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %704 = amdgpu.mfma %651 * %arg12 + %699 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %705 = vector.load %alloc[%49, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %706 = amdgpu.mfma %650 * %arg27 + %701 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %707 = amdgpu.mfma %650 * %arg23 + %702 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %708 = amdgpu.mfma %650 * %arg18 + %703 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %709 = amdgpu.mfma %650 * %arg13 + %704 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %710 = vector.load %alloc_0[%41, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %711 = amdgpu.mfma %649 * %arg28 + %706 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %712 = amdgpu.mfma %649 * %arg24 + %707 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %713 = amdgpu.mfma %649 * %arg19 + %708 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %714 = amdgpu.mfma %649 * %arg14 + %709 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %715 = vector.load %alloc[%49, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %716 = amdgpu.mfma %652 * %arg6 + %arg33 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %717 = vector.load %alloc[%59, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %718 = amdgpu.mfma %651 * %arg7 + %716 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %719 = vector.load %alloc[%59, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %720 = amdgpu.mfma %650 * %arg8 + %718 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %721 = vector.load %alloc[%59, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %722 = amdgpu.mfma %649 * %arg9 + %720 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %723 = amdgpu.mfma %710 * %715 + %arg52 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %724 = vector.load %alloc[%59, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %725 = amdgpu.mfma %700 * %705 + %723 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %726 = vector.load %alloc[%66, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %727 = amdgpu.mfma %694 * %695 + %725 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %728 = vector.load %alloc[%66, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %729 = amdgpu.mfma %692 * %693 + %727 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %730 = vector.load %alloc[%66, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %731 = amdgpu.mfma %710 * %724 + %arg51 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %732 = vector.load %alloc[%66, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %733 = amdgpu.mfma %700 * %721 + %731 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %734 = vector.load %alloc[%75, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %735 = amdgpu.mfma %694 * %719 + %733 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %736 = vector.load %alloc[%75, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %737 = amdgpu.mfma %692 * %717 + %735 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %738 = vector.load %alloc[%75, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %739 = amdgpu.mfma %710 * %732 + %arg50 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %740 = vector.load %alloc[%75, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %741 = amdgpu.mfma %700 * %730 + %739 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %742 = vector.load %alloc[%48, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %743 = amdgpu.mfma %694 * %728 + %741 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %744 = vector.load %alloc[%48, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %745 = amdgpu.mfma %692 * %726 + %743 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %746 = vector.load %alloc[%48, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %747 = amdgpu.mfma %710 * %740 + %arg49 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %748 = vector.load %alloc[%48, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %749 = amdgpu.mfma %700 * %738 + %747 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %750 = vector.load %alloc_0[%92, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %751 = amdgpu.mfma %694 * %736 + %749 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %752 = vector.load %alloc_0[%92, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %753 = amdgpu.mfma %692 * %734 + %751 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %754 = vector.load %alloc_0[%92, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %755 = amdgpu.mfma %710 * %748 + %arg48 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %756 = vector.load %alloc_0[%92, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %757 = amdgpu.mfma %700 * %746 + %755 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %758 = vector.load %alloc_0[%101, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %759 = amdgpu.mfma %694 * %744 + %757 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %760 = vector.load %alloc_0[%101, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %761 = amdgpu.mfma %692 * %742 + %759 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %762 = vector.load %alloc_0[%101, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          scf.yield %748, %746, %744, %742, %762, %740, %738, %736, %734, %760, %732, %730, %728, %726, %758, %724, %721, %719, %717, %715, %705, %695, %693, %756, %754, %752, %750, %722, %714, %713, %712, %711, %691, %690, %683, %682, %681, %680, %667, %666, %665, %664, %761, %753, %745, %737, %729 : vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
        }
        %108 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>
        %109 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<5120xi32, strided<[1], offset: ?>>
        %110 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<5120xf32, strided<[1], offset: ?>>
        %111 = arith.muli %43, %c4 : index
        %112 = arith.addi %6, %38 : index
        %113 = arith.addi %112, %111 : index
        %114 = arith.addi %113, %c48 : index
        %115 = arith.addi %39, %18 : index
        %116 = arith.addi %115, %47 : index
        %117 = vector.extract_strided_slice %107#42 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %118 = vector.load %109[%116] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %119 = vector.load %110[%116] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %120 = arith.addi %117, %118 : vector<1xi32>
        %121 = arith.sitofp %120 : vector<1xi32> to vector<1xf32>
        %122 = arith.mulf %121, %119 : vector<1xf32>
        %123 = arith.truncf %122 : vector<1xf32> to vector<1xf16>
        vector.store %123, %108[%workgroup_id_2, %114, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %124 = vector.extract_strided_slice %107#42 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %125 = arith.addi %113, %c49 : index
        %126 = arith.addi %124, %118 : vector<1xi32>
        %127 = arith.sitofp %126 : vector<1xi32> to vector<1xf32>
        %128 = arith.mulf %127, %119 : vector<1xf32>
        %129 = arith.truncf %128 : vector<1xf32> to vector<1xf16>
        vector.store %129, %108[%workgroup_id_2, %125, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %130 = vector.extract_strided_slice %107#42 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %131 = arith.addi %113, %c50 : index
        %132 = arith.addi %130, %118 : vector<1xi32>
        %133 = arith.sitofp %132 : vector<1xi32> to vector<1xf32>
        %134 = arith.mulf %133, %119 : vector<1xf32>
        %135 = arith.truncf %134 : vector<1xf32> to vector<1xf16>
        vector.store %135, %108[%workgroup_id_2, %131, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %136 = vector.extract_strided_slice %107#42 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %137 = arith.addi %113, %c51 : index
        %138 = arith.addi %136, %118 : vector<1xi32>
        %139 = arith.sitofp %138 : vector<1xi32> to vector<1xf32>
        %140 = arith.mulf %139, %119 : vector<1xf32>
        %141 = arith.truncf %140 : vector<1xf32> to vector<1xf16>
        vector.store %141, %108[%workgroup_id_2, %137, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %142 = arith.addi %116, %c16 : index
        %143 = vector.extract_strided_slice %107#43 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %144 = vector.load %109[%142] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %145 = vector.load %110[%142] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %146 = arith.addi %143, %144 : vector<1xi32>
        %147 = arith.sitofp %146 : vector<1xi32> to vector<1xf32>
        %148 = arith.mulf %147, %145 : vector<1xf32>
        %149 = arith.truncf %148 : vector<1xf32> to vector<1xf16>
        vector.store %149, %108[%workgroup_id_2, %114, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %150 = vector.extract_strided_slice %107#43 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %151 = arith.addi %150, %144 : vector<1xi32>
        %152 = arith.sitofp %151 : vector<1xi32> to vector<1xf32>
        %153 = arith.mulf %152, %145 : vector<1xf32>
        %154 = arith.truncf %153 : vector<1xf32> to vector<1xf16>
        vector.store %154, %108[%workgroup_id_2, %125, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %155 = vector.extract_strided_slice %107#43 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %156 = arith.addi %155, %144 : vector<1xi32>
        %157 = arith.sitofp %156 : vector<1xi32> to vector<1xf32>
        %158 = arith.mulf %157, %145 : vector<1xf32>
        %159 = arith.truncf %158 : vector<1xf32> to vector<1xf16>
        vector.store %159, %108[%workgroup_id_2, %131, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %160 = vector.extract_strided_slice %107#43 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %161 = arith.addi %160, %144 : vector<1xi32>
        %162 = arith.sitofp %161 : vector<1xi32> to vector<1xf32>
        %163 = arith.mulf %162, %145 : vector<1xf32>
        %164 = arith.truncf %163 : vector<1xf32> to vector<1xf16>
        vector.store %164, %108[%workgroup_id_2, %137, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %165 = arith.addi %116, %c32 : index
        %166 = vector.extract_strided_slice %107#44 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %167 = vector.load %109[%165] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %168 = vector.load %110[%165] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %169 = arith.addi %166, %167 : vector<1xi32>
        %170 = arith.sitofp %169 : vector<1xi32> to vector<1xf32>
        %171 = arith.mulf %170, %168 : vector<1xf32>
        %172 = arith.truncf %171 : vector<1xf32> to vector<1xf16>
        vector.store %172, %108[%workgroup_id_2, %114, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %173 = vector.extract_strided_slice %107#44 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %174 = arith.addi %173, %167 : vector<1xi32>
        %175 = arith.sitofp %174 : vector<1xi32> to vector<1xf32>
        %176 = arith.mulf %175, %168 : vector<1xf32>
        %177 = arith.truncf %176 : vector<1xf32> to vector<1xf16>
        vector.store %177, %108[%workgroup_id_2, %125, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %178 = vector.extract_strided_slice %107#44 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %179 = arith.addi %178, %167 : vector<1xi32>
        %180 = arith.sitofp %179 : vector<1xi32> to vector<1xf32>
        %181 = arith.mulf %180, %168 : vector<1xf32>
        %182 = arith.truncf %181 : vector<1xf32> to vector<1xf16>
        vector.store %182, %108[%workgroup_id_2, %131, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %183 = vector.extract_strided_slice %107#44 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %184 = arith.addi %183, %167 : vector<1xi32>
        %185 = arith.sitofp %184 : vector<1xi32> to vector<1xf32>
        %186 = arith.mulf %185, %168 : vector<1xf32>
        %187 = arith.truncf %186 : vector<1xf32> to vector<1xf16>
        vector.store %187, %108[%workgroup_id_2, %137, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %188 = arith.addi %116, %c48 : index
        %189 = vector.extract_strided_slice %107#45 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %190 = vector.load %109[%188] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %191 = vector.load %110[%188] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %192 = arith.addi %189, %190 : vector<1xi32>
        %193 = arith.sitofp %192 : vector<1xi32> to vector<1xf32>
        %194 = arith.mulf %193, %191 : vector<1xf32>
        %195 = arith.truncf %194 : vector<1xf32> to vector<1xf16>
        vector.store %195, %108[%workgroup_id_2, %114, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %196 = vector.extract_strided_slice %107#45 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %197 = arith.addi %196, %190 : vector<1xi32>
        %198 = arith.sitofp %197 : vector<1xi32> to vector<1xf32>
        %199 = arith.mulf %198, %191 : vector<1xf32>
        %200 = arith.truncf %199 : vector<1xf32> to vector<1xf16>
        vector.store %200, %108[%workgroup_id_2, %125, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %201 = vector.extract_strided_slice %107#45 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %202 = arith.addi %201, %190 : vector<1xi32>
        %203 = arith.sitofp %202 : vector<1xi32> to vector<1xf32>
        %204 = arith.mulf %203, %191 : vector<1xf32>
        %205 = arith.truncf %204 : vector<1xf32> to vector<1xf16>
        vector.store %205, %108[%workgroup_id_2, %131, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %206 = vector.extract_strided_slice %107#45 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %207 = arith.addi %206, %190 : vector<1xi32>
        %208 = arith.sitofp %207 : vector<1xi32> to vector<1xf32>
        %209 = arith.mulf %208, %191 : vector<1xf32>
        %210 = arith.truncf %209 : vector<1xf32> to vector<1xf16>
        vector.store %210, %108[%workgroup_id_2, %137, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %211 = arith.addi %116, %c64 : index
        %212 = vector.extract_strided_slice %107#46 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %213 = vector.load %109[%211] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %214 = vector.load %110[%211] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %215 = arith.addi %212, %213 : vector<1xi32>
        %216 = arith.sitofp %215 : vector<1xi32> to vector<1xf32>
        %217 = arith.mulf %216, %214 : vector<1xf32>
        %218 = arith.truncf %217 : vector<1xf32> to vector<1xf16>
        vector.store %218, %108[%workgroup_id_2, %114, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %219 = vector.extract_strided_slice %107#46 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %220 = arith.addi %219, %213 : vector<1xi32>
        %221 = arith.sitofp %220 : vector<1xi32> to vector<1xf32>
        %222 = arith.mulf %221, %214 : vector<1xf32>
        %223 = arith.truncf %222 : vector<1xf32> to vector<1xf16>
        vector.store %223, %108[%workgroup_id_2, %125, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %224 = vector.extract_strided_slice %107#46 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %225 = arith.addi %224, %213 : vector<1xi32>
        %226 = arith.sitofp %225 : vector<1xi32> to vector<1xf32>
        %227 = arith.mulf %226, %214 : vector<1xf32>
        %228 = arith.truncf %227 : vector<1xf32> to vector<1xf16>
        vector.store %228, %108[%workgroup_id_2, %131, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %229 = vector.extract_strided_slice %107#46 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %230 = arith.addi %229, %213 : vector<1xi32>
        %231 = arith.sitofp %230 : vector<1xi32> to vector<1xf32>
        %232 = arith.mulf %231, %214 : vector<1xf32>
        %233 = arith.truncf %232 : vector<1xf32> to vector<1xf16>
        vector.store %233, %108[%workgroup_id_2, %137, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %234 = amdgpu.mfma %107#23 * %107#19 + %107#41 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %235 = amdgpu.mfma %107#23 * %107#15 + %107#40 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %236 = amdgpu.mfma %107#23 * %107#10 + %107#39 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %237 = amdgpu.mfma %107#23 * %107#5 + %107#38 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %238 = vector.load %alloc_0[%101, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %239 = vector.load %alloc_0[%40, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %240 = vector.load %alloc_0[%40, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %241 = vector.load %alloc_0[%40, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %242 = vector.load %alloc_0[%40, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %243 = amdgpu.mfma %107#24 * %107#20 + %234 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %244 = amdgpu.mfma %107#24 * %107#16 + %235 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %245 = amdgpu.mfma %107#24 * %107#11 + %236 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %246 = amdgpu.mfma %107#24 * %107#6 + %237 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %247 = amdgpu.mfma %107#25 * %107#21 + %243 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %248 = amdgpu.mfma %107#25 * %107#17 + %244 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %249 = amdgpu.mfma %107#25 * %107#12 + %245 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %250 = amdgpu.mfma %107#25 * %107#7 + %246 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %251 = amdgpu.mfma %107#26 * %107#22 + %247 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %252 = arith.addi %113, %c32 : index
        %253 = vector.extract_strided_slice %251 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %254 = arith.addi %253, %213 : vector<1xi32>
        %255 = arith.sitofp %254 : vector<1xi32> to vector<1xf32>
        %256 = arith.mulf %255, %214 : vector<1xf32>
        %257 = arith.truncf %256 : vector<1xf32> to vector<1xf16>
        vector.store %257, %108[%workgroup_id_2, %252, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %258 = vector.extract_strided_slice %251 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %259 = arith.addi %113, %c33 : index
        %260 = arith.addi %258, %213 : vector<1xi32>
        %261 = arith.sitofp %260 : vector<1xi32> to vector<1xf32>
        %262 = arith.mulf %261, %214 : vector<1xf32>
        %263 = arith.truncf %262 : vector<1xf32> to vector<1xf16>
        vector.store %263, %108[%workgroup_id_2, %259, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %264 = vector.extract_strided_slice %251 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %265 = arith.addi %113, %c34 : index
        %266 = arith.addi %264, %213 : vector<1xi32>
        %267 = arith.sitofp %266 : vector<1xi32> to vector<1xf32>
        %268 = arith.mulf %267, %214 : vector<1xf32>
        %269 = arith.truncf %268 : vector<1xf32> to vector<1xf16>
        vector.store %269, %108[%workgroup_id_2, %265, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %270 = vector.extract_strided_slice %251 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %271 = arith.addi %113, %c35 : index
        %272 = arith.addi %270, %213 : vector<1xi32>
        %273 = arith.sitofp %272 : vector<1xi32> to vector<1xf32>
        %274 = arith.mulf %273, %214 : vector<1xf32>
        %275 = arith.truncf %274 : vector<1xf32> to vector<1xf16>
        vector.store %275, %108[%workgroup_id_2, %271, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %276 = amdgpu.mfma %107#26 * %107#18 + %248 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %277 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %278 = vector.load %109[%188] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %279 = vector.load %110[%188] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %280 = arith.addi %277, %278 : vector<1xi32>
        %281 = arith.sitofp %280 : vector<1xi32> to vector<1xf32>
        %282 = arith.mulf %281, %279 : vector<1xf32>
        %283 = arith.truncf %282 : vector<1xf32> to vector<1xf16>
        vector.store %283, %108[%workgroup_id_2, %252, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %284 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %285 = arith.addi %284, %278 : vector<1xi32>
        %286 = arith.sitofp %285 : vector<1xi32> to vector<1xf32>
        %287 = arith.mulf %286, %279 : vector<1xf32>
        %288 = arith.truncf %287 : vector<1xf32> to vector<1xf16>
        vector.store %288, %108[%workgroup_id_2, %259, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %289 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %290 = arith.addi %289, %278 : vector<1xi32>
        %291 = arith.sitofp %290 : vector<1xi32> to vector<1xf32>
        %292 = arith.mulf %291, %279 : vector<1xf32>
        %293 = arith.truncf %292 : vector<1xf32> to vector<1xf16>
        vector.store %293, %108[%workgroup_id_2, %265, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %294 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %295 = arith.addi %294, %278 : vector<1xi32>
        %296 = arith.sitofp %295 : vector<1xi32> to vector<1xf32>
        %297 = arith.mulf %296, %279 : vector<1xf32>
        %298 = arith.truncf %297 : vector<1xf32> to vector<1xf16>
        vector.store %298, %108[%workgroup_id_2, %271, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %299 = amdgpu.mfma %107#26 * %107#13 + %249 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %300 = vector.extract_strided_slice %299 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %301 = vector.load %109[%165] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %302 = vector.load %110[%165] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %303 = arith.addi %300, %301 : vector<1xi32>
        %304 = arith.sitofp %303 : vector<1xi32> to vector<1xf32>
        %305 = arith.mulf %304, %302 : vector<1xf32>
        %306 = arith.truncf %305 : vector<1xf32> to vector<1xf16>
        vector.store %306, %108[%workgroup_id_2, %252, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %307 = vector.extract_strided_slice %299 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %308 = arith.addi %307, %301 : vector<1xi32>
        %309 = arith.sitofp %308 : vector<1xi32> to vector<1xf32>
        %310 = arith.mulf %309, %302 : vector<1xf32>
        %311 = arith.truncf %310 : vector<1xf32> to vector<1xf16>
        vector.store %311, %108[%workgroup_id_2, %259, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %312 = vector.extract_strided_slice %299 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %313 = arith.addi %312, %301 : vector<1xi32>
        %314 = arith.sitofp %313 : vector<1xi32> to vector<1xf32>
        %315 = arith.mulf %314, %302 : vector<1xf32>
        %316 = arith.truncf %315 : vector<1xf32> to vector<1xf16>
        vector.store %316, %108[%workgroup_id_2, %265, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %317 = vector.extract_strided_slice %299 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %318 = arith.addi %317, %301 : vector<1xi32>
        %319 = arith.sitofp %318 : vector<1xi32> to vector<1xf32>
        %320 = arith.mulf %319, %302 : vector<1xf32>
        %321 = arith.truncf %320 : vector<1xf32> to vector<1xf16>
        vector.store %321, %108[%workgroup_id_2, %271, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %322 = amdgpu.mfma %107#26 * %107#8 + %250 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %323 = vector.extract_strided_slice %322 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %324 = vector.load %109[%142] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %325 = vector.load %110[%142] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %326 = arith.addi %323, %324 : vector<1xi32>
        %327 = arith.sitofp %326 : vector<1xi32> to vector<1xf32>
        %328 = arith.mulf %327, %325 : vector<1xf32>
        %329 = arith.truncf %328 : vector<1xf32> to vector<1xf16>
        vector.store %329, %108[%workgroup_id_2, %252, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %330 = vector.extract_strided_slice %322 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %331 = arith.addi %330, %324 : vector<1xi32>
        %332 = arith.sitofp %331 : vector<1xi32> to vector<1xf32>
        %333 = arith.mulf %332, %325 : vector<1xf32>
        %334 = arith.truncf %333 : vector<1xf32> to vector<1xf16>
        vector.store %334, %108[%workgroup_id_2, %259, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %335 = vector.extract_strided_slice %322 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %336 = arith.addi %335, %324 : vector<1xi32>
        %337 = arith.sitofp %336 : vector<1xi32> to vector<1xf32>
        %338 = arith.mulf %337, %325 : vector<1xf32>
        %339 = arith.truncf %338 : vector<1xf32> to vector<1xf16>
        vector.store %339, %108[%workgroup_id_2, %265, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %340 = vector.extract_strided_slice %322 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %341 = arith.addi %340, %324 : vector<1xi32>
        %342 = arith.sitofp %341 : vector<1xi32> to vector<1xf32>
        %343 = arith.mulf %342, %325 : vector<1xf32>
        %344 = arith.truncf %343 : vector<1xf32> to vector<1xf16>
        vector.store %344, %108[%workgroup_id_2, %271, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %345 = amdgpu.mfma %107#23 * %107#0 + %107#37 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %346 = amdgpu.mfma %238 * %107#19 + %107#36 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %347 = amdgpu.mfma %238 * %107#15 + %107#35 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %348 = amdgpu.mfma %238 * %107#10 + %107#34 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %349 = amdgpu.mfma %107#24 * %107#1 + %345 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %350 = amdgpu.mfma %107#4 * %107#20 + %346 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %351 = amdgpu.mfma %107#4 * %107#16 + %347 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %352 = amdgpu.mfma %107#4 * %107#11 + %348 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %353 = amdgpu.mfma %107#25 * %107#2 + %349 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %354 = amdgpu.mfma %107#9 * %107#21 + %350 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %355 = amdgpu.mfma %107#9 * %107#17 + %351 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %356 = amdgpu.mfma %107#9 * %107#12 + %352 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %357 = amdgpu.mfma %107#26 * %107#3 + %353 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %358 = vector.extract_strided_slice %357 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %359 = vector.load %109[%116] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %360 = vector.load %110[%116] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %361 = arith.addi %358, %359 : vector<1xi32>
        %362 = arith.sitofp %361 : vector<1xi32> to vector<1xf32>
        %363 = arith.mulf %362, %360 : vector<1xf32>
        %364 = arith.truncf %363 : vector<1xf32> to vector<1xf16>
        vector.store %364, %108[%workgroup_id_2, %252, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %365 = vector.extract_strided_slice %357 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %366 = arith.addi %365, %359 : vector<1xi32>
        %367 = arith.sitofp %366 : vector<1xi32> to vector<1xf32>
        %368 = arith.mulf %367, %360 : vector<1xf32>
        %369 = arith.truncf %368 : vector<1xf32> to vector<1xf16>
        vector.store %369, %108[%workgroup_id_2, %259, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %370 = vector.extract_strided_slice %357 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %371 = arith.addi %370, %359 : vector<1xi32>
        %372 = arith.sitofp %371 : vector<1xi32> to vector<1xf32>
        %373 = arith.mulf %372, %360 : vector<1xf32>
        %374 = arith.truncf %373 : vector<1xf32> to vector<1xf16>
        vector.store %374, %108[%workgroup_id_2, %265, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %375 = vector.extract_strided_slice %357 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %376 = arith.addi %375, %359 : vector<1xi32>
        %377 = arith.sitofp %376 : vector<1xi32> to vector<1xf32>
        %378 = arith.mulf %377, %360 : vector<1xf32>
        %379 = arith.truncf %378 : vector<1xf32> to vector<1xf16>
        vector.store %379, %108[%workgroup_id_2, %271, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %380 = amdgpu.mfma %107#14 * %107#22 + %354 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %381 = arith.addi %113, %c16 : index
        %382 = vector.extract_strided_slice %380 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %383 = vector.load %109[%211] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %384 = vector.load %110[%211] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %385 = arith.addi %382, %383 : vector<1xi32>
        %386 = arith.sitofp %385 : vector<1xi32> to vector<1xf32>
        %387 = arith.mulf %386, %384 : vector<1xf32>
        %388 = arith.truncf %387 : vector<1xf32> to vector<1xf16>
        vector.store %388, %108[%workgroup_id_2, %381, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %389 = vector.extract_strided_slice %380 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %390 = arith.addi %113, %c17 : index
        %391 = arith.addi %389, %383 : vector<1xi32>
        %392 = arith.sitofp %391 : vector<1xi32> to vector<1xf32>
        %393 = arith.mulf %392, %384 : vector<1xf32>
        %394 = arith.truncf %393 : vector<1xf32> to vector<1xf16>
        vector.store %394, %108[%workgroup_id_2, %390, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %395 = vector.extract_strided_slice %380 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %396 = arith.addi %113, %c18 : index
        %397 = arith.addi %395, %383 : vector<1xi32>
        %398 = arith.sitofp %397 : vector<1xi32> to vector<1xf32>
        %399 = arith.mulf %398, %384 : vector<1xf32>
        %400 = arith.truncf %399 : vector<1xf32> to vector<1xf16>
        vector.store %400, %108[%workgroup_id_2, %396, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %401 = vector.extract_strided_slice %380 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %402 = arith.addi %113, %c19 : index
        %403 = arith.addi %401, %383 : vector<1xi32>
        %404 = arith.sitofp %403 : vector<1xi32> to vector<1xf32>
        %405 = arith.mulf %404, %384 : vector<1xf32>
        %406 = arith.truncf %405 : vector<1xf32> to vector<1xf16>
        vector.store %406, %108[%workgroup_id_2, %402, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %407 = amdgpu.mfma %107#14 * %107#18 + %355 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %408 = vector.extract_strided_slice %407 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %409 = vector.load %109[%188] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %410 = vector.load %110[%188] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %411 = arith.addi %408, %409 : vector<1xi32>
        %412 = arith.sitofp %411 : vector<1xi32> to vector<1xf32>
        %413 = arith.mulf %412, %410 : vector<1xf32>
        %414 = arith.truncf %413 : vector<1xf32> to vector<1xf16>
        vector.store %414, %108[%workgroup_id_2, %381, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %415 = vector.extract_strided_slice %407 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %416 = arith.addi %415, %409 : vector<1xi32>
        %417 = arith.sitofp %416 : vector<1xi32> to vector<1xf32>
        %418 = arith.mulf %417, %410 : vector<1xf32>
        %419 = arith.truncf %418 : vector<1xf32> to vector<1xf16>
        vector.store %419, %108[%workgroup_id_2, %390, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %420 = vector.extract_strided_slice %407 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %421 = arith.addi %420, %409 : vector<1xi32>
        %422 = arith.sitofp %421 : vector<1xi32> to vector<1xf32>
        %423 = arith.mulf %422, %410 : vector<1xf32>
        %424 = arith.truncf %423 : vector<1xf32> to vector<1xf16>
        vector.store %424, %108[%workgroup_id_2, %396, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %425 = vector.extract_strided_slice %407 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %426 = arith.addi %425, %409 : vector<1xi32>
        %427 = arith.sitofp %426 : vector<1xi32> to vector<1xf32>
        %428 = arith.mulf %427, %410 : vector<1xf32>
        %429 = arith.truncf %428 : vector<1xf32> to vector<1xf16>
        vector.store %429, %108[%workgroup_id_2, %402, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %430 = amdgpu.mfma %107#14 * %107#13 + %356 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %431 = vector.extract_strided_slice %430 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %432 = vector.load %109[%165] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %433 = vector.load %110[%165] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %434 = arith.addi %431, %432 : vector<1xi32>
        %435 = arith.sitofp %434 : vector<1xi32> to vector<1xf32>
        %436 = arith.mulf %435, %433 : vector<1xf32>
        %437 = arith.truncf %436 : vector<1xf32> to vector<1xf16>
        vector.store %437, %108[%workgroup_id_2, %381, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %438 = vector.extract_strided_slice %430 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %439 = arith.addi %438, %432 : vector<1xi32>
        %440 = arith.sitofp %439 : vector<1xi32> to vector<1xf32>
        %441 = arith.mulf %440, %433 : vector<1xf32>
        %442 = arith.truncf %441 : vector<1xf32> to vector<1xf16>
        vector.store %442, %108[%workgroup_id_2, %390, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %443 = vector.extract_strided_slice %430 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %444 = arith.addi %443, %432 : vector<1xi32>
        %445 = arith.sitofp %444 : vector<1xi32> to vector<1xf32>
        %446 = arith.mulf %445, %433 : vector<1xf32>
        %447 = arith.truncf %446 : vector<1xf32> to vector<1xf16>
        vector.store %447, %108[%workgroup_id_2, %396, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %448 = vector.extract_strided_slice %430 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %449 = arith.addi %448, %432 : vector<1xi32>
        %450 = arith.sitofp %449 : vector<1xi32> to vector<1xf32>
        %451 = arith.mulf %450, %433 : vector<1xf32>
        %452 = arith.truncf %451 : vector<1xf32> to vector<1xf16>
        vector.store %452, %108[%workgroup_id_2, %402, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %453 = amdgpu.mfma %238 * %107#5 + %107#33 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %454 = amdgpu.mfma %238 * %107#0 + %107#32 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %455 = amdgpu.mfma %107#4 * %107#6 + %453 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %456 = amdgpu.mfma %107#4 * %107#1 + %454 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %457 = amdgpu.mfma %107#9 * %107#7 + %455 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %458 = amdgpu.mfma %107#9 * %107#2 + %456 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %459 = amdgpu.mfma %107#14 * %107#8 + %457 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %460 = vector.extract_strided_slice %459 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %461 = vector.load %109[%142] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %462 = vector.load %110[%142] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %463 = arith.addi %460, %461 : vector<1xi32>
        %464 = arith.sitofp %463 : vector<1xi32> to vector<1xf32>
        %465 = arith.mulf %464, %462 : vector<1xf32>
        %466 = arith.truncf %465 : vector<1xf32> to vector<1xf16>
        vector.store %466, %108[%workgroup_id_2, %381, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %467 = vector.extract_strided_slice %459 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %468 = arith.addi %467, %461 : vector<1xi32>
        %469 = arith.sitofp %468 : vector<1xi32> to vector<1xf32>
        %470 = arith.mulf %469, %462 : vector<1xf32>
        %471 = arith.truncf %470 : vector<1xf32> to vector<1xf16>
        vector.store %471, %108[%workgroup_id_2, %390, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %472 = vector.extract_strided_slice %459 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %473 = arith.addi %472, %461 : vector<1xi32>
        %474 = arith.sitofp %473 : vector<1xi32> to vector<1xf32>
        %475 = arith.mulf %474, %462 : vector<1xf32>
        %476 = arith.truncf %475 : vector<1xf32> to vector<1xf16>
        vector.store %476, %108[%workgroup_id_2, %396, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %477 = vector.extract_strided_slice %459 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %478 = arith.addi %477, %461 : vector<1xi32>
        %479 = arith.sitofp %478 : vector<1xi32> to vector<1xf32>
        %480 = arith.mulf %479, %462 : vector<1xf32>
        %481 = arith.truncf %480 : vector<1xf32> to vector<1xf16>
        vector.store %481, %108[%workgroup_id_2, %402, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %482 = amdgpu.mfma %107#14 * %107#3 + %458 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %483 = vector.extract_strided_slice %482 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %484 = vector.load %109[%116] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %485 = vector.load %110[%116] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %486 = arith.addi %483, %484 : vector<1xi32>
        %487 = arith.sitofp %486 : vector<1xi32> to vector<1xf32>
        %488 = arith.mulf %487, %485 : vector<1xf32>
        %489 = arith.truncf %488 : vector<1xf32> to vector<1xf16>
        vector.store %489, %108[%workgroup_id_2, %381, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %490 = vector.extract_strided_slice %482 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %491 = arith.addi %490, %484 : vector<1xi32>
        %492 = arith.sitofp %491 : vector<1xi32> to vector<1xf32>
        %493 = arith.mulf %492, %485 : vector<1xf32>
        %494 = arith.truncf %493 : vector<1xf32> to vector<1xf16>
        vector.store %494, %108[%workgroup_id_2, %390, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %495 = vector.extract_strided_slice %482 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %496 = arith.addi %495, %484 : vector<1xi32>
        %497 = arith.sitofp %496 : vector<1xi32> to vector<1xf32>
        %498 = arith.mulf %497, %485 : vector<1xf32>
        %499 = arith.truncf %498 : vector<1xf32> to vector<1xf16>
        vector.store %499, %108[%workgroup_id_2, %396, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %500 = vector.extract_strided_slice %482 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %501 = arith.addi %500, %484 : vector<1xi32>
        %502 = arith.sitofp %501 : vector<1xi32> to vector<1xf32>
        %503 = arith.mulf %502, %485 : vector<1xf32>
        %504 = arith.truncf %503 : vector<1xf32> to vector<1xf16>
        vector.store %504, %108[%workgroup_id_2, %402, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %505 = amdgpu.mfma %242 * %107#19 + %107#31 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %506 = amdgpu.mfma %242 * %107#15 + %107#30 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %507 = amdgpu.mfma %242 * %107#10 + %107#29 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %508 = amdgpu.mfma %242 * %107#5 + %107#28 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %509 = amdgpu.mfma %241 * %107#20 + %505 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %510 = amdgpu.mfma %241 * %107#16 + %506 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %511 = amdgpu.mfma %241 * %107#11 + %507 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %512 = amdgpu.mfma %241 * %107#6 + %508 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %513 = amdgpu.mfma %240 * %107#21 + %509 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %514 = amdgpu.mfma %240 * %107#17 + %510 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %515 = amdgpu.mfma %240 * %107#12 + %511 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %516 = amdgpu.mfma %240 * %107#7 + %512 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %517 = amdgpu.mfma %239 * %107#22 + %513 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %518 = vector.extract_strided_slice %517 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %519 = vector.load %109[%211] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %520 = vector.load %110[%211] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %521 = arith.addi %518, %519 : vector<1xi32>
        %522 = arith.sitofp %521 : vector<1xi32> to vector<1xf32>
        %523 = arith.mulf %522, %520 : vector<1xf32>
        %524 = arith.truncf %523 : vector<1xf32> to vector<1xf16>
        vector.store %524, %108[%workgroup_id_2, %113, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %525 = vector.extract_strided_slice %517 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %526 = arith.addi %113, %c1 : index
        %527 = arith.addi %525, %519 : vector<1xi32>
        %528 = arith.sitofp %527 : vector<1xi32> to vector<1xf32>
        %529 = arith.mulf %528, %520 : vector<1xf32>
        %530 = arith.truncf %529 : vector<1xf32> to vector<1xf16>
        vector.store %530, %108[%workgroup_id_2, %526, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %531 = vector.extract_strided_slice %517 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %532 = arith.addi %113, %c2 : index
        %533 = arith.addi %531, %519 : vector<1xi32>
        %534 = arith.sitofp %533 : vector<1xi32> to vector<1xf32>
        %535 = arith.mulf %534, %520 : vector<1xf32>
        %536 = arith.truncf %535 : vector<1xf32> to vector<1xf16>
        vector.store %536, %108[%workgroup_id_2, %532, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %537 = vector.extract_strided_slice %517 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %538 = arith.addi %113, %c3 : index
        %539 = arith.addi %537, %519 : vector<1xi32>
        %540 = arith.sitofp %539 : vector<1xi32> to vector<1xf32>
        %541 = arith.mulf %540, %520 : vector<1xf32>
        %542 = arith.truncf %541 : vector<1xf32> to vector<1xf16>
        vector.store %542, %108[%workgroup_id_2, %538, %211] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %543 = amdgpu.mfma %239 * %107#18 + %514 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %544 = vector.extract_strided_slice %543 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %545 = vector.load %109[%188] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %546 = vector.load %110[%188] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %547 = arith.addi %544, %545 : vector<1xi32>
        %548 = arith.sitofp %547 : vector<1xi32> to vector<1xf32>
        %549 = arith.mulf %548, %546 : vector<1xf32>
        %550 = arith.truncf %549 : vector<1xf32> to vector<1xf16>
        vector.store %550, %108[%workgroup_id_2, %113, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %551 = vector.extract_strided_slice %543 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %552 = arith.addi %551, %545 : vector<1xi32>
        %553 = arith.sitofp %552 : vector<1xi32> to vector<1xf32>
        %554 = arith.mulf %553, %546 : vector<1xf32>
        %555 = arith.truncf %554 : vector<1xf32> to vector<1xf16>
        vector.store %555, %108[%workgroup_id_2, %526, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %556 = vector.extract_strided_slice %543 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %557 = arith.addi %556, %545 : vector<1xi32>
        %558 = arith.sitofp %557 : vector<1xi32> to vector<1xf32>
        %559 = arith.mulf %558, %546 : vector<1xf32>
        %560 = arith.truncf %559 : vector<1xf32> to vector<1xf16>
        vector.store %560, %108[%workgroup_id_2, %532, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %561 = vector.extract_strided_slice %543 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %562 = arith.addi %561, %545 : vector<1xi32>
        %563 = arith.sitofp %562 : vector<1xi32> to vector<1xf32>
        %564 = arith.mulf %563, %546 : vector<1xf32>
        %565 = arith.truncf %564 : vector<1xf32> to vector<1xf16>
        vector.store %565, %108[%workgroup_id_2, %538, %188] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %566 = amdgpu.mfma %239 * %107#13 + %515 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %567 = vector.extract_strided_slice %566 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %568 = vector.load %109[%165] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %569 = vector.load %110[%165] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %570 = arith.addi %567, %568 : vector<1xi32>
        %571 = arith.sitofp %570 : vector<1xi32> to vector<1xf32>
        %572 = arith.mulf %571, %569 : vector<1xf32>
        %573 = arith.truncf %572 : vector<1xf32> to vector<1xf16>
        vector.store %573, %108[%workgroup_id_2, %113, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %574 = vector.extract_strided_slice %566 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %575 = arith.addi %574, %568 : vector<1xi32>
        %576 = arith.sitofp %575 : vector<1xi32> to vector<1xf32>
        %577 = arith.mulf %576, %569 : vector<1xf32>
        %578 = arith.truncf %577 : vector<1xf32> to vector<1xf16>
        vector.store %578, %108[%workgroup_id_2, %526, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %579 = vector.extract_strided_slice %566 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %580 = arith.addi %579, %568 : vector<1xi32>
        %581 = arith.sitofp %580 : vector<1xi32> to vector<1xf32>
        %582 = arith.mulf %581, %569 : vector<1xf32>
        %583 = arith.truncf %582 : vector<1xf32> to vector<1xf16>
        vector.store %583, %108[%workgroup_id_2, %532, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %584 = vector.extract_strided_slice %566 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %585 = arith.addi %584, %568 : vector<1xi32>
        %586 = arith.sitofp %585 : vector<1xi32> to vector<1xf32>
        %587 = arith.mulf %586, %569 : vector<1xf32>
        %588 = arith.truncf %587 : vector<1xf32> to vector<1xf16>
        vector.store %588, %108[%workgroup_id_2, %538, %165] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %589 = amdgpu.mfma %239 * %107#8 + %516 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %590 = vector.extract_strided_slice %589 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %591 = vector.load %109[%142] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %592 = vector.load %110[%142] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %593 = arith.addi %590, %591 : vector<1xi32>
        %594 = arith.sitofp %593 : vector<1xi32> to vector<1xf32>
        %595 = arith.mulf %594, %592 : vector<1xf32>
        %596 = arith.truncf %595 : vector<1xf32> to vector<1xf16>
        vector.store %596, %108[%workgroup_id_2, %113, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %597 = vector.extract_strided_slice %589 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %598 = arith.addi %597, %591 : vector<1xi32>
        %599 = arith.sitofp %598 : vector<1xi32> to vector<1xf32>
        %600 = arith.mulf %599, %592 : vector<1xf32>
        %601 = arith.truncf %600 : vector<1xf32> to vector<1xf16>
        vector.store %601, %108[%workgroup_id_2, %526, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %602 = vector.extract_strided_slice %589 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %603 = arith.addi %602, %591 : vector<1xi32>
        %604 = arith.sitofp %603 : vector<1xi32> to vector<1xf32>
        %605 = arith.mulf %604, %592 : vector<1xf32>
        %606 = arith.truncf %605 : vector<1xf32> to vector<1xf16>
        vector.store %606, %108[%workgroup_id_2, %532, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %607 = vector.extract_strided_slice %589 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %608 = arith.addi %607, %591 : vector<1xi32>
        %609 = arith.sitofp %608 : vector<1xi32> to vector<1xf32>
        %610 = arith.mulf %609, %592 : vector<1xf32>
        %611 = arith.truncf %610 : vector<1xf32> to vector<1xf16>
        vector.store %611, %108[%workgroup_id_2, %538, %142] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %612 = amdgpu.mfma %242 * %107#0 + %107#27 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %613 = amdgpu.mfma %241 * %107#1 + %612 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %614 = amdgpu.mfma %240 * %107#2 + %613 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %615 = amdgpu.mfma %239 * %107#3 + %614 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %616 = vector.extract_strided_slice %615 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %617 = vector.load %109[%116] : memref<5120xi32, strided<[1], offset: ?>>, vector<1xi32>
        %618 = vector.load %110[%116] : memref<5120xf32, strided<[1], offset: ?>>, vector<1xf32>
        %619 = arith.addi %616, %617 : vector<1xi32>
        %620 = arith.sitofp %619 : vector<1xi32> to vector<1xf32>
        %621 = arith.mulf %620, %618 : vector<1xf32>
        %622 = arith.truncf %621 : vector<1xf32> to vector<1xf16>
        vector.store %622, %108[%workgroup_id_2, %113, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %623 = vector.extract_strided_slice %615 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %624 = arith.addi %623, %617 : vector<1xi32>
        %625 = arith.sitofp %624 : vector<1xi32> to vector<1xf32>
        %626 = arith.mulf %625, %618 : vector<1xf32>
        %627 = arith.truncf %626 : vector<1xf32> to vector<1xf16>
        vector.store %627, %108[%workgroup_id_2, %526, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %628 = vector.extract_strided_slice %615 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %629 = arith.addi %628, %617 : vector<1xi32>
        %630 = arith.sitofp %629 : vector<1xi32> to vector<1xf32>
        %631 = arith.mulf %630, %618 : vector<1xf32>
        %632 = arith.truncf %631 : vector<1xf32> to vector<1xf16>
        vector.store %632, %108[%workgroup_id_2, %532, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        %633 = vector.extract_strided_slice %615 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %634 = arith.addi %633, %617 : vector<1xi32>
        %635 = arith.sitofp %634 : vector<1xi32> to vector<1xf32>
        %636 = arith.mulf %635, %618 : vector<1xf32>
        %637 = arith.truncf %636 : vector<1xf32> to vector<1xf16>
        vector.store %637, %108[%workgroup_id_2, %538, %116] : memref<2x4096x5120xf16, strided<[20971520, 5120, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

