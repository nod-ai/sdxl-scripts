#translation = #iree_codegen.translation_info<None workgroup_size = [128, 4, 1] subgroup_size = 64>
module {
  flow.executable private @tk_gemm_fused_2x1024x10240x1280 {
    flow.executable.export public @tk_gemm_fused_2x1024x10240x1280 workgroups() -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      flow.return %c8, %c32, %c2 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2x1024x10240x1280(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
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
        %c3_i32 = arith.constant 3 : i32
        %c512_i32 = arith.constant 512 : i32
        %c5_i32 = arith.constant 5 : i32
        %c256_i32 = arith.constant 256 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c2_i32 = arith.constant 2 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c10 = arith.constant 10 : index
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
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c16 : index
        %2 = arith.muli %thread_id_z, %c64 : index
        %3 = arith.muli %workgroup_id_1, %c8 : index
        %4 = arith.addi %3, %workgroup_id_0 : index
        %5 = arith.divsi %4, %c32 : index
        %6 = arith.muli %5, %c128 : index
        %7 = arith.divsi %thread_id_x, %c8 : index
        %8 = arith.addi %7, %6 : index
        %9 = arith.addi %8, %2 : index
        %10 = arith.addi %9, %1 : index
        %11 = arith.remsi %thread_id_x, %c8 : index
        %12 = arith.muli %11, %c16 : index
        %13 = vector.load %0[%workgroup_id_2, %10, %12] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
        %14 = arith.addi %10, %c64 : index
        %15 = vector.load %0[%workgroup_id_2, %14, %12] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
        %16 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<10240x1280xi8, strided<[1280, 1], offset: ?>>
        %17 = arith.remsi %4, %c32 : index
        %18 = arith.muli %17, %c320 : index
        %19 = arith.addi %7, %18 : index
        %20 = arith.addi %19, %2 : index
        %21 = arith.addi %20, %1 : index
        %22 = vector.load %16[%21, %12] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
        %23 = arith.addi %21, %c64 : index
        %24 = vector.load %16[%23, %12] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
        %25 = arith.addi %21, %c128 : index
        %26 = vector.load %16[%25, %12] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
        %27 = arith.addi %21, %c192 : index
        %28 = vector.load %16[%27, %12] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
        %29 = arith.addi %21, %c256 : index
        %30 = vector.load %16[%29, %12] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
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
        %59 = amdgpu.mfma %57 * %58 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %60 = arith.addi %48, %c48 : index
        %61 = vector.load %alloc[%60, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %62 = vector.load %alloc[%60, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %63 = amdgpu.mfma %55 * %56 + %59 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %64 = vector.load %alloc[%60, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %65 = vector.load %alloc[%60, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %66 = amdgpu.mfma %52 * %53 + %63 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %67 = amdgpu.mfma %57 * %65 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %68 = arith.addi %48, %c32 : index
        %69 = vector.load %alloc[%68, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %70 = vector.load %alloc[%68, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %71 = amdgpu.mfma %46 * %50 + %66 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %72 = amdgpu.mfma %55 * %64 + %67 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %73 = vector.load %alloc[%68, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %74 = vector.load %alloc[%68, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %75 = amdgpu.mfma %52 * %62 + %72 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %76 = amdgpu.mfma %57 * %74 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %77 = arith.addi %48, %c16 : index
        %78 = vector.load %alloc[%77, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %79 = vector.load %alloc[%77, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %80 = amdgpu.mfma %46 * %61 + %75 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %81 = amdgpu.mfma %55 * %73 + %76 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %82 = vector.load %alloc[%77, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %83 = vector.load %alloc[%77, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %84 = amdgpu.mfma %52 * %70 + %81 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %85 = amdgpu.mfma %57 * %83 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %86 = vector.load %alloc[%48, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %87 = vector.load %alloc[%48, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %88 = amdgpu.mfma %46 * %69 + %84 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %89 = amdgpu.mfma %55 * %82 + %85 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %90 = vector.load %alloc[%48, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %91 = vector.load %alloc[%48, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %92 = amdgpu.mfma %52 * %79 + %89 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %93 = amdgpu.mfma %57 * %91 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %94 = arith.addi %40, %c32 : index
        %95 = vector.load %alloc_0[%94, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %96 = vector.load %alloc_0[%94, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %97 = amdgpu.mfma %46 * %78 + %92 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %98 = amdgpu.mfma %55 * %90 + %93 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %99 = vector.load %alloc_0[%94, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %100 = vector.load %alloc_0[%94, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %101 = amdgpu.mfma %52 * %87 + %98 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %102 = amdgpu.mfma %100 * %58 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %103 = amdgpu.mfma %100 * %65 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %104 = amdgpu.mfma %100 * %74 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %105 = arith.addi %40, %c16 : index
        %106 = vector.load %alloc_0[%105, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %107 = vector.load %alloc_0[%105, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %108 = amdgpu.mfma %46 * %86 + %101 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %109 = amdgpu.mfma %99 * %56 + %102 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %110 = amdgpu.mfma %99 * %64 + %103 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %111 = amdgpu.mfma %99 * %73 + %104 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %112 = vector.load %alloc_0[%105, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %113 = vector.load %alloc_0[%105, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %114:48 = scf.for %arg5 = %c1 to %c10 step %c1 iter_args(%arg6 = %111, %arg7 = %110, %arg8 = %109, %arg9 = %113, %arg10 = %91, %arg11 = %90, %arg12 = %87, %arg13 = %86, %arg14 = %112, %arg15 = %83, %arg16 = %82, %arg17 = %79, %arg18 = %78, %arg19 = %107, %arg20 = %74, %arg21 = %73, %arg22 = %70, %arg23 = %69, %arg24 = %106, %arg25 = %65, %arg26 = %64, %arg27 = %62, %arg28 = %61, %arg29 = %58, %arg30 = %56, %arg31 = %53, %arg32 = %50, %arg33 = %100, %arg34 = %99, %arg35 = %96, %arg36 = %95, %arg37 = %cst, %arg38 = %cst, %arg39 = %cst, %arg40 = %cst, %arg41 = %cst, %arg42 = %cst, %arg43 = %cst, %arg44 = %cst, %arg45 = %cst, %arg46 = %cst, %arg47 = %cst, %arg48 = %cst, %arg49 = %108, %arg50 = %97, %arg51 = %88, %arg52 = %80, %arg53 = %71) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
          %638 = arith.muli %arg5, %c128 : index
          %639 = arith.addi %638, %12 : index
          %640 = vector.load %0[%workgroup_id_2, %10, %639] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
          %641 = vector.load %0[%workgroup_id_2, %14, %639] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
          %642 = amdgpu.mfma %arg35 * %arg31 + %arg8 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %643 = amdgpu.mfma %arg35 * %arg27 + %arg7 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %644 = amdgpu.mfma %arg35 * %arg22 + %arg6 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %645 = amdgpu.mfma %arg33 * %arg15 + %arg48 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %646 = vector.load %alloc_0[%40, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %647 = vector.load %alloc_0[%40, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %648 = vector.load %alloc_0[%40, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %649 = vector.load %alloc_0[%40, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %650 = vector.load %16[%21, %639] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %651 = vector.load %16[%23, %639] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %652 = vector.load %16[%25, %639] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %653 = vector.load %16[%27, %639] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %654 = vector.load %16[%29, %639] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %655 = amdgpu.mfma %arg36 * %arg32 + %642 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %656 = amdgpu.mfma %arg36 * %arg28 + %643 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %657 = amdgpu.mfma %arg36 * %arg23 + %644 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %658 = amdgpu.mfma %arg34 * %arg16 + %645 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c5_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          vector.store %640, %alloc_0[%32, %12] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %641, %alloc_0[%33, %12] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %659 = amdgpu.mfma %arg35 * %arg17 + %658 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %660 = amdgpu.mfma %arg33 * %arg10 + %arg47 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %661 = amdgpu.mfma %arg9 * %arg29 + %arg46 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %662 = amdgpu.mfma %arg9 * %arg25 + %arg45 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %663 = amdgpu.mfma %arg36 * %arg18 + %659 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %664 = amdgpu.mfma %arg34 * %arg11 + %660 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %665 = amdgpu.mfma %arg14 * %arg30 + %661 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %666 = amdgpu.mfma %arg14 * %arg26 + %662 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %667 = amdgpu.mfma %arg35 * %arg12 + %664 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %668 = amdgpu.mfma %arg19 * %arg31 + %665 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %669 = amdgpu.mfma %arg19 * %arg27 + %666 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %670 = amdgpu.mfma %arg9 * %arg20 + %arg44 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %650, %alloc[%32, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %651, %alloc[%33, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %652, %alloc[%34, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %653, %alloc[%35, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %654, %alloc[%36, %12] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %671 = amdgpu.mfma %arg36 * %arg13 + %667 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %672 = amdgpu.mfma %arg24 * %arg32 + %668 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %673 = amdgpu.mfma %arg24 * %arg28 + %669 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %674 = amdgpu.mfma %arg14 * %arg21 + %670 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c5_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %675 = vector.load %alloc_0[%41, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %676 = vector.load %alloc[%49, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %677 = amdgpu.mfma %arg19 * %arg22 + %674 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %678 = amdgpu.mfma %arg9 * %arg15 + %arg43 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %679 = amdgpu.mfma %arg9 * %arg10 + %arg42 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %680 = amdgpu.mfma %649 * %arg29 + %arg41 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %681 = vector.load %alloc_0[%41, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %682 = vector.load %alloc[%49, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %683 = amdgpu.mfma %arg24 * %arg23 + %677 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %684 = amdgpu.mfma %arg14 * %arg16 + %678 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %685 = amdgpu.mfma %arg14 * %arg11 + %679 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %686 = amdgpu.mfma %648 * %arg30 + %680 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %687 = vector.load %alloc_0[%41, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %688 = vector.load %alloc[%49, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %689 = amdgpu.mfma %arg19 * %arg17 + %684 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %690 = amdgpu.mfma %arg19 * %arg12 + %685 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %691 = amdgpu.mfma %647 * %arg31 + %686 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %692 = amdgpu.mfma %649 * %arg25 + %arg40 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %693 = vector.load %alloc_0[%41, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %694 = vector.load %alloc[%49, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %695 = amdgpu.mfma %arg24 * %arg18 + %689 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %696 = amdgpu.mfma %arg24 * %arg13 + %690 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %697 = amdgpu.mfma %646 * %arg32 + %691 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %698 = amdgpu.mfma %648 * %arg26 + %692 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %699 = amdgpu.mfma %693 * %694 + %arg53 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %700 = vector.load %alloc[%60, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %701 = vector.load %alloc[%60, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %702 = amdgpu.mfma %647 * %arg27 + %698 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %703 = amdgpu.mfma %649 * %arg20 + %arg39 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %704 = amdgpu.mfma %649 * %arg15 + %arg38 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %705 = amdgpu.mfma %687 * %688 + %699 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %706 = vector.load %alloc[%60, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %707 = vector.load %alloc[%60, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %708 = amdgpu.mfma %646 * %arg28 + %702 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %709 = amdgpu.mfma %648 * %arg21 + %703 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %710 = amdgpu.mfma %648 * %arg16 + %704 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %711 = amdgpu.mfma %681 * %682 + %705 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %712 = amdgpu.mfma %693 * %707 + %arg52 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %713 = vector.load %alloc[%68, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %714 = vector.load %alloc[%68, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %715 = amdgpu.mfma %647 * %arg22 + %709 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %716 = amdgpu.mfma %647 * %arg17 + %710 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %717 = amdgpu.mfma %675 * %676 + %711 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %718 = amdgpu.mfma %687 * %706 + %712 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %719 = vector.load %alloc[%68, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %720 = vector.load %alloc[%68, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %721 = amdgpu.mfma %646 * %arg23 + %715 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %722 = amdgpu.mfma %646 * %arg18 + %716 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %723 = amdgpu.mfma %681 * %701 + %718 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %724 = amdgpu.mfma %693 * %720 + %arg51 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %725 = vector.load %alloc[%77, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %726 = vector.load %alloc[%77, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %727 = amdgpu.mfma %649 * %arg10 + %arg37 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %728 = amdgpu.mfma %675 * %700 + %723 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %729 = amdgpu.mfma %687 * %719 + %724 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %730 = vector.load %alloc[%77, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %731 = vector.load %alloc[%77, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %732 = amdgpu.mfma %648 * %arg11 + %727 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %733 = amdgpu.mfma %681 * %714 + %729 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %734 = amdgpu.mfma %693 * %731 + %arg50 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %735 = vector.load %alloc[%48, %45] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %736 = vector.load %alloc[%48, %51] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %737 = amdgpu.mfma %647 * %arg12 + %732 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %738 = amdgpu.mfma %675 * %713 + %733 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %739 = amdgpu.mfma %687 * %730 + %734 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %740 = vector.load %alloc[%48, %54] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %741 = vector.load %alloc[%48, %44] : memref<320x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %742 = amdgpu.mfma %646 * %arg13 + %737 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %743 = amdgpu.mfma %681 * %726 + %739 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %744 = amdgpu.mfma %693 * %741 + %arg49 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %745 = vector.load %alloc_0[%94, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %746 = vector.load %alloc_0[%94, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %747 = amdgpu.mfma %675 * %725 + %743 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %748 = amdgpu.mfma %687 * %740 + %744 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %749 = vector.load %alloc_0[%94, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %750 = vector.load %alloc_0[%94, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %751 = amdgpu.mfma %681 * %736 + %748 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %752 = amdgpu.mfma %750 * %694 + %655 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %753 = amdgpu.mfma %750 * %707 + %656 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %754 = amdgpu.mfma %750 * %720 + %657 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %755 = vector.load %alloc_0[%105, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %756 = vector.load %alloc_0[%105, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          %757 = amdgpu.mfma %675 * %735 + %751 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %758 = amdgpu.mfma %749 * %688 + %752 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %759 = amdgpu.mfma %749 * %706 + %753 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %760 = amdgpu.mfma %749 * %719 + %754 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %761 = vector.load %alloc_0[%105, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %762 = vector.load %alloc_0[%105, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          scf.yield %760, %759, %758, %762, %741, %740, %736, %735, %761, %731, %730, %726, %725, %756, %720, %719, %714, %713, %755, %707, %706, %701, %700, %694, %688, %682, %676, %750, %749, %746, %745, %742, %722, %721, %708, %697, %696, %695, %683, %673, %672, %671, %663, %757, %747, %738, %728, %717 : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
        }
        %115 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>
        %116 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<10240xi32, strided<[1], offset: ?>>
        %117 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10240xf32, strided<[1], offset: ?>>
        %118 = arith.muli %43, %c4 : index
        %119 = arith.addi %6, %38 : index
        %120 = arith.addi %119, %118 : index
        %121 = arith.addi %120, %c48 : index
        %122 = arith.addi %39, %18 : index
        %123 = arith.addi %122, %47 : index
        %124 = vector.extract_strided_slice %114#43 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %125 = vector.load %116[%123] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %126 = vector.load %117[%123] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %127 = arith.addi %124, %125 : vector<1xi32>
        %128 = arith.sitofp %127 : vector<1xi32> to vector<1xf32>
        %129 = arith.mulf %128, %126 : vector<1xf32>
        %130 = arith.truncf %129 : vector<1xf32> to vector<1xf16>
        vector.store %130, %115[%workgroup_id_2, %121, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %131 = vector.extract_strided_slice %114#43 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %132 = arith.addi %120, %c49 : index
        %133 = arith.addi %131, %125 : vector<1xi32>
        %134 = arith.sitofp %133 : vector<1xi32> to vector<1xf32>
        %135 = arith.mulf %134, %126 : vector<1xf32>
        %136 = arith.truncf %135 : vector<1xf32> to vector<1xf16>
        vector.store %136, %115[%workgroup_id_2, %132, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %137 = vector.extract_strided_slice %114#43 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %138 = arith.addi %120, %c50 : index
        %139 = arith.addi %137, %125 : vector<1xi32>
        %140 = arith.sitofp %139 : vector<1xi32> to vector<1xf32>
        %141 = arith.mulf %140, %126 : vector<1xf32>
        %142 = arith.truncf %141 : vector<1xf32> to vector<1xf16>
        vector.store %142, %115[%workgroup_id_2, %138, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %143 = vector.extract_strided_slice %114#43 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %144 = arith.addi %120, %c51 : index
        %145 = arith.addi %143, %125 : vector<1xi32>
        %146 = arith.sitofp %145 : vector<1xi32> to vector<1xf32>
        %147 = arith.mulf %146, %126 : vector<1xf32>
        %148 = arith.truncf %147 : vector<1xf32> to vector<1xf16>
        vector.store %148, %115[%workgroup_id_2, %144, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %149 = arith.addi %123, %c16 : index
        %150 = vector.extract_strided_slice %114#44 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %151 = vector.load %116[%149] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %152 = vector.load %117[%149] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %153 = arith.addi %150, %151 : vector<1xi32>
        %154 = arith.sitofp %153 : vector<1xi32> to vector<1xf32>
        %155 = arith.mulf %154, %152 : vector<1xf32>
        %156 = arith.truncf %155 : vector<1xf32> to vector<1xf16>
        vector.store %156, %115[%workgroup_id_2, %121, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %157 = vector.extract_strided_slice %114#44 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %158 = arith.addi %157, %151 : vector<1xi32>
        %159 = arith.sitofp %158 : vector<1xi32> to vector<1xf32>
        %160 = arith.mulf %159, %152 : vector<1xf32>
        %161 = arith.truncf %160 : vector<1xf32> to vector<1xf16>
        vector.store %161, %115[%workgroup_id_2, %132, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %162 = vector.extract_strided_slice %114#44 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %163 = arith.addi %162, %151 : vector<1xi32>
        %164 = arith.sitofp %163 : vector<1xi32> to vector<1xf32>
        %165 = arith.mulf %164, %152 : vector<1xf32>
        %166 = arith.truncf %165 : vector<1xf32> to vector<1xf16>
        vector.store %166, %115[%workgroup_id_2, %138, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %167 = vector.extract_strided_slice %114#44 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %168 = arith.addi %167, %151 : vector<1xi32>
        %169 = arith.sitofp %168 : vector<1xi32> to vector<1xf32>
        %170 = arith.mulf %169, %152 : vector<1xf32>
        %171 = arith.truncf %170 : vector<1xf32> to vector<1xf16>
        vector.store %171, %115[%workgroup_id_2, %144, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %172 = arith.addi %123, %c32 : index
        %173 = vector.extract_strided_slice %114#45 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %174 = vector.load %116[%172] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %175 = vector.load %117[%172] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %176 = arith.addi %173, %174 : vector<1xi32>
        %177 = arith.sitofp %176 : vector<1xi32> to vector<1xf32>
        %178 = arith.mulf %177, %175 : vector<1xf32>
        %179 = arith.truncf %178 : vector<1xf32> to vector<1xf16>
        vector.store %179, %115[%workgroup_id_2, %121, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %180 = vector.extract_strided_slice %114#45 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %181 = arith.addi %180, %174 : vector<1xi32>
        %182 = arith.sitofp %181 : vector<1xi32> to vector<1xf32>
        %183 = arith.mulf %182, %175 : vector<1xf32>
        %184 = arith.truncf %183 : vector<1xf32> to vector<1xf16>
        vector.store %184, %115[%workgroup_id_2, %132, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %185 = vector.extract_strided_slice %114#45 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %186 = arith.addi %185, %174 : vector<1xi32>
        %187 = arith.sitofp %186 : vector<1xi32> to vector<1xf32>
        %188 = arith.mulf %187, %175 : vector<1xf32>
        %189 = arith.truncf %188 : vector<1xf32> to vector<1xf16>
        vector.store %189, %115[%workgroup_id_2, %138, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %190 = vector.extract_strided_slice %114#45 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %191 = arith.addi %190, %174 : vector<1xi32>
        %192 = arith.sitofp %191 : vector<1xi32> to vector<1xf32>
        %193 = arith.mulf %192, %175 : vector<1xf32>
        %194 = arith.truncf %193 : vector<1xf32> to vector<1xf16>
        vector.store %194, %115[%workgroup_id_2, %144, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %195 = arith.addi %123, %c48 : index
        %196 = vector.extract_strided_slice %114#46 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %197 = vector.load %116[%195] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %198 = vector.load %117[%195] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %199 = arith.addi %196, %197 : vector<1xi32>
        %200 = arith.sitofp %199 : vector<1xi32> to vector<1xf32>
        %201 = arith.mulf %200, %198 : vector<1xf32>
        %202 = arith.truncf %201 : vector<1xf32> to vector<1xf16>
        vector.store %202, %115[%workgroup_id_2, %121, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %203 = vector.extract_strided_slice %114#46 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %204 = arith.addi %203, %197 : vector<1xi32>
        %205 = arith.sitofp %204 : vector<1xi32> to vector<1xf32>
        %206 = arith.mulf %205, %198 : vector<1xf32>
        %207 = arith.truncf %206 : vector<1xf32> to vector<1xf16>
        vector.store %207, %115[%workgroup_id_2, %132, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %208 = vector.extract_strided_slice %114#46 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %209 = arith.addi %208, %197 : vector<1xi32>
        %210 = arith.sitofp %209 : vector<1xi32> to vector<1xf32>
        %211 = arith.mulf %210, %198 : vector<1xf32>
        %212 = arith.truncf %211 : vector<1xf32> to vector<1xf16>
        vector.store %212, %115[%workgroup_id_2, %138, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %213 = vector.extract_strided_slice %114#46 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %214 = arith.addi %213, %197 : vector<1xi32>
        %215 = arith.sitofp %214 : vector<1xi32> to vector<1xf32>
        %216 = arith.mulf %215, %198 : vector<1xf32>
        %217 = arith.truncf %216 : vector<1xf32> to vector<1xf16>
        vector.store %217, %115[%workgroup_id_2, %144, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %218 = arith.addi %123, %c64 : index
        %219 = vector.extract_strided_slice %114#47 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %220 = vector.load %116[%218] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %221 = vector.load %117[%218] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %222 = arith.addi %219, %220 : vector<1xi32>
        %223 = arith.sitofp %222 : vector<1xi32> to vector<1xf32>
        %224 = arith.mulf %223, %221 : vector<1xf32>
        %225 = arith.truncf %224 : vector<1xf32> to vector<1xf16>
        vector.store %225, %115[%workgroup_id_2, %121, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %226 = vector.extract_strided_slice %114#47 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %227 = arith.addi %226, %220 : vector<1xi32>
        %228 = arith.sitofp %227 : vector<1xi32> to vector<1xf32>
        %229 = arith.mulf %228, %221 : vector<1xf32>
        %230 = arith.truncf %229 : vector<1xf32> to vector<1xf16>
        vector.store %230, %115[%workgroup_id_2, %132, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %231 = vector.extract_strided_slice %114#47 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %232 = arith.addi %231, %220 : vector<1xi32>
        %233 = arith.sitofp %232 : vector<1xi32> to vector<1xf32>
        %234 = arith.mulf %233, %221 : vector<1xf32>
        %235 = arith.truncf %234 : vector<1xf32> to vector<1xf16>
        vector.store %235, %115[%workgroup_id_2, %138, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %236 = vector.extract_strided_slice %114#47 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %237 = arith.addi %236, %220 : vector<1xi32>
        %238 = arith.sitofp %237 : vector<1xi32> to vector<1xf32>
        %239 = arith.mulf %238, %221 : vector<1xf32>
        %240 = arith.truncf %239 : vector<1xf32> to vector<1xf16>
        vector.store %240, %115[%workgroup_id_2, %144, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %241 = amdgpu.mfma %114#29 * %114#25 + %114#2 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %242 = amdgpu.mfma %114#29 * %114#21 + %114#1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %243 = amdgpu.mfma %114#29 * %114#16 + %114#0 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %244 = amdgpu.mfma %114#27 * %114#9 + %114#42 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %245 = vector.load %alloc_0[%40, %45] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %246 = vector.load %alloc_0[%40, %51] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %247 = vector.load %alloc_0[%40, %54] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %248 = vector.load %alloc_0[%40, %44] : memref<128x136xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %249 = amdgpu.mfma %114#30 * %114#26 + %241 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %250 = arith.addi %120, %c32 : index
        %251 = vector.extract_strided_slice %249 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %252 = arith.addi %251, %220 : vector<1xi32>
        %253 = arith.sitofp %252 : vector<1xi32> to vector<1xf32>
        %254 = arith.mulf %253, %221 : vector<1xf32>
        %255 = arith.truncf %254 : vector<1xf32> to vector<1xf16>
        vector.store %255, %115[%workgroup_id_2, %250, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %256 = vector.extract_strided_slice %249 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %257 = arith.addi %120, %c33 : index
        %258 = arith.addi %256, %220 : vector<1xi32>
        %259 = arith.sitofp %258 : vector<1xi32> to vector<1xf32>
        %260 = arith.mulf %259, %221 : vector<1xf32>
        %261 = arith.truncf %260 : vector<1xf32> to vector<1xf16>
        vector.store %261, %115[%workgroup_id_2, %257, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %262 = vector.extract_strided_slice %249 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %263 = arith.addi %120, %c34 : index
        %264 = arith.addi %262, %220 : vector<1xi32>
        %265 = arith.sitofp %264 : vector<1xi32> to vector<1xf32>
        %266 = arith.mulf %265, %221 : vector<1xf32>
        %267 = arith.truncf %266 : vector<1xf32> to vector<1xf16>
        vector.store %267, %115[%workgroup_id_2, %263, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %268 = vector.extract_strided_slice %249 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %269 = arith.addi %120, %c35 : index
        %270 = arith.addi %268, %220 : vector<1xi32>
        %271 = arith.sitofp %270 : vector<1xi32> to vector<1xf32>
        %272 = arith.mulf %271, %221 : vector<1xf32>
        %273 = arith.truncf %272 : vector<1xf32> to vector<1xf16>
        vector.store %273, %115[%workgroup_id_2, %269, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %274 = amdgpu.mfma %114#30 * %114#22 + %242 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %275 = vector.extract_strided_slice %274 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %276 = vector.load %116[%195] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %277 = vector.load %117[%195] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %278 = arith.addi %275, %276 : vector<1xi32>
        %279 = arith.sitofp %278 : vector<1xi32> to vector<1xf32>
        %280 = arith.mulf %279, %277 : vector<1xf32>
        %281 = arith.truncf %280 : vector<1xf32> to vector<1xf16>
        vector.store %281, %115[%workgroup_id_2, %250, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %282 = vector.extract_strided_slice %274 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %283 = arith.addi %282, %276 : vector<1xi32>
        %284 = arith.sitofp %283 : vector<1xi32> to vector<1xf32>
        %285 = arith.mulf %284, %277 : vector<1xf32>
        %286 = arith.truncf %285 : vector<1xf32> to vector<1xf16>
        vector.store %286, %115[%workgroup_id_2, %257, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %287 = vector.extract_strided_slice %274 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %288 = arith.addi %287, %276 : vector<1xi32>
        %289 = arith.sitofp %288 : vector<1xi32> to vector<1xf32>
        %290 = arith.mulf %289, %277 : vector<1xf32>
        %291 = arith.truncf %290 : vector<1xf32> to vector<1xf16>
        vector.store %291, %115[%workgroup_id_2, %263, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %292 = vector.extract_strided_slice %274 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %293 = arith.addi %292, %276 : vector<1xi32>
        %294 = arith.sitofp %293 : vector<1xi32> to vector<1xf32>
        %295 = arith.mulf %294, %277 : vector<1xf32>
        %296 = arith.truncf %295 : vector<1xf32> to vector<1xf16>
        vector.store %296, %115[%workgroup_id_2, %269, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %297 = amdgpu.mfma %114#30 * %114#17 + %243 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %298 = vector.extract_strided_slice %297 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %299 = vector.load %116[%172] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %300 = vector.load %117[%172] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %301 = arith.addi %298, %299 : vector<1xi32>
        %302 = arith.sitofp %301 : vector<1xi32> to vector<1xf32>
        %303 = arith.mulf %302, %300 : vector<1xf32>
        %304 = arith.truncf %303 : vector<1xf32> to vector<1xf16>
        vector.store %304, %115[%workgroup_id_2, %250, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %305 = vector.extract_strided_slice %297 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %306 = arith.addi %305, %299 : vector<1xi32>
        %307 = arith.sitofp %306 : vector<1xi32> to vector<1xf32>
        %308 = arith.mulf %307, %300 : vector<1xf32>
        %309 = arith.truncf %308 : vector<1xf32> to vector<1xf16>
        vector.store %309, %115[%workgroup_id_2, %257, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %310 = vector.extract_strided_slice %297 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %311 = arith.addi %310, %299 : vector<1xi32>
        %312 = arith.sitofp %311 : vector<1xi32> to vector<1xf32>
        %313 = arith.mulf %312, %300 : vector<1xf32>
        %314 = arith.truncf %313 : vector<1xf32> to vector<1xf16>
        vector.store %314, %115[%workgroup_id_2, %263, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %315 = vector.extract_strided_slice %297 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %316 = arith.addi %315, %299 : vector<1xi32>
        %317 = arith.sitofp %316 : vector<1xi32> to vector<1xf32>
        %318 = arith.mulf %317, %300 : vector<1xf32>
        %319 = arith.truncf %318 : vector<1xf32> to vector<1xf16>
        vector.store %319, %115[%workgroup_id_2, %269, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %320 = amdgpu.mfma %114#28 * %114#10 + %244 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %321 = amdgpu.mfma %114#29 * %114#11 + %320 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %322 = amdgpu.mfma %114#27 * %114#4 + %114#41 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %323 = amdgpu.mfma %114#3 * %114#23 + %114#40 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %324 = amdgpu.mfma %114#3 * %114#19 + %114#39 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %325 = amdgpu.mfma %114#30 * %114#12 + %321 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %326 = vector.extract_strided_slice %325 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %327 = vector.load %116[%149] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %328 = vector.load %117[%149] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %329 = arith.addi %326, %327 : vector<1xi32>
        %330 = arith.sitofp %329 : vector<1xi32> to vector<1xf32>
        %331 = arith.mulf %330, %328 : vector<1xf32>
        %332 = arith.truncf %331 : vector<1xf32> to vector<1xf16>
        vector.store %332, %115[%workgroup_id_2, %250, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %333 = vector.extract_strided_slice %325 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %334 = arith.addi %333, %327 : vector<1xi32>
        %335 = arith.sitofp %334 : vector<1xi32> to vector<1xf32>
        %336 = arith.mulf %335, %328 : vector<1xf32>
        %337 = arith.truncf %336 : vector<1xf32> to vector<1xf16>
        vector.store %337, %115[%workgroup_id_2, %257, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %338 = vector.extract_strided_slice %325 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %339 = arith.addi %338, %327 : vector<1xi32>
        %340 = arith.sitofp %339 : vector<1xi32> to vector<1xf32>
        %341 = arith.mulf %340, %328 : vector<1xf32>
        %342 = arith.truncf %341 : vector<1xf32> to vector<1xf16>
        vector.store %342, %115[%workgroup_id_2, %263, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %343 = vector.extract_strided_slice %325 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %344 = arith.addi %343, %327 : vector<1xi32>
        %345 = arith.sitofp %344 : vector<1xi32> to vector<1xf32>
        %346 = arith.mulf %345, %328 : vector<1xf32>
        %347 = arith.truncf %346 : vector<1xf32> to vector<1xf16>
        vector.store %347, %115[%workgroup_id_2, %269, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %348 = amdgpu.mfma %114#28 * %114#5 + %322 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %349 = amdgpu.mfma %114#8 * %114#24 + %323 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %350 = amdgpu.mfma %114#8 * %114#20 + %324 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %351 = amdgpu.mfma %114#29 * %114#6 + %348 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %352 = amdgpu.mfma %114#13 * %114#25 + %349 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %353 = amdgpu.mfma %114#13 * %114#21 + %350 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %354 = amdgpu.mfma %114#3 * %114#14 + %114#38 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %355 = amdgpu.mfma %114#30 * %114#7 + %351 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %356 = vector.extract_strided_slice %355 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %357 = vector.load %116[%123] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %358 = vector.load %117[%123] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %359 = arith.addi %356, %357 : vector<1xi32>
        %360 = arith.sitofp %359 : vector<1xi32> to vector<1xf32>
        %361 = arith.mulf %360, %358 : vector<1xf32>
        %362 = arith.truncf %361 : vector<1xf32> to vector<1xf16>
        vector.store %362, %115[%workgroup_id_2, %250, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %363 = vector.extract_strided_slice %355 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %364 = arith.addi %363, %357 : vector<1xi32>
        %365 = arith.sitofp %364 : vector<1xi32> to vector<1xf32>
        %366 = arith.mulf %365, %358 : vector<1xf32>
        %367 = arith.truncf %366 : vector<1xf32> to vector<1xf16>
        vector.store %367, %115[%workgroup_id_2, %257, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %368 = vector.extract_strided_slice %355 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %369 = arith.addi %368, %357 : vector<1xi32>
        %370 = arith.sitofp %369 : vector<1xi32> to vector<1xf32>
        %371 = arith.mulf %370, %358 : vector<1xf32>
        %372 = arith.truncf %371 : vector<1xf32> to vector<1xf16>
        vector.store %372, %115[%workgroup_id_2, %263, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %373 = vector.extract_strided_slice %355 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %374 = arith.addi %373, %357 : vector<1xi32>
        %375 = arith.sitofp %374 : vector<1xi32> to vector<1xf32>
        %376 = arith.mulf %375, %358 : vector<1xf32>
        %377 = arith.truncf %376 : vector<1xf32> to vector<1xf16>
        vector.store %377, %115[%workgroup_id_2, %269, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %378 = amdgpu.mfma %114#18 * %114#26 + %352 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %379 = arith.addi %120, %c16 : index
        %380 = vector.extract_strided_slice %378 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %381 = vector.load %116[%218] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %382 = vector.load %117[%218] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %383 = arith.addi %380, %381 : vector<1xi32>
        %384 = arith.sitofp %383 : vector<1xi32> to vector<1xf32>
        %385 = arith.mulf %384, %382 : vector<1xf32>
        %386 = arith.truncf %385 : vector<1xf32> to vector<1xf16>
        vector.store %386, %115[%workgroup_id_2, %379, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %387 = vector.extract_strided_slice %378 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %388 = arith.addi %120, %c17 : index
        %389 = arith.addi %387, %381 : vector<1xi32>
        %390 = arith.sitofp %389 : vector<1xi32> to vector<1xf32>
        %391 = arith.mulf %390, %382 : vector<1xf32>
        %392 = arith.truncf %391 : vector<1xf32> to vector<1xf16>
        vector.store %392, %115[%workgroup_id_2, %388, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %393 = vector.extract_strided_slice %378 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %394 = arith.addi %120, %c18 : index
        %395 = arith.addi %393, %381 : vector<1xi32>
        %396 = arith.sitofp %395 : vector<1xi32> to vector<1xf32>
        %397 = arith.mulf %396, %382 : vector<1xf32>
        %398 = arith.truncf %397 : vector<1xf32> to vector<1xf16>
        vector.store %398, %115[%workgroup_id_2, %394, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %399 = vector.extract_strided_slice %378 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %400 = arith.addi %120, %c19 : index
        %401 = arith.addi %399, %381 : vector<1xi32>
        %402 = arith.sitofp %401 : vector<1xi32> to vector<1xf32>
        %403 = arith.mulf %402, %382 : vector<1xf32>
        %404 = arith.truncf %403 : vector<1xf32> to vector<1xf16>
        vector.store %404, %115[%workgroup_id_2, %400, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %405 = amdgpu.mfma %114#18 * %114#22 + %353 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %406 = vector.extract_strided_slice %405 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %407 = vector.load %116[%195] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %408 = vector.load %117[%195] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %409 = arith.addi %406, %407 : vector<1xi32>
        %410 = arith.sitofp %409 : vector<1xi32> to vector<1xf32>
        %411 = arith.mulf %410, %408 : vector<1xf32>
        %412 = arith.truncf %411 : vector<1xf32> to vector<1xf16>
        vector.store %412, %115[%workgroup_id_2, %379, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %413 = vector.extract_strided_slice %405 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %414 = arith.addi %413, %407 : vector<1xi32>
        %415 = arith.sitofp %414 : vector<1xi32> to vector<1xf32>
        %416 = arith.mulf %415, %408 : vector<1xf32>
        %417 = arith.truncf %416 : vector<1xf32> to vector<1xf16>
        vector.store %417, %115[%workgroup_id_2, %388, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %418 = vector.extract_strided_slice %405 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %419 = arith.addi %418, %407 : vector<1xi32>
        %420 = arith.sitofp %419 : vector<1xi32> to vector<1xf32>
        %421 = arith.mulf %420, %408 : vector<1xf32>
        %422 = arith.truncf %421 : vector<1xf32> to vector<1xf16>
        vector.store %422, %115[%workgroup_id_2, %394, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %423 = vector.extract_strided_slice %405 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %424 = arith.addi %423, %407 : vector<1xi32>
        %425 = arith.sitofp %424 : vector<1xi32> to vector<1xf32>
        %426 = arith.mulf %425, %408 : vector<1xf32>
        %427 = arith.truncf %426 : vector<1xf32> to vector<1xf16>
        vector.store %427, %115[%workgroup_id_2, %400, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %428 = amdgpu.mfma %114#8 * %114#15 + %354 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %429 = amdgpu.mfma %114#13 * %114#16 + %428 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %430 = amdgpu.mfma %114#3 * %114#9 + %114#37 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %431 = amdgpu.mfma %114#3 * %114#4 + %114#36 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %432 = amdgpu.mfma %248 * %114#23 + %114#35 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %433 = amdgpu.mfma %114#18 * %114#17 + %429 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %434 = vector.extract_strided_slice %433 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %435 = vector.load %116[%172] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %436 = vector.load %117[%172] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %437 = arith.addi %434, %435 : vector<1xi32>
        %438 = arith.sitofp %437 : vector<1xi32> to vector<1xf32>
        %439 = arith.mulf %438, %436 : vector<1xf32>
        %440 = arith.truncf %439 : vector<1xf32> to vector<1xf16>
        vector.store %440, %115[%workgroup_id_2, %379, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %441 = vector.extract_strided_slice %433 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %442 = arith.addi %441, %435 : vector<1xi32>
        %443 = arith.sitofp %442 : vector<1xi32> to vector<1xf32>
        %444 = arith.mulf %443, %436 : vector<1xf32>
        %445 = arith.truncf %444 : vector<1xf32> to vector<1xf16>
        vector.store %445, %115[%workgroup_id_2, %388, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %446 = vector.extract_strided_slice %433 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %447 = arith.addi %446, %435 : vector<1xi32>
        %448 = arith.sitofp %447 : vector<1xi32> to vector<1xf32>
        %449 = arith.mulf %448, %436 : vector<1xf32>
        %450 = arith.truncf %449 : vector<1xf32> to vector<1xf16>
        vector.store %450, %115[%workgroup_id_2, %394, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %451 = vector.extract_strided_slice %433 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %452 = arith.addi %451, %435 : vector<1xi32>
        %453 = arith.sitofp %452 : vector<1xi32> to vector<1xf32>
        %454 = arith.mulf %453, %436 : vector<1xf32>
        %455 = arith.truncf %454 : vector<1xf32> to vector<1xf16>
        vector.store %455, %115[%workgroup_id_2, %400, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %456 = amdgpu.mfma %114#8 * %114#10 + %430 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %457 = amdgpu.mfma %114#8 * %114#5 + %431 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %458 = amdgpu.mfma %247 * %114#24 + %432 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %459 = amdgpu.mfma %114#13 * %114#11 + %456 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %460 = amdgpu.mfma %114#13 * %114#6 + %457 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %461 = amdgpu.mfma %246 * %114#25 + %458 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %462 = amdgpu.mfma %248 * %114#19 + %114#34 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %463 = amdgpu.mfma %114#18 * %114#12 + %459 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %464 = vector.extract_strided_slice %463 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %465 = vector.load %116[%149] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %466 = vector.load %117[%149] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %467 = arith.addi %464, %465 : vector<1xi32>
        %468 = arith.sitofp %467 : vector<1xi32> to vector<1xf32>
        %469 = arith.mulf %468, %466 : vector<1xf32>
        %470 = arith.truncf %469 : vector<1xf32> to vector<1xf16>
        vector.store %470, %115[%workgroup_id_2, %379, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %471 = vector.extract_strided_slice %463 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %472 = arith.addi %471, %465 : vector<1xi32>
        %473 = arith.sitofp %472 : vector<1xi32> to vector<1xf32>
        %474 = arith.mulf %473, %466 : vector<1xf32>
        %475 = arith.truncf %474 : vector<1xf32> to vector<1xf16>
        vector.store %475, %115[%workgroup_id_2, %388, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %476 = vector.extract_strided_slice %463 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %477 = arith.addi %476, %465 : vector<1xi32>
        %478 = arith.sitofp %477 : vector<1xi32> to vector<1xf32>
        %479 = arith.mulf %478, %466 : vector<1xf32>
        %480 = arith.truncf %479 : vector<1xf32> to vector<1xf16>
        vector.store %480, %115[%workgroup_id_2, %394, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %481 = vector.extract_strided_slice %463 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %482 = arith.addi %481, %465 : vector<1xi32>
        %483 = arith.sitofp %482 : vector<1xi32> to vector<1xf32>
        %484 = arith.mulf %483, %466 : vector<1xf32>
        %485 = arith.truncf %484 : vector<1xf32> to vector<1xf16>
        vector.store %485, %115[%workgroup_id_2, %400, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %486 = amdgpu.mfma %114#18 * %114#7 + %460 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %487 = vector.extract_strided_slice %486 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %488 = vector.load %116[%123] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %489 = vector.load %117[%123] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %490 = arith.addi %487, %488 : vector<1xi32>
        %491 = arith.sitofp %490 : vector<1xi32> to vector<1xf32>
        %492 = arith.mulf %491, %489 : vector<1xf32>
        %493 = arith.truncf %492 : vector<1xf32> to vector<1xf16>
        vector.store %493, %115[%workgroup_id_2, %379, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %494 = vector.extract_strided_slice %486 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %495 = arith.addi %494, %488 : vector<1xi32>
        %496 = arith.sitofp %495 : vector<1xi32> to vector<1xf32>
        %497 = arith.mulf %496, %489 : vector<1xf32>
        %498 = arith.truncf %497 : vector<1xf32> to vector<1xf16>
        vector.store %498, %115[%workgroup_id_2, %388, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %499 = vector.extract_strided_slice %486 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %500 = arith.addi %499, %488 : vector<1xi32>
        %501 = arith.sitofp %500 : vector<1xi32> to vector<1xf32>
        %502 = arith.mulf %501, %489 : vector<1xf32>
        %503 = arith.truncf %502 : vector<1xf32> to vector<1xf16>
        vector.store %503, %115[%workgroup_id_2, %394, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %504 = vector.extract_strided_slice %486 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %505 = arith.addi %504, %488 : vector<1xi32>
        %506 = arith.sitofp %505 : vector<1xi32> to vector<1xf32>
        %507 = arith.mulf %506, %489 : vector<1xf32>
        %508 = arith.truncf %507 : vector<1xf32> to vector<1xf16>
        vector.store %508, %115[%workgroup_id_2, %400, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %509 = amdgpu.mfma %245 * %114#26 + %461 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %510 = vector.extract_strided_slice %509 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %511 = vector.load %116[%218] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %512 = vector.load %117[%218] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %513 = arith.addi %510, %511 : vector<1xi32>
        %514 = arith.sitofp %513 : vector<1xi32> to vector<1xf32>
        %515 = arith.mulf %514, %512 : vector<1xf32>
        %516 = arith.truncf %515 : vector<1xf32> to vector<1xf16>
        vector.store %516, %115[%workgroup_id_2, %120, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %517 = vector.extract_strided_slice %509 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %518 = arith.addi %120, %c1 : index
        %519 = arith.addi %517, %511 : vector<1xi32>
        %520 = arith.sitofp %519 : vector<1xi32> to vector<1xf32>
        %521 = arith.mulf %520, %512 : vector<1xf32>
        %522 = arith.truncf %521 : vector<1xf32> to vector<1xf16>
        vector.store %522, %115[%workgroup_id_2, %518, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %523 = vector.extract_strided_slice %509 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %524 = arith.addi %120, %c2 : index
        %525 = arith.addi %523, %511 : vector<1xi32>
        %526 = arith.sitofp %525 : vector<1xi32> to vector<1xf32>
        %527 = arith.mulf %526, %512 : vector<1xf32>
        %528 = arith.truncf %527 : vector<1xf32> to vector<1xf16>
        vector.store %528, %115[%workgroup_id_2, %524, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %529 = vector.extract_strided_slice %509 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %530 = arith.addi %120, %c3 : index
        %531 = arith.addi %529, %511 : vector<1xi32>
        %532 = arith.sitofp %531 : vector<1xi32> to vector<1xf32>
        %533 = arith.mulf %532, %512 : vector<1xf32>
        %534 = arith.truncf %533 : vector<1xf32> to vector<1xf16>
        vector.store %534, %115[%workgroup_id_2, %530, %218] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %535 = amdgpu.mfma %247 * %114#20 + %462 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %536 = amdgpu.mfma %246 * %114#21 + %535 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %537 = amdgpu.mfma %248 * %114#14 + %114#33 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %538 = amdgpu.mfma %248 * %114#9 + %114#32 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %539 = amdgpu.mfma %245 * %114#22 + %536 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %540 = vector.extract_strided_slice %539 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %541 = vector.load %116[%195] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %542 = vector.load %117[%195] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %543 = arith.addi %540, %541 : vector<1xi32>
        %544 = arith.sitofp %543 : vector<1xi32> to vector<1xf32>
        %545 = arith.mulf %544, %542 : vector<1xf32>
        %546 = arith.truncf %545 : vector<1xf32> to vector<1xf16>
        vector.store %546, %115[%workgroup_id_2, %120, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %547 = vector.extract_strided_slice %539 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %548 = arith.addi %547, %541 : vector<1xi32>
        %549 = arith.sitofp %548 : vector<1xi32> to vector<1xf32>
        %550 = arith.mulf %549, %542 : vector<1xf32>
        %551 = arith.truncf %550 : vector<1xf32> to vector<1xf16>
        vector.store %551, %115[%workgroup_id_2, %518, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %552 = vector.extract_strided_slice %539 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %553 = arith.addi %552, %541 : vector<1xi32>
        %554 = arith.sitofp %553 : vector<1xi32> to vector<1xf32>
        %555 = arith.mulf %554, %542 : vector<1xf32>
        %556 = arith.truncf %555 : vector<1xf32> to vector<1xf16>
        vector.store %556, %115[%workgroup_id_2, %524, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %557 = vector.extract_strided_slice %539 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %558 = arith.addi %557, %541 : vector<1xi32>
        %559 = arith.sitofp %558 : vector<1xi32> to vector<1xf32>
        %560 = arith.mulf %559, %542 : vector<1xf32>
        %561 = arith.truncf %560 : vector<1xf32> to vector<1xf16>
        vector.store %561, %115[%workgroup_id_2, %530, %195] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %562 = amdgpu.mfma %247 * %114#15 + %537 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %563 = amdgpu.mfma %247 * %114#10 + %538 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %564 = amdgpu.mfma %246 * %114#16 + %562 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %565 = amdgpu.mfma %246 * %114#11 + %563 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %566 = amdgpu.mfma %245 * %114#17 + %564 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %567 = vector.extract_strided_slice %566 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %568 = vector.load %116[%172] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %569 = vector.load %117[%172] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %570 = arith.addi %567, %568 : vector<1xi32>
        %571 = arith.sitofp %570 : vector<1xi32> to vector<1xf32>
        %572 = arith.mulf %571, %569 : vector<1xf32>
        %573 = arith.truncf %572 : vector<1xf32> to vector<1xf16>
        vector.store %573, %115[%workgroup_id_2, %120, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %574 = vector.extract_strided_slice %566 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %575 = arith.addi %574, %568 : vector<1xi32>
        %576 = arith.sitofp %575 : vector<1xi32> to vector<1xf32>
        %577 = arith.mulf %576, %569 : vector<1xf32>
        %578 = arith.truncf %577 : vector<1xf32> to vector<1xf16>
        vector.store %578, %115[%workgroup_id_2, %518, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %579 = vector.extract_strided_slice %566 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %580 = arith.addi %579, %568 : vector<1xi32>
        %581 = arith.sitofp %580 : vector<1xi32> to vector<1xf32>
        %582 = arith.mulf %581, %569 : vector<1xf32>
        %583 = arith.truncf %582 : vector<1xf32> to vector<1xf16>
        vector.store %583, %115[%workgroup_id_2, %524, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %584 = vector.extract_strided_slice %566 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %585 = arith.addi %584, %568 : vector<1xi32>
        %586 = arith.sitofp %585 : vector<1xi32> to vector<1xf32>
        %587 = arith.mulf %586, %569 : vector<1xf32>
        %588 = arith.truncf %587 : vector<1xf32> to vector<1xf16>
        vector.store %588, %115[%workgroup_id_2, %530, %172] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %589 = amdgpu.mfma %245 * %114#12 + %565 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %590 = vector.extract_strided_slice %589 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %591 = vector.load %116[%149] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %592 = vector.load %117[%149] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %593 = arith.addi %590, %591 : vector<1xi32>
        %594 = arith.sitofp %593 : vector<1xi32> to vector<1xf32>
        %595 = arith.mulf %594, %592 : vector<1xf32>
        %596 = arith.truncf %595 : vector<1xf32> to vector<1xf16>
        vector.store %596, %115[%workgroup_id_2, %120, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %597 = vector.extract_strided_slice %589 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %598 = arith.addi %597, %591 : vector<1xi32>
        %599 = arith.sitofp %598 : vector<1xi32> to vector<1xf32>
        %600 = arith.mulf %599, %592 : vector<1xf32>
        %601 = arith.truncf %600 : vector<1xf32> to vector<1xf16>
        vector.store %601, %115[%workgroup_id_2, %518, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %602 = vector.extract_strided_slice %589 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %603 = arith.addi %602, %591 : vector<1xi32>
        %604 = arith.sitofp %603 : vector<1xi32> to vector<1xf32>
        %605 = arith.mulf %604, %592 : vector<1xf32>
        %606 = arith.truncf %605 : vector<1xf32> to vector<1xf16>
        vector.store %606, %115[%workgroup_id_2, %524, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %607 = vector.extract_strided_slice %589 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %608 = arith.addi %607, %591 : vector<1xi32>
        %609 = arith.sitofp %608 : vector<1xi32> to vector<1xf32>
        %610 = arith.mulf %609, %592 : vector<1xf32>
        %611 = arith.truncf %610 : vector<1xf32> to vector<1xf16>
        vector.store %611, %115[%workgroup_id_2, %530, %149] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %612 = amdgpu.mfma %248 * %114#4 + %114#31 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %613 = amdgpu.mfma %247 * %114#5 + %612 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %614 = amdgpu.mfma %246 * %114#6 + %613 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %615 = amdgpu.mfma %245 * %114#7 + %614 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %616 = vector.extract_strided_slice %615 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %617 = vector.load %116[%123] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %618 = vector.load %117[%123] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %619 = arith.addi %616, %617 : vector<1xi32>
        %620 = arith.sitofp %619 : vector<1xi32> to vector<1xf32>
        %621 = arith.mulf %620, %618 : vector<1xf32>
        %622 = arith.truncf %621 : vector<1xf32> to vector<1xf16>
        vector.store %622, %115[%workgroup_id_2, %120, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %623 = vector.extract_strided_slice %615 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %624 = arith.addi %623, %617 : vector<1xi32>
        %625 = arith.sitofp %624 : vector<1xi32> to vector<1xf32>
        %626 = arith.mulf %625, %618 : vector<1xf32>
        %627 = arith.truncf %626 : vector<1xf32> to vector<1xf16>
        vector.store %627, %115[%workgroup_id_2, %518, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %628 = vector.extract_strided_slice %615 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %629 = arith.addi %628, %617 : vector<1xi32>
        %630 = arith.sitofp %629 : vector<1xi32> to vector<1xf32>
        %631 = arith.mulf %630, %618 : vector<1xf32>
        %632 = arith.truncf %631 : vector<1xf32> to vector<1xf16>
        vector.store %632, %115[%workgroup_id_2, %524, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %633 = vector.extract_strided_slice %615 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %634 = arith.addi %633, %617 : vector<1xi32>
        %635 = arith.sitofp %634 : vector<1xi32> to vector<1xf32>
        %636 = arith.mulf %635, %618 : vector<1xf32>
        %637 = arith.truncf %636 : vector<1xf32> to vector<1xf16>
        vector.store %637, %115[%workgroup_id_2, %530, %123] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

