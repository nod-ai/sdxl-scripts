#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64>
module {
  flow.executable private @tk_gemm_fused_2x1024x10240x1280 {
    flow.executable.export public @tk_gemm_fused_2x1024x10240x1280 workgroups() -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c80 = arith.constant 80 : index
      %c2 = arith.constant 2 : index
      flow.return %c8, %c80, %c2 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2x1024x10240x1280(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: !stream.binding, %arg6: !stream.binding) attributes {translation_info = #translation} {
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
        %c4 = arith.constant 4 : index
        %c40 = arith.constant 40 : index
        %c1 = arith.constant 1 : index
        %c48 = arith.constant 48 : index
        %c32 = arith.constant 32 : index
        %c16 = arith.constant 16 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c80 = arith.constant 80 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0> : vector<4xi32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<128x36xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<128x36xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c64 : index
        %2 = arith.muli %thread_id_z, %c128 : index
        %3 = arith.muli %workgroup_id_1, %c8 : index
        %4 = arith.addi %3, %workgroup_id_0 : index
        %5 = arith.divsi %4, %c80 : index
        %6 = arith.muli %5, %c128 : index
        %7 = arith.divsi %thread_id_x, %c2 : index
        %8 = arith.addi %7, %6 : index
        %9 = arith.addi %8, %2 : index
        %10 = arith.addi %9, %1 : index
        %11 = arith.remsi %thread_id_x, %c2 : index
        %12 = arith.muli %11, %c16 : index
        %13 = vector.load %0[%workgroup_id_2, %10, %12] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
        %14 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<10240x1280xi8, strided<[1280, 1], offset: ?>>
        %15 = arith.remsi %4, %c80 : index
        %16 = arith.muli %15, %c128 : index
        %17 = arith.addi %7, %16 : index
        %18 = arith.addi %17, %2 : index
        %19 = arith.addi %18, %1 : index
        %20 = vector.load %14[%19, %12] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
        %21 = arith.addi %7, %2 : index
        %22 = arith.addi %21, %1 : index
        vector.store %13, %alloc_0[%22, %12] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %20, %alloc[%22, %12] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        amdgpu.lds_barrier
        amdgpu.lds_barrier
        %23 = arith.divsi %thread_id_x, %c64 : index
        %24 = arith.muli %23, %c64 : index
        %25 = arith.remsi %thread_id_x, %c16 : index
        %26 = arith.addi %25, %24 : index
        %27 = arith.addi %26, %c48 : index
        %28 = arith.remsi %thread_id_x, %c64 : index
        %29 = arith.divsi %28, %c16 : index
        %30 = arith.muli %29, %c8 : index
        %31 = vector.load %alloc_0[%27, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %32 = arith.addi %25, %1 : index
        %33 = arith.addi %32, %c48 : index
        %34 = vector.load %alloc[%33, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %35 = amdgpu.mfma %31 * %34 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %36 = arith.addi %32, %c32 : index
        %37 = vector.load %alloc[%36, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %38 = arith.addi %32, %c16 : index
        %39 = vector.load %alloc[%38, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %40:20 = scf.for %arg7 = %c1 to %c40 step %c1 iter_args(%arg8 = %39, %arg9 = %37, %arg10 = %34, %arg11 = %31, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %cst, %arg26 = %cst, %arg27 = %35) -> (vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
          %666 = arith.muli %arg7, %c32 : index
          %667 = arith.addi %666, %12 : index
          %668 = vector.load %0[%workgroup_id_2, %10, %667] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
          %669 = vector.load %14[%19, %667] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %670 = amdgpu.mfma %arg11 * %arg9 + %arg26 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %671 = amdgpu.mfma %arg11 * %arg8 + %arg25 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %672 = vector.load %alloc[%32, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %673 = arith.addi %26, %c32 : index
          %674 = vector.load %alloc_0[%673, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %675 = amdgpu.mfma %arg11 * %672 + %arg24 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %676 = amdgpu.mfma %674 * %arg10 + %arg23 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %677 = arith.addi %26, %c16 : index
          %678 = vector.load %alloc_0[%677, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %679 = vector.load %alloc_0[%26, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %680 = amdgpu.mfma %674 * %arg9 + %arg22 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %681 = amdgpu.mfma %674 * %arg8 + %arg21 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %682 = amdgpu.mfma %674 * %672 + %arg20 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %683 = amdgpu.mfma %678 * %arg10 + %arg19 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %684 = amdgpu.mfma %678 * %arg9 + %arg18 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %685 = amdgpu.mfma %678 * %arg8 + %arg17 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          amdgpu.lds_barrier
          vector.store %668, %alloc_0[%22, %12] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %669, %alloc[%22, %12] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %686 = amdgpu.mfma %678 * %672 + %arg16 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %687 = amdgpu.mfma %679 * %arg10 + %arg15 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %688 = vector.load %alloc_0[%27, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %689 = vector.load %alloc[%33, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %690 = amdgpu.mfma %679 * %arg9 + %arg14 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %691 = amdgpu.mfma %679 * %arg8 + %arg13 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %692 = amdgpu.mfma %688 * %689 + %arg27 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %693 = vector.load %alloc[%36, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %694 = vector.load %alloc[%38, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %695 = amdgpu.mfma %679 * %672 + %arg12 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          scf.yield %694, %693, %689, %688, %695, %691, %690, %687, %686, %685, %684, %683, %682, %681, %680, %676, %675, %671, %670, %692 : vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
        }
        %41 = stream.binding.subspan %arg6[%c0] : !stream.binding -> memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>
        %42 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<2x1024xi32, strided<[1024, 1], offset: ?>>
        %43 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10240xi8, strided<[1], offset: ?>>
        %44 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<10240xi32, strided<[1], offset: ?>>
        %45 = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<10240xf32, strided<[1], offset: ?>>
        %46 = arith.muli %29, %c4 : index
        %47 = arith.addi %6, %24 : index
        %48 = arith.addi %47, %46 : index
        %49 = arith.addi %48, %c48 : index
        %50 = arith.addi %25, %16 : index
        %51 = arith.addi %50, %1 : index
        %52 = arith.addi %51, %c48 : index
        %53 = vector.extract_strided_slice %40#19 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %54 = vector.load %42[%workgroup_id_2, %49] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %55 = vector.load %43[%52] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %56 = vector.load %44[%52] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %57 = vector.load %45[%52] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %58 = arith.extsi %55 : vector<1xi8> to vector<1xi32>
        %59 = arith.muli %54, %58 : vector<1xi32>
        %60 = arith.subi %53, %59 : vector<1xi32>
        %61 = arith.addi %60, %56 : vector<1xi32>
        %62 = arith.sitofp %61 : vector<1xi32> to vector<1xf32>
        %63 = arith.mulf %62, %57 : vector<1xf32>
        %64 = arith.truncf %63 : vector<1xf32> to vector<1xf16>
        vector.store %64, %41[%workgroup_id_2, %49, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %65 = vector.extract_strided_slice %40#19 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %66 = arith.addi %48, %c49 : index
        %67 = vector.load %42[%workgroup_id_2, %66] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %68 = arith.muli %67, %58 : vector<1xi32>
        %69 = arith.subi %65, %68 : vector<1xi32>
        %70 = arith.addi %69, %56 : vector<1xi32>
        %71 = arith.sitofp %70 : vector<1xi32> to vector<1xf32>
        %72 = arith.mulf %71, %57 : vector<1xf32>
        %73 = arith.truncf %72 : vector<1xf32> to vector<1xf16>
        vector.store %73, %41[%workgroup_id_2, %66, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %74 = vector.extract_strided_slice %40#19 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %75 = arith.addi %48, %c50 : index
        %76 = vector.load %42[%workgroup_id_2, %75] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %77 = arith.muli %76, %58 : vector<1xi32>
        %78 = arith.subi %74, %77 : vector<1xi32>
        %79 = arith.addi %78, %56 : vector<1xi32>
        %80 = arith.sitofp %79 : vector<1xi32> to vector<1xf32>
        %81 = arith.mulf %80, %57 : vector<1xf32>
        %82 = arith.truncf %81 : vector<1xf32> to vector<1xf16>
        vector.store %82, %41[%workgroup_id_2, %75, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %83 = vector.extract_strided_slice %40#19 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %84 = arith.addi %48, %c51 : index
        %85 = vector.load %42[%workgroup_id_2, %84] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %86 = arith.muli %85, %58 : vector<1xi32>
        %87 = arith.subi %83, %86 : vector<1xi32>
        %88 = arith.addi %87, %56 : vector<1xi32>
        %89 = arith.sitofp %88 : vector<1xi32> to vector<1xf32>
        %90 = arith.mulf %89, %57 : vector<1xf32>
        %91 = arith.truncf %90 : vector<1xf32> to vector<1xf16>
        vector.store %91, %41[%workgroup_id_2, %84, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %92 = amdgpu.mfma %40#3 * %40#1 + %40#18 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %93 = arith.addi %51, %c32 : index
        %94 = vector.extract_strided_slice %92 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %95 = vector.load %42[%workgroup_id_2, %49] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %96 = vector.load %43[%93] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %97 = vector.load %44[%93] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %98 = vector.load %45[%93] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %99 = arith.extsi %96 : vector<1xi8> to vector<1xi32>
        %100 = arith.muli %95, %99 : vector<1xi32>
        %101 = arith.subi %94, %100 : vector<1xi32>
        %102 = arith.addi %101, %97 : vector<1xi32>
        %103 = arith.sitofp %102 : vector<1xi32> to vector<1xf32>
        %104 = arith.mulf %103, %98 : vector<1xf32>
        %105 = arith.truncf %104 : vector<1xf32> to vector<1xf16>
        vector.store %105, %41[%workgroup_id_2, %49, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %106 = vector.extract_strided_slice %92 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %107 = vector.load %42[%workgroup_id_2, %66] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %108 = arith.muli %107, %99 : vector<1xi32>
        %109 = arith.subi %106, %108 : vector<1xi32>
        %110 = arith.addi %109, %97 : vector<1xi32>
        %111 = arith.sitofp %110 : vector<1xi32> to vector<1xf32>
        %112 = arith.mulf %111, %98 : vector<1xf32>
        %113 = arith.truncf %112 : vector<1xf32> to vector<1xf16>
        vector.store %113, %41[%workgroup_id_2, %66, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %114 = vector.extract_strided_slice %92 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %115 = vector.load %42[%workgroup_id_2, %75] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %116 = arith.muli %115, %99 : vector<1xi32>
        %117 = arith.subi %114, %116 : vector<1xi32>
        %118 = arith.addi %117, %97 : vector<1xi32>
        %119 = arith.sitofp %118 : vector<1xi32> to vector<1xf32>
        %120 = arith.mulf %119, %98 : vector<1xf32>
        %121 = arith.truncf %120 : vector<1xf32> to vector<1xf16>
        vector.store %121, %41[%workgroup_id_2, %75, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %122 = vector.extract_strided_slice %92 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %123 = vector.load %42[%workgroup_id_2, %84] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %124 = arith.muli %123, %99 : vector<1xi32>
        %125 = arith.subi %122, %124 : vector<1xi32>
        %126 = arith.addi %125, %97 : vector<1xi32>
        %127 = arith.sitofp %126 : vector<1xi32> to vector<1xf32>
        %128 = arith.mulf %127, %98 : vector<1xf32>
        %129 = arith.truncf %128 : vector<1xf32> to vector<1xf16>
        vector.store %129, %41[%workgroup_id_2, %84, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %130 = amdgpu.mfma %40#3 * %40#0 + %40#17 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %131 = arith.addi %51, %c16 : index
        %132 = vector.extract_strided_slice %130 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %133 = vector.load %42[%workgroup_id_2, %49] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %134 = vector.load %43[%131] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %135 = vector.load %44[%131] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %136 = vector.load %45[%131] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %137 = arith.extsi %134 : vector<1xi8> to vector<1xi32>
        %138 = arith.muli %133, %137 : vector<1xi32>
        %139 = arith.subi %132, %138 : vector<1xi32>
        %140 = arith.addi %139, %135 : vector<1xi32>
        %141 = arith.sitofp %140 : vector<1xi32> to vector<1xf32>
        %142 = arith.mulf %141, %136 : vector<1xf32>
        %143 = arith.truncf %142 : vector<1xf32> to vector<1xf16>
        vector.store %143, %41[%workgroup_id_2, %49, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %144 = vector.extract_strided_slice %130 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %145 = vector.load %42[%workgroup_id_2, %66] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %146 = arith.muli %145, %137 : vector<1xi32>
        %147 = arith.subi %144, %146 : vector<1xi32>
        %148 = arith.addi %147, %135 : vector<1xi32>
        %149 = arith.sitofp %148 : vector<1xi32> to vector<1xf32>
        %150 = arith.mulf %149, %136 : vector<1xf32>
        %151 = arith.truncf %150 : vector<1xf32> to vector<1xf16>
        vector.store %151, %41[%workgroup_id_2, %66, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %152 = vector.extract_strided_slice %130 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %153 = vector.load %42[%workgroup_id_2, %75] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %154 = arith.muli %153, %137 : vector<1xi32>
        %155 = arith.subi %152, %154 : vector<1xi32>
        %156 = arith.addi %155, %135 : vector<1xi32>
        %157 = arith.sitofp %156 : vector<1xi32> to vector<1xf32>
        %158 = arith.mulf %157, %136 : vector<1xf32>
        %159 = arith.truncf %158 : vector<1xf32> to vector<1xf16>
        vector.store %159, %41[%workgroup_id_2, %75, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %160 = vector.extract_strided_slice %130 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %161 = vector.load %42[%workgroup_id_2, %84] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %162 = arith.muli %161, %137 : vector<1xi32>
        %163 = arith.subi %160, %162 : vector<1xi32>
        %164 = arith.addi %163, %135 : vector<1xi32>
        %165 = arith.sitofp %164 : vector<1xi32> to vector<1xf32>
        %166 = arith.mulf %165, %136 : vector<1xf32>
        %167 = arith.truncf %166 : vector<1xf32> to vector<1xf16>
        vector.store %167, %41[%workgroup_id_2, %84, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %168 = vector.load %alloc[%32, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %169 = arith.addi %26, %c32 : index
        %170 = vector.load %alloc_0[%169, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %171 = amdgpu.mfma %40#3 * %168 + %40#16 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %172 = vector.extract_strided_slice %171 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %173 = vector.load %42[%workgroup_id_2, %49] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %174 = vector.load %43[%51] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %175 = vector.load %44[%51] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %176 = vector.load %45[%51] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %177 = arith.extsi %174 : vector<1xi8> to vector<1xi32>
        %178 = arith.muli %173, %177 : vector<1xi32>
        %179 = arith.subi %172, %178 : vector<1xi32>
        %180 = arith.addi %179, %175 : vector<1xi32>
        %181 = arith.sitofp %180 : vector<1xi32> to vector<1xf32>
        %182 = arith.mulf %181, %176 : vector<1xf32>
        %183 = arith.truncf %182 : vector<1xf32> to vector<1xf16>
        vector.store %183, %41[%workgroup_id_2, %49, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %184 = vector.extract_strided_slice %171 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %185 = vector.load %42[%workgroup_id_2, %66] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %186 = arith.muli %185, %177 : vector<1xi32>
        %187 = arith.subi %184, %186 : vector<1xi32>
        %188 = arith.addi %187, %175 : vector<1xi32>
        %189 = arith.sitofp %188 : vector<1xi32> to vector<1xf32>
        %190 = arith.mulf %189, %176 : vector<1xf32>
        %191 = arith.truncf %190 : vector<1xf32> to vector<1xf16>
        vector.store %191, %41[%workgroup_id_2, %66, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %192 = vector.extract_strided_slice %171 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %193 = vector.load %42[%workgroup_id_2, %75] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %194 = arith.muli %193, %177 : vector<1xi32>
        %195 = arith.subi %192, %194 : vector<1xi32>
        %196 = arith.addi %195, %175 : vector<1xi32>
        %197 = arith.sitofp %196 : vector<1xi32> to vector<1xf32>
        %198 = arith.mulf %197, %176 : vector<1xf32>
        %199 = arith.truncf %198 : vector<1xf32> to vector<1xf16>
        vector.store %199, %41[%workgroup_id_2, %75, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %200 = vector.extract_strided_slice %171 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %201 = vector.load %42[%workgroup_id_2, %84] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %202 = arith.muli %201, %177 : vector<1xi32>
        %203 = arith.subi %200, %202 : vector<1xi32>
        %204 = arith.addi %203, %175 : vector<1xi32>
        %205 = arith.sitofp %204 : vector<1xi32> to vector<1xf32>
        %206 = arith.mulf %205, %176 : vector<1xf32>
        %207 = arith.truncf %206 : vector<1xf32> to vector<1xf16>
        vector.store %207, %41[%workgroup_id_2, %84, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %208 = amdgpu.mfma %170 * %40#2 + %40#15 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %209 = arith.addi %48, %c32 : index
        %210 = vector.extract_strided_slice %208 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %211 = vector.load %42[%workgroup_id_2, %209] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %212 = vector.load %43[%52] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %213 = vector.load %44[%52] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %214 = vector.load %45[%52] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %215 = arith.extsi %212 : vector<1xi8> to vector<1xi32>
        %216 = arith.muli %211, %215 : vector<1xi32>
        %217 = arith.subi %210, %216 : vector<1xi32>
        %218 = arith.addi %217, %213 : vector<1xi32>
        %219 = arith.sitofp %218 : vector<1xi32> to vector<1xf32>
        %220 = arith.mulf %219, %214 : vector<1xf32>
        %221 = arith.truncf %220 : vector<1xf32> to vector<1xf16>
        vector.store %221, %41[%workgroup_id_2, %209, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %222 = vector.extract_strided_slice %208 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %223 = arith.addi %48, %c33 : index
        %224 = vector.load %42[%workgroup_id_2, %223] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %225 = arith.muli %224, %215 : vector<1xi32>
        %226 = arith.subi %222, %225 : vector<1xi32>
        %227 = arith.addi %226, %213 : vector<1xi32>
        %228 = arith.sitofp %227 : vector<1xi32> to vector<1xf32>
        %229 = arith.mulf %228, %214 : vector<1xf32>
        %230 = arith.truncf %229 : vector<1xf32> to vector<1xf16>
        vector.store %230, %41[%workgroup_id_2, %223, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %231 = vector.extract_strided_slice %208 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %232 = arith.addi %48, %c34 : index
        %233 = vector.load %42[%workgroup_id_2, %232] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %234 = arith.muli %233, %215 : vector<1xi32>
        %235 = arith.subi %231, %234 : vector<1xi32>
        %236 = arith.addi %235, %213 : vector<1xi32>
        %237 = arith.sitofp %236 : vector<1xi32> to vector<1xf32>
        %238 = arith.mulf %237, %214 : vector<1xf32>
        %239 = arith.truncf %238 : vector<1xf32> to vector<1xf16>
        vector.store %239, %41[%workgroup_id_2, %232, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %240 = vector.extract_strided_slice %208 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %241 = arith.addi %48, %c35 : index
        %242 = vector.load %42[%workgroup_id_2, %241] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %243 = arith.muli %242, %215 : vector<1xi32>
        %244 = arith.subi %240, %243 : vector<1xi32>
        %245 = arith.addi %244, %213 : vector<1xi32>
        %246 = arith.sitofp %245 : vector<1xi32> to vector<1xf32>
        %247 = arith.mulf %246, %214 : vector<1xf32>
        %248 = arith.truncf %247 : vector<1xf32> to vector<1xf16>
        vector.store %248, %41[%workgroup_id_2, %241, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %249 = arith.addi %26, %c16 : index
        %250 = vector.load %alloc_0[%249, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %251 = vector.load %alloc_0[%26, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %252 = amdgpu.mfma %170 * %40#1 + %40#14 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %253 = vector.extract_strided_slice %252 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %254 = vector.load %42[%workgroup_id_2, %209] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %255 = vector.load %43[%93] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %256 = vector.load %44[%93] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %257 = vector.load %45[%93] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %258 = arith.extsi %255 : vector<1xi8> to vector<1xi32>
        %259 = arith.muli %254, %258 : vector<1xi32>
        %260 = arith.subi %253, %259 : vector<1xi32>
        %261 = arith.addi %260, %256 : vector<1xi32>
        %262 = arith.sitofp %261 : vector<1xi32> to vector<1xf32>
        %263 = arith.mulf %262, %257 : vector<1xf32>
        %264 = arith.truncf %263 : vector<1xf32> to vector<1xf16>
        vector.store %264, %41[%workgroup_id_2, %209, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %265 = vector.extract_strided_slice %252 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %266 = vector.load %42[%workgroup_id_2, %223] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %267 = arith.muli %266, %258 : vector<1xi32>
        %268 = arith.subi %265, %267 : vector<1xi32>
        %269 = arith.addi %268, %256 : vector<1xi32>
        %270 = arith.sitofp %269 : vector<1xi32> to vector<1xf32>
        %271 = arith.mulf %270, %257 : vector<1xf32>
        %272 = arith.truncf %271 : vector<1xf32> to vector<1xf16>
        vector.store %272, %41[%workgroup_id_2, %223, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %273 = vector.extract_strided_slice %252 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %274 = vector.load %42[%workgroup_id_2, %232] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %275 = arith.muli %274, %258 : vector<1xi32>
        %276 = arith.subi %273, %275 : vector<1xi32>
        %277 = arith.addi %276, %256 : vector<1xi32>
        %278 = arith.sitofp %277 : vector<1xi32> to vector<1xf32>
        %279 = arith.mulf %278, %257 : vector<1xf32>
        %280 = arith.truncf %279 : vector<1xf32> to vector<1xf16>
        vector.store %280, %41[%workgroup_id_2, %232, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %281 = vector.extract_strided_slice %252 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %282 = vector.load %42[%workgroup_id_2, %241] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %283 = arith.muli %282, %258 : vector<1xi32>
        %284 = arith.subi %281, %283 : vector<1xi32>
        %285 = arith.addi %284, %256 : vector<1xi32>
        %286 = arith.sitofp %285 : vector<1xi32> to vector<1xf32>
        %287 = arith.mulf %286, %257 : vector<1xf32>
        %288 = arith.truncf %287 : vector<1xf32> to vector<1xf16>
        vector.store %288, %41[%workgroup_id_2, %241, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %289 = amdgpu.mfma %170 * %40#0 + %40#13 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %290 = vector.extract_strided_slice %289 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %291 = vector.load %42[%workgroup_id_2, %209] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %292 = vector.load %43[%131] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %293 = vector.load %44[%131] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %294 = vector.load %45[%131] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %295 = arith.extsi %292 : vector<1xi8> to vector<1xi32>
        %296 = arith.muli %291, %295 : vector<1xi32>
        %297 = arith.subi %290, %296 : vector<1xi32>
        %298 = arith.addi %297, %293 : vector<1xi32>
        %299 = arith.sitofp %298 : vector<1xi32> to vector<1xf32>
        %300 = arith.mulf %299, %294 : vector<1xf32>
        %301 = arith.truncf %300 : vector<1xf32> to vector<1xf16>
        vector.store %301, %41[%workgroup_id_2, %209, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %302 = vector.extract_strided_slice %289 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %303 = vector.load %42[%workgroup_id_2, %223] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %304 = arith.muli %303, %295 : vector<1xi32>
        %305 = arith.subi %302, %304 : vector<1xi32>
        %306 = arith.addi %305, %293 : vector<1xi32>
        %307 = arith.sitofp %306 : vector<1xi32> to vector<1xf32>
        %308 = arith.mulf %307, %294 : vector<1xf32>
        %309 = arith.truncf %308 : vector<1xf32> to vector<1xf16>
        vector.store %309, %41[%workgroup_id_2, %223, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %310 = vector.extract_strided_slice %289 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %311 = vector.load %42[%workgroup_id_2, %232] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %312 = arith.muli %311, %295 : vector<1xi32>
        %313 = arith.subi %310, %312 : vector<1xi32>
        %314 = arith.addi %313, %293 : vector<1xi32>
        %315 = arith.sitofp %314 : vector<1xi32> to vector<1xf32>
        %316 = arith.mulf %315, %294 : vector<1xf32>
        %317 = arith.truncf %316 : vector<1xf32> to vector<1xf16>
        vector.store %317, %41[%workgroup_id_2, %232, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %318 = vector.extract_strided_slice %289 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %319 = vector.load %42[%workgroup_id_2, %241] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %320 = arith.muli %319, %295 : vector<1xi32>
        %321 = arith.subi %318, %320 : vector<1xi32>
        %322 = arith.addi %321, %293 : vector<1xi32>
        %323 = arith.sitofp %322 : vector<1xi32> to vector<1xf32>
        %324 = arith.mulf %323, %294 : vector<1xf32>
        %325 = arith.truncf %324 : vector<1xf32> to vector<1xf16>
        vector.store %325, %41[%workgroup_id_2, %241, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %326 = amdgpu.mfma %170 * %168 + %40#12 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %327 = vector.extract_strided_slice %326 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %328 = vector.load %42[%workgroup_id_2, %209] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %329 = vector.load %43[%51] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %330 = vector.load %44[%51] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %331 = vector.load %45[%51] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %332 = arith.extsi %329 : vector<1xi8> to vector<1xi32>
        %333 = arith.muli %328, %332 : vector<1xi32>
        %334 = arith.subi %327, %333 : vector<1xi32>
        %335 = arith.addi %334, %330 : vector<1xi32>
        %336 = arith.sitofp %335 : vector<1xi32> to vector<1xf32>
        %337 = arith.mulf %336, %331 : vector<1xf32>
        %338 = arith.truncf %337 : vector<1xf32> to vector<1xf16>
        vector.store %338, %41[%workgroup_id_2, %209, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %339 = vector.extract_strided_slice %326 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %340 = vector.load %42[%workgroup_id_2, %223] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %341 = arith.muli %340, %332 : vector<1xi32>
        %342 = arith.subi %339, %341 : vector<1xi32>
        %343 = arith.addi %342, %330 : vector<1xi32>
        %344 = arith.sitofp %343 : vector<1xi32> to vector<1xf32>
        %345 = arith.mulf %344, %331 : vector<1xf32>
        %346 = arith.truncf %345 : vector<1xf32> to vector<1xf16>
        vector.store %346, %41[%workgroup_id_2, %223, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %347 = vector.extract_strided_slice %326 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %348 = vector.load %42[%workgroup_id_2, %232] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %349 = arith.muli %348, %332 : vector<1xi32>
        %350 = arith.subi %347, %349 : vector<1xi32>
        %351 = arith.addi %350, %330 : vector<1xi32>
        %352 = arith.sitofp %351 : vector<1xi32> to vector<1xf32>
        %353 = arith.mulf %352, %331 : vector<1xf32>
        %354 = arith.truncf %353 : vector<1xf32> to vector<1xf16>
        vector.store %354, %41[%workgroup_id_2, %232, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %355 = vector.extract_strided_slice %326 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %356 = vector.load %42[%workgroup_id_2, %241] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %357 = arith.muli %356, %332 : vector<1xi32>
        %358 = arith.subi %355, %357 : vector<1xi32>
        %359 = arith.addi %358, %330 : vector<1xi32>
        %360 = arith.sitofp %359 : vector<1xi32> to vector<1xf32>
        %361 = arith.mulf %360, %331 : vector<1xf32>
        %362 = arith.truncf %361 : vector<1xf32> to vector<1xf16>
        vector.store %362, %41[%workgroup_id_2, %241, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %363 = amdgpu.mfma %250 * %40#2 + %40#11 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %364 = arith.addi %48, %c16 : index
        %365 = vector.extract_strided_slice %363 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %366 = vector.load %42[%workgroup_id_2, %364] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %367 = vector.load %43[%52] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %368 = vector.load %44[%52] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %369 = vector.load %45[%52] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %370 = arith.extsi %367 : vector<1xi8> to vector<1xi32>
        %371 = arith.muli %366, %370 : vector<1xi32>
        %372 = arith.subi %365, %371 : vector<1xi32>
        %373 = arith.addi %372, %368 : vector<1xi32>
        %374 = arith.sitofp %373 : vector<1xi32> to vector<1xf32>
        %375 = arith.mulf %374, %369 : vector<1xf32>
        %376 = arith.truncf %375 : vector<1xf32> to vector<1xf16>
        vector.store %376, %41[%workgroup_id_2, %364, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %377 = vector.extract_strided_slice %363 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %378 = arith.addi %48, %c17 : index
        %379 = vector.load %42[%workgroup_id_2, %378] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %380 = arith.muli %379, %370 : vector<1xi32>
        %381 = arith.subi %377, %380 : vector<1xi32>
        %382 = arith.addi %381, %368 : vector<1xi32>
        %383 = arith.sitofp %382 : vector<1xi32> to vector<1xf32>
        %384 = arith.mulf %383, %369 : vector<1xf32>
        %385 = arith.truncf %384 : vector<1xf32> to vector<1xf16>
        vector.store %385, %41[%workgroup_id_2, %378, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %386 = vector.extract_strided_slice %363 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %387 = arith.addi %48, %c18 : index
        %388 = vector.load %42[%workgroup_id_2, %387] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %389 = arith.muli %388, %370 : vector<1xi32>
        %390 = arith.subi %386, %389 : vector<1xi32>
        %391 = arith.addi %390, %368 : vector<1xi32>
        %392 = arith.sitofp %391 : vector<1xi32> to vector<1xf32>
        %393 = arith.mulf %392, %369 : vector<1xf32>
        %394 = arith.truncf %393 : vector<1xf32> to vector<1xf16>
        vector.store %394, %41[%workgroup_id_2, %387, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %395 = vector.extract_strided_slice %363 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %396 = arith.addi %48, %c19 : index
        %397 = vector.load %42[%workgroup_id_2, %396] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %398 = arith.muli %397, %370 : vector<1xi32>
        %399 = arith.subi %395, %398 : vector<1xi32>
        %400 = arith.addi %399, %368 : vector<1xi32>
        %401 = arith.sitofp %400 : vector<1xi32> to vector<1xf32>
        %402 = arith.mulf %401, %369 : vector<1xf32>
        %403 = arith.truncf %402 : vector<1xf32> to vector<1xf16>
        vector.store %403, %41[%workgroup_id_2, %396, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %404 = amdgpu.mfma %250 * %40#1 + %40#10 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %405 = vector.extract_strided_slice %404 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %406 = vector.load %42[%workgroup_id_2, %364] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %407 = vector.load %43[%93] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %408 = vector.load %44[%93] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %409 = vector.load %45[%93] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %410 = arith.extsi %407 : vector<1xi8> to vector<1xi32>
        %411 = arith.muli %406, %410 : vector<1xi32>
        %412 = arith.subi %405, %411 : vector<1xi32>
        %413 = arith.addi %412, %408 : vector<1xi32>
        %414 = arith.sitofp %413 : vector<1xi32> to vector<1xf32>
        %415 = arith.mulf %414, %409 : vector<1xf32>
        %416 = arith.truncf %415 : vector<1xf32> to vector<1xf16>
        vector.store %416, %41[%workgroup_id_2, %364, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %417 = vector.extract_strided_slice %404 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %418 = vector.load %42[%workgroup_id_2, %378] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %419 = arith.muli %418, %410 : vector<1xi32>
        %420 = arith.subi %417, %419 : vector<1xi32>
        %421 = arith.addi %420, %408 : vector<1xi32>
        %422 = arith.sitofp %421 : vector<1xi32> to vector<1xf32>
        %423 = arith.mulf %422, %409 : vector<1xf32>
        %424 = arith.truncf %423 : vector<1xf32> to vector<1xf16>
        vector.store %424, %41[%workgroup_id_2, %378, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %425 = vector.extract_strided_slice %404 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %426 = vector.load %42[%workgroup_id_2, %387] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %427 = arith.muli %426, %410 : vector<1xi32>
        %428 = arith.subi %425, %427 : vector<1xi32>
        %429 = arith.addi %428, %408 : vector<1xi32>
        %430 = arith.sitofp %429 : vector<1xi32> to vector<1xf32>
        %431 = arith.mulf %430, %409 : vector<1xf32>
        %432 = arith.truncf %431 : vector<1xf32> to vector<1xf16>
        vector.store %432, %41[%workgroup_id_2, %387, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %433 = vector.extract_strided_slice %404 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %434 = vector.load %42[%workgroup_id_2, %396] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %435 = arith.muli %434, %410 : vector<1xi32>
        %436 = arith.subi %433, %435 : vector<1xi32>
        %437 = arith.addi %436, %408 : vector<1xi32>
        %438 = arith.sitofp %437 : vector<1xi32> to vector<1xf32>
        %439 = arith.mulf %438, %409 : vector<1xf32>
        %440 = arith.truncf %439 : vector<1xf32> to vector<1xf16>
        vector.store %440, %41[%workgroup_id_2, %396, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %441 = amdgpu.mfma %250 * %40#0 + %40#9 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %442 = vector.extract_strided_slice %441 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %443 = vector.load %42[%workgroup_id_2, %364] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %444 = vector.load %43[%131] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %445 = vector.load %44[%131] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %446 = vector.load %45[%131] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %447 = arith.extsi %444 : vector<1xi8> to vector<1xi32>
        %448 = arith.muli %443, %447 : vector<1xi32>
        %449 = arith.subi %442, %448 : vector<1xi32>
        %450 = arith.addi %449, %445 : vector<1xi32>
        %451 = arith.sitofp %450 : vector<1xi32> to vector<1xf32>
        %452 = arith.mulf %451, %446 : vector<1xf32>
        %453 = arith.truncf %452 : vector<1xf32> to vector<1xf16>
        vector.store %453, %41[%workgroup_id_2, %364, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %454 = vector.extract_strided_slice %441 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %455 = vector.load %42[%workgroup_id_2, %378] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %456 = arith.muli %455, %447 : vector<1xi32>
        %457 = arith.subi %454, %456 : vector<1xi32>
        %458 = arith.addi %457, %445 : vector<1xi32>
        %459 = arith.sitofp %458 : vector<1xi32> to vector<1xf32>
        %460 = arith.mulf %459, %446 : vector<1xf32>
        %461 = arith.truncf %460 : vector<1xf32> to vector<1xf16>
        vector.store %461, %41[%workgroup_id_2, %378, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %462 = vector.extract_strided_slice %441 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %463 = vector.load %42[%workgroup_id_2, %387] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %464 = arith.muli %463, %447 : vector<1xi32>
        %465 = arith.subi %462, %464 : vector<1xi32>
        %466 = arith.addi %465, %445 : vector<1xi32>
        %467 = arith.sitofp %466 : vector<1xi32> to vector<1xf32>
        %468 = arith.mulf %467, %446 : vector<1xf32>
        %469 = arith.truncf %468 : vector<1xf32> to vector<1xf16>
        vector.store %469, %41[%workgroup_id_2, %387, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %470 = vector.extract_strided_slice %441 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %471 = vector.load %42[%workgroup_id_2, %396] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %472 = arith.muli %471, %447 : vector<1xi32>
        %473 = arith.subi %470, %472 : vector<1xi32>
        %474 = arith.addi %473, %445 : vector<1xi32>
        %475 = arith.sitofp %474 : vector<1xi32> to vector<1xf32>
        %476 = arith.mulf %475, %446 : vector<1xf32>
        %477 = arith.truncf %476 : vector<1xf32> to vector<1xf16>
        vector.store %477, %41[%workgroup_id_2, %396, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %478 = amdgpu.mfma %250 * %168 + %40#8 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %479 = vector.extract_strided_slice %478 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %480 = vector.load %42[%workgroup_id_2, %364] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %481 = vector.load %43[%51] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %482 = vector.load %44[%51] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %483 = vector.load %45[%51] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %484 = arith.extsi %481 : vector<1xi8> to vector<1xi32>
        %485 = arith.muli %480, %484 : vector<1xi32>
        %486 = arith.subi %479, %485 : vector<1xi32>
        %487 = arith.addi %486, %482 : vector<1xi32>
        %488 = arith.sitofp %487 : vector<1xi32> to vector<1xf32>
        %489 = arith.mulf %488, %483 : vector<1xf32>
        %490 = arith.truncf %489 : vector<1xf32> to vector<1xf16>
        vector.store %490, %41[%workgroup_id_2, %364, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %491 = vector.extract_strided_slice %478 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %492 = vector.load %42[%workgroup_id_2, %378] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %493 = arith.muli %492, %484 : vector<1xi32>
        %494 = arith.subi %491, %493 : vector<1xi32>
        %495 = arith.addi %494, %482 : vector<1xi32>
        %496 = arith.sitofp %495 : vector<1xi32> to vector<1xf32>
        %497 = arith.mulf %496, %483 : vector<1xf32>
        %498 = arith.truncf %497 : vector<1xf32> to vector<1xf16>
        vector.store %498, %41[%workgroup_id_2, %378, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %499 = vector.extract_strided_slice %478 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %500 = vector.load %42[%workgroup_id_2, %387] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %501 = arith.muli %500, %484 : vector<1xi32>
        %502 = arith.subi %499, %501 : vector<1xi32>
        %503 = arith.addi %502, %482 : vector<1xi32>
        %504 = arith.sitofp %503 : vector<1xi32> to vector<1xf32>
        %505 = arith.mulf %504, %483 : vector<1xf32>
        %506 = arith.truncf %505 : vector<1xf32> to vector<1xf16>
        vector.store %506, %41[%workgroup_id_2, %387, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %507 = vector.extract_strided_slice %478 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %508 = vector.load %42[%workgroup_id_2, %396] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %509 = arith.muli %508, %484 : vector<1xi32>
        %510 = arith.subi %507, %509 : vector<1xi32>
        %511 = arith.addi %510, %482 : vector<1xi32>
        %512 = arith.sitofp %511 : vector<1xi32> to vector<1xf32>
        %513 = arith.mulf %512, %483 : vector<1xf32>
        %514 = arith.truncf %513 : vector<1xf32> to vector<1xf16>
        vector.store %514, %41[%workgroup_id_2, %396, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %515 = amdgpu.mfma %251 * %40#2 + %40#7 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %516 = vector.extract_strided_slice %515 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %517 = vector.load %42[%workgroup_id_2, %48] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %518 = vector.load %43[%52] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %519 = vector.load %44[%52] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %520 = vector.load %45[%52] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %521 = arith.extsi %518 : vector<1xi8> to vector<1xi32>
        %522 = arith.muli %517, %521 : vector<1xi32>
        %523 = arith.subi %516, %522 : vector<1xi32>
        %524 = arith.addi %523, %519 : vector<1xi32>
        %525 = arith.sitofp %524 : vector<1xi32> to vector<1xf32>
        %526 = arith.mulf %525, %520 : vector<1xf32>
        %527 = arith.truncf %526 : vector<1xf32> to vector<1xf16>
        vector.store %527, %41[%workgroup_id_2, %48, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %528 = vector.extract_strided_slice %515 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %529 = arith.addi %48, %c1 : index
        %530 = vector.load %42[%workgroup_id_2, %529] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %531 = arith.muli %530, %521 : vector<1xi32>
        %532 = arith.subi %528, %531 : vector<1xi32>
        %533 = arith.addi %532, %519 : vector<1xi32>
        %534 = arith.sitofp %533 : vector<1xi32> to vector<1xf32>
        %535 = arith.mulf %534, %520 : vector<1xf32>
        %536 = arith.truncf %535 : vector<1xf32> to vector<1xf16>
        vector.store %536, %41[%workgroup_id_2, %529, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %537 = vector.extract_strided_slice %515 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %538 = arith.addi %48, %c2 : index
        %539 = vector.load %42[%workgroup_id_2, %538] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %540 = arith.muli %539, %521 : vector<1xi32>
        %541 = arith.subi %537, %540 : vector<1xi32>
        %542 = arith.addi %541, %519 : vector<1xi32>
        %543 = arith.sitofp %542 : vector<1xi32> to vector<1xf32>
        %544 = arith.mulf %543, %520 : vector<1xf32>
        %545 = arith.truncf %544 : vector<1xf32> to vector<1xf16>
        vector.store %545, %41[%workgroup_id_2, %538, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %546 = vector.extract_strided_slice %515 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %547 = arith.addi %48, %c3 : index
        %548 = vector.load %42[%workgroup_id_2, %547] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %549 = arith.muli %548, %521 : vector<1xi32>
        %550 = arith.subi %546, %549 : vector<1xi32>
        %551 = arith.addi %550, %519 : vector<1xi32>
        %552 = arith.sitofp %551 : vector<1xi32> to vector<1xf32>
        %553 = arith.mulf %552, %520 : vector<1xf32>
        %554 = arith.truncf %553 : vector<1xf32> to vector<1xf16>
        vector.store %554, %41[%workgroup_id_2, %547, %52] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %555 = amdgpu.mfma %251 * %40#1 + %40#6 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %556 = vector.extract_strided_slice %555 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %557 = vector.load %42[%workgroup_id_2, %48] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %558 = vector.load %43[%93] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %559 = vector.load %44[%93] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %560 = vector.load %45[%93] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %561 = arith.extsi %558 : vector<1xi8> to vector<1xi32>
        %562 = arith.muli %557, %561 : vector<1xi32>
        %563 = arith.subi %556, %562 : vector<1xi32>
        %564 = arith.addi %563, %559 : vector<1xi32>
        %565 = arith.sitofp %564 : vector<1xi32> to vector<1xf32>
        %566 = arith.mulf %565, %560 : vector<1xf32>
        %567 = arith.truncf %566 : vector<1xf32> to vector<1xf16>
        vector.store %567, %41[%workgroup_id_2, %48, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %568 = vector.extract_strided_slice %555 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %569 = vector.load %42[%workgroup_id_2, %529] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %570 = arith.muli %569, %561 : vector<1xi32>
        %571 = arith.subi %568, %570 : vector<1xi32>
        %572 = arith.addi %571, %559 : vector<1xi32>
        %573 = arith.sitofp %572 : vector<1xi32> to vector<1xf32>
        %574 = arith.mulf %573, %560 : vector<1xf32>
        %575 = arith.truncf %574 : vector<1xf32> to vector<1xf16>
        vector.store %575, %41[%workgroup_id_2, %529, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %576 = vector.extract_strided_slice %555 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %577 = vector.load %42[%workgroup_id_2, %538] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %578 = arith.muli %577, %561 : vector<1xi32>
        %579 = arith.subi %576, %578 : vector<1xi32>
        %580 = arith.addi %579, %559 : vector<1xi32>
        %581 = arith.sitofp %580 : vector<1xi32> to vector<1xf32>
        %582 = arith.mulf %581, %560 : vector<1xf32>
        %583 = arith.truncf %582 : vector<1xf32> to vector<1xf16>
        vector.store %583, %41[%workgroup_id_2, %538, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %584 = vector.extract_strided_slice %555 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %585 = vector.load %42[%workgroup_id_2, %547] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %586 = arith.muli %585, %561 : vector<1xi32>
        %587 = arith.subi %584, %586 : vector<1xi32>
        %588 = arith.addi %587, %559 : vector<1xi32>
        %589 = arith.sitofp %588 : vector<1xi32> to vector<1xf32>
        %590 = arith.mulf %589, %560 : vector<1xf32>
        %591 = arith.truncf %590 : vector<1xf32> to vector<1xf16>
        vector.store %591, %41[%workgroup_id_2, %547, %93] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %592 = amdgpu.mfma %251 * %40#0 + %40#5 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %593 = vector.extract_strided_slice %592 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %594 = vector.load %42[%workgroup_id_2, %48] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %595 = vector.load %43[%131] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %596 = vector.load %44[%131] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %597 = vector.load %45[%131] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %598 = arith.extsi %595 : vector<1xi8> to vector<1xi32>
        %599 = arith.muli %594, %598 : vector<1xi32>
        %600 = arith.subi %593, %599 : vector<1xi32>
        %601 = arith.addi %600, %596 : vector<1xi32>
        %602 = arith.sitofp %601 : vector<1xi32> to vector<1xf32>
        %603 = arith.mulf %602, %597 : vector<1xf32>
        %604 = arith.truncf %603 : vector<1xf32> to vector<1xf16>
        vector.store %604, %41[%workgroup_id_2, %48, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %605 = vector.extract_strided_slice %592 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %606 = vector.load %42[%workgroup_id_2, %529] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %607 = arith.muli %606, %598 : vector<1xi32>
        %608 = arith.subi %605, %607 : vector<1xi32>
        %609 = arith.addi %608, %596 : vector<1xi32>
        %610 = arith.sitofp %609 : vector<1xi32> to vector<1xf32>
        %611 = arith.mulf %610, %597 : vector<1xf32>
        %612 = arith.truncf %611 : vector<1xf32> to vector<1xf16>
        vector.store %612, %41[%workgroup_id_2, %529, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %613 = vector.extract_strided_slice %592 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %614 = vector.load %42[%workgroup_id_2, %538] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %615 = arith.muli %614, %598 : vector<1xi32>
        %616 = arith.subi %613, %615 : vector<1xi32>
        %617 = arith.addi %616, %596 : vector<1xi32>
        %618 = arith.sitofp %617 : vector<1xi32> to vector<1xf32>
        %619 = arith.mulf %618, %597 : vector<1xf32>
        %620 = arith.truncf %619 : vector<1xf32> to vector<1xf16>
        vector.store %620, %41[%workgroup_id_2, %538, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %621 = vector.extract_strided_slice %592 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %622 = vector.load %42[%workgroup_id_2, %547] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %623 = arith.muli %622, %598 : vector<1xi32>
        %624 = arith.subi %621, %623 : vector<1xi32>
        %625 = arith.addi %624, %596 : vector<1xi32>
        %626 = arith.sitofp %625 : vector<1xi32> to vector<1xf32>
        %627 = arith.mulf %626, %597 : vector<1xf32>
        %628 = arith.truncf %627 : vector<1xf32> to vector<1xf16>
        vector.store %628, %41[%workgroup_id_2, %547, %131] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %629 = amdgpu.mfma %251 * %168 + %40#4 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %630 = vector.extract_strided_slice %629 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %631 = vector.load %42[%workgroup_id_2, %48] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %632 = vector.load %43[%51] : memref<10240xi8, strided<[1], offset: ?>>, vector<1xi8>
        %633 = vector.load %44[%51] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %634 = vector.load %45[%51] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %635 = arith.extsi %632 : vector<1xi8> to vector<1xi32>
        %636 = arith.muli %631, %635 : vector<1xi32>
        %637 = arith.subi %630, %636 : vector<1xi32>
        %638 = arith.addi %637, %633 : vector<1xi32>
        %639 = arith.sitofp %638 : vector<1xi32> to vector<1xf32>
        %640 = arith.mulf %639, %634 : vector<1xf32>
        %641 = arith.truncf %640 : vector<1xf32> to vector<1xf16>
        vector.store %641, %41[%workgroup_id_2, %48, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %642 = vector.extract_strided_slice %629 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %643 = vector.load %42[%workgroup_id_2, %529] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %644 = arith.muli %643, %635 : vector<1xi32>
        %645 = arith.subi %642, %644 : vector<1xi32>
        %646 = arith.addi %645, %633 : vector<1xi32>
        %647 = arith.sitofp %646 : vector<1xi32> to vector<1xf32>
        %648 = arith.mulf %647, %634 : vector<1xf32>
        %649 = arith.truncf %648 : vector<1xf32> to vector<1xf16>
        vector.store %649, %41[%workgroup_id_2, %529, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %650 = vector.extract_strided_slice %629 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %651 = vector.load %42[%workgroup_id_2, %538] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %652 = arith.muli %651, %635 : vector<1xi32>
        %653 = arith.subi %650, %652 : vector<1xi32>
        %654 = arith.addi %653, %633 : vector<1xi32>
        %655 = arith.sitofp %654 : vector<1xi32> to vector<1xf32>
        %656 = arith.mulf %655, %634 : vector<1xf32>
        %657 = arith.truncf %656 : vector<1xf32> to vector<1xf16>
        vector.store %657, %41[%workgroup_id_2, %538, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %658 = vector.extract_strided_slice %629 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %659 = vector.load %42[%workgroup_id_2, %547] : memref<2x1024xi32, strided<[1024, 1], offset: ?>>, vector<1xi32>
        %660 = arith.muli %659, %635 : vector<1xi32>
        %661 = arith.subi %658, %660 : vector<1xi32>
        %662 = arith.addi %661, %633 : vector<1xi32>
        %663 = arith.sitofp %662 : vector<1xi32> to vector<1xf32>
        %664 = arith.mulf %663, %634 : vector<1xf32>
        %665 = arith.truncf %664 : vector<1xf32> to vector<1xf16>
        vector.store %665, %41[%workgroup_id_2, %547, %51] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

