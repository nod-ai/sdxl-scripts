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
        %40:20 = scf.for %arg5 = %c1 to %c40 step %c1 iter_args(%arg6 = %39, %arg7 = %37, %arg8 = %34, %arg9 = %31, %arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst, %arg20 = %cst, %arg21 = %cst, %arg22 = %cst, %arg23 = %cst, %arg24 = %cst, %arg25 = %35) -> (vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
          %440 = arith.muli %arg5, %c32 : index
          %441 = arith.addi %440, %12 : index
          %442 = vector.load %0[%workgroup_id_2, %10, %441] : memref<2x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<16xi8>
          %443 = vector.load %14[%19, %441] : memref<10240x1280xi8, strided<[1280, 1], offset: ?>>, vector<16xi8>
          %444 = amdgpu.mfma %arg9 * %arg7 + %arg24 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %445 = amdgpu.mfma %arg9 * %arg6 + %arg23 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %446 = vector.load %alloc[%32, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %447 = arith.addi %26, %c32 : index
          %448 = vector.load %alloc_0[%447, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %449 = amdgpu.mfma %arg9 * %446 + %arg22 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %450 = amdgpu.mfma %448 * %arg8 + %arg21 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %451 = arith.addi %26, %c16 : index
          %452 = vector.load %alloc_0[%451, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %453 = vector.load %alloc_0[%26, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %454 = amdgpu.mfma %448 * %arg7 + %arg20 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %455 = amdgpu.mfma %448 * %arg6 + %arg19 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %456 = amdgpu.mfma %448 * %446 + %arg18 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %457 = amdgpu.mfma %452 * %arg8 + %arg17 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %458 = amdgpu.mfma %452 * %arg7 + %arg16 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %459 = amdgpu.mfma %452 * %arg6 + %arg15 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          amdgpu.lds_barrier
          vector.store %442, %alloc_0[%22, %12] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %443, %alloc[%22, %12] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %460 = amdgpu.mfma %452 * %446 + %arg14 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %461 = amdgpu.mfma %453 * %arg8 + %arg13 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %462 = vector.load %alloc_0[%27, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %463 = vector.load %alloc[%33, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %464 = amdgpu.mfma %453 * %arg7 + %arg12 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %465 = amdgpu.mfma %453 * %arg6 + %arg11 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %466 = amdgpu.mfma %462 * %463 + %arg25 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %467 = vector.load %alloc[%36, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %468 = vector.load %alloc[%38, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %469 = amdgpu.mfma %453 * %446 + %arg10 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          scf.yield %468, %467, %463, %462, %469, %465, %464, %461, %460, %459, %458, %457, %456, %455, %454, %450, %449, %445, %444, %466 : vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
        }
        %41 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>
        %42 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<10240xi32, strided<[1], offset: ?>>
        %43 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<10240xf32, strided<[1], offset: ?>>
        %44 = arith.muli %29, %c4 : index
        %45 = arith.addi %6, %24 : index
        %46 = arith.addi %45, %44 : index
        %47 = arith.addi %46, %c48 : index
        %48 = arith.addi %25, %16 : index
        %49 = arith.addi %48, %1 : index
        %50 = arith.addi %49, %c48 : index
        %51 = vector.extract_strided_slice %40#19 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %52 = vector.load %42[%50] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %53 = vector.load %43[%50] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %54 = arith.addi %51, %52 : vector<1xi32>
        %55 = arith.sitofp %54 : vector<1xi32> to vector<1xf32>
        %56 = arith.mulf %55, %53 : vector<1xf32>
        %57 = arith.truncf %56 : vector<1xf32> to vector<1xf16>
        vector.store %57, %41[%workgroup_id_2, %47, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %58 = vector.extract_strided_slice %40#19 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %59 = arith.addi %46, %c49 : index
        %60 = arith.addi %58, %52 : vector<1xi32>
        %61 = arith.sitofp %60 : vector<1xi32> to vector<1xf32>
        %62 = arith.mulf %61, %53 : vector<1xf32>
        %63 = arith.truncf %62 : vector<1xf32> to vector<1xf16>
        vector.store %63, %41[%workgroup_id_2, %59, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %64 = vector.extract_strided_slice %40#19 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %65 = arith.addi %46, %c50 : index
        %66 = arith.addi %64, %52 : vector<1xi32>
        %67 = arith.sitofp %66 : vector<1xi32> to vector<1xf32>
        %68 = arith.mulf %67, %53 : vector<1xf32>
        %69 = arith.truncf %68 : vector<1xf32> to vector<1xf16>
        vector.store %69, %41[%workgroup_id_2, %65, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %70 = vector.extract_strided_slice %40#19 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %71 = arith.addi %46, %c51 : index
        %72 = arith.addi %70, %52 : vector<1xi32>
        %73 = arith.sitofp %72 : vector<1xi32> to vector<1xf32>
        %74 = arith.mulf %73, %53 : vector<1xf32>
        %75 = arith.truncf %74 : vector<1xf32> to vector<1xf16>
        vector.store %75, %41[%workgroup_id_2, %71, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %76 = amdgpu.mfma %40#3 * %40#1 + %40#18 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %77 = arith.addi %49, %c32 : index
        %78 = vector.extract_strided_slice %76 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %79 = vector.load %42[%77] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %80 = vector.load %43[%77] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %81 = arith.addi %78, %79 : vector<1xi32>
        %82 = arith.sitofp %81 : vector<1xi32> to vector<1xf32>
        %83 = arith.mulf %82, %80 : vector<1xf32>
        %84 = arith.truncf %83 : vector<1xf32> to vector<1xf16>
        vector.store %84, %41[%workgroup_id_2, %47, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %85 = vector.extract_strided_slice %76 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %86 = arith.addi %85, %79 : vector<1xi32>
        %87 = arith.sitofp %86 : vector<1xi32> to vector<1xf32>
        %88 = arith.mulf %87, %80 : vector<1xf32>
        %89 = arith.truncf %88 : vector<1xf32> to vector<1xf16>
        vector.store %89, %41[%workgroup_id_2, %59, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %90 = vector.extract_strided_slice %76 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %91 = arith.addi %90, %79 : vector<1xi32>
        %92 = arith.sitofp %91 : vector<1xi32> to vector<1xf32>
        %93 = arith.mulf %92, %80 : vector<1xf32>
        %94 = arith.truncf %93 : vector<1xf32> to vector<1xf16>
        vector.store %94, %41[%workgroup_id_2, %65, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %95 = vector.extract_strided_slice %76 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %96 = arith.addi %95, %79 : vector<1xi32>
        %97 = arith.sitofp %96 : vector<1xi32> to vector<1xf32>
        %98 = arith.mulf %97, %80 : vector<1xf32>
        %99 = arith.truncf %98 : vector<1xf32> to vector<1xf16>
        vector.store %99, %41[%workgroup_id_2, %71, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %100 = amdgpu.mfma %40#3 * %40#0 + %40#17 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %101 = arith.addi %49, %c16 : index
        %102 = vector.extract_strided_slice %100 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %103 = vector.load %42[%101] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %104 = vector.load %43[%101] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %105 = arith.addi %102, %103 : vector<1xi32>
        %106 = arith.sitofp %105 : vector<1xi32> to vector<1xf32>
        %107 = arith.mulf %106, %104 : vector<1xf32>
        %108 = arith.truncf %107 : vector<1xf32> to vector<1xf16>
        vector.store %108, %41[%workgroup_id_2, %47, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %109 = vector.extract_strided_slice %100 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %110 = arith.addi %109, %103 : vector<1xi32>
        %111 = arith.sitofp %110 : vector<1xi32> to vector<1xf32>
        %112 = arith.mulf %111, %104 : vector<1xf32>
        %113 = arith.truncf %112 : vector<1xf32> to vector<1xf16>
        vector.store %113, %41[%workgroup_id_2, %59, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %114 = vector.extract_strided_slice %100 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %115 = arith.addi %114, %103 : vector<1xi32>
        %116 = arith.sitofp %115 : vector<1xi32> to vector<1xf32>
        %117 = arith.mulf %116, %104 : vector<1xf32>
        %118 = arith.truncf %117 : vector<1xf32> to vector<1xf16>
        vector.store %118, %41[%workgroup_id_2, %65, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %119 = vector.extract_strided_slice %100 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %120 = arith.addi %119, %103 : vector<1xi32>
        %121 = arith.sitofp %120 : vector<1xi32> to vector<1xf32>
        %122 = arith.mulf %121, %104 : vector<1xf32>
        %123 = arith.truncf %122 : vector<1xf32> to vector<1xf16>
        vector.store %123, %41[%workgroup_id_2, %71, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %124 = vector.load %alloc[%32, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %125 = arith.addi %26, %c32 : index
        %126 = vector.load %alloc_0[%125, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %127 = amdgpu.mfma %40#3 * %124 + %40#16 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %128 = vector.extract_strided_slice %127 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %129 = vector.load %42[%49] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %130 = vector.load %43[%49] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %131 = arith.addi %128, %129 : vector<1xi32>
        %132 = arith.sitofp %131 : vector<1xi32> to vector<1xf32>
        %133 = arith.mulf %132, %130 : vector<1xf32>
        %134 = arith.truncf %133 : vector<1xf32> to vector<1xf16>
        vector.store %134, %41[%workgroup_id_2, %47, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %135 = vector.extract_strided_slice %127 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %136 = arith.addi %135, %129 : vector<1xi32>
        %137 = arith.sitofp %136 : vector<1xi32> to vector<1xf32>
        %138 = arith.mulf %137, %130 : vector<1xf32>
        %139 = arith.truncf %138 : vector<1xf32> to vector<1xf16>
        vector.store %139, %41[%workgroup_id_2, %59, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %140 = vector.extract_strided_slice %127 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %141 = arith.addi %140, %129 : vector<1xi32>
        %142 = arith.sitofp %141 : vector<1xi32> to vector<1xf32>
        %143 = arith.mulf %142, %130 : vector<1xf32>
        %144 = arith.truncf %143 : vector<1xf32> to vector<1xf16>
        vector.store %144, %41[%workgroup_id_2, %65, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %145 = vector.extract_strided_slice %127 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %146 = arith.addi %145, %129 : vector<1xi32>
        %147 = arith.sitofp %146 : vector<1xi32> to vector<1xf32>
        %148 = arith.mulf %147, %130 : vector<1xf32>
        %149 = arith.truncf %148 : vector<1xf32> to vector<1xf16>
        vector.store %149, %41[%workgroup_id_2, %71, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %150 = amdgpu.mfma %126 * %40#2 + %40#15 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %151 = arith.addi %46, %c32 : index
        %152 = vector.extract_strided_slice %150 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %153 = vector.load %42[%50] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %154 = vector.load %43[%50] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %155 = arith.addi %152, %153 : vector<1xi32>
        %156 = arith.sitofp %155 : vector<1xi32> to vector<1xf32>
        %157 = arith.mulf %156, %154 : vector<1xf32>
        %158 = arith.truncf %157 : vector<1xf32> to vector<1xf16>
        vector.store %158, %41[%workgroup_id_2, %151, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %159 = vector.extract_strided_slice %150 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %160 = arith.addi %46, %c33 : index
        %161 = arith.addi %159, %153 : vector<1xi32>
        %162 = arith.sitofp %161 : vector<1xi32> to vector<1xf32>
        %163 = arith.mulf %162, %154 : vector<1xf32>
        %164 = arith.truncf %163 : vector<1xf32> to vector<1xf16>
        vector.store %164, %41[%workgroup_id_2, %160, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %165 = vector.extract_strided_slice %150 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %166 = arith.addi %46, %c34 : index
        %167 = arith.addi %165, %153 : vector<1xi32>
        %168 = arith.sitofp %167 : vector<1xi32> to vector<1xf32>
        %169 = arith.mulf %168, %154 : vector<1xf32>
        %170 = arith.truncf %169 : vector<1xf32> to vector<1xf16>
        vector.store %170, %41[%workgroup_id_2, %166, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %171 = vector.extract_strided_slice %150 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %172 = arith.addi %46, %c35 : index
        %173 = arith.addi %171, %153 : vector<1xi32>
        %174 = arith.sitofp %173 : vector<1xi32> to vector<1xf32>
        %175 = arith.mulf %174, %154 : vector<1xf32>
        %176 = arith.truncf %175 : vector<1xf32> to vector<1xf16>
        vector.store %176, %41[%workgroup_id_2, %172, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %177 = arith.addi %26, %c16 : index
        %178 = vector.load %alloc_0[%177, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %179 = vector.load %alloc_0[%26, %30] : memref<128x36xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %180 = amdgpu.mfma %126 * %40#1 + %40#14 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %181 = vector.extract_strided_slice %180 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %182 = vector.load %42[%77] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %183 = vector.load %43[%77] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %184 = arith.addi %181, %182 : vector<1xi32>
        %185 = arith.sitofp %184 : vector<1xi32> to vector<1xf32>
        %186 = arith.mulf %185, %183 : vector<1xf32>
        %187 = arith.truncf %186 : vector<1xf32> to vector<1xf16>
        vector.store %187, %41[%workgroup_id_2, %151, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %188 = vector.extract_strided_slice %180 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %189 = arith.addi %188, %182 : vector<1xi32>
        %190 = arith.sitofp %189 : vector<1xi32> to vector<1xf32>
        %191 = arith.mulf %190, %183 : vector<1xf32>
        %192 = arith.truncf %191 : vector<1xf32> to vector<1xf16>
        vector.store %192, %41[%workgroup_id_2, %160, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %193 = vector.extract_strided_slice %180 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %194 = arith.addi %193, %182 : vector<1xi32>
        %195 = arith.sitofp %194 : vector<1xi32> to vector<1xf32>
        %196 = arith.mulf %195, %183 : vector<1xf32>
        %197 = arith.truncf %196 : vector<1xf32> to vector<1xf16>
        vector.store %197, %41[%workgroup_id_2, %166, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %198 = vector.extract_strided_slice %180 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %199 = arith.addi %198, %182 : vector<1xi32>
        %200 = arith.sitofp %199 : vector<1xi32> to vector<1xf32>
        %201 = arith.mulf %200, %183 : vector<1xf32>
        %202 = arith.truncf %201 : vector<1xf32> to vector<1xf16>
        vector.store %202, %41[%workgroup_id_2, %172, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %203 = amdgpu.mfma %126 * %40#0 + %40#13 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %204 = vector.extract_strided_slice %203 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %205 = vector.load %42[%101] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %206 = vector.load %43[%101] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %207 = arith.addi %204, %205 : vector<1xi32>
        %208 = arith.sitofp %207 : vector<1xi32> to vector<1xf32>
        %209 = arith.mulf %208, %206 : vector<1xf32>
        %210 = arith.truncf %209 : vector<1xf32> to vector<1xf16>
        vector.store %210, %41[%workgroup_id_2, %151, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %211 = vector.extract_strided_slice %203 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %212 = arith.addi %211, %205 : vector<1xi32>
        %213 = arith.sitofp %212 : vector<1xi32> to vector<1xf32>
        %214 = arith.mulf %213, %206 : vector<1xf32>
        %215 = arith.truncf %214 : vector<1xf32> to vector<1xf16>
        vector.store %215, %41[%workgroup_id_2, %160, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %216 = vector.extract_strided_slice %203 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %217 = arith.addi %216, %205 : vector<1xi32>
        %218 = arith.sitofp %217 : vector<1xi32> to vector<1xf32>
        %219 = arith.mulf %218, %206 : vector<1xf32>
        %220 = arith.truncf %219 : vector<1xf32> to vector<1xf16>
        vector.store %220, %41[%workgroup_id_2, %166, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %221 = vector.extract_strided_slice %203 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %222 = arith.addi %221, %205 : vector<1xi32>
        %223 = arith.sitofp %222 : vector<1xi32> to vector<1xf32>
        %224 = arith.mulf %223, %206 : vector<1xf32>
        %225 = arith.truncf %224 : vector<1xf32> to vector<1xf16>
        vector.store %225, %41[%workgroup_id_2, %172, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %226 = amdgpu.mfma %126 * %124 + %40#12 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %227 = vector.extract_strided_slice %226 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %228 = vector.load %42[%49] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %229 = vector.load %43[%49] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %230 = arith.addi %227, %228 : vector<1xi32>
        %231 = arith.sitofp %230 : vector<1xi32> to vector<1xf32>
        %232 = arith.mulf %231, %229 : vector<1xf32>
        %233 = arith.truncf %232 : vector<1xf32> to vector<1xf16>
        vector.store %233, %41[%workgroup_id_2, %151, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %234 = vector.extract_strided_slice %226 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %235 = arith.addi %234, %228 : vector<1xi32>
        %236 = arith.sitofp %235 : vector<1xi32> to vector<1xf32>
        %237 = arith.mulf %236, %229 : vector<1xf32>
        %238 = arith.truncf %237 : vector<1xf32> to vector<1xf16>
        vector.store %238, %41[%workgroup_id_2, %160, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %239 = vector.extract_strided_slice %226 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %240 = arith.addi %239, %228 : vector<1xi32>
        %241 = arith.sitofp %240 : vector<1xi32> to vector<1xf32>
        %242 = arith.mulf %241, %229 : vector<1xf32>
        %243 = arith.truncf %242 : vector<1xf32> to vector<1xf16>
        vector.store %243, %41[%workgroup_id_2, %166, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %244 = vector.extract_strided_slice %226 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %245 = arith.addi %244, %228 : vector<1xi32>
        %246 = arith.sitofp %245 : vector<1xi32> to vector<1xf32>
        %247 = arith.mulf %246, %229 : vector<1xf32>
        %248 = arith.truncf %247 : vector<1xf32> to vector<1xf16>
        vector.store %248, %41[%workgroup_id_2, %172, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %249 = amdgpu.mfma %178 * %40#2 + %40#11 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %250 = arith.addi %46, %c16 : index
        %251 = vector.extract_strided_slice %249 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %252 = vector.load %42[%50] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %253 = vector.load %43[%50] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %254 = arith.addi %251, %252 : vector<1xi32>
        %255 = arith.sitofp %254 : vector<1xi32> to vector<1xf32>
        %256 = arith.mulf %255, %253 : vector<1xf32>
        %257 = arith.truncf %256 : vector<1xf32> to vector<1xf16>
        vector.store %257, %41[%workgroup_id_2, %250, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %258 = vector.extract_strided_slice %249 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %259 = arith.addi %46, %c17 : index
        %260 = arith.addi %258, %252 : vector<1xi32>
        %261 = arith.sitofp %260 : vector<1xi32> to vector<1xf32>
        %262 = arith.mulf %261, %253 : vector<1xf32>
        %263 = arith.truncf %262 : vector<1xf32> to vector<1xf16>
        vector.store %263, %41[%workgroup_id_2, %259, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %264 = vector.extract_strided_slice %249 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %265 = arith.addi %46, %c18 : index
        %266 = arith.addi %264, %252 : vector<1xi32>
        %267 = arith.sitofp %266 : vector<1xi32> to vector<1xf32>
        %268 = arith.mulf %267, %253 : vector<1xf32>
        %269 = arith.truncf %268 : vector<1xf32> to vector<1xf16>
        vector.store %269, %41[%workgroup_id_2, %265, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %270 = vector.extract_strided_slice %249 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %271 = arith.addi %46, %c19 : index
        %272 = arith.addi %270, %252 : vector<1xi32>
        %273 = arith.sitofp %272 : vector<1xi32> to vector<1xf32>
        %274 = arith.mulf %273, %253 : vector<1xf32>
        %275 = arith.truncf %274 : vector<1xf32> to vector<1xf16>
        vector.store %275, %41[%workgroup_id_2, %271, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %276 = amdgpu.mfma %178 * %40#1 + %40#10 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %277 = vector.extract_strided_slice %276 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %278 = vector.load %42[%77] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %279 = vector.load %43[%77] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %280 = arith.addi %277, %278 : vector<1xi32>
        %281 = arith.sitofp %280 : vector<1xi32> to vector<1xf32>
        %282 = arith.mulf %281, %279 : vector<1xf32>
        %283 = arith.truncf %282 : vector<1xf32> to vector<1xf16>
        vector.store %283, %41[%workgroup_id_2, %250, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %284 = vector.extract_strided_slice %276 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %285 = arith.addi %284, %278 : vector<1xi32>
        %286 = arith.sitofp %285 : vector<1xi32> to vector<1xf32>
        %287 = arith.mulf %286, %279 : vector<1xf32>
        %288 = arith.truncf %287 : vector<1xf32> to vector<1xf16>
        vector.store %288, %41[%workgroup_id_2, %259, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %289 = vector.extract_strided_slice %276 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %290 = arith.addi %289, %278 : vector<1xi32>
        %291 = arith.sitofp %290 : vector<1xi32> to vector<1xf32>
        %292 = arith.mulf %291, %279 : vector<1xf32>
        %293 = arith.truncf %292 : vector<1xf32> to vector<1xf16>
        vector.store %293, %41[%workgroup_id_2, %265, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %294 = vector.extract_strided_slice %276 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %295 = arith.addi %294, %278 : vector<1xi32>
        %296 = arith.sitofp %295 : vector<1xi32> to vector<1xf32>
        %297 = arith.mulf %296, %279 : vector<1xf32>
        %298 = arith.truncf %297 : vector<1xf32> to vector<1xf16>
        vector.store %298, %41[%workgroup_id_2, %271, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %299 = amdgpu.mfma %178 * %40#0 + %40#9 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %300 = vector.extract_strided_slice %299 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %301 = vector.load %42[%101] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %302 = vector.load %43[%101] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %303 = arith.addi %300, %301 : vector<1xi32>
        %304 = arith.sitofp %303 : vector<1xi32> to vector<1xf32>
        %305 = arith.mulf %304, %302 : vector<1xf32>
        %306 = arith.truncf %305 : vector<1xf32> to vector<1xf16>
        vector.store %306, %41[%workgroup_id_2, %250, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %307 = vector.extract_strided_slice %299 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %308 = arith.addi %307, %301 : vector<1xi32>
        %309 = arith.sitofp %308 : vector<1xi32> to vector<1xf32>
        %310 = arith.mulf %309, %302 : vector<1xf32>
        %311 = arith.truncf %310 : vector<1xf32> to vector<1xf16>
        vector.store %311, %41[%workgroup_id_2, %259, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %312 = vector.extract_strided_slice %299 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %313 = arith.addi %312, %301 : vector<1xi32>
        %314 = arith.sitofp %313 : vector<1xi32> to vector<1xf32>
        %315 = arith.mulf %314, %302 : vector<1xf32>
        %316 = arith.truncf %315 : vector<1xf32> to vector<1xf16>
        vector.store %316, %41[%workgroup_id_2, %265, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %317 = vector.extract_strided_slice %299 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %318 = arith.addi %317, %301 : vector<1xi32>
        %319 = arith.sitofp %318 : vector<1xi32> to vector<1xf32>
        %320 = arith.mulf %319, %302 : vector<1xf32>
        %321 = arith.truncf %320 : vector<1xf32> to vector<1xf16>
        vector.store %321, %41[%workgroup_id_2, %271, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %322 = amdgpu.mfma %178 * %124 + %40#8 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %323 = vector.extract_strided_slice %322 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %324 = vector.load %42[%49] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %325 = vector.load %43[%49] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %326 = arith.addi %323, %324 : vector<1xi32>
        %327 = arith.sitofp %326 : vector<1xi32> to vector<1xf32>
        %328 = arith.mulf %327, %325 : vector<1xf32>
        %329 = arith.truncf %328 : vector<1xf32> to vector<1xf16>
        vector.store %329, %41[%workgroup_id_2, %250, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %330 = vector.extract_strided_slice %322 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %331 = arith.addi %330, %324 : vector<1xi32>
        %332 = arith.sitofp %331 : vector<1xi32> to vector<1xf32>
        %333 = arith.mulf %332, %325 : vector<1xf32>
        %334 = arith.truncf %333 : vector<1xf32> to vector<1xf16>
        vector.store %334, %41[%workgroup_id_2, %259, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %335 = vector.extract_strided_slice %322 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %336 = arith.addi %335, %324 : vector<1xi32>
        %337 = arith.sitofp %336 : vector<1xi32> to vector<1xf32>
        %338 = arith.mulf %337, %325 : vector<1xf32>
        %339 = arith.truncf %338 : vector<1xf32> to vector<1xf16>
        vector.store %339, %41[%workgroup_id_2, %265, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %340 = vector.extract_strided_slice %322 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %341 = arith.addi %340, %324 : vector<1xi32>
        %342 = arith.sitofp %341 : vector<1xi32> to vector<1xf32>
        %343 = arith.mulf %342, %325 : vector<1xf32>
        %344 = arith.truncf %343 : vector<1xf32> to vector<1xf16>
        vector.store %344, %41[%workgroup_id_2, %271, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %345 = amdgpu.mfma %179 * %40#2 + %40#7 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %346 = vector.extract_strided_slice %345 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %347 = vector.load %42[%50] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %348 = vector.load %43[%50] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %349 = arith.addi %346, %347 : vector<1xi32>
        %350 = arith.sitofp %349 : vector<1xi32> to vector<1xf32>
        %351 = arith.mulf %350, %348 : vector<1xf32>
        %352 = arith.truncf %351 : vector<1xf32> to vector<1xf16>
        vector.store %352, %41[%workgroup_id_2, %46, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %353 = vector.extract_strided_slice %345 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %354 = arith.addi %46, %c1 : index
        %355 = arith.addi %353, %347 : vector<1xi32>
        %356 = arith.sitofp %355 : vector<1xi32> to vector<1xf32>
        %357 = arith.mulf %356, %348 : vector<1xf32>
        %358 = arith.truncf %357 : vector<1xf32> to vector<1xf16>
        vector.store %358, %41[%workgroup_id_2, %354, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %359 = vector.extract_strided_slice %345 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %360 = arith.addi %46, %c2 : index
        %361 = arith.addi %359, %347 : vector<1xi32>
        %362 = arith.sitofp %361 : vector<1xi32> to vector<1xf32>
        %363 = arith.mulf %362, %348 : vector<1xf32>
        %364 = arith.truncf %363 : vector<1xf32> to vector<1xf16>
        vector.store %364, %41[%workgroup_id_2, %360, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %365 = vector.extract_strided_slice %345 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %366 = arith.addi %46, %c3 : index
        %367 = arith.addi %365, %347 : vector<1xi32>
        %368 = arith.sitofp %367 : vector<1xi32> to vector<1xf32>
        %369 = arith.mulf %368, %348 : vector<1xf32>
        %370 = arith.truncf %369 : vector<1xf32> to vector<1xf16>
        vector.store %370, %41[%workgroup_id_2, %366, %50] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %371 = amdgpu.mfma %179 * %40#1 + %40#6 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %372 = vector.extract_strided_slice %371 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %373 = vector.load %42[%77] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %374 = vector.load %43[%77] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %375 = arith.addi %372, %373 : vector<1xi32>
        %376 = arith.sitofp %375 : vector<1xi32> to vector<1xf32>
        %377 = arith.mulf %376, %374 : vector<1xf32>
        %378 = arith.truncf %377 : vector<1xf32> to vector<1xf16>
        vector.store %378, %41[%workgroup_id_2, %46, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %379 = vector.extract_strided_slice %371 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %380 = arith.addi %379, %373 : vector<1xi32>
        %381 = arith.sitofp %380 : vector<1xi32> to vector<1xf32>
        %382 = arith.mulf %381, %374 : vector<1xf32>
        %383 = arith.truncf %382 : vector<1xf32> to vector<1xf16>
        vector.store %383, %41[%workgroup_id_2, %354, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %384 = vector.extract_strided_slice %371 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %385 = arith.addi %384, %373 : vector<1xi32>
        %386 = arith.sitofp %385 : vector<1xi32> to vector<1xf32>
        %387 = arith.mulf %386, %374 : vector<1xf32>
        %388 = arith.truncf %387 : vector<1xf32> to vector<1xf16>
        vector.store %388, %41[%workgroup_id_2, %360, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %389 = vector.extract_strided_slice %371 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %390 = arith.addi %389, %373 : vector<1xi32>
        %391 = arith.sitofp %390 : vector<1xi32> to vector<1xf32>
        %392 = arith.mulf %391, %374 : vector<1xf32>
        %393 = arith.truncf %392 : vector<1xf32> to vector<1xf16>
        vector.store %393, %41[%workgroup_id_2, %366, %77] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %394 = amdgpu.mfma %179 * %40#0 + %40#5 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %395 = vector.extract_strided_slice %394 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %396 = vector.load %42[%101] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %397 = vector.load %43[%101] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %398 = arith.addi %395, %396 : vector<1xi32>
        %399 = arith.sitofp %398 : vector<1xi32> to vector<1xf32>
        %400 = arith.mulf %399, %397 : vector<1xf32>
        %401 = arith.truncf %400 : vector<1xf32> to vector<1xf16>
        vector.store %401, %41[%workgroup_id_2, %46, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %402 = vector.extract_strided_slice %394 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %403 = arith.addi %402, %396 : vector<1xi32>
        %404 = arith.sitofp %403 : vector<1xi32> to vector<1xf32>
        %405 = arith.mulf %404, %397 : vector<1xf32>
        %406 = arith.truncf %405 : vector<1xf32> to vector<1xf16>
        vector.store %406, %41[%workgroup_id_2, %354, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %407 = vector.extract_strided_slice %394 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %408 = arith.addi %407, %396 : vector<1xi32>
        %409 = arith.sitofp %408 : vector<1xi32> to vector<1xf32>
        %410 = arith.mulf %409, %397 : vector<1xf32>
        %411 = arith.truncf %410 : vector<1xf32> to vector<1xf16>
        vector.store %411, %41[%workgroup_id_2, %360, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %412 = vector.extract_strided_slice %394 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %413 = arith.addi %412, %396 : vector<1xi32>
        %414 = arith.sitofp %413 : vector<1xi32> to vector<1xf32>
        %415 = arith.mulf %414, %397 : vector<1xf32>
        %416 = arith.truncf %415 : vector<1xf32> to vector<1xf16>
        vector.store %416, %41[%workgroup_id_2, %366, %101] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %417 = amdgpu.mfma %179 * %124 + %40#4 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %418 = vector.extract_strided_slice %417 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %419 = vector.load %42[%49] : memref<10240xi32, strided<[1], offset: ?>>, vector<1xi32>
        %420 = vector.load %43[%49] : memref<10240xf32, strided<[1], offset: ?>>, vector<1xf32>
        %421 = arith.addi %418, %419 : vector<1xi32>
        %422 = arith.sitofp %421 : vector<1xi32> to vector<1xf32>
        %423 = arith.mulf %422, %420 : vector<1xf32>
        %424 = arith.truncf %423 : vector<1xf32> to vector<1xf16>
        vector.store %424, %41[%workgroup_id_2, %46, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %425 = vector.extract_strided_slice %417 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %426 = arith.addi %425, %419 : vector<1xi32>
        %427 = arith.sitofp %426 : vector<1xi32> to vector<1xf32>
        %428 = arith.mulf %427, %420 : vector<1xf32>
        %429 = arith.truncf %428 : vector<1xf32> to vector<1xf16>
        vector.store %429, %41[%workgroup_id_2, %354, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %430 = vector.extract_strided_slice %417 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %431 = arith.addi %430, %419 : vector<1xi32>
        %432 = arith.sitofp %431 : vector<1xi32> to vector<1xf32>
        %433 = arith.mulf %432, %420 : vector<1xf32>
        %434 = arith.truncf %433 : vector<1xf32> to vector<1xf16>
        vector.store %434, %41[%workgroup_id_2, %360, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        %435 = vector.extract_strided_slice %417 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %436 = arith.addi %435, %419 : vector<1xi32>
        %437 = arith.sitofp %436 : vector<1xi32> to vector<1xf32>
        %438 = arith.mulf %437, %420 : vector<1xf32>
        %439 = arith.truncf %438 : vector<1xf32> to vector<1xf16>
        vector.store %439, %41[%workgroup_id_2, %366, %49] : memref<2x1024x10240xf16, strided<[10485760, 10240, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

