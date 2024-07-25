#translation = #iree_codegen.translation_info<None workgroup_size = [256, 1, 1] subgroup_size = 64>
module {
  flow.executable private @tk_gemm_fused_4x1024x1280x5120_bias1 {
    flow.executable.export public @tk_gemm_fused_4x1024x1280x5120_bias1 workgroups() -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      flow.return %c8, %c16, %c4 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_4x1024x1280x5120_bias1(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: !stream.binding, %arg6: !stream.binding, %arg7: !stream.binding) attributes {translation_info = #translation} {
        %cst = arith.constant dense<1.270000e+02> : vector<1xf16>
        %cst_0 = arith.constant dense<-1.280000e+02> : vector<1xf16>
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c3_i32 = arith.constant 3 : i32
        %c512_i32 = arith.constant 512 : i32
        %c256_i32 = arith.constant 256 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c2_i32 = arith.constant 2 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c20 = arith.constant 20 : index
        %c1 = arith.constant 1 : index
        %c160 = arith.constant 160 : index
        %c192 = arith.constant 192 : index
        %c224 = arith.constant 224 : index
        %c112 = arith.constant 112 : index
        %c96 = arith.constant 96 : index
        %c80 = arith.constant 80 : index
        %c64 = arith.constant 64 : index
        %c48 = arith.constant 48 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %c8 = arith.constant 8 : index
        %c128 = arith.constant 128 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst_1 = arith.constant dense<0> : vector<4xi32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<80x264xi8, #gpu.address_space<workgroup>>
        %alloc_2 = memref.alloc() : memref<128x264xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c16 : index
        %2 = arith.muli %thread_id_z, %c16 : index
        %3 = arith.muli %workgroup_id_1, %c8 : index
        %4 = arith.addi %3, %workgroup_id_0 : index
        %5 = arith.divsi %4, %c16 : index
        %6 = arith.muli %5, %c128 : index
        %7 = arith.divsi %thread_id_x, %c16 : index
        %8 = arith.addi %7, %6 : index
        %9 = arith.addi %8, %2 : index
        %10 = arith.addi %9, %1 : index
        %11 = arith.remsi %thread_id_x, %c16 : index
        %12 = arith.muli %11, %c16 : index
        %13 = vector.load %0[%workgroup_id_2, %10, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %14 = arith.addi %10, %c16 : index
        %15 = vector.load %0[%workgroup_id_2, %14, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %16 = arith.addi %10, %c32 : index
        %17 = vector.load %0[%workgroup_id_2, %16, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %18 = arith.addi %10, %c48 : index
        %19 = vector.load %0[%workgroup_id_2, %18, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %20 = arith.addi %10, %c64 : index
        %21 = vector.load %0[%workgroup_id_2, %20, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %22 = arith.addi %10, %c80 : index
        %23 = vector.load %0[%workgroup_id_2, %22, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %24 = arith.addi %7, %2 : index
        %25 = arith.addi %24, %1 : index
        vector.store %13, %alloc_2[%25, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %26 = arith.addi %25, %c16 : index
        vector.store %15, %alloc_2[%26, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %27 = arith.addi %10, %c96 : index
        %28 = vector.load %0[%workgroup_id_2, %27, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %29 = arith.addi %10, %c112 : index
        %30 = vector.load %0[%workgroup_id_2, %29, %12] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %31 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<1280x5120xi8, strided<[5120, 1], offset: ?>>
        %32 = arith.remsi %4, %c16 : index
        %33 = arith.muli %32, %c80 : index
        %34 = arith.addi %7, %33 : index
        %35 = arith.addi %34, %2 : index
        %36 = arith.addi %35, %1 : index
        %37 = vector.load %31[%36, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %38 = arith.addi %36, %c16 : index
        %39 = vector.load %31[%38, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %40 = arith.addi %25, %c32 : index
        vector.store %17, %alloc_2[%40, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %41 = arith.addi %25, %c48 : index
        vector.store %19, %alloc_2[%41, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %42 = arith.addi %25, %c64 : index
        vector.store %21, %alloc_2[%42, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %43 = arith.addi %25, %c80 : index
        vector.store %23, %alloc_2[%43, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %44 = arith.addi %36, %c32 : index
        %45 = vector.load %31[%44, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %46 = arith.addi %36, %c48 : index
        %47 = vector.load %31[%46, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %48 = arith.addi %36, %c64 : index
        %49 = vector.load %31[%48, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %50 = arith.addi %25, %c96 : index
        vector.store %28, %alloc_2[%50, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %51 = arith.addi %25, %c112 : index
        vector.store %30, %alloc_2[%51, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %37, %alloc[%25, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %39, %alloc[%26, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %45, %alloc[%40, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %47, %alloc[%41, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        vector.store %49, %alloc[%42, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        amdgpu.lds_barrier
        amdgpu.lds_barrier
        %52 = arith.divsi %thread_id_x, %c64 : index
        %53 = arith.muli %52, %c32 : index
        %54 = arith.addi %11, %53 : index
        %55 = arith.addi %54, %c16 : index
        %56 = arith.remsi %thread_id_x, %c64 : index
        %57 = arith.divsi %56, %c16 : index
        %58 = arith.muli %57, %c8 : index
        %59 = arith.addi %58, %c224 : index
        %60 = vector.load %alloc_2[%55, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %61 = arith.muli %thread_id_y, %c80 : index
        %62 = arith.addi %11, %61 : index
        %63 = arith.addi %62, %c64 : index
        %64 = vector.load %alloc[%63, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %65 = arith.addi %58, %c192 : index
        %66 = vector.load %alloc_2[%55, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %67 = vector.load %alloc[%63, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %68 = arith.addi %58, %c160 : index
        %69 = vector.load %alloc_2[%55, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %70 = vector.load %alloc[%63, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %71 = arith.addi %58, %c128 : index
        %72 = vector.load %alloc_2[%55, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %73 = vector.load %alloc[%63, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %74 = arith.addi %58, %c96 : index
        %75 = vector.load %alloc_2[%55, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %76 = vector.load %alloc[%63, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %77 = arith.addi %58, %c64 : index
        %78 = vector.load %alloc_2[%55, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %79 = vector.load %alloc[%63, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %80 = arith.addi %58, %c32 : index
        %81 = vector.load %alloc_2[%55, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %82 = vector.load %alloc[%63, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %83 = vector.load %alloc_2[%55, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %84 = vector.load %alloc[%63, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %85 = amdgpu.mfma %83 * %84 + %cst_1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %86 = arith.addi %62, %c48 : index
        %87 = vector.load %alloc[%86, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %88 = vector.load %alloc[%86, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %89 = vector.load %alloc[%86, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %90 = vector.load %alloc[%86, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %91 = amdgpu.mfma %81 * %82 + %85 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %92 = vector.load %alloc[%86, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %93 = vector.load %alloc[%86, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %94 = vector.load %alloc[%86, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %95 = vector.load %alloc[%86, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %96 = amdgpu.mfma %78 * %79 + %91 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %97 = amdgpu.mfma %83 * %95 + %cst_1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %98 = arith.addi %62, %c32 : index
        %99 = vector.load %alloc[%98, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %100 = vector.load %alloc[%98, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %101 = vector.load %alloc[%98, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %102 = vector.load %alloc[%98, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %103 = amdgpu.mfma %75 * %76 + %96 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %104 = amdgpu.mfma %81 * %94 + %97 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %105 = vector.load %alloc[%98, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %106 = vector.load %alloc[%98, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %107 = vector.load %alloc[%98, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %108 = vector.load %alloc[%98, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %109 = amdgpu.mfma %72 * %73 + %103 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %110 = amdgpu.mfma %78 * %93 + %104 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %111 = amdgpu.mfma %83 * %108 + %cst_1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %112 = arith.addi %62, %c16 : index
        %113 = vector.load %alloc[%112, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %114 = vector.load %alloc[%112, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %115 = vector.load %alloc[%112, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %116 = vector.load %alloc[%112, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %117 = amdgpu.mfma %69 * %70 + %109 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %118 = amdgpu.mfma %75 * %92 + %110 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %119 = amdgpu.mfma %81 * %107 + %111 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %120 = vector.load %alloc[%112, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %121 = vector.load %alloc[%112, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %122 = vector.load %alloc[%112, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %123 = vector.load %alloc[%112, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %124 = amdgpu.mfma %66 * %67 + %117 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %125 = amdgpu.mfma %72 * %90 + %118 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %126 = amdgpu.mfma %78 * %106 + %119 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %127 = amdgpu.mfma %83 * %123 + %cst_1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %128 = vector.load %alloc[%62, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %129 = vector.load %alloc[%62, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %130 = vector.load %alloc[%62, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %131 = vector.load %alloc[%62, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %132 = amdgpu.mfma %60 * %64 + %124 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %133 = amdgpu.mfma %69 * %89 + %125 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %134 = amdgpu.mfma %75 * %105 + %126 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %135 = amdgpu.mfma %81 * %122 + %127 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %136 = vector.load %alloc[%62, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %137 = vector.load %alloc[%62, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %138 = vector.load %alloc[%62, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %139 = vector.load %alloc[%62, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %140:58 = scf.for %arg8 = %c1 to %c20 step %c1 iter_args(%arg9 = %135, %arg10 = %134, %arg11 = %133, %arg12 = %83, %arg13 = %139, %arg14 = %138, %arg15 = %137, %arg16 = %136, %arg17 = %131, %arg18 = %130, %arg19 = %129, %arg20 = %128, %arg21 = %81, %arg22 = %123, %arg23 = %122, %arg24 = %121, %arg25 = %120, %arg26 = %116, %arg27 = %115, %arg28 = %114, %arg29 = %113, %arg30 = %78, %arg31 = %108, %arg32 = %107, %arg33 = %106, %arg34 = %105, %arg35 = %102, %arg36 = %101, %arg37 = %100, %arg38 = %99, %arg39 = %75, %arg40 = %95, %arg41 = %94, %arg42 = %93, %arg43 = %92, %arg44 = %90, %arg45 = %89, %arg46 = %88, %arg47 = %87, %arg48 = %72, %arg49 = %84, %arg50 = %82, %arg51 = %79, %arg52 = %76, %arg53 = %73, %arg54 = %70, %arg55 = %67, %arg56 = %64, %arg57 = %69, %arg58 = %66, %arg59 = %60, %arg60 = %cst_1, %arg61 = %cst_1, %arg62 = %cst_1, %arg63 = %cst_1, %arg64 = %cst_1, %arg65 = %cst_1, %arg66 = %132) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
          %863 = arith.muli %arg8, %c256 : index
          %864 = arith.addi %863, %12 : index
          %865 = vector.load %0[%workgroup_id_2, %10, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %866 = vector.load %0[%workgroup_id_2, %14, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %867 = amdgpu.mfma %arg58 * %arg46 + %arg11 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %868 = amdgpu.mfma %arg48 * %arg35 + %arg10 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %869 = amdgpu.mfma %arg30 * %arg24 + %arg9 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %870 = amdgpu.mfma %arg12 * %arg13 + %arg65 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %871 = vector.load %alloc_2[%54, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %872 = vector.load %alloc_2[%54, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %873 = vector.load %alloc_2[%54, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %874 = vector.load %alloc_2[%54, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %875 = vector.load %0[%workgroup_id_2, %16, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %876 = vector.load %0[%workgroup_id_2, %18, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %877 = vector.load %0[%workgroup_id_2, %20, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %878 = vector.load %0[%workgroup_id_2, %22, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %879 = amdgpu.mfma %arg59 * %arg47 + %867 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %880 = amdgpu.mfma %arg57 * %arg36 + %868 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %881 = amdgpu.mfma %arg39 * %arg25 + %869 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %882 = amdgpu.mfma %arg21 * %arg14 + %870 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %883 = vector.load %alloc_2[%54, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %884 = vector.load %alloc_2[%54, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %885 = vector.load %alloc_2[%54, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %886 = vector.load %alloc_2[%54, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %887 = amdgpu.mfma %arg58 * %arg37 + %880 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %888 = amdgpu.mfma %arg48 * %arg26 + %881 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %889 = amdgpu.mfma %arg30 * %arg15 + %882 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %890 = amdgpu.mfma %886 * %arg49 + %arg64 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          vector.store %865, %alloc_2[%25, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %866, %alloc_2[%26, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %891 = vector.load %0[%workgroup_id_2, %27, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %892 = vector.load %0[%workgroup_id_2, %29, %864] : memref<4x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %893 = vector.load %31[%36, %864] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %894 = vector.load %31[%38, %864] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %895 = amdgpu.mfma %arg59 * %arg38 + %887 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %896 = amdgpu.mfma %arg57 * %arg27 + %888 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %897 = amdgpu.mfma %arg39 * %arg16 + %889 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %898 = amdgpu.mfma %885 * %arg50 + %890 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %875, %alloc_2[%40, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %876, %alloc_2[%41, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %899 = amdgpu.mfma %arg58 * %arg28 + %896 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %900 = amdgpu.mfma %arg48 * %arg17 + %897 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %901 = amdgpu.mfma %884 * %arg51 + %898 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %902 = amdgpu.mfma %886 * %arg40 + %arg63 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %877, %alloc_2[%42, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %878, %alloc_2[%43, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %903 = vector.load %31[%44, %864] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %904 = vector.load %31[%46, %864] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %905 = vector.load %31[%48, %864] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %906 = amdgpu.mfma %arg59 * %arg29 + %899 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %907 = amdgpu.mfma %arg57 * %arg18 + %900 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %908 = amdgpu.mfma %883 * %arg52 + %901 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %909 = amdgpu.mfma %885 * %arg41 + %902 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %891, %alloc_2[%50, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %892, %alloc_2[%51, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %910 = amdgpu.mfma %arg58 * %arg19 + %907 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %911 = amdgpu.mfma %874 * %arg53 + %908 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %912 = amdgpu.mfma %884 * %arg42 + %909 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %913 = amdgpu.mfma %886 * %arg31 + %arg62 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %893, %alloc[%25, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %894, %alloc[%26, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %914 = amdgpu.mfma %arg59 * %arg20 + %910 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %915 = amdgpu.mfma %873 * %arg54 + %911 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %916 = amdgpu.mfma %883 * %arg43 + %912 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %917 = amdgpu.mfma %885 * %arg32 + %913 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %903, %alloc[%40, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %904, %alloc[%41, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %905, %alloc[%42, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %918 = amdgpu.mfma %872 * %arg55 + %915 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %919 = amdgpu.mfma %874 * %arg44 + %916 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %920 = amdgpu.mfma %884 * %arg33 + %917 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %921 = amdgpu.mfma %886 * %arg22 + %arg61 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %922 = vector.load %alloc_2[%55, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %923 = vector.load %alloc[%63, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %924 = vector.load %alloc_2[%55, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %925 = vector.load %alloc[%63, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %926 = amdgpu.mfma %871 * %arg56 + %918 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %927 = amdgpu.mfma %873 * %arg45 + %919 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %928 = amdgpu.mfma %883 * %arg34 + %920 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %929 = amdgpu.mfma %885 * %arg23 + %921 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %930 = vector.load %alloc_2[%55, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %931 = vector.load %alloc[%63, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %932 = vector.load %alloc_2[%55, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %933 = vector.load %alloc[%63, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %934 = amdgpu.mfma %872 * %arg46 + %927 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %935 = amdgpu.mfma %874 * %arg35 + %928 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %936 = amdgpu.mfma %884 * %arg24 + %929 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %937 = amdgpu.mfma %886 * %arg13 + %arg60 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %938 = vector.load %alloc_2[%55, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %939 = vector.load %alloc[%63, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %940 = vector.load %alloc_2[%55, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %941 = vector.load %alloc[%63, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %942 = amdgpu.mfma %871 * %arg47 + %934 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %943 = amdgpu.mfma %873 * %arg36 + %935 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %944 = amdgpu.mfma %883 * %arg25 + %936 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %945 = amdgpu.mfma %885 * %arg14 + %937 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %946 = vector.load %alloc_2[%55, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %947 = vector.load %alloc[%63, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %948 = vector.load %alloc_2[%55, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %949 = vector.load %alloc[%63, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %950 = amdgpu.mfma %872 * %arg37 + %943 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %951 = amdgpu.mfma %874 * %arg26 + %944 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %952 = amdgpu.mfma %884 * %arg15 + %945 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          %953 = amdgpu.mfma %948 * %949 + %arg66 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %954 = vector.load %alloc[%86, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %955 = vector.load %alloc[%86, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %956 = vector.load %alloc[%86, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %957 = vector.load %alloc[%86, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %958 = amdgpu.mfma %871 * %arg38 + %950 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %959 = amdgpu.mfma %873 * %arg27 + %951 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %960 = amdgpu.mfma %883 * %arg16 + %952 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %961 = amdgpu.mfma %946 * %947 + %953 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %962 = vector.load %alloc[%86, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %963 = vector.load %alloc[%86, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %964 = vector.load %alloc[%86, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %965 = vector.load %alloc[%86, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %966 = amdgpu.mfma %872 * %arg28 + %959 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %967 = amdgpu.mfma %874 * %arg17 + %960 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %968 = amdgpu.mfma %940 * %941 + %961 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %969 = amdgpu.mfma %948 * %965 + %879 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %970 = vector.load %alloc[%98, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %971 = vector.load %alloc[%98, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %972 = vector.load %alloc[%98, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %973 = vector.load %alloc[%98, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %974 = amdgpu.mfma %871 * %arg29 + %966 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %975 = amdgpu.mfma %873 * %arg18 + %967 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %976 = amdgpu.mfma %938 * %939 + %968 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %977 = amdgpu.mfma %946 * %964 + %969 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %978 = vector.load %alloc[%98, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %979 = vector.load %alloc[%98, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %980 = vector.load %alloc[%98, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %981 = vector.load %alloc[%98, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %982 = amdgpu.mfma %872 * %arg19 + %975 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %983 = amdgpu.mfma %932 * %933 + %976 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %984 = amdgpu.mfma %940 * %963 + %977 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %985 = amdgpu.mfma %948 * %981 + %895 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %986 = vector.load %alloc[%112, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %987 = vector.load %alloc[%112, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %988 = vector.load %alloc[%112, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %989 = vector.load %alloc[%112, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %990 = amdgpu.mfma %871 * %arg20 + %982 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %991 = amdgpu.mfma %930 * %931 + %983 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %992 = amdgpu.mfma %938 * %962 + %984 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %993 = amdgpu.mfma %946 * %980 + %985 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %994 = vector.load %alloc[%112, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %995 = vector.load %alloc[%112, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %996 = vector.load %alloc[%112, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %997 = vector.load %alloc[%112, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %998 = amdgpu.mfma %924 * %925 + %991 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %999 = amdgpu.mfma %932 * %957 + %992 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1000 = amdgpu.mfma %940 * %979 + %993 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1001 = amdgpu.mfma %948 * %997 + %906 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1002 = vector.load %alloc[%62, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %1003 = vector.load %alloc[%62, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %1004 = vector.load %alloc[%62, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %1005 = vector.load %alloc[%62, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %1006 = amdgpu.mfma %922 * %923 + %998 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1007 = amdgpu.mfma %930 * %956 + %999 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1008 = amdgpu.mfma %938 * %978 + %1000 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1009 = amdgpu.mfma %946 * %996 + %1001 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %1010 = vector.load %alloc[%62, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %1011 = vector.load %alloc[%62, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %1012 = vector.load %alloc[%62, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %1013 = vector.load %alloc[%62, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          scf.yield %1009, %1008, %1007, %948, %1013, %1012, %1011, %1010, %1005, %1004, %1003, %1002, %946, %997, %996, %995, %994, %989, %988, %987, %986, %940, %981, %980, %979, %978, %973, %972, %971, %970, %938, %965, %964, %963, %962, %957, %956, %955, %954, %932, %949, %947, %941, %939, %933, %931, %925, %923, %930, %924, %922, %990, %974, %958, %942, %926, %914, %1006 : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
        }
        %141 = stream.binding.subspan %arg7[%c0] : !stream.binding -> memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>
        %142 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1280xi32, strided<[1], offset: ?>>
        %143 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<1280xf32, strided<[1], offset: ?>>
        %144 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>
        %145 = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<1280xf16, strided<[1], offset: ?>>
        %146 = stream.binding.subspan %arg6[%c0] : !stream.binding -> memref<f32, strided<[], offset: ?>>
        %147 = arith.muli %57, %c4 : index
        %148 = arith.addi %6, %53 : index
        %149 = arith.addi %148, %147 : index
        %150 = arith.addi %149, %c16 : index
        %151 = arith.addi %11, %33 : index
        %152 = arith.addi %151, %61 : index
        %153 = arith.addi %152, %c64 : index
        %154 = vector.extract_strided_slice %140#57 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %155 = vector.load %142[%153] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %156 = vector.load %143[%153] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %157 = vector.load %144[%workgroup_id_2, %150, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %158 = vector.load %145[%153] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %159 = vector.load %146[] : memref<f32, strided<[], offset: ?>>, vector<1xf32>
        %160 = arith.addi %154, %155 : vector<1xi32>
        %161 = arith.sitofp %160 : vector<1xi32> to vector<1xf32>
        %162 = arith.mulf %161, %156 : vector<1xf32>
        %163 = arith.truncf %162 : vector<1xf32> to vector<1xf16>
        %164 = arith.addf %163, %157 : vector<1xf16>
        %165 = arith.mulf %164, %158 : vector<1xf16>
        %166 = arith.truncf %159 : vector<1xf32> to vector<1xf16>
        %167 = arith.mulf %165, %166 : vector<1xf16>
        %168 = math.roundeven %167 : vector<1xf16>
        %169 = arith.cmpf ult, %168, %cst_0 : vector<1xf16>
        %170 = arith.select %169, %cst_0, %168 : vector<1xi1>, vector<1xf16>
        %171 = arith.cmpf ugt, %170, %cst : vector<1xf16>
        %172 = arith.select %171, %cst, %170 : vector<1xi1>, vector<1xf16>
        %173 = arith.fptosi %172 : vector<1xf16> to vector<1xi8>
        vector.store %173, %141[%workgroup_id_2, %150, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %174 = vector.extract_strided_slice %140#57 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %175 = arith.addi %149, %c17 : index
        %176 = vector.load %144[%workgroup_id_2, %175, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %177 = arith.addi %174, %155 : vector<1xi32>
        %178 = arith.sitofp %177 : vector<1xi32> to vector<1xf32>
        %179 = arith.mulf %178, %156 : vector<1xf32>
        %180 = arith.truncf %179 : vector<1xf32> to vector<1xf16>
        %181 = arith.addf %180, %176 : vector<1xf16>
        %182 = arith.mulf %181, %158 : vector<1xf16>
        %183 = arith.mulf %182, %166 : vector<1xf16>
        %184 = math.roundeven %183 : vector<1xf16>
        %185 = arith.cmpf ult, %184, %cst_0 : vector<1xf16>
        %186 = arith.select %185, %cst_0, %184 : vector<1xi1>, vector<1xf16>
        %187 = arith.cmpf ugt, %186, %cst : vector<1xf16>
        %188 = arith.select %187, %cst, %186 : vector<1xi1>, vector<1xf16>
        %189 = arith.fptosi %188 : vector<1xf16> to vector<1xi8>
        vector.store %189, %141[%workgroup_id_2, %175, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %190 = vector.extract_strided_slice %140#57 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %191 = arith.addi %149, %c18 : index
        %192 = vector.load %144[%workgroup_id_2, %191, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %193 = arith.addi %190, %155 : vector<1xi32>
        %194 = arith.sitofp %193 : vector<1xi32> to vector<1xf32>
        %195 = arith.mulf %194, %156 : vector<1xf32>
        %196 = arith.truncf %195 : vector<1xf32> to vector<1xf16>
        %197 = arith.addf %196, %192 : vector<1xf16>
        %198 = arith.mulf %197, %158 : vector<1xf16>
        %199 = arith.mulf %198, %166 : vector<1xf16>
        %200 = math.roundeven %199 : vector<1xf16>
        %201 = arith.cmpf ult, %200, %cst_0 : vector<1xf16>
        %202 = arith.select %201, %cst_0, %200 : vector<1xi1>, vector<1xf16>
        %203 = arith.cmpf ugt, %202, %cst : vector<1xf16>
        %204 = arith.select %203, %cst, %202 : vector<1xi1>, vector<1xf16>
        %205 = arith.fptosi %204 : vector<1xf16> to vector<1xi8>
        vector.store %205, %141[%workgroup_id_2, %191, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %206 = vector.extract_strided_slice %140#57 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %207 = arith.addi %149, %c19 : index
        %208 = vector.load %144[%workgroup_id_2, %207, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %209 = arith.addi %206, %155 : vector<1xi32>
        %210 = arith.sitofp %209 : vector<1xi32> to vector<1xf32>
        %211 = arith.mulf %210, %156 : vector<1xf32>
        %212 = arith.truncf %211 : vector<1xf32> to vector<1xf16>
        %213 = arith.addf %212, %208 : vector<1xf16>
        %214 = arith.mulf %213, %158 : vector<1xf16>
        %215 = arith.mulf %214, %166 : vector<1xf16>
        %216 = math.roundeven %215 : vector<1xf16>
        %217 = arith.cmpf ult, %216, %cst_0 : vector<1xf16>
        %218 = arith.select %217, %cst_0, %216 : vector<1xi1>, vector<1xf16>
        %219 = arith.cmpf ugt, %218, %cst : vector<1xf16>
        %220 = arith.select %219, %cst, %218 : vector<1xi1>, vector<1xf16>
        %221 = arith.fptosi %220 : vector<1xf16> to vector<1xi8>
        vector.store %221, %141[%workgroup_id_2, %207, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %222 = amdgpu.mfma %140#49 * %140#37 + %140#2 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %223 = amdgpu.mfma %140#39 * %140#26 + %140#1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %224 = amdgpu.mfma %140#21 * %140#15 + %140#0 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %225 = amdgpu.mfma %140#3 * %140#4 + %140#56 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %226 = vector.load %alloc_2[%54, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %227 = vector.load %alloc_2[%54, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %228 = vector.load %alloc_2[%54, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %229 = vector.load %alloc_2[%54, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %230 = amdgpu.mfma %140#50 * %140#38 + %222 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %231 = arith.addi %152, %c48 : index
        %232 = vector.extract_strided_slice %230 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %233 = vector.load %142[%231] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %234 = vector.load %143[%231] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %235 = vector.load %144[%workgroup_id_2, %150, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %236 = vector.load %145[%231] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %237 = arith.addi %232, %233 : vector<1xi32>
        %238 = arith.sitofp %237 : vector<1xi32> to vector<1xf32>
        %239 = arith.mulf %238, %234 : vector<1xf32>
        %240 = arith.truncf %239 : vector<1xf32> to vector<1xf16>
        %241 = arith.addf %240, %235 : vector<1xf16>
        %242 = arith.mulf %241, %236 : vector<1xf16>
        %243 = arith.mulf %242, %166 : vector<1xf16>
        %244 = math.roundeven %243 : vector<1xf16>
        %245 = arith.cmpf ult, %244, %cst_0 : vector<1xf16>
        %246 = arith.select %245, %cst_0, %244 : vector<1xi1>, vector<1xf16>
        %247 = arith.cmpf ugt, %246, %cst : vector<1xf16>
        %248 = arith.select %247, %cst, %246 : vector<1xi1>, vector<1xf16>
        %249 = arith.fptosi %248 : vector<1xf16> to vector<1xi8>
        vector.store %249, %141[%workgroup_id_2, %150, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %250 = vector.extract_strided_slice %230 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %251 = vector.load %144[%workgroup_id_2, %175, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %252 = arith.addi %250, %233 : vector<1xi32>
        %253 = arith.sitofp %252 : vector<1xi32> to vector<1xf32>
        %254 = arith.mulf %253, %234 : vector<1xf32>
        %255 = arith.truncf %254 : vector<1xf32> to vector<1xf16>
        %256 = arith.addf %255, %251 : vector<1xf16>
        %257 = arith.mulf %256, %236 : vector<1xf16>
        %258 = arith.mulf %257, %166 : vector<1xf16>
        %259 = math.roundeven %258 : vector<1xf16>
        %260 = arith.cmpf ult, %259, %cst_0 : vector<1xf16>
        %261 = arith.select %260, %cst_0, %259 : vector<1xi1>, vector<1xf16>
        %262 = arith.cmpf ugt, %261, %cst : vector<1xf16>
        %263 = arith.select %262, %cst, %261 : vector<1xi1>, vector<1xf16>
        %264 = arith.fptosi %263 : vector<1xf16> to vector<1xi8>
        vector.store %264, %141[%workgroup_id_2, %175, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %265 = vector.extract_strided_slice %230 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %266 = vector.load %144[%workgroup_id_2, %191, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %267 = arith.addi %265, %233 : vector<1xi32>
        %268 = arith.sitofp %267 : vector<1xi32> to vector<1xf32>
        %269 = arith.mulf %268, %234 : vector<1xf32>
        %270 = arith.truncf %269 : vector<1xf32> to vector<1xf16>
        %271 = arith.addf %270, %266 : vector<1xf16>
        %272 = arith.mulf %271, %236 : vector<1xf16>
        %273 = arith.mulf %272, %166 : vector<1xf16>
        %274 = math.roundeven %273 : vector<1xf16>
        %275 = arith.cmpf ult, %274, %cst_0 : vector<1xf16>
        %276 = arith.select %275, %cst_0, %274 : vector<1xi1>, vector<1xf16>
        %277 = arith.cmpf ugt, %276, %cst : vector<1xf16>
        %278 = arith.select %277, %cst, %276 : vector<1xi1>, vector<1xf16>
        %279 = arith.fptosi %278 : vector<1xf16> to vector<1xi8>
        vector.store %279, %141[%workgroup_id_2, %191, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %280 = vector.extract_strided_slice %230 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %281 = vector.load %144[%workgroup_id_2, %207, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %282 = arith.addi %280, %233 : vector<1xi32>
        %283 = arith.sitofp %282 : vector<1xi32> to vector<1xf32>
        %284 = arith.mulf %283, %234 : vector<1xf32>
        %285 = arith.truncf %284 : vector<1xf32> to vector<1xf16>
        %286 = arith.addf %285, %281 : vector<1xf16>
        %287 = arith.mulf %286, %236 : vector<1xf16>
        %288 = arith.mulf %287, %166 : vector<1xf16>
        %289 = math.roundeven %288 : vector<1xf16>
        %290 = arith.cmpf ult, %289, %cst_0 : vector<1xf16>
        %291 = arith.select %290, %cst_0, %289 : vector<1xi1>, vector<1xf16>
        %292 = arith.cmpf ugt, %291, %cst : vector<1xf16>
        %293 = arith.select %292, %cst, %291 : vector<1xi1>, vector<1xf16>
        %294 = arith.fptosi %293 : vector<1xf16> to vector<1xi8>
        vector.store %294, %141[%workgroup_id_2, %207, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %295 = amdgpu.mfma %140#48 * %140#27 + %223 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %296 = amdgpu.mfma %140#30 * %140#16 + %224 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %297 = amdgpu.mfma %140#12 * %140#5 + %225 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %298 = vector.load %alloc_2[%54, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %299 = vector.load %alloc_2[%54, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %300 = vector.load %alloc_2[%54, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %301 = vector.load %alloc_2[%54, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %302 = amdgpu.mfma %140#49 * %140#28 + %295 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %303 = amdgpu.mfma %140#39 * %140#17 + %296 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %304 = amdgpu.mfma %140#21 * %140#6 + %297 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %305 = amdgpu.mfma %301 * %140#40 + %140#55 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %306 = amdgpu.mfma %140#50 * %140#29 + %302 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %307 = arith.addi %152, %c32 : index
        %308 = vector.extract_strided_slice %306 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %309 = vector.load %142[%307] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %310 = vector.load %143[%307] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %311 = vector.load %144[%workgroup_id_2, %150, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %312 = vector.load %145[%307] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %313 = arith.addi %308, %309 : vector<1xi32>
        %314 = arith.sitofp %313 : vector<1xi32> to vector<1xf32>
        %315 = arith.mulf %314, %310 : vector<1xf32>
        %316 = arith.truncf %315 : vector<1xf32> to vector<1xf16>
        %317 = arith.addf %316, %311 : vector<1xf16>
        %318 = arith.mulf %317, %312 : vector<1xf16>
        %319 = arith.mulf %318, %166 : vector<1xf16>
        %320 = math.roundeven %319 : vector<1xf16>
        %321 = arith.cmpf ult, %320, %cst_0 : vector<1xf16>
        %322 = arith.select %321, %cst_0, %320 : vector<1xi1>, vector<1xf16>
        %323 = arith.cmpf ugt, %322, %cst : vector<1xf16>
        %324 = arith.select %323, %cst, %322 : vector<1xi1>, vector<1xf16>
        %325 = arith.fptosi %324 : vector<1xf16> to vector<1xi8>
        vector.store %325, %141[%workgroup_id_2, %150, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %326 = vector.extract_strided_slice %306 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %327 = vector.load %144[%workgroup_id_2, %175, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %328 = arith.addi %326, %309 : vector<1xi32>
        %329 = arith.sitofp %328 : vector<1xi32> to vector<1xf32>
        %330 = arith.mulf %329, %310 : vector<1xf32>
        %331 = arith.truncf %330 : vector<1xf32> to vector<1xf16>
        %332 = arith.addf %331, %327 : vector<1xf16>
        %333 = arith.mulf %332, %312 : vector<1xf16>
        %334 = arith.mulf %333, %166 : vector<1xf16>
        %335 = math.roundeven %334 : vector<1xf16>
        %336 = arith.cmpf ult, %335, %cst_0 : vector<1xf16>
        %337 = arith.select %336, %cst_0, %335 : vector<1xi1>, vector<1xf16>
        %338 = arith.cmpf ugt, %337, %cst : vector<1xf16>
        %339 = arith.select %338, %cst, %337 : vector<1xi1>, vector<1xf16>
        %340 = arith.fptosi %339 : vector<1xf16> to vector<1xi8>
        vector.store %340, %141[%workgroup_id_2, %175, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %341 = vector.extract_strided_slice %306 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %342 = vector.load %144[%workgroup_id_2, %191, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %343 = arith.addi %341, %309 : vector<1xi32>
        %344 = arith.sitofp %343 : vector<1xi32> to vector<1xf32>
        %345 = arith.mulf %344, %310 : vector<1xf32>
        %346 = arith.truncf %345 : vector<1xf32> to vector<1xf16>
        %347 = arith.addf %346, %342 : vector<1xf16>
        %348 = arith.mulf %347, %312 : vector<1xf16>
        %349 = arith.mulf %348, %166 : vector<1xf16>
        %350 = math.roundeven %349 : vector<1xf16>
        %351 = arith.cmpf ult, %350, %cst_0 : vector<1xf16>
        %352 = arith.select %351, %cst_0, %350 : vector<1xi1>, vector<1xf16>
        %353 = arith.cmpf ugt, %352, %cst : vector<1xf16>
        %354 = arith.select %353, %cst, %352 : vector<1xi1>, vector<1xf16>
        %355 = arith.fptosi %354 : vector<1xf16> to vector<1xi8>
        vector.store %355, %141[%workgroup_id_2, %191, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %356 = vector.extract_strided_slice %306 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %357 = vector.load %144[%workgroup_id_2, %207, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %358 = arith.addi %356, %309 : vector<1xi32>
        %359 = arith.sitofp %358 : vector<1xi32> to vector<1xf32>
        %360 = arith.mulf %359, %310 : vector<1xf32>
        %361 = arith.truncf %360 : vector<1xf32> to vector<1xf16>
        %362 = arith.addf %361, %357 : vector<1xf16>
        %363 = arith.mulf %362, %312 : vector<1xf16>
        %364 = arith.mulf %363, %166 : vector<1xf16>
        %365 = math.roundeven %364 : vector<1xf16>
        %366 = arith.cmpf ult, %365, %cst_0 : vector<1xf16>
        %367 = arith.select %366, %cst_0, %365 : vector<1xi1>, vector<1xf16>
        %368 = arith.cmpf ugt, %367, %cst : vector<1xf16>
        %369 = arith.select %368, %cst, %367 : vector<1xi1>, vector<1xf16>
        %370 = arith.fptosi %369 : vector<1xf16> to vector<1xi8>
        vector.store %370, %141[%workgroup_id_2, %207, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %371 = amdgpu.mfma %140#48 * %140#18 + %303 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %372 = amdgpu.mfma %140#30 * %140#7 + %304 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %373 = amdgpu.mfma %300 * %140#41 + %305 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %374 = amdgpu.mfma %140#49 * %140#19 + %371 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %375 = amdgpu.mfma %140#39 * %140#8 + %372 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %376 = amdgpu.mfma %299 * %140#42 + %373 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %377 = amdgpu.mfma %301 * %140#31 + %140#54 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %378 = amdgpu.mfma %140#50 * %140#20 + %374 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %379 = arith.addi %152, %c16 : index
        %380 = vector.extract_strided_slice %378 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %381 = vector.load %142[%379] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %382 = vector.load %143[%379] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %383 = vector.load %144[%workgroup_id_2, %150, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %384 = vector.load %145[%379] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %385 = arith.addi %380, %381 : vector<1xi32>
        %386 = arith.sitofp %385 : vector<1xi32> to vector<1xf32>
        %387 = arith.mulf %386, %382 : vector<1xf32>
        %388 = arith.truncf %387 : vector<1xf32> to vector<1xf16>
        %389 = arith.addf %388, %383 : vector<1xf16>
        %390 = arith.mulf %389, %384 : vector<1xf16>
        %391 = arith.mulf %390, %166 : vector<1xf16>
        %392 = math.roundeven %391 : vector<1xf16>
        %393 = arith.cmpf ult, %392, %cst_0 : vector<1xf16>
        %394 = arith.select %393, %cst_0, %392 : vector<1xi1>, vector<1xf16>
        %395 = arith.cmpf ugt, %394, %cst : vector<1xf16>
        %396 = arith.select %395, %cst, %394 : vector<1xi1>, vector<1xf16>
        %397 = arith.fptosi %396 : vector<1xf16> to vector<1xi8>
        vector.store %397, %141[%workgroup_id_2, %150, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %398 = vector.extract_strided_slice %378 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %399 = vector.load %144[%workgroup_id_2, %175, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %400 = arith.addi %398, %381 : vector<1xi32>
        %401 = arith.sitofp %400 : vector<1xi32> to vector<1xf32>
        %402 = arith.mulf %401, %382 : vector<1xf32>
        %403 = arith.truncf %402 : vector<1xf32> to vector<1xf16>
        %404 = arith.addf %403, %399 : vector<1xf16>
        %405 = arith.mulf %404, %384 : vector<1xf16>
        %406 = arith.mulf %405, %166 : vector<1xf16>
        %407 = math.roundeven %406 : vector<1xf16>
        %408 = arith.cmpf ult, %407, %cst_0 : vector<1xf16>
        %409 = arith.select %408, %cst_0, %407 : vector<1xi1>, vector<1xf16>
        %410 = arith.cmpf ugt, %409, %cst : vector<1xf16>
        %411 = arith.select %410, %cst, %409 : vector<1xi1>, vector<1xf16>
        %412 = arith.fptosi %411 : vector<1xf16> to vector<1xi8>
        vector.store %412, %141[%workgroup_id_2, %175, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %413 = vector.extract_strided_slice %378 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %414 = vector.load %144[%workgroup_id_2, %191, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %415 = arith.addi %413, %381 : vector<1xi32>
        %416 = arith.sitofp %415 : vector<1xi32> to vector<1xf32>
        %417 = arith.mulf %416, %382 : vector<1xf32>
        %418 = arith.truncf %417 : vector<1xf32> to vector<1xf16>
        %419 = arith.addf %418, %414 : vector<1xf16>
        %420 = arith.mulf %419, %384 : vector<1xf16>
        %421 = arith.mulf %420, %166 : vector<1xf16>
        %422 = math.roundeven %421 : vector<1xf16>
        %423 = arith.cmpf ult, %422, %cst_0 : vector<1xf16>
        %424 = arith.select %423, %cst_0, %422 : vector<1xi1>, vector<1xf16>
        %425 = arith.cmpf ugt, %424, %cst : vector<1xf16>
        %426 = arith.select %425, %cst, %424 : vector<1xi1>, vector<1xf16>
        %427 = arith.fptosi %426 : vector<1xf16> to vector<1xi8>
        vector.store %427, %141[%workgroup_id_2, %191, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %428 = vector.extract_strided_slice %378 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %429 = vector.load %144[%workgroup_id_2, %207, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %430 = arith.addi %428, %381 : vector<1xi32>
        %431 = arith.sitofp %430 : vector<1xi32> to vector<1xf32>
        %432 = arith.mulf %431, %382 : vector<1xf32>
        %433 = arith.truncf %432 : vector<1xf32> to vector<1xf16>
        %434 = arith.addf %433, %429 : vector<1xf16>
        %435 = arith.mulf %434, %384 : vector<1xf16>
        %436 = arith.mulf %435, %166 : vector<1xf16>
        %437 = math.roundeven %436 : vector<1xf16>
        %438 = arith.cmpf ult, %437, %cst_0 : vector<1xf16>
        %439 = arith.select %438, %cst_0, %437 : vector<1xi1>, vector<1xf16>
        %440 = arith.cmpf ugt, %439, %cst : vector<1xf16>
        %441 = arith.select %440, %cst, %439 : vector<1xi1>, vector<1xf16>
        %442 = arith.fptosi %441 : vector<1xf16> to vector<1xi8>
        vector.store %442, %141[%workgroup_id_2, %207, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %443 = amdgpu.mfma %140#48 * %140#9 + %375 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %444 = amdgpu.mfma %298 * %140#43 + %376 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %445 = amdgpu.mfma %300 * %140#32 + %377 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %446 = amdgpu.mfma %140#49 * %140#10 + %443 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %447 = amdgpu.mfma %229 * %140#44 + %444 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %448 = amdgpu.mfma %299 * %140#33 + %445 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %449 = amdgpu.mfma %301 * %140#22 + %140#53 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %450 = amdgpu.mfma %140#50 * %140#11 + %446 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %451 = vector.extract_strided_slice %450 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %452 = vector.load %142[%152] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %453 = vector.load %143[%152] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %454 = vector.load %144[%workgroup_id_2, %150, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %455 = vector.load %145[%152] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %456 = arith.addi %451, %452 : vector<1xi32>
        %457 = arith.sitofp %456 : vector<1xi32> to vector<1xf32>
        %458 = arith.mulf %457, %453 : vector<1xf32>
        %459 = arith.truncf %458 : vector<1xf32> to vector<1xf16>
        %460 = arith.addf %459, %454 : vector<1xf16>
        %461 = arith.mulf %460, %455 : vector<1xf16>
        %462 = arith.mulf %461, %166 : vector<1xf16>
        %463 = math.roundeven %462 : vector<1xf16>
        %464 = arith.cmpf ult, %463, %cst_0 : vector<1xf16>
        %465 = arith.select %464, %cst_0, %463 : vector<1xi1>, vector<1xf16>
        %466 = arith.cmpf ugt, %465, %cst : vector<1xf16>
        %467 = arith.select %466, %cst, %465 : vector<1xi1>, vector<1xf16>
        %468 = arith.fptosi %467 : vector<1xf16> to vector<1xi8>
        vector.store %468, %141[%workgroup_id_2, %150, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %469 = vector.extract_strided_slice %450 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %470 = vector.load %144[%workgroup_id_2, %175, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %471 = arith.addi %469, %452 : vector<1xi32>
        %472 = arith.sitofp %471 : vector<1xi32> to vector<1xf32>
        %473 = arith.mulf %472, %453 : vector<1xf32>
        %474 = arith.truncf %473 : vector<1xf32> to vector<1xf16>
        %475 = arith.addf %474, %470 : vector<1xf16>
        %476 = arith.mulf %475, %455 : vector<1xf16>
        %477 = arith.mulf %476, %166 : vector<1xf16>
        %478 = math.roundeven %477 : vector<1xf16>
        %479 = arith.cmpf ult, %478, %cst_0 : vector<1xf16>
        %480 = arith.select %479, %cst_0, %478 : vector<1xi1>, vector<1xf16>
        %481 = arith.cmpf ugt, %480, %cst : vector<1xf16>
        %482 = arith.select %481, %cst, %480 : vector<1xi1>, vector<1xf16>
        %483 = arith.fptosi %482 : vector<1xf16> to vector<1xi8>
        vector.store %483, %141[%workgroup_id_2, %175, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %484 = vector.extract_strided_slice %450 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %485 = vector.load %144[%workgroup_id_2, %191, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %486 = arith.addi %484, %452 : vector<1xi32>
        %487 = arith.sitofp %486 : vector<1xi32> to vector<1xf32>
        %488 = arith.mulf %487, %453 : vector<1xf32>
        %489 = arith.truncf %488 : vector<1xf32> to vector<1xf16>
        %490 = arith.addf %489, %485 : vector<1xf16>
        %491 = arith.mulf %490, %455 : vector<1xf16>
        %492 = arith.mulf %491, %166 : vector<1xf16>
        %493 = math.roundeven %492 : vector<1xf16>
        %494 = arith.cmpf ult, %493, %cst_0 : vector<1xf16>
        %495 = arith.select %494, %cst_0, %493 : vector<1xi1>, vector<1xf16>
        %496 = arith.cmpf ugt, %495, %cst : vector<1xf16>
        %497 = arith.select %496, %cst, %495 : vector<1xi1>, vector<1xf16>
        %498 = arith.fptosi %497 : vector<1xf16> to vector<1xi8>
        vector.store %498, %141[%workgroup_id_2, %191, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %499 = vector.extract_strided_slice %450 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %500 = vector.load %144[%workgroup_id_2, %207, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %501 = arith.addi %499, %452 : vector<1xi32>
        %502 = arith.sitofp %501 : vector<1xi32> to vector<1xf32>
        %503 = arith.mulf %502, %453 : vector<1xf32>
        %504 = arith.truncf %503 : vector<1xf32> to vector<1xf16>
        %505 = arith.addf %504, %500 : vector<1xf16>
        %506 = arith.mulf %505, %455 : vector<1xf16>
        %507 = arith.mulf %506, %166 : vector<1xf16>
        %508 = math.roundeven %507 : vector<1xf16>
        %509 = arith.cmpf ult, %508, %cst_0 : vector<1xf16>
        %510 = arith.select %509, %cst_0, %508 : vector<1xi1>, vector<1xf16>
        %511 = arith.cmpf ugt, %510, %cst : vector<1xf16>
        %512 = arith.select %511, %cst, %510 : vector<1xi1>, vector<1xf16>
        %513 = arith.fptosi %512 : vector<1xf16> to vector<1xi8>
        vector.store %513, %141[%workgroup_id_2, %207, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %514 = amdgpu.mfma %228 * %140#45 + %447 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %515 = amdgpu.mfma %298 * %140#34 + %448 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %516 = amdgpu.mfma %300 * %140#23 + %449 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %517 = amdgpu.mfma %227 * %140#46 + %514 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %518 = amdgpu.mfma %229 * %140#35 + %515 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %519 = amdgpu.mfma %299 * %140#24 + %516 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %520 = amdgpu.mfma %301 * %140#13 + %140#52 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %521 = amdgpu.mfma %226 * %140#47 + %517 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %522 = vector.extract_strided_slice %521 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %523 = vector.load %142[%153] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %524 = vector.load %143[%153] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %525 = vector.load %144[%workgroup_id_2, %149, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %526 = vector.load %145[%153] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %527 = arith.addi %522, %523 : vector<1xi32>
        %528 = arith.sitofp %527 : vector<1xi32> to vector<1xf32>
        %529 = arith.mulf %528, %524 : vector<1xf32>
        %530 = arith.truncf %529 : vector<1xf32> to vector<1xf16>
        %531 = arith.addf %530, %525 : vector<1xf16>
        %532 = arith.mulf %531, %526 : vector<1xf16>
        %533 = arith.mulf %532, %166 : vector<1xf16>
        %534 = math.roundeven %533 : vector<1xf16>
        %535 = arith.cmpf ult, %534, %cst_0 : vector<1xf16>
        %536 = arith.select %535, %cst_0, %534 : vector<1xi1>, vector<1xf16>
        %537 = arith.cmpf ugt, %536, %cst : vector<1xf16>
        %538 = arith.select %537, %cst, %536 : vector<1xi1>, vector<1xf16>
        %539 = arith.fptosi %538 : vector<1xf16> to vector<1xi8>
        vector.store %539, %141[%workgroup_id_2, %149, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %540 = vector.extract_strided_slice %521 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %541 = arith.addi %149, %c1 : index
        %542 = vector.load %144[%workgroup_id_2, %541, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %543 = arith.addi %540, %523 : vector<1xi32>
        %544 = arith.sitofp %543 : vector<1xi32> to vector<1xf32>
        %545 = arith.mulf %544, %524 : vector<1xf32>
        %546 = arith.truncf %545 : vector<1xf32> to vector<1xf16>
        %547 = arith.addf %546, %542 : vector<1xf16>
        %548 = arith.mulf %547, %526 : vector<1xf16>
        %549 = arith.mulf %548, %166 : vector<1xf16>
        %550 = math.roundeven %549 : vector<1xf16>
        %551 = arith.cmpf ult, %550, %cst_0 : vector<1xf16>
        %552 = arith.select %551, %cst_0, %550 : vector<1xi1>, vector<1xf16>
        %553 = arith.cmpf ugt, %552, %cst : vector<1xf16>
        %554 = arith.select %553, %cst, %552 : vector<1xi1>, vector<1xf16>
        %555 = arith.fptosi %554 : vector<1xf16> to vector<1xi8>
        vector.store %555, %141[%workgroup_id_2, %541, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %556 = vector.extract_strided_slice %521 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %557 = arith.addi %149, %c2 : index
        %558 = vector.load %144[%workgroup_id_2, %557, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %559 = arith.addi %556, %523 : vector<1xi32>
        %560 = arith.sitofp %559 : vector<1xi32> to vector<1xf32>
        %561 = arith.mulf %560, %524 : vector<1xf32>
        %562 = arith.truncf %561 : vector<1xf32> to vector<1xf16>
        %563 = arith.addf %562, %558 : vector<1xf16>
        %564 = arith.mulf %563, %526 : vector<1xf16>
        %565 = arith.mulf %564, %166 : vector<1xf16>
        %566 = math.roundeven %565 : vector<1xf16>
        %567 = arith.cmpf ult, %566, %cst_0 : vector<1xf16>
        %568 = arith.select %567, %cst_0, %566 : vector<1xi1>, vector<1xf16>
        %569 = arith.cmpf ugt, %568, %cst : vector<1xf16>
        %570 = arith.select %569, %cst, %568 : vector<1xi1>, vector<1xf16>
        %571 = arith.fptosi %570 : vector<1xf16> to vector<1xi8>
        vector.store %571, %141[%workgroup_id_2, %557, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %572 = vector.extract_strided_slice %521 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %573 = arith.addi %149, %c3 : index
        %574 = vector.load %144[%workgroup_id_2, %573, %153] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %575 = arith.addi %572, %523 : vector<1xi32>
        %576 = arith.sitofp %575 : vector<1xi32> to vector<1xf32>
        %577 = arith.mulf %576, %524 : vector<1xf32>
        %578 = arith.truncf %577 : vector<1xf32> to vector<1xf16>
        %579 = arith.addf %578, %574 : vector<1xf16>
        %580 = arith.mulf %579, %526 : vector<1xf16>
        %581 = arith.mulf %580, %166 : vector<1xf16>
        %582 = math.roundeven %581 : vector<1xf16>
        %583 = arith.cmpf ult, %582, %cst_0 : vector<1xf16>
        %584 = arith.select %583, %cst_0, %582 : vector<1xi1>, vector<1xf16>
        %585 = arith.cmpf ugt, %584, %cst : vector<1xf16>
        %586 = arith.select %585, %cst, %584 : vector<1xi1>, vector<1xf16>
        %587 = arith.fptosi %586 : vector<1xf16> to vector<1xi8>
        vector.store %587, %141[%workgroup_id_2, %573, %153] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %588 = amdgpu.mfma %228 * %140#36 + %518 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %589 = amdgpu.mfma %298 * %140#25 + %519 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %590 = amdgpu.mfma %300 * %140#14 + %520 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %591 = amdgpu.mfma %227 * %140#37 + %588 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %592 = amdgpu.mfma %229 * %140#26 + %589 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %593 = amdgpu.mfma %299 * %140#15 + %590 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %594 = amdgpu.mfma %301 * %140#4 + %140#51 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %595 = amdgpu.mfma %226 * %140#38 + %591 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %596 = vector.extract_strided_slice %595 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %597 = vector.load %142[%231] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %598 = vector.load %143[%231] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %599 = vector.load %144[%workgroup_id_2, %149, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %600 = vector.load %145[%231] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %601 = arith.addi %596, %597 : vector<1xi32>
        %602 = arith.sitofp %601 : vector<1xi32> to vector<1xf32>
        %603 = arith.mulf %602, %598 : vector<1xf32>
        %604 = arith.truncf %603 : vector<1xf32> to vector<1xf16>
        %605 = arith.addf %604, %599 : vector<1xf16>
        %606 = arith.mulf %605, %600 : vector<1xf16>
        %607 = arith.mulf %606, %166 : vector<1xf16>
        %608 = math.roundeven %607 : vector<1xf16>
        %609 = arith.cmpf ult, %608, %cst_0 : vector<1xf16>
        %610 = arith.select %609, %cst_0, %608 : vector<1xi1>, vector<1xf16>
        %611 = arith.cmpf ugt, %610, %cst : vector<1xf16>
        %612 = arith.select %611, %cst, %610 : vector<1xi1>, vector<1xf16>
        %613 = arith.fptosi %612 : vector<1xf16> to vector<1xi8>
        vector.store %613, %141[%workgroup_id_2, %149, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %614 = vector.extract_strided_slice %595 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %615 = vector.load %144[%workgroup_id_2, %541, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %616 = arith.addi %614, %597 : vector<1xi32>
        %617 = arith.sitofp %616 : vector<1xi32> to vector<1xf32>
        %618 = arith.mulf %617, %598 : vector<1xf32>
        %619 = arith.truncf %618 : vector<1xf32> to vector<1xf16>
        %620 = arith.addf %619, %615 : vector<1xf16>
        %621 = arith.mulf %620, %600 : vector<1xf16>
        %622 = arith.mulf %621, %166 : vector<1xf16>
        %623 = math.roundeven %622 : vector<1xf16>
        %624 = arith.cmpf ult, %623, %cst_0 : vector<1xf16>
        %625 = arith.select %624, %cst_0, %623 : vector<1xi1>, vector<1xf16>
        %626 = arith.cmpf ugt, %625, %cst : vector<1xf16>
        %627 = arith.select %626, %cst, %625 : vector<1xi1>, vector<1xf16>
        %628 = arith.fptosi %627 : vector<1xf16> to vector<1xi8>
        vector.store %628, %141[%workgroup_id_2, %541, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %629 = vector.extract_strided_slice %595 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %630 = vector.load %144[%workgroup_id_2, %557, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %631 = arith.addi %629, %597 : vector<1xi32>
        %632 = arith.sitofp %631 : vector<1xi32> to vector<1xf32>
        %633 = arith.mulf %632, %598 : vector<1xf32>
        %634 = arith.truncf %633 : vector<1xf32> to vector<1xf16>
        %635 = arith.addf %634, %630 : vector<1xf16>
        %636 = arith.mulf %635, %600 : vector<1xf16>
        %637 = arith.mulf %636, %166 : vector<1xf16>
        %638 = math.roundeven %637 : vector<1xf16>
        %639 = arith.cmpf ult, %638, %cst_0 : vector<1xf16>
        %640 = arith.select %639, %cst_0, %638 : vector<1xi1>, vector<1xf16>
        %641 = arith.cmpf ugt, %640, %cst : vector<1xf16>
        %642 = arith.select %641, %cst, %640 : vector<1xi1>, vector<1xf16>
        %643 = arith.fptosi %642 : vector<1xf16> to vector<1xi8>
        vector.store %643, %141[%workgroup_id_2, %557, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %644 = vector.extract_strided_slice %595 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %645 = vector.load %144[%workgroup_id_2, %573, %231] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %646 = arith.addi %644, %597 : vector<1xi32>
        %647 = arith.sitofp %646 : vector<1xi32> to vector<1xf32>
        %648 = arith.mulf %647, %598 : vector<1xf32>
        %649 = arith.truncf %648 : vector<1xf32> to vector<1xf16>
        %650 = arith.addf %649, %645 : vector<1xf16>
        %651 = arith.mulf %650, %600 : vector<1xf16>
        %652 = arith.mulf %651, %166 : vector<1xf16>
        %653 = math.roundeven %652 : vector<1xf16>
        %654 = arith.cmpf ult, %653, %cst_0 : vector<1xf16>
        %655 = arith.select %654, %cst_0, %653 : vector<1xi1>, vector<1xf16>
        %656 = arith.cmpf ugt, %655, %cst : vector<1xf16>
        %657 = arith.select %656, %cst, %655 : vector<1xi1>, vector<1xf16>
        %658 = arith.fptosi %657 : vector<1xf16> to vector<1xi8>
        vector.store %658, %141[%workgroup_id_2, %573, %231] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %659 = amdgpu.mfma %228 * %140#27 + %592 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %660 = amdgpu.mfma %298 * %140#16 + %593 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %661 = amdgpu.mfma %300 * %140#5 + %594 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %662 = amdgpu.mfma %227 * %140#28 + %659 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %663 = amdgpu.mfma %229 * %140#17 + %660 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %664 = amdgpu.mfma %299 * %140#6 + %661 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %665 = amdgpu.mfma %226 * %140#29 + %662 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %666 = vector.extract_strided_slice %665 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %667 = vector.load %142[%307] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %668 = vector.load %143[%307] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %669 = vector.load %144[%workgroup_id_2, %149, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %670 = vector.load %145[%307] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %671 = arith.addi %666, %667 : vector<1xi32>
        %672 = arith.sitofp %671 : vector<1xi32> to vector<1xf32>
        %673 = arith.mulf %672, %668 : vector<1xf32>
        %674 = arith.truncf %673 : vector<1xf32> to vector<1xf16>
        %675 = arith.addf %674, %669 : vector<1xf16>
        %676 = arith.mulf %675, %670 : vector<1xf16>
        %677 = arith.mulf %676, %166 : vector<1xf16>
        %678 = math.roundeven %677 : vector<1xf16>
        %679 = arith.cmpf ult, %678, %cst_0 : vector<1xf16>
        %680 = arith.select %679, %cst_0, %678 : vector<1xi1>, vector<1xf16>
        %681 = arith.cmpf ugt, %680, %cst : vector<1xf16>
        %682 = arith.select %681, %cst, %680 : vector<1xi1>, vector<1xf16>
        %683 = arith.fptosi %682 : vector<1xf16> to vector<1xi8>
        vector.store %683, %141[%workgroup_id_2, %149, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %684 = vector.extract_strided_slice %665 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %685 = vector.load %144[%workgroup_id_2, %541, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %686 = arith.addi %684, %667 : vector<1xi32>
        %687 = arith.sitofp %686 : vector<1xi32> to vector<1xf32>
        %688 = arith.mulf %687, %668 : vector<1xf32>
        %689 = arith.truncf %688 : vector<1xf32> to vector<1xf16>
        %690 = arith.addf %689, %685 : vector<1xf16>
        %691 = arith.mulf %690, %670 : vector<1xf16>
        %692 = arith.mulf %691, %166 : vector<1xf16>
        %693 = math.roundeven %692 : vector<1xf16>
        %694 = arith.cmpf ult, %693, %cst_0 : vector<1xf16>
        %695 = arith.select %694, %cst_0, %693 : vector<1xi1>, vector<1xf16>
        %696 = arith.cmpf ugt, %695, %cst : vector<1xf16>
        %697 = arith.select %696, %cst, %695 : vector<1xi1>, vector<1xf16>
        %698 = arith.fptosi %697 : vector<1xf16> to vector<1xi8>
        vector.store %698, %141[%workgroup_id_2, %541, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %699 = vector.extract_strided_slice %665 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %700 = vector.load %144[%workgroup_id_2, %557, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %701 = arith.addi %699, %667 : vector<1xi32>
        %702 = arith.sitofp %701 : vector<1xi32> to vector<1xf32>
        %703 = arith.mulf %702, %668 : vector<1xf32>
        %704 = arith.truncf %703 : vector<1xf32> to vector<1xf16>
        %705 = arith.addf %704, %700 : vector<1xf16>
        %706 = arith.mulf %705, %670 : vector<1xf16>
        %707 = arith.mulf %706, %166 : vector<1xf16>
        %708 = math.roundeven %707 : vector<1xf16>
        %709 = arith.cmpf ult, %708, %cst_0 : vector<1xf16>
        %710 = arith.select %709, %cst_0, %708 : vector<1xi1>, vector<1xf16>
        %711 = arith.cmpf ugt, %710, %cst : vector<1xf16>
        %712 = arith.select %711, %cst, %710 : vector<1xi1>, vector<1xf16>
        %713 = arith.fptosi %712 : vector<1xf16> to vector<1xi8>
        vector.store %713, %141[%workgroup_id_2, %557, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %714 = vector.extract_strided_slice %665 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %715 = vector.load %144[%workgroup_id_2, %573, %307] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %716 = arith.addi %714, %667 : vector<1xi32>
        %717 = arith.sitofp %716 : vector<1xi32> to vector<1xf32>
        %718 = arith.mulf %717, %668 : vector<1xf32>
        %719 = arith.truncf %718 : vector<1xf32> to vector<1xf16>
        %720 = arith.addf %719, %715 : vector<1xf16>
        %721 = arith.mulf %720, %670 : vector<1xf16>
        %722 = arith.mulf %721, %166 : vector<1xf16>
        %723 = math.roundeven %722 : vector<1xf16>
        %724 = arith.cmpf ult, %723, %cst_0 : vector<1xf16>
        %725 = arith.select %724, %cst_0, %723 : vector<1xi1>, vector<1xf16>
        %726 = arith.cmpf ugt, %725, %cst : vector<1xf16>
        %727 = arith.select %726, %cst, %725 : vector<1xi1>, vector<1xf16>
        %728 = arith.fptosi %727 : vector<1xf16> to vector<1xi8>
        vector.store %728, %141[%workgroup_id_2, %573, %307] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %729 = amdgpu.mfma %228 * %140#18 + %663 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %730 = amdgpu.mfma %298 * %140#7 + %664 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %731 = amdgpu.mfma %227 * %140#19 + %729 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %732 = amdgpu.mfma %229 * %140#8 + %730 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %733 = amdgpu.mfma %226 * %140#20 + %731 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %734 = vector.extract_strided_slice %733 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %735 = vector.load %142[%379] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %736 = vector.load %143[%379] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %737 = vector.load %144[%workgroup_id_2, %149, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %738 = vector.load %145[%379] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %739 = arith.addi %734, %735 : vector<1xi32>
        %740 = arith.sitofp %739 : vector<1xi32> to vector<1xf32>
        %741 = arith.mulf %740, %736 : vector<1xf32>
        %742 = arith.truncf %741 : vector<1xf32> to vector<1xf16>
        %743 = arith.addf %742, %737 : vector<1xf16>
        %744 = arith.mulf %743, %738 : vector<1xf16>
        %745 = arith.mulf %744, %166 : vector<1xf16>
        %746 = math.roundeven %745 : vector<1xf16>
        %747 = arith.cmpf ult, %746, %cst_0 : vector<1xf16>
        %748 = arith.select %747, %cst_0, %746 : vector<1xi1>, vector<1xf16>
        %749 = arith.cmpf ugt, %748, %cst : vector<1xf16>
        %750 = arith.select %749, %cst, %748 : vector<1xi1>, vector<1xf16>
        %751 = arith.fptosi %750 : vector<1xf16> to vector<1xi8>
        vector.store %751, %141[%workgroup_id_2, %149, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %752 = vector.extract_strided_slice %733 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %753 = vector.load %144[%workgroup_id_2, %541, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %754 = arith.addi %752, %735 : vector<1xi32>
        %755 = arith.sitofp %754 : vector<1xi32> to vector<1xf32>
        %756 = arith.mulf %755, %736 : vector<1xf32>
        %757 = arith.truncf %756 : vector<1xf32> to vector<1xf16>
        %758 = arith.addf %757, %753 : vector<1xf16>
        %759 = arith.mulf %758, %738 : vector<1xf16>
        %760 = arith.mulf %759, %166 : vector<1xf16>
        %761 = math.roundeven %760 : vector<1xf16>
        %762 = arith.cmpf ult, %761, %cst_0 : vector<1xf16>
        %763 = arith.select %762, %cst_0, %761 : vector<1xi1>, vector<1xf16>
        %764 = arith.cmpf ugt, %763, %cst : vector<1xf16>
        %765 = arith.select %764, %cst, %763 : vector<1xi1>, vector<1xf16>
        %766 = arith.fptosi %765 : vector<1xf16> to vector<1xi8>
        vector.store %766, %141[%workgroup_id_2, %541, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %767 = vector.extract_strided_slice %733 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %768 = vector.load %144[%workgroup_id_2, %557, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %769 = arith.addi %767, %735 : vector<1xi32>
        %770 = arith.sitofp %769 : vector<1xi32> to vector<1xf32>
        %771 = arith.mulf %770, %736 : vector<1xf32>
        %772 = arith.truncf %771 : vector<1xf32> to vector<1xf16>
        %773 = arith.addf %772, %768 : vector<1xf16>
        %774 = arith.mulf %773, %738 : vector<1xf16>
        %775 = arith.mulf %774, %166 : vector<1xf16>
        %776 = math.roundeven %775 : vector<1xf16>
        %777 = arith.cmpf ult, %776, %cst_0 : vector<1xf16>
        %778 = arith.select %777, %cst_0, %776 : vector<1xi1>, vector<1xf16>
        %779 = arith.cmpf ugt, %778, %cst : vector<1xf16>
        %780 = arith.select %779, %cst, %778 : vector<1xi1>, vector<1xf16>
        %781 = arith.fptosi %780 : vector<1xf16> to vector<1xi8>
        vector.store %781, %141[%workgroup_id_2, %557, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %782 = vector.extract_strided_slice %733 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %783 = vector.load %144[%workgroup_id_2, %573, %379] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %784 = arith.addi %782, %735 : vector<1xi32>
        %785 = arith.sitofp %784 : vector<1xi32> to vector<1xf32>
        %786 = arith.mulf %785, %736 : vector<1xf32>
        %787 = arith.truncf %786 : vector<1xf32> to vector<1xf16>
        %788 = arith.addf %787, %783 : vector<1xf16>
        %789 = arith.mulf %788, %738 : vector<1xf16>
        %790 = arith.mulf %789, %166 : vector<1xf16>
        %791 = math.roundeven %790 : vector<1xf16>
        %792 = arith.cmpf ult, %791, %cst_0 : vector<1xf16>
        %793 = arith.select %792, %cst_0, %791 : vector<1xi1>, vector<1xf16>
        %794 = arith.cmpf ugt, %793, %cst : vector<1xf16>
        %795 = arith.select %794, %cst, %793 : vector<1xi1>, vector<1xf16>
        %796 = arith.fptosi %795 : vector<1xf16> to vector<1xi8>
        vector.store %796, %141[%workgroup_id_2, %573, %379] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %797 = amdgpu.mfma %228 * %140#9 + %732 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %798 = amdgpu.mfma %227 * %140#10 + %797 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %799 = amdgpu.mfma %226 * %140#11 + %798 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %800 = vector.extract_strided_slice %799 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %801 = vector.load %142[%152] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %802 = vector.load %143[%152] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %803 = vector.load %144[%workgroup_id_2, %149, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %804 = vector.load %145[%152] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %805 = arith.addi %800, %801 : vector<1xi32>
        %806 = arith.sitofp %805 : vector<1xi32> to vector<1xf32>
        %807 = arith.mulf %806, %802 : vector<1xf32>
        %808 = arith.truncf %807 : vector<1xf32> to vector<1xf16>
        %809 = arith.addf %808, %803 : vector<1xf16>
        %810 = arith.mulf %809, %804 : vector<1xf16>
        %811 = arith.mulf %810, %166 : vector<1xf16>
        %812 = math.roundeven %811 : vector<1xf16>
        %813 = arith.cmpf ult, %812, %cst_0 : vector<1xf16>
        %814 = arith.select %813, %cst_0, %812 : vector<1xi1>, vector<1xf16>
        %815 = arith.cmpf ugt, %814, %cst : vector<1xf16>
        %816 = arith.select %815, %cst, %814 : vector<1xi1>, vector<1xf16>
        %817 = arith.fptosi %816 : vector<1xf16> to vector<1xi8>
        vector.store %817, %141[%workgroup_id_2, %149, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %818 = vector.extract_strided_slice %799 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %819 = vector.load %144[%workgroup_id_2, %541, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %820 = arith.addi %818, %801 : vector<1xi32>
        %821 = arith.sitofp %820 : vector<1xi32> to vector<1xf32>
        %822 = arith.mulf %821, %802 : vector<1xf32>
        %823 = arith.truncf %822 : vector<1xf32> to vector<1xf16>
        %824 = arith.addf %823, %819 : vector<1xf16>
        %825 = arith.mulf %824, %804 : vector<1xf16>
        %826 = arith.mulf %825, %166 : vector<1xf16>
        %827 = math.roundeven %826 : vector<1xf16>
        %828 = arith.cmpf ult, %827, %cst_0 : vector<1xf16>
        %829 = arith.select %828, %cst_0, %827 : vector<1xi1>, vector<1xf16>
        %830 = arith.cmpf ugt, %829, %cst : vector<1xf16>
        %831 = arith.select %830, %cst, %829 : vector<1xi1>, vector<1xf16>
        %832 = arith.fptosi %831 : vector<1xf16> to vector<1xi8>
        vector.store %832, %141[%workgroup_id_2, %541, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %833 = vector.extract_strided_slice %799 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %834 = vector.load %144[%workgroup_id_2, %557, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %835 = arith.addi %833, %801 : vector<1xi32>
        %836 = arith.sitofp %835 : vector<1xi32> to vector<1xf32>
        %837 = arith.mulf %836, %802 : vector<1xf32>
        %838 = arith.truncf %837 : vector<1xf32> to vector<1xf16>
        %839 = arith.addf %838, %834 : vector<1xf16>
        %840 = arith.mulf %839, %804 : vector<1xf16>
        %841 = arith.mulf %840, %166 : vector<1xf16>
        %842 = math.roundeven %841 : vector<1xf16>
        %843 = arith.cmpf ult, %842, %cst_0 : vector<1xf16>
        %844 = arith.select %843, %cst_0, %842 : vector<1xi1>, vector<1xf16>
        %845 = arith.cmpf ugt, %844, %cst : vector<1xf16>
        %846 = arith.select %845, %cst, %844 : vector<1xi1>, vector<1xf16>
        %847 = arith.fptosi %846 : vector<1xf16> to vector<1xi8>
        vector.store %847, %141[%workgroup_id_2, %557, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        %848 = vector.extract_strided_slice %799 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %849 = vector.load %144[%workgroup_id_2, %573, %152] : memref<4x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %850 = arith.addi %848, %801 : vector<1xi32>
        %851 = arith.sitofp %850 : vector<1xi32> to vector<1xf32>
        %852 = arith.mulf %851, %802 : vector<1xf32>
        %853 = arith.truncf %852 : vector<1xf32> to vector<1xf16>
        %854 = arith.addf %853, %849 : vector<1xf16>
        %855 = arith.mulf %854, %804 : vector<1xf16>
        %856 = arith.mulf %855, %166 : vector<1xf16>
        %857 = math.roundeven %856 : vector<1xf16>
        %858 = arith.cmpf ult, %857, %cst_0 : vector<1xf16>
        %859 = arith.select %858, %cst_0, %857 : vector<1xi1>, vector<1xf16>
        %860 = arith.cmpf ugt, %859, %cst : vector<1xf16>
        %861 = arith.select %860, %cst, %859 : vector<1xi1>, vector<1xf16>
        %862 = arith.fptosi %861 : vector<1xf16> to vector<1xi8>
        vector.store %862, %141[%workgroup_id_2, %573, %152] : memref<4x1024x1280xi8, strided<[1310720, 1280, 1], offset: ?>>, vector<1xi8>
        return
      }
    }
  }
}

