#translation = #iree_codegen.translation_info<None workgroup_size = [256, 1, 1] subgroup_size = 64>
module {
  flow.executable private @tk_gemm_fused_16x1024x1280x5120_bias0 {
    flow.executable.export public @tk_gemm_fused_16x1024x1280x5120_bias0 workgroups() -> (index, index, index) {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      flow.return %c8, %c16, %c16 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_16x1024x1280x5120_bias0(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: !stream.binding) attributes {translation_info = #translation} {
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
        %cst = arith.constant dense<0> : vector<4xi32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %workgroup_id_2 = stream.dispatch.workgroup.id[2] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<80x264xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<128x264xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>
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
        %13 = vector.load %0[%workgroup_id_2, %10, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %14 = arith.addi %10, %c16 : index
        %15 = vector.load %0[%workgroup_id_2, %14, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %16 = arith.addi %10, %c32 : index
        %17 = vector.load %0[%workgroup_id_2, %16, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %18 = arith.addi %10, %c48 : index
        %19 = vector.load %0[%workgroup_id_2, %18, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %20 = arith.addi %10, %c64 : index
        %21 = vector.load %0[%workgroup_id_2, %20, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %22 = arith.addi %10, %c80 : index
        %23 = vector.load %0[%workgroup_id_2, %22, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %24 = arith.addi %7, %2 : index
        %25 = arith.addi %24, %1 : index
        vector.store %13, %alloc_0[%25, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %26 = arith.addi %25, %c16 : index
        vector.store %15, %alloc_0[%26, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %27 = arith.addi %10, %c96 : index
        %28 = vector.load %0[%workgroup_id_2, %27, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
        %29 = arith.addi %10, %c112 : index
        %30 = vector.load %0[%workgroup_id_2, %29, %12] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
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
        vector.store %17, %alloc_0[%40, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %41 = arith.addi %25, %c48 : index
        vector.store %19, %alloc_0[%41, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %42 = arith.addi %25, %c64 : index
        vector.store %21, %alloc_0[%42, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %43 = arith.addi %25, %c80 : index
        vector.store %23, %alloc_0[%43, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %44 = arith.addi %36, %c32 : index
        %45 = vector.load %31[%44, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %46 = arith.addi %36, %c48 : index
        %47 = vector.load %31[%46, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %48 = arith.addi %36, %c64 : index
        %49 = vector.load %31[%48, %12] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
        %50 = arith.addi %25, %c96 : index
        vector.store %28, %alloc_0[%50, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
        %51 = arith.addi %25, %c112 : index
        vector.store %30, %alloc_0[%51, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
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
        %60 = vector.load %alloc_0[%55, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %61 = arith.muli %thread_id_y, %c80 : index
        %62 = arith.addi %11, %61 : index
        %63 = arith.addi %62, %c64 : index
        %64 = vector.load %alloc[%63, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %65 = arith.addi %58, %c192 : index
        %66 = vector.load %alloc_0[%55, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %67 = vector.load %alloc[%63, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %68 = arith.addi %58, %c160 : index
        %69 = vector.load %alloc_0[%55, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %70 = vector.load %alloc[%63, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %71 = arith.addi %58, %c128 : index
        %72 = vector.load %alloc_0[%55, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %73 = vector.load %alloc[%63, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %74 = arith.addi %58, %c96 : index
        %75 = vector.load %alloc_0[%55, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %76 = vector.load %alloc[%63, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %77 = arith.addi %58, %c64 : index
        %78 = vector.load %alloc_0[%55, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %79 = vector.load %alloc[%63, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %80 = arith.addi %58, %c32 : index
        %81 = vector.load %alloc_0[%55, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %82 = vector.load %alloc[%63, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %83 = vector.load %alloc_0[%55, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %84 = vector.load %alloc[%63, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %85 = amdgpu.mfma %83 * %84 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
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
        %97 = amdgpu.mfma %83 * %95 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
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
        %111 = amdgpu.mfma %83 * %108 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
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
        %127 = amdgpu.mfma %83 * %123 + %cst {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
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
        %140:58 = scf.for %arg6 = %c1 to %c20 step %c1 iter_args(%arg7 = %135, %arg8 = %134, %arg9 = %133, %arg10 = %83, %arg11 = %139, %arg12 = %138, %arg13 = %137, %arg14 = %136, %arg15 = %131, %arg16 = %130, %arg17 = %129, %arg18 = %128, %arg19 = %81, %arg20 = %123, %arg21 = %122, %arg22 = %121, %arg23 = %120, %arg24 = %116, %arg25 = %115, %arg26 = %114, %arg27 = %113, %arg28 = %78, %arg29 = %108, %arg30 = %107, %arg31 = %106, %arg32 = %105, %arg33 = %102, %arg34 = %101, %arg35 = %100, %arg36 = %99, %arg37 = %75, %arg38 = %95, %arg39 = %94, %arg40 = %93, %arg41 = %92, %arg42 = %90, %arg43 = %89, %arg44 = %88, %arg45 = %87, %arg46 = %72, %arg47 = %84, %arg48 = %82, %arg49 = %79, %arg50 = %76, %arg51 = %73, %arg52 = %70, %arg53 = %67, %arg54 = %64, %arg55 = %69, %arg56 = %66, %arg57 = %60, %arg58 = %cst, %arg59 = %cst, %arg60 = %cst, %arg61 = %cst, %arg62 = %cst, %arg63 = %cst, %arg64 = %132) -> (vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>) {
          %529 = arith.muli %arg6, %c256 : index
          %530 = arith.addi %529, %12 : index
          %531 = vector.load %0[%workgroup_id_2, %10, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %532 = vector.load %0[%workgroup_id_2, %14, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %533 = amdgpu.mfma %arg56 * %arg44 + %arg9 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %534 = amdgpu.mfma %arg46 * %arg33 + %arg8 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %535 = amdgpu.mfma %arg28 * %arg22 + %arg7 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %536 = amdgpu.mfma %arg10 * %arg11 + %arg63 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %537 = vector.load %alloc_0[%54, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %538 = vector.load %alloc_0[%54, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %539 = vector.load %alloc_0[%54, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %540 = vector.load %alloc_0[%54, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %541 = vector.load %0[%workgroup_id_2, %16, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %542 = vector.load %0[%workgroup_id_2, %18, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %543 = vector.load %0[%workgroup_id_2, %20, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %544 = vector.load %0[%workgroup_id_2, %22, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %545 = amdgpu.mfma %arg57 * %arg45 + %533 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %546 = amdgpu.mfma %arg55 * %arg34 + %534 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %547 = amdgpu.mfma %arg37 * %arg23 + %535 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %548 = amdgpu.mfma %arg19 * %arg12 + %536 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %549 = vector.load %alloc_0[%54, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %550 = vector.load %alloc_0[%54, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %551 = vector.load %alloc_0[%54, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %552 = vector.load %alloc_0[%54, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %553 = amdgpu.mfma %arg56 * %arg35 + %546 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %554 = amdgpu.mfma %arg46 * %arg24 + %547 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %555 = amdgpu.mfma %arg28 * %arg13 + %548 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %556 = amdgpu.mfma %552 * %arg47 + %arg62 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          vector.store %531, %alloc_0[%25, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %532, %alloc_0[%26, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %557 = vector.load %0[%workgroup_id_2, %27, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %558 = vector.load %0[%workgroup_id_2, %29, %530] : memref<16x1024x5120xi8, strided<[5242880, 5120, 1], offset: ?>>, vector<16xi8>
          %559 = vector.load %31[%36, %530] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %560 = vector.load %31[%38, %530] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %561 = amdgpu.mfma %arg57 * %arg36 + %553 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %562 = amdgpu.mfma %arg55 * %arg25 + %554 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %563 = amdgpu.mfma %arg37 * %arg14 + %555 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %564 = amdgpu.mfma %551 * %arg48 + %556 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %541, %alloc_0[%40, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %542, %alloc_0[%41, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %565 = amdgpu.mfma %arg56 * %arg26 + %562 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %566 = amdgpu.mfma %arg46 * %arg15 + %563 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %567 = amdgpu.mfma %550 * %arg49 + %564 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %568 = amdgpu.mfma %552 * %arg38 + %arg61 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %543, %alloc_0[%42, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %544, %alloc_0[%43, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %569 = vector.load %31[%44, %530] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %570 = vector.load %31[%46, %530] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %571 = vector.load %31[%48, %530] : memref<1280x5120xi8, strided<[5120, 1], offset: ?>>, vector<16xi8>
          %572 = amdgpu.mfma %arg57 * %arg27 + %565 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %573 = amdgpu.mfma %arg55 * %arg16 + %566 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %574 = amdgpu.mfma %549 * %arg50 + %567 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %575 = amdgpu.mfma %551 * %arg39 + %568 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %557, %alloc_0[%50, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %558, %alloc_0[%51, %12] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %576 = amdgpu.mfma %arg56 * %arg17 + %573 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %577 = amdgpu.mfma %540 * %arg51 + %574 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %578 = amdgpu.mfma %550 * %arg40 + %575 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %579 = amdgpu.mfma %552 * %arg29 + %arg60 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %559, %alloc[%25, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %560, %alloc[%26, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %580 = amdgpu.mfma %arg57 * %arg18 + %576 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %581 = amdgpu.mfma %539 * %arg52 + %577 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %582 = amdgpu.mfma %549 * %arg41 + %578 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %583 = amdgpu.mfma %551 * %arg30 + %579 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c2_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %569, %alloc[%40, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %570, %alloc[%41, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          vector.store %571, %alloc[%42, %12] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<16xi8>
          %584 = amdgpu.mfma %538 * %arg53 + %581 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %585 = amdgpu.mfma %540 * %arg42 + %582 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %586 = amdgpu.mfma %550 * %arg31 + %583 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %587 = amdgpu.mfma %552 * %arg20 + %arg59 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %588 = vector.load %alloc_0[%55, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %589 = vector.load %alloc[%63, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %590 = vector.load %alloc_0[%55, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %591 = vector.load %alloc[%63, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %592 = amdgpu.mfma %537 * %arg54 + %584 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %593 = amdgpu.mfma %539 * %arg43 + %585 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %594 = amdgpu.mfma %549 * %arg32 + %586 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %595 = amdgpu.mfma %551 * %arg21 + %587 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %596 = vector.load %alloc_0[%55, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %597 = vector.load %alloc[%63, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %598 = vector.load %alloc_0[%55, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %599 = vector.load %alloc[%63, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %600 = amdgpu.mfma %538 * %arg44 + %593 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %601 = amdgpu.mfma %540 * %arg33 + %594 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %602 = amdgpu.mfma %550 * %arg22 + %595 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %603 = amdgpu.mfma %552 * %arg11 + %arg58 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %604 = vector.load %alloc_0[%55, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %605 = vector.load %alloc[%63, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %606 = vector.load %alloc_0[%55, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %607 = vector.load %alloc[%63, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %608 = amdgpu.mfma %537 * %arg45 + %600 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %609 = amdgpu.mfma %539 * %arg34 + %601 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %610 = amdgpu.mfma %549 * %arg23 + %602 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %611 = amdgpu.mfma %551 * %arg12 + %603 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %612 = vector.load %alloc_0[%55, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %613 = vector.load %alloc[%63, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %614 = vector.load %alloc_0[%55, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %615 = vector.load %alloc[%63, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %616 = amdgpu.mfma %538 * %arg35 + %609 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %617 = amdgpu.mfma %540 * %arg24 + %610 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %618 = amdgpu.mfma %550 * %arg13 + %611 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          %619 = amdgpu.mfma %614 * %615 + %arg64 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %620 = vector.load %alloc[%86, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %621 = vector.load %alloc[%86, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %622 = vector.load %alloc[%86, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %623 = vector.load %alloc[%86, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %624 = amdgpu.mfma %537 * %arg36 + %616 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %625 = amdgpu.mfma %539 * %arg25 + %617 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %626 = amdgpu.mfma %549 * %arg14 + %618 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %627 = amdgpu.mfma %612 * %613 + %619 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %628 = vector.load %alloc[%86, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %629 = vector.load %alloc[%86, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %630 = vector.load %alloc[%86, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %631 = vector.load %alloc[%86, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %632 = amdgpu.mfma %538 * %arg26 + %625 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %633 = amdgpu.mfma %540 * %arg15 + %626 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %634 = amdgpu.mfma %606 * %607 + %627 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %635 = amdgpu.mfma %614 * %631 + %545 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %636 = vector.load %alloc[%98, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %637 = vector.load %alloc[%98, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %638 = vector.load %alloc[%98, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %639 = vector.load %alloc[%98, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %640 = amdgpu.mfma %537 * %arg27 + %632 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %641 = amdgpu.mfma %539 * %arg16 + %633 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %642 = amdgpu.mfma %604 * %605 + %634 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %643 = amdgpu.mfma %612 * %630 + %635 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %644 = vector.load %alloc[%98, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %645 = vector.load %alloc[%98, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %646 = vector.load %alloc[%98, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %647 = vector.load %alloc[%98, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %648 = amdgpu.mfma %538 * %arg17 + %641 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %649 = amdgpu.mfma %598 * %599 + %642 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %650 = amdgpu.mfma %606 * %629 + %643 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %651 = amdgpu.mfma %614 * %647 + %561 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %652 = vector.load %alloc[%112, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %653 = vector.load %alloc[%112, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %654 = vector.load %alloc[%112, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %655 = vector.load %alloc[%112, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %656 = amdgpu.mfma %537 * %arg18 + %648 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %657 = amdgpu.mfma %596 * %597 + %649 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %658 = amdgpu.mfma %604 * %628 + %650 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %659 = amdgpu.mfma %612 * %646 + %651 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %660 = vector.load %alloc[%112, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %661 = vector.load %alloc[%112, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %662 = vector.load %alloc[%112, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %663 = vector.load %alloc[%112, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c3_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %664 = amdgpu.mfma %590 * %591 + %657 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %665 = amdgpu.mfma %598 * %623 + %658 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %666 = amdgpu.mfma %606 * %645 + %659 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %667 = amdgpu.mfma %614 * %663 + %572 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %668 = vector.load %alloc[%62, %59] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %669 = vector.load %alloc[%62, %65] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %670 = vector.load %alloc[%62, %68] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %671 = vector.load %alloc[%62, %71] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          %672 = amdgpu.mfma %588 * %589 + %664 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %673 = amdgpu.mfma %596 * %622 + %665 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %674 = amdgpu.mfma %604 * %644 + %666 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %675 = amdgpu.mfma %612 * %662 + %667 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
          %676 = vector.load %alloc[%62, %74] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %677 = vector.load %alloc[%62, %77] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %678 = vector.load %alloc[%62, %80] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          %679 = vector.load %alloc[%62, %58] : memref<80x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c4_i32, %c0_i32) : (i32, i32, i32) -> ()
          scf.yield %675, %674, %673, %614, %679, %678, %677, %676, %671, %670, %669, %668, %612, %663, %662, %661, %660, %655, %654, %653, %652, %606, %647, %646, %645, %644, %639, %638, %637, %636, %604, %631, %630, %629, %628, %623, %622, %621, %620, %598, %615, %613, %607, %605, %599, %597, %591, %589, %596, %590, %588, %656, %640, %624, %608, %592, %580, %672 : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<8xi8>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
        }
        %141 = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>
        %142 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1280xi32, strided<[1], offset: ?>>
        %143 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<1280xf32, strided<[1], offset: ?>>
        %144 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>
        %145 = arith.muli %57, %c4 : index
        %146 = arith.addi %6, %53 : index
        %147 = arith.addi %146, %145 : index
        %148 = arith.addi %147, %c16 : index
        %149 = arith.addi %11, %33 : index
        %150 = arith.addi %149, %61 : index
        %151 = arith.addi %150, %c64 : index
        %152 = vector.extract_strided_slice %140#57 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %153 = vector.load %142[%151] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %154 = vector.load %143[%151] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %155 = vector.load %144[%workgroup_id_2, %148, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %156 = arith.addi %152, %153 : vector<1xi32>
        %157 = arith.sitofp %156 : vector<1xi32> to vector<1xf32>
        %158 = arith.mulf %157, %154 : vector<1xf32>
        %159 = arith.truncf %158 : vector<1xf32> to vector<1xf16>
        %160 = arith.addf %159, %155 : vector<1xf16>
        vector.store %160, %141[%workgroup_id_2, %148, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %161 = vector.extract_strided_slice %140#57 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %162 = arith.addi %147, %c17 : index
        %163 = vector.load %144[%workgroup_id_2, %162, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %164 = arith.addi %161, %153 : vector<1xi32>
        %165 = arith.sitofp %164 : vector<1xi32> to vector<1xf32>
        %166 = arith.mulf %165, %154 : vector<1xf32>
        %167 = arith.truncf %166 : vector<1xf32> to vector<1xf16>
        %168 = arith.addf %167, %163 : vector<1xf16>
        vector.store %168, %141[%workgroup_id_2, %162, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %169 = vector.extract_strided_slice %140#57 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %170 = arith.addi %147, %c18 : index
        %171 = vector.load %144[%workgroup_id_2, %170, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %172 = arith.addi %169, %153 : vector<1xi32>
        %173 = arith.sitofp %172 : vector<1xi32> to vector<1xf32>
        %174 = arith.mulf %173, %154 : vector<1xf32>
        %175 = arith.truncf %174 : vector<1xf32> to vector<1xf16>
        %176 = arith.addf %175, %171 : vector<1xf16>
        vector.store %176, %141[%workgroup_id_2, %170, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %177 = vector.extract_strided_slice %140#57 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %178 = arith.addi %147, %c19 : index
        %179 = vector.load %144[%workgroup_id_2, %178, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %180 = arith.addi %177, %153 : vector<1xi32>
        %181 = arith.sitofp %180 : vector<1xi32> to vector<1xf32>
        %182 = arith.mulf %181, %154 : vector<1xf32>
        %183 = arith.truncf %182 : vector<1xf32> to vector<1xf16>
        %184 = arith.addf %183, %179 : vector<1xf16>
        vector.store %184, %141[%workgroup_id_2, %178, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %185 = amdgpu.mfma %140#49 * %140#37 + %140#2 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %186 = amdgpu.mfma %140#39 * %140#26 + %140#1 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %187 = amdgpu.mfma %140#21 * %140#15 + %140#0 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %188 = amdgpu.mfma %140#3 * %140#4 + %140#56 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %189 = vector.load %alloc_0[%54, %59] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %190 = vector.load %alloc_0[%54, %65] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %191 = vector.load %alloc_0[%54, %68] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %192 = vector.load %alloc_0[%54, %71] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %193 = amdgpu.mfma %140#50 * %140#38 + %185 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %194 = arith.addi %150, %c48 : index
        %195 = vector.extract_strided_slice %193 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %196 = vector.load %142[%194] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %197 = vector.load %143[%194] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %198 = vector.load %144[%workgroup_id_2, %148, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %199 = arith.addi %195, %196 : vector<1xi32>
        %200 = arith.sitofp %199 : vector<1xi32> to vector<1xf32>
        %201 = arith.mulf %200, %197 : vector<1xf32>
        %202 = arith.truncf %201 : vector<1xf32> to vector<1xf16>
        %203 = arith.addf %202, %198 : vector<1xf16>
        vector.store %203, %141[%workgroup_id_2, %148, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %204 = vector.extract_strided_slice %193 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %205 = vector.load %144[%workgroup_id_2, %162, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %206 = arith.addi %204, %196 : vector<1xi32>
        %207 = arith.sitofp %206 : vector<1xi32> to vector<1xf32>
        %208 = arith.mulf %207, %197 : vector<1xf32>
        %209 = arith.truncf %208 : vector<1xf32> to vector<1xf16>
        %210 = arith.addf %209, %205 : vector<1xf16>
        vector.store %210, %141[%workgroup_id_2, %162, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %211 = vector.extract_strided_slice %193 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %212 = vector.load %144[%workgroup_id_2, %170, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %213 = arith.addi %211, %196 : vector<1xi32>
        %214 = arith.sitofp %213 : vector<1xi32> to vector<1xf32>
        %215 = arith.mulf %214, %197 : vector<1xf32>
        %216 = arith.truncf %215 : vector<1xf32> to vector<1xf16>
        %217 = arith.addf %216, %212 : vector<1xf16>
        vector.store %217, %141[%workgroup_id_2, %170, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %218 = vector.extract_strided_slice %193 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %219 = vector.load %144[%workgroup_id_2, %178, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %220 = arith.addi %218, %196 : vector<1xi32>
        %221 = arith.sitofp %220 : vector<1xi32> to vector<1xf32>
        %222 = arith.mulf %221, %197 : vector<1xf32>
        %223 = arith.truncf %222 : vector<1xf32> to vector<1xf16>
        %224 = arith.addf %223, %219 : vector<1xf16>
        vector.store %224, %141[%workgroup_id_2, %178, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %225 = amdgpu.mfma %140#48 * %140#27 + %186 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %226 = amdgpu.mfma %140#30 * %140#16 + %187 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %227 = amdgpu.mfma %140#12 * %140#5 + %188 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %228 = vector.load %alloc_0[%54, %74] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %229 = vector.load %alloc_0[%54, %77] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %230 = vector.load %alloc_0[%54, %80] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %231 = vector.load %alloc_0[%54, %58] : memref<128x264xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        %232 = amdgpu.mfma %140#49 * %140#28 + %225 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %233 = amdgpu.mfma %140#39 * %140#17 + %226 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %234 = amdgpu.mfma %140#21 * %140#6 + %227 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %235 = amdgpu.mfma %231 * %140#40 + %140#55 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %236 = amdgpu.mfma %140#50 * %140#29 + %232 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %237 = arith.addi %150, %c32 : index
        %238 = vector.extract_strided_slice %236 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %239 = vector.load %142[%237] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %240 = vector.load %143[%237] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %241 = vector.load %144[%workgroup_id_2, %148, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %242 = arith.addi %238, %239 : vector<1xi32>
        %243 = arith.sitofp %242 : vector<1xi32> to vector<1xf32>
        %244 = arith.mulf %243, %240 : vector<1xf32>
        %245 = arith.truncf %244 : vector<1xf32> to vector<1xf16>
        %246 = arith.addf %245, %241 : vector<1xf16>
        vector.store %246, %141[%workgroup_id_2, %148, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %247 = vector.extract_strided_slice %236 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %248 = vector.load %144[%workgroup_id_2, %162, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %249 = arith.addi %247, %239 : vector<1xi32>
        %250 = arith.sitofp %249 : vector<1xi32> to vector<1xf32>
        %251 = arith.mulf %250, %240 : vector<1xf32>
        %252 = arith.truncf %251 : vector<1xf32> to vector<1xf16>
        %253 = arith.addf %252, %248 : vector<1xf16>
        vector.store %253, %141[%workgroup_id_2, %162, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %254 = vector.extract_strided_slice %236 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %255 = vector.load %144[%workgroup_id_2, %170, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %256 = arith.addi %254, %239 : vector<1xi32>
        %257 = arith.sitofp %256 : vector<1xi32> to vector<1xf32>
        %258 = arith.mulf %257, %240 : vector<1xf32>
        %259 = arith.truncf %258 : vector<1xf32> to vector<1xf16>
        %260 = arith.addf %259, %255 : vector<1xf16>
        vector.store %260, %141[%workgroup_id_2, %170, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %261 = vector.extract_strided_slice %236 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %262 = vector.load %144[%workgroup_id_2, %178, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %263 = arith.addi %261, %239 : vector<1xi32>
        %264 = arith.sitofp %263 : vector<1xi32> to vector<1xf32>
        %265 = arith.mulf %264, %240 : vector<1xf32>
        %266 = arith.truncf %265 : vector<1xf32> to vector<1xf16>
        %267 = arith.addf %266, %262 : vector<1xf16>
        vector.store %267, %141[%workgroup_id_2, %178, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %268 = amdgpu.mfma %140#48 * %140#18 + %233 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %269 = amdgpu.mfma %140#30 * %140#7 + %234 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %270 = amdgpu.mfma %230 * %140#41 + %235 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %271 = amdgpu.mfma %140#49 * %140#19 + %268 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %272 = amdgpu.mfma %140#39 * %140#8 + %269 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %273 = amdgpu.mfma %229 * %140#42 + %270 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %274 = amdgpu.mfma %231 * %140#31 + %140#54 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %275 = amdgpu.mfma %140#50 * %140#20 + %271 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %276 = arith.addi %150, %c16 : index
        %277 = vector.extract_strided_slice %275 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %278 = vector.load %142[%276] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %279 = vector.load %143[%276] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %280 = vector.load %144[%workgroup_id_2, %148, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %281 = arith.addi %277, %278 : vector<1xi32>
        %282 = arith.sitofp %281 : vector<1xi32> to vector<1xf32>
        %283 = arith.mulf %282, %279 : vector<1xf32>
        %284 = arith.truncf %283 : vector<1xf32> to vector<1xf16>
        %285 = arith.addf %284, %280 : vector<1xf16>
        vector.store %285, %141[%workgroup_id_2, %148, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %286 = vector.extract_strided_slice %275 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %287 = vector.load %144[%workgroup_id_2, %162, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %288 = arith.addi %286, %278 : vector<1xi32>
        %289 = arith.sitofp %288 : vector<1xi32> to vector<1xf32>
        %290 = arith.mulf %289, %279 : vector<1xf32>
        %291 = arith.truncf %290 : vector<1xf32> to vector<1xf16>
        %292 = arith.addf %291, %287 : vector<1xf16>
        vector.store %292, %141[%workgroup_id_2, %162, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %293 = vector.extract_strided_slice %275 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %294 = vector.load %144[%workgroup_id_2, %170, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %295 = arith.addi %293, %278 : vector<1xi32>
        %296 = arith.sitofp %295 : vector<1xi32> to vector<1xf32>
        %297 = arith.mulf %296, %279 : vector<1xf32>
        %298 = arith.truncf %297 : vector<1xf32> to vector<1xf16>
        %299 = arith.addf %298, %294 : vector<1xf16>
        vector.store %299, %141[%workgroup_id_2, %170, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %300 = vector.extract_strided_slice %275 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %301 = vector.load %144[%workgroup_id_2, %178, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %302 = arith.addi %300, %278 : vector<1xi32>
        %303 = arith.sitofp %302 : vector<1xi32> to vector<1xf32>
        %304 = arith.mulf %303, %279 : vector<1xf32>
        %305 = arith.truncf %304 : vector<1xf32> to vector<1xf16>
        %306 = arith.addf %305, %301 : vector<1xf16>
        vector.store %306, %141[%workgroup_id_2, %178, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %307 = amdgpu.mfma %140#48 * %140#9 + %272 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %308 = amdgpu.mfma %228 * %140#43 + %273 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %309 = amdgpu.mfma %230 * %140#32 + %274 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %310 = amdgpu.mfma %140#49 * %140#10 + %307 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %311 = amdgpu.mfma %192 * %140#44 + %308 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %312 = amdgpu.mfma %229 * %140#33 + %309 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %313 = amdgpu.mfma %231 * %140#22 + %140#53 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %314 = amdgpu.mfma %140#50 * %140#11 + %310 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %315 = vector.extract_strided_slice %314 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %316 = vector.load %142[%150] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %317 = vector.load %143[%150] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %318 = vector.load %144[%workgroup_id_2, %148, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %319 = arith.addi %315, %316 : vector<1xi32>
        %320 = arith.sitofp %319 : vector<1xi32> to vector<1xf32>
        %321 = arith.mulf %320, %317 : vector<1xf32>
        %322 = arith.truncf %321 : vector<1xf32> to vector<1xf16>
        %323 = arith.addf %322, %318 : vector<1xf16>
        vector.store %323, %141[%workgroup_id_2, %148, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %324 = vector.extract_strided_slice %314 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %325 = vector.load %144[%workgroup_id_2, %162, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %326 = arith.addi %324, %316 : vector<1xi32>
        %327 = arith.sitofp %326 : vector<1xi32> to vector<1xf32>
        %328 = arith.mulf %327, %317 : vector<1xf32>
        %329 = arith.truncf %328 : vector<1xf32> to vector<1xf16>
        %330 = arith.addf %329, %325 : vector<1xf16>
        vector.store %330, %141[%workgroup_id_2, %162, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %331 = vector.extract_strided_slice %314 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %332 = vector.load %144[%workgroup_id_2, %170, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %333 = arith.addi %331, %316 : vector<1xi32>
        %334 = arith.sitofp %333 : vector<1xi32> to vector<1xf32>
        %335 = arith.mulf %334, %317 : vector<1xf32>
        %336 = arith.truncf %335 : vector<1xf32> to vector<1xf16>
        %337 = arith.addf %336, %332 : vector<1xf16>
        vector.store %337, %141[%workgroup_id_2, %170, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %338 = vector.extract_strided_slice %314 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %339 = vector.load %144[%workgroup_id_2, %178, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %340 = arith.addi %338, %316 : vector<1xi32>
        %341 = arith.sitofp %340 : vector<1xi32> to vector<1xf32>
        %342 = arith.mulf %341, %317 : vector<1xf32>
        %343 = arith.truncf %342 : vector<1xf32> to vector<1xf16>
        %344 = arith.addf %343, %339 : vector<1xf16>
        vector.store %344, %141[%workgroup_id_2, %178, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %345 = amdgpu.mfma %191 * %140#45 + %311 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %346 = amdgpu.mfma %228 * %140#34 + %312 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %347 = amdgpu.mfma %230 * %140#23 + %313 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %348 = amdgpu.mfma %190 * %140#46 + %345 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %349 = amdgpu.mfma %192 * %140#35 + %346 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %350 = amdgpu.mfma %229 * %140#24 + %347 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %351 = amdgpu.mfma %231 * %140#13 + %140#52 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %352 = amdgpu.mfma %189 * %140#47 + %348 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %353 = vector.extract_strided_slice %352 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %354 = vector.load %142[%151] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %355 = vector.load %143[%151] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %356 = vector.load %144[%workgroup_id_2, %147, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %357 = arith.addi %353, %354 : vector<1xi32>
        %358 = arith.sitofp %357 : vector<1xi32> to vector<1xf32>
        %359 = arith.mulf %358, %355 : vector<1xf32>
        %360 = arith.truncf %359 : vector<1xf32> to vector<1xf16>
        %361 = arith.addf %360, %356 : vector<1xf16>
        vector.store %361, %141[%workgroup_id_2, %147, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %362 = vector.extract_strided_slice %352 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %363 = arith.addi %147, %c1 : index
        %364 = vector.load %144[%workgroup_id_2, %363, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %365 = arith.addi %362, %354 : vector<1xi32>
        %366 = arith.sitofp %365 : vector<1xi32> to vector<1xf32>
        %367 = arith.mulf %366, %355 : vector<1xf32>
        %368 = arith.truncf %367 : vector<1xf32> to vector<1xf16>
        %369 = arith.addf %368, %364 : vector<1xf16>
        vector.store %369, %141[%workgroup_id_2, %363, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %370 = vector.extract_strided_slice %352 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %371 = arith.addi %147, %c2 : index
        %372 = vector.load %144[%workgroup_id_2, %371, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %373 = arith.addi %370, %354 : vector<1xi32>
        %374 = arith.sitofp %373 : vector<1xi32> to vector<1xf32>
        %375 = arith.mulf %374, %355 : vector<1xf32>
        %376 = arith.truncf %375 : vector<1xf32> to vector<1xf16>
        %377 = arith.addf %376, %372 : vector<1xf16>
        vector.store %377, %141[%workgroup_id_2, %371, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %378 = vector.extract_strided_slice %352 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %379 = arith.addi %147, %c3 : index
        %380 = vector.load %144[%workgroup_id_2, %379, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %381 = arith.addi %378, %354 : vector<1xi32>
        %382 = arith.sitofp %381 : vector<1xi32> to vector<1xf32>
        %383 = arith.mulf %382, %355 : vector<1xf32>
        %384 = arith.truncf %383 : vector<1xf32> to vector<1xf16>
        %385 = arith.addf %384, %380 : vector<1xf16>
        vector.store %385, %141[%workgroup_id_2, %379, %151] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %386 = amdgpu.mfma %191 * %140#36 + %349 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %387 = amdgpu.mfma %228 * %140#25 + %350 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %388 = amdgpu.mfma %230 * %140#14 + %351 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %389 = amdgpu.mfma %190 * %140#37 + %386 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %390 = amdgpu.mfma %192 * %140#26 + %387 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %391 = amdgpu.mfma %229 * %140#15 + %388 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %392 = amdgpu.mfma %231 * %140#4 + %140#51 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %393 = amdgpu.mfma %189 * %140#38 + %389 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %394 = vector.extract_strided_slice %393 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %395 = vector.load %142[%194] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %396 = vector.load %143[%194] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %397 = vector.load %144[%workgroup_id_2, %147, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %398 = arith.addi %394, %395 : vector<1xi32>
        %399 = arith.sitofp %398 : vector<1xi32> to vector<1xf32>
        %400 = arith.mulf %399, %396 : vector<1xf32>
        %401 = arith.truncf %400 : vector<1xf32> to vector<1xf16>
        %402 = arith.addf %401, %397 : vector<1xf16>
        vector.store %402, %141[%workgroup_id_2, %147, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %403 = vector.extract_strided_slice %393 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %404 = vector.load %144[%workgroup_id_2, %363, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %405 = arith.addi %403, %395 : vector<1xi32>
        %406 = arith.sitofp %405 : vector<1xi32> to vector<1xf32>
        %407 = arith.mulf %406, %396 : vector<1xf32>
        %408 = arith.truncf %407 : vector<1xf32> to vector<1xf16>
        %409 = arith.addf %408, %404 : vector<1xf16>
        vector.store %409, %141[%workgroup_id_2, %363, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %410 = vector.extract_strided_slice %393 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %411 = vector.load %144[%workgroup_id_2, %371, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %412 = arith.addi %410, %395 : vector<1xi32>
        %413 = arith.sitofp %412 : vector<1xi32> to vector<1xf32>
        %414 = arith.mulf %413, %396 : vector<1xf32>
        %415 = arith.truncf %414 : vector<1xf32> to vector<1xf16>
        %416 = arith.addf %415, %411 : vector<1xf16>
        vector.store %416, %141[%workgroup_id_2, %371, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %417 = vector.extract_strided_slice %393 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %418 = vector.load %144[%workgroup_id_2, %379, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %419 = arith.addi %417, %395 : vector<1xi32>
        %420 = arith.sitofp %419 : vector<1xi32> to vector<1xf32>
        %421 = arith.mulf %420, %396 : vector<1xf32>
        %422 = arith.truncf %421 : vector<1xf32> to vector<1xf16>
        %423 = arith.addf %422, %418 : vector<1xf16>
        vector.store %423, %141[%workgroup_id_2, %379, %194] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %424 = amdgpu.mfma %191 * %140#27 + %390 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %425 = amdgpu.mfma %228 * %140#16 + %391 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %426 = amdgpu.mfma %230 * %140#5 + %392 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %427 = amdgpu.mfma %190 * %140#28 + %424 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %428 = amdgpu.mfma %192 * %140#17 + %425 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %429 = amdgpu.mfma %229 * %140#6 + %426 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %430 = amdgpu.mfma %189 * %140#29 + %427 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %431 = vector.extract_strided_slice %430 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %432 = vector.load %142[%237] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %433 = vector.load %143[%237] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %434 = vector.load %144[%workgroup_id_2, %147, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %435 = arith.addi %431, %432 : vector<1xi32>
        %436 = arith.sitofp %435 : vector<1xi32> to vector<1xf32>
        %437 = arith.mulf %436, %433 : vector<1xf32>
        %438 = arith.truncf %437 : vector<1xf32> to vector<1xf16>
        %439 = arith.addf %438, %434 : vector<1xf16>
        vector.store %439, %141[%workgroup_id_2, %147, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %440 = vector.extract_strided_slice %430 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %441 = vector.load %144[%workgroup_id_2, %363, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %442 = arith.addi %440, %432 : vector<1xi32>
        %443 = arith.sitofp %442 : vector<1xi32> to vector<1xf32>
        %444 = arith.mulf %443, %433 : vector<1xf32>
        %445 = arith.truncf %444 : vector<1xf32> to vector<1xf16>
        %446 = arith.addf %445, %441 : vector<1xf16>
        vector.store %446, %141[%workgroup_id_2, %363, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %447 = vector.extract_strided_slice %430 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %448 = vector.load %144[%workgroup_id_2, %371, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %449 = arith.addi %447, %432 : vector<1xi32>
        %450 = arith.sitofp %449 : vector<1xi32> to vector<1xf32>
        %451 = arith.mulf %450, %433 : vector<1xf32>
        %452 = arith.truncf %451 : vector<1xf32> to vector<1xf16>
        %453 = arith.addf %452, %448 : vector<1xf16>
        vector.store %453, %141[%workgroup_id_2, %371, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %454 = vector.extract_strided_slice %430 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %455 = vector.load %144[%workgroup_id_2, %379, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %456 = arith.addi %454, %432 : vector<1xi32>
        %457 = arith.sitofp %456 : vector<1xi32> to vector<1xf32>
        %458 = arith.mulf %457, %433 : vector<1xf32>
        %459 = arith.truncf %458 : vector<1xf32> to vector<1xf16>
        %460 = arith.addf %459, %455 : vector<1xf16>
        vector.store %460, %141[%workgroup_id_2, %379, %237] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %461 = amdgpu.mfma %191 * %140#18 + %428 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %462 = amdgpu.mfma %228 * %140#7 + %429 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %463 = amdgpu.mfma %190 * %140#19 + %461 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %464 = amdgpu.mfma %192 * %140#8 + %462 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %465 = amdgpu.mfma %189 * %140#20 + %463 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %466 = vector.extract_strided_slice %465 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %467 = vector.load %142[%276] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %468 = vector.load %143[%276] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %469 = vector.load %144[%workgroup_id_2, %147, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %470 = arith.addi %466, %467 : vector<1xi32>
        %471 = arith.sitofp %470 : vector<1xi32> to vector<1xf32>
        %472 = arith.mulf %471, %468 : vector<1xf32>
        %473 = arith.truncf %472 : vector<1xf32> to vector<1xf16>
        %474 = arith.addf %473, %469 : vector<1xf16>
        vector.store %474, %141[%workgroup_id_2, %147, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %475 = vector.extract_strided_slice %465 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %476 = vector.load %144[%workgroup_id_2, %363, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %477 = arith.addi %475, %467 : vector<1xi32>
        %478 = arith.sitofp %477 : vector<1xi32> to vector<1xf32>
        %479 = arith.mulf %478, %468 : vector<1xf32>
        %480 = arith.truncf %479 : vector<1xf32> to vector<1xf16>
        %481 = arith.addf %480, %476 : vector<1xf16>
        vector.store %481, %141[%workgroup_id_2, %363, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %482 = vector.extract_strided_slice %465 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %483 = vector.load %144[%workgroup_id_2, %371, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %484 = arith.addi %482, %467 : vector<1xi32>
        %485 = arith.sitofp %484 : vector<1xi32> to vector<1xf32>
        %486 = arith.mulf %485, %468 : vector<1xf32>
        %487 = arith.truncf %486 : vector<1xf32> to vector<1xf16>
        %488 = arith.addf %487, %483 : vector<1xf16>
        vector.store %488, %141[%workgroup_id_2, %371, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %489 = vector.extract_strided_slice %465 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %490 = vector.load %144[%workgroup_id_2, %379, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %491 = arith.addi %489, %467 : vector<1xi32>
        %492 = arith.sitofp %491 : vector<1xi32> to vector<1xf32>
        %493 = arith.mulf %492, %468 : vector<1xf32>
        %494 = arith.truncf %493 : vector<1xf32> to vector<1xf16>
        %495 = arith.addf %494, %490 : vector<1xf16>
        vector.store %495, %141[%workgroup_id_2, %379, %276] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %496 = amdgpu.mfma %191 * %140#9 + %464 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %497 = amdgpu.mfma %190 * %140#10 + %496 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %498 = amdgpu.mfma %189 * %140#11 + %497 {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
        %499 = vector.extract_strided_slice %498 {offsets = [0], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %500 = vector.load %142[%150] : memref<1280xi32, strided<[1], offset: ?>>, vector<1xi32>
        %501 = vector.load %143[%150] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %502 = vector.load %144[%workgroup_id_2, %147, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %503 = arith.addi %499, %500 : vector<1xi32>
        %504 = arith.sitofp %503 : vector<1xi32> to vector<1xf32>
        %505 = arith.mulf %504, %501 : vector<1xf32>
        %506 = arith.truncf %505 : vector<1xf32> to vector<1xf16>
        %507 = arith.addf %506, %502 : vector<1xf16>
        vector.store %507, %141[%workgroup_id_2, %147, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %508 = vector.extract_strided_slice %498 {offsets = [1], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %509 = vector.load %144[%workgroup_id_2, %363, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %510 = arith.addi %508, %500 : vector<1xi32>
        %511 = arith.sitofp %510 : vector<1xi32> to vector<1xf32>
        %512 = arith.mulf %511, %501 : vector<1xf32>
        %513 = arith.truncf %512 : vector<1xf32> to vector<1xf16>
        %514 = arith.addf %513, %509 : vector<1xf16>
        vector.store %514, %141[%workgroup_id_2, %363, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %515 = vector.extract_strided_slice %498 {offsets = [2], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %516 = vector.load %144[%workgroup_id_2, %371, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %517 = arith.addi %515, %500 : vector<1xi32>
        %518 = arith.sitofp %517 : vector<1xi32> to vector<1xf32>
        %519 = arith.mulf %518, %501 : vector<1xf32>
        %520 = arith.truncf %519 : vector<1xf32> to vector<1xf16>
        %521 = arith.addf %520, %516 : vector<1xf16>
        vector.store %521, %141[%workgroup_id_2, %371, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %522 = vector.extract_strided_slice %498 {offsets = [3], sizes = [1], strides = [1]} : vector<4xi32> to vector<1xi32>
        %523 = vector.load %144[%workgroup_id_2, %379, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        %524 = arith.addi %522, %500 : vector<1xi32>
        %525 = arith.sitofp %524 : vector<1xi32> to vector<1xf32>
        %526 = arith.mulf %525, %501 : vector<1xf32>
        %527 = arith.truncf %526 : vector<1xf32> to vector<1xf16>
        %528 = arith.addf %527, %523 : vector<1xf16>
        vector.store %528, %141[%workgroup_id_2, %379, %150] : memref<16x1024x1280xf16, strided<[1310720, 1280, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

