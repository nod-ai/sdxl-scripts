#translation = #iree_codegen.translation_info<None workgroup_size = [256, 1, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
module {
  flow.executable private @tk_gemm_fused_2048x1280x5120 {
    flow.executable.export public @tk_gemm_fused_2048x1280x5120 workgroups() -> (index, index, index) {
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      flow.return %c16, %c16, %c1 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2048x1280x5120(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c512_i32 = arith.constant 512 : i32
        %c256_i32 = arith.constant 256 : i32
        %c8_i32 = arith.constant 8 : i32
        %c1_i32 = arith.constant 1 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0_i32 = arith.constant 0 : i32
        %c40 = arith.constant 40 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c112 = arith.constant 112 : index
        %c96 = arith.constant 96 : index
        %c80 = arith.constant 80 : index
        %c64 = arith.constant 64 : index
        %c48 = arith.constant 48 : index
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c128 = arith.constant 128 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<80x132xf16, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<128x132xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2048x5120xf16, strided<[5120, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c16 : index
        %2 = arith.muli %thread_id_z, %c16 : index
        %3 = arith.muli %workgroup_id_0, %c128 : index
        %4 = arith.divsi %thread_id_x, %c16 : index
        %5 = arith.addi %4, %3 : index
        %6 = arith.addi %5, %2 : index
        %7 = arith.addi %6, %1 : index
        %8 = arith.remsi %thread_id_x, %c16 : index
        %9 = arith.muli %8, %c8 : index
        %10 = vector.load %0[%7, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %11 = arith.addi %7, %c16 : index
        %12 = vector.load %0[%11, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %13 = arith.addi %7, %c32 : index
        %14 = vector.load %0[%13, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %15 = arith.addi %7, %c48 : index
        %16 = vector.load %0[%15, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %17 = arith.addi %7, %c64 : index
        %18 = vector.load %0[%17, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %19 = arith.addi %7, %c80 : index
        %20 = vector.load %0[%19, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %21 = arith.addi %4, %2 : index
        %22 = arith.addi %21, %1 : index
        vector.store %10, %alloc_0[%22, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %23 = arith.addi %7, %c96 : index
        %24 = vector.load %0[%23, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %25 = arith.addi %7, %c112 : index
        %26 = vector.load %0[%25, %9] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %27 = arith.addi %22, %c16 : index
        vector.store %12, %alloc_0[%27, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %28 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<1280x5120xf16, strided<[5120, 1], offset: ?>>
        %29 = arith.muli %workgroup_id_1, %c80 : index
        %30 = arith.addi %4, %29 : index
        %31 = arith.addi %30, %2 : index
        %32 = arith.addi %31, %1 : index
        %33 = vector.load %28[%32, %9] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %34 = arith.addi %32, %c16 : index
        %35 = vector.load %28[%34, %9] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %36 = arith.addi %22, %c32 : index
        vector.store %14, %alloc_0[%36, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %37 = arith.addi %22, %c48 : index
        vector.store %16, %alloc_0[%37, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %38 = arith.addi %22, %c64 : index
        vector.store %18, %alloc_0[%38, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %39 = arith.addi %32, %c32 : index
        %40 = vector.load %28[%39, %9] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %41 = arith.addi %32, %c48 : index
        %42 = vector.load %28[%41, %9] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %43 = arith.addi %22, %c80 : index
        vector.store %20, %alloc_0[%43, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %44 = arith.addi %22, %c96 : index
        vector.store %24, %alloc_0[%44, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %45 = arith.addi %32, %c64 : index
        %46 = vector.load %28[%45, %9] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
        %47 = arith.addi %22, %c112 : index
        vector.store %26, %alloc_0[%47, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %33, %alloc[%22, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %35, %alloc[%27, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %40, %alloc[%36, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %42, %alloc[%37, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %46, %alloc[%38, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        amdgpu.lds_barrier
        %48 = arith.divsi %thread_id_x, %c64 : index
        %49 = arith.muli %48, %c32 : index
        %50 = arith.addi %8, %49 : index
        %51 = arith.addi %50, %c16 : index
        %52 = arith.remsi %thread_id_x, %c64 : index
        %53 = arith.divsi %52, %c16 : index
        %54 = arith.muli %53, %c4 : index
        %55 = arith.addi %54, %c112 : index
        %56 = vector.load %alloc_0[%51, %55] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %57 = arith.muli %thread_id_y, %c80 : index
        %58 = arith.addi %8, %57 : index
        %59 = arith.addi %58, %c64 : index
        %60 = vector.load %alloc[%59, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %61 = arith.addi %54, %c96 : index
        %62 = vector.load %alloc_0[%51, %61] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %63 = vector.load %alloc[%59, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %64 = arith.addi %54, %c80 : index
        %65 = vector.load %alloc_0[%51, %64] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %66 = vector.load %alloc[%59, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %67 = arith.addi %54, %c64 : index
        %68 = vector.load %alloc_0[%51, %67] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %69 = vector.load %alloc[%59, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %70 = arith.addi %54, %c48 : index
        %71 = vector.load %alloc_0[%51, %70] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %72 = vector.load %alloc[%59, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %73 = arith.addi %54, %c32 : index
        %74 = vector.load %alloc_0[%51, %73] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %75 = vector.load %alloc[%59, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %76 = arith.addi %54, %c16 : index
        %77 = vector.load %alloc_0[%51, %76] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %78 = vector.load %alloc[%59, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %79 = vector.load %alloc_0[%51, %54] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %80 = vector.load %alloc[%59, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %81 = amdgpu.mfma %79 * %80 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %82 = arith.addi %58, %c48 : index
        %83 = vector.load %alloc[%82, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %84 = vector.load %alloc[%82, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %85 = amdgpu.mfma %77 * %78 + %81 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %86 = vector.load %alloc[%82, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %87 = vector.load %alloc[%82, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %88 = amdgpu.mfma %74 * %75 + %85 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %89 = vector.load %alloc[%82, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %90 = vector.load %alloc[%82, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %91 = amdgpu.mfma %71 * %72 + %88 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %92 = vector.load %alloc[%82, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %93 = vector.load %alloc[%82, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %94 = amdgpu.mfma %68 * %69 + %91 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %95 = amdgpu.mfma %79 * %93 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %96 = arith.addi %58, %c32 : index
        %97 = vector.load %alloc[%96, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %98 = vector.load %alloc[%96, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %99 = amdgpu.mfma %65 * %66 + %94 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %100 = amdgpu.mfma %77 * %92 + %95 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %101 = vector.load %alloc[%96, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %102 = vector.load %alloc[%96, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %103 = amdgpu.mfma %62 * %63 + %99 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %104 = amdgpu.mfma %74 * %90 + %100 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %105 = vector.load %alloc[%96, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %106 = vector.load %alloc[%96, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %107 = amdgpu.mfma %56 * %60 + %103 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %108 = amdgpu.mfma %71 * %89 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %109 = vector.load %alloc[%96, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %110 = vector.load %alloc[%96, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %111 = amdgpu.mfma %68 * %87 + %108 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %112 = amdgpu.mfma %79 * %110 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %113 = arith.addi %58, %c16 : index
        %114 = vector.load %alloc[%113, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %115 = vector.load %alloc[%113, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %116 = amdgpu.mfma %65 * %86 + %111 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %117 = amdgpu.mfma %77 * %109 + %112 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %118 = vector.load %alloc[%113, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %119 = vector.load %alloc[%113, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %120 = amdgpu.mfma %62 * %84 + %116 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %121 = amdgpu.mfma %74 * %106 + %117 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %122 = vector.load %alloc[%113, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %123 = vector.load %alloc[%113, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %124 = amdgpu.mfma %56 * %83 + %120 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %125 = amdgpu.mfma %71 * %105 + %121 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %126 = vector.load %alloc[%113, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %127 = vector.load %alloc[%113, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %128 = amdgpu.mfma %68 * %102 + %125 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %129 = amdgpu.mfma %79 * %127 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %130 = vector.load %alloc[%58, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %131 = vector.load %alloc[%58, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %132 = amdgpu.mfma %65 * %101 + %128 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %133 = amdgpu.mfma %77 * %126 + %129 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %134 = vector.load %alloc[%58, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %135 = vector.load %alloc[%58, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %136 = amdgpu.mfma %62 * %98 + %132 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %137 = amdgpu.mfma %74 * %123 + %133 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %138 = vector.load %alloc[%58, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %139 = vector.load %alloc[%58, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %140 = amdgpu.mfma %56 * %97 + %136 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %141 = amdgpu.mfma %71 * %122 + %137 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %142 = vector.load %alloc[%58, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %143 = vector.load %alloc[%58, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %144 = amdgpu.mfma %68 * %119 + %141 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %145 = amdgpu.mfma %79 * %143 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %146 = vector.load %alloc_0[%50, %55] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %147 = vector.load %alloc_0[%50, %61] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %148 = amdgpu.mfma %65 * %118 + %144 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %149 = amdgpu.mfma %77 * %142 + %145 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %150 = vector.load %alloc_0[%50, %64] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %151 = vector.load %alloc_0[%50, %67] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %152:60 = scf.for %arg5 = %c1 to %c40 step %c1 iter_args(%arg6 = %149, %arg7 = %148, %arg8 = %151, %arg9 = %150, %arg10 = %147, %arg11 = %146, %arg12 = %143, %arg13 = %142, %arg14 = %139, %arg15 = %138, %arg16 = %135, %arg17 = %134, %arg18 = %131, %arg19 = %130, %arg20 = %127, %arg21 = %126, %arg22 = %123, %arg23 = %122, %arg24 = %119, %arg25 = %118, %arg26 = %115, %arg27 = %114, %arg28 = %74, %arg29 = %110, %arg30 = %109, %arg31 = %106, %arg32 = %105, %arg33 = %102, %arg34 = %101, %arg35 = %98, %arg36 = %97, %arg37 = %71, %arg38 = %93, %arg39 = %92, %arg40 = %90, %arg41 = %89, %arg42 = %87, %arg43 = %86, %arg44 = %84, %arg45 = %83, %arg46 = %68, %arg47 = %80, %arg48 = %78, %arg49 = %75, %arg50 = %72, %arg51 = %69, %arg52 = %66, %arg53 = %63, %arg54 = %60, %arg55 = %65, %arg56 = %62, %arg57 = %56, %arg58 = %cst, %arg59 = %cst, %arg60 = %cst, %arg61 = %cst, %arg62 = %cst, %arg63 = %140, %arg64 = %124, %arg65 = %107) -> (vector<4xf32>, vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %433 = arith.muli %arg5, %c128 : index
          %434 = arith.addi %433, %9 : index
          %435 = vector.load %0[%7, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %436 = vector.load %0[%11, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %437 = amdgpu.mfma %arg56 * %arg26 + %arg7 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %438 = amdgpu.mfma %arg28 * %arg14 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %439 = vector.load %alloc_0[%50, %70] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %440 = vector.load %alloc_0[%50, %73] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %441 = vector.load %0[%13, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %442 = vector.load %0[%15, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %443 = amdgpu.mfma %arg57 * %arg27 + %437 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %444 = amdgpu.mfma %arg37 * %arg15 + %438 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %445 = vector.load %alloc_0[%50, %76] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %446 = vector.load %alloc_0[%50, %54] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %447 = vector.load %0[%17, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %448 = vector.load %0[%19, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %449 = amdgpu.mfma %arg46 * %arg16 + %444 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %450 = amdgpu.mfma %446 * %arg47 + %arg62 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %451 = amdgpu.mfma %446 * %arg38 + %arg61 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %452 = amdgpu.mfma %446 * %arg29 + %arg60 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          vector.store %435, %alloc_0[%22, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %453 = vector.load %0[%23, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %454 = vector.load %0[%25, %434] : memref<2048x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %455 = amdgpu.mfma %arg55 * %arg17 + %449 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %456 = amdgpu.mfma %445 * %arg48 + %450 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %457 = amdgpu.mfma %445 * %arg39 + %451 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %458 = amdgpu.mfma %445 * %arg30 + %452 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %436, %alloc_0[%27, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %459 = vector.load %28[%32, %434] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %460 = vector.load %28[%34, %434] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %461 = amdgpu.mfma %arg56 * %arg18 + %455 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %462 = amdgpu.mfma %440 * %arg49 + %456 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %463 = amdgpu.mfma %440 * %arg40 + %457 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %464 = amdgpu.mfma %440 * %arg31 + %458 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %441, %alloc_0[%36, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %442, %alloc_0[%37, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %447, %alloc_0[%38, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %465 = vector.load %28[%39, %434] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %466 = vector.load %28[%41, %434] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %467 = amdgpu.mfma %arg57 * %arg19 + %461 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %468 = amdgpu.mfma %439 * %arg50 + %462 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %469 = amdgpu.mfma %439 * %arg41 + %463 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %470 = amdgpu.mfma %439 * %arg32 + %464 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %448, %alloc_0[%43, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %453, %alloc_0[%44, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %471 = vector.load %28[%45, %434] : memref<1280x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
          %472 = amdgpu.mfma %arg8 * %arg51 + %468 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %473 = amdgpu.mfma %arg8 * %arg42 + %469 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %474 = amdgpu.mfma %arg8 * %arg33 + %470 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %475 = amdgpu.mfma %446 * %arg20 + %arg59 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c32_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %454, %alloc_0[%47, %9] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %459, %alloc[%22, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %460, %alloc[%27, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %476 = amdgpu.mfma %arg9 * %arg52 + %472 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %477 = amdgpu.mfma %arg9 * %arg43 + %473 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %478 = amdgpu.mfma %arg9 * %arg34 + %474 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %479 = amdgpu.mfma %445 * %arg21 + %475 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          vector.store %465, %alloc[%36, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %466, %alloc[%37, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %471, %alloc[%38, %9] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %480 = amdgpu.mfma %arg10 * %arg53 + %476 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %481 = amdgpu.mfma %arg10 * %arg44 + %477 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %482 = amdgpu.mfma %arg10 * %arg35 + %478 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %483 = amdgpu.mfma %440 * %arg22 + %479 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c512_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %484 = vector.load %alloc_0[%51, %55] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %485 = vector.load %alloc[%59, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %486 = amdgpu.mfma %arg11 * %arg54 + %480 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %487 = amdgpu.mfma %arg11 * %arg45 + %481 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %488 = amdgpu.mfma %arg11 * %arg36 + %482 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %489 = amdgpu.mfma %439 * %arg23 + %483 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %490 = vector.load %alloc_0[%51, %61] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %491 = vector.load %alloc[%59, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %492 = amdgpu.mfma %arg8 * %arg24 + %489 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %493 = amdgpu.mfma %446 * %arg12 + %arg58 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %494 = vector.load %alloc_0[%51, %64] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %495 = vector.load %alloc[%59, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %496 = amdgpu.mfma %arg9 * %arg25 + %492 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %497 = amdgpu.mfma %445 * %arg13 + %493 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %498 = vector.load %alloc_0[%51, %67] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %499 = vector.load %alloc[%59, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %500 = amdgpu.mfma %arg10 * %arg26 + %496 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %501 = amdgpu.mfma %440 * %arg14 + %497 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %502 = vector.load %alloc_0[%51, %70] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %503 = vector.load %alloc[%59, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %504 = amdgpu.mfma %arg11 * %arg27 + %500 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %505 = amdgpu.mfma %439 * %arg15 + %501 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %506 = vector.load %alloc_0[%51, %73] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %507 = vector.load %alloc[%59, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %508 = amdgpu.mfma %arg8 * %arg16 + %505 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %509 = vector.load %alloc_0[%51, %76] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %510 = vector.load %alloc[%59, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %511 = amdgpu.mfma %arg9 * %arg17 + %508 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %512 = vector.load %alloc_0[%51, %54] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %513 = vector.load %alloc[%59, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %514 = amdgpu.mfma %arg10 * %arg18 + %511 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %515 = amdgpu.mfma %512 * %513 + %arg65 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %516 = vector.load %alloc[%82, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %517 = vector.load %alloc[%82, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %518 = amdgpu.mfma %arg11 * %arg19 + %514 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %519 = amdgpu.mfma %509 * %510 + %515 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %520 = vector.load %alloc[%82, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %521 = vector.load %alloc[%82, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %522 = amdgpu.mfma %506 * %507 + %519 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %523 = vector.load %alloc[%82, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %524 = vector.load %alloc[%82, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %525 = amdgpu.mfma %502 * %503 + %522 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %526 = vector.load %alloc[%82, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %527 = vector.load %alloc[%82, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %528 = amdgpu.mfma %498 * %499 + %525 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %529 = amdgpu.mfma %512 * %527 + %arg64 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %530 = vector.load %alloc[%96, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %531 = vector.load %alloc[%96, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %532 = amdgpu.mfma %494 * %495 + %528 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %533 = amdgpu.mfma %509 * %526 + %529 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %534 = vector.load %alloc[%96, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %535 = vector.load %alloc[%96, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %536 = amdgpu.mfma %490 * %491 + %532 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %537 = amdgpu.mfma %506 * %524 + %533 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %538 = vector.load %alloc[%96, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %539 = vector.load %alloc[%96, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %540 = amdgpu.mfma %484 * %485 + %536 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %541 = amdgpu.mfma %502 * %523 + %537 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %542 = vector.load %alloc[%96, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %543 = vector.load %alloc[%96, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %544 = amdgpu.mfma %498 * %521 + %541 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %545 = amdgpu.mfma %512 * %543 + %arg63 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %546 = vector.load %alloc[%113, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %547 = vector.load %alloc[%113, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %548 = amdgpu.mfma %494 * %520 + %544 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %549 = amdgpu.mfma %509 * %542 + %545 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %550 = vector.load %alloc[%113, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %551 = vector.load %alloc[%113, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %552 = amdgpu.mfma %490 * %517 + %548 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %553 = amdgpu.mfma %506 * %539 + %549 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %554 = vector.load %alloc[%113, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %555 = vector.load %alloc[%113, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %556 = amdgpu.mfma %484 * %516 + %552 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %557 = amdgpu.mfma %502 * %538 + %553 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %558 = vector.load %alloc[%113, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %559 = vector.load %alloc[%113, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %560 = amdgpu.mfma %498 * %535 + %557 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %561 = amdgpu.mfma %512 * %559 + %443 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %562 = vector.load %alloc[%58, %55] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %563 = vector.load %alloc[%58, %61] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %564 = amdgpu.mfma %494 * %534 + %560 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %565 = amdgpu.mfma %509 * %558 + %561 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %566 = vector.load %alloc[%58, %64] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %567 = vector.load %alloc[%58, %67] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %568 = amdgpu.mfma %490 * %531 + %564 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %569 = amdgpu.mfma %506 * %555 + %565 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %570 = vector.load %alloc[%58, %70] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %571 = vector.load %alloc[%58, %73] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %572 = amdgpu.mfma %484 * %530 + %568 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %573 = amdgpu.mfma %502 * %554 + %569 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %574 = vector.load %alloc[%58, %76] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %575 = vector.load %alloc[%58, %54] : memref<80x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %576 = amdgpu.mfma %498 * %551 + %573 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %577 = amdgpu.mfma %512 * %575 + %467 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %578 = vector.load %alloc_0[%50, %55] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %579 = vector.load %alloc_0[%50, %61] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          %580 = amdgpu.mfma %494 * %550 + %576 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %581 = amdgpu.mfma %509 * %574 + %577 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %582 = vector.load %alloc_0[%50, %64] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %583 = vector.load %alloc_0[%50, %67] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c8_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"(%c256_i32, %c1_i32, %c0_i32) : (i32, i32, i32) -> ()
          scf.yield %581, %580, %583, %582, %579, %578, %575, %574, %571, %570, %567, %566, %563, %562, %559, %558, %555, %554, %551, %550, %547, %546, %506, %543, %542, %539, %538, %535, %534, %531, %530, %502, %527, %526, %524, %523, %521, %520, %517, %516, %498, %513, %510, %507, %503, %499, %495, %491, %485, %494, %490, %484, %518, %504, %488, %487, %486, %572, %556, %540 : vector<4xf32>, vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %153 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %154 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1280xf32, strided<[1], offset: ?>>
        %155 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %156 = arith.addi %3, %49 : index
        %157 = arith.addi %156, %54 : index
        %158 = arith.addi %157, %c16 : index
        %159 = arith.addi %8, %29 : index
        %160 = arith.addi %159, %57 : index
        %161 = arith.addi %160, %c32 : index
        %162 = vector.extract_strided_slice %152#57 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %163 = vector.load %154[%161] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %164 = arith.addf %162, %163 : vector<1xf32>
        %165 = arith.truncf %164 : vector<1xf32> to vector<1xf16>
        %166 = vector.load %155[%158, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %167 = arith.addf %165, %166 : vector<1xf16>
        vector.store %167, %153[%158, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %168 = vector.extract_strided_slice %152#57 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %169 = arith.addi %157, %c17 : index
        %170 = arith.addf %168, %163 : vector<1xf32>
        %171 = arith.truncf %170 : vector<1xf32> to vector<1xf16>
        %172 = vector.load %155[%169, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %173 = arith.addf %171, %172 : vector<1xf16>
        vector.store %173, %153[%169, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %174 = vector.extract_strided_slice %152#57 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %175 = arith.addi %157, %c18 : index
        %176 = arith.addf %174, %163 : vector<1xf32>
        %177 = arith.truncf %176 : vector<1xf32> to vector<1xf16>
        %178 = vector.load %155[%175, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %179 = arith.addf %177, %178 : vector<1xf16>
        vector.store %179, %153[%175, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %180 = vector.extract_strided_slice %152#57 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %181 = arith.addi %157, %c19 : index
        %182 = arith.addf %180, %163 : vector<1xf32>
        %183 = arith.truncf %182 : vector<1xf32> to vector<1xf16>
        %184 = vector.load %155[%181, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %185 = arith.addf %183, %184 : vector<1xf16>
        vector.store %185, %153[%181, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %186 = arith.addi %160, %c48 : index
        %187 = vector.extract_strided_slice %152#58 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %188 = vector.load %154[%186] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %189 = arith.addf %187, %188 : vector<1xf32>
        %190 = arith.truncf %189 : vector<1xf32> to vector<1xf16>
        %191 = vector.load %155[%158, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %192 = arith.addf %190, %191 : vector<1xf16>
        vector.store %192, %153[%158, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %193 = vector.extract_strided_slice %152#58 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %194 = arith.addf %193, %188 : vector<1xf32>
        %195 = arith.truncf %194 : vector<1xf32> to vector<1xf16>
        %196 = vector.load %155[%169, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %197 = arith.addf %195, %196 : vector<1xf16>
        vector.store %197, %153[%169, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %198 = vector.extract_strided_slice %152#58 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %199 = arith.addf %198, %188 : vector<1xf32>
        %200 = arith.truncf %199 : vector<1xf32> to vector<1xf16>
        %201 = vector.load %155[%175, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %202 = arith.addf %200, %201 : vector<1xf16>
        vector.store %202, %153[%175, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %203 = vector.extract_strided_slice %152#58 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %204 = arith.addf %203, %188 : vector<1xf32>
        %205 = arith.truncf %204 : vector<1xf32> to vector<1xf16>
        %206 = vector.load %155[%181, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %207 = arith.addf %205, %206 : vector<1xf16>
        vector.store %207, %153[%181, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %208 = arith.addi %160, %c64 : index
        %209 = vector.extract_strided_slice %152#59 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %210 = vector.load %154[%208] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %211 = arith.addf %209, %210 : vector<1xf32>
        %212 = arith.truncf %211 : vector<1xf32> to vector<1xf16>
        %213 = vector.load %155[%158, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %214 = arith.addf %212, %213 : vector<1xf16>
        vector.store %214, %153[%158, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %215 = vector.extract_strided_slice %152#59 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %216 = arith.addf %215, %210 : vector<1xf32>
        %217 = arith.truncf %216 : vector<1xf32> to vector<1xf16>
        %218 = vector.load %155[%169, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %219 = arith.addf %217, %218 : vector<1xf16>
        vector.store %219, %153[%169, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %220 = vector.extract_strided_slice %152#59 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %221 = arith.addf %220, %210 : vector<1xf32>
        %222 = arith.truncf %221 : vector<1xf32> to vector<1xf16>
        %223 = vector.load %155[%175, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %224 = arith.addf %222, %223 : vector<1xf16>
        vector.store %224, %153[%175, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %225 = vector.extract_strided_slice %152#59 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %226 = arith.addf %225, %210 : vector<1xf32>
        %227 = arith.truncf %226 : vector<1xf32> to vector<1xf16>
        %228 = vector.load %155[%181, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %229 = arith.addf %227, %228 : vector<1xf16>
        vector.store %229, %153[%181, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %230 = amdgpu.mfma %152#50 * %152#20 + %152#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %231 = amdgpu.mfma %152#22 * %152#8 + %152#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %232 = vector.load %alloc_0[%50, %70] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %233 = vector.load %alloc_0[%50, %73] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %234 = amdgpu.mfma %152#51 * %152#21 + %230 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %235 = arith.addi %160, %c16 : index
        %236 = vector.extract_strided_slice %234 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %237 = vector.load %154[%235] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %238 = arith.addf %236, %237 : vector<1xf32>
        %239 = arith.truncf %238 : vector<1xf32> to vector<1xf16>
        %240 = vector.load %155[%158, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %241 = arith.addf %239, %240 : vector<1xf16>
        vector.store %241, %153[%158, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %242 = vector.extract_strided_slice %234 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %243 = arith.addf %242, %237 : vector<1xf32>
        %244 = arith.truncf %243 : vector<1xf32> to vector<1xf16>
        %245 = vector.load %155[%169, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %246 = arith.addf %244, %245 : vector<1xf16>
        vector.store %246, %153[%169, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %247 = vector.extract_strided_slice %234 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %248 = arith.addf %247, %237 : vector<1xf32>
        %249 = arith.truncf %248 : vector<1xf32> to vector<1xf16>
        %250 = vector.load %155[%175, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %251 = arith.addf %249, %250 : vector<1xf16>
        vector.store %251, %153[%175, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %252 = vector.extract_strided_slice %234 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %253 = arith.addf %252, %237 : vector<1xf32>
        %254 = arith.truncf %253 : vector<1xf32> to vector<1xf16>
        %255 = vector.load %155[%181, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %256 = arith.addf %254, %255 : vector<1xf16>
        vector.store %256, %153[%181, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %257 = amdgpu.mfma %152#31 * %152#9 + %231 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %258 = vector.load %alloc_0[%50, %76] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %259 = vector.load %alloc_0[%50, %54] : memref<128x132xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %260 = amdgpu.mfma %152#40 * %152#10 + %257 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %261 = amdgpu.mfma %259 * %152#41 + %152#56 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %262 = amdgpu.mfma %259 * %152#32 + %152#55 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %263 = amdgpu.mfma %259 * %152#23 + %152#54 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %264 = amdgpu.mfma %152#49 * %152#11 + %260 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %265 = amdgpu.mfma %258 * %152#42 + %261 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %266 = amdgpu.mfma %258 * %152#33 + %262 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %267 = amdgpu.mfma %258 * %152#24 + %263 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %268 = amdgpu.mfma %152#50 * %152#12 + %264 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %269 = amdgpu.mfma %233 * %152#43 + %265 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %270 = amdgpu.mfma %233 * %152#34 + %266 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %271 = amdgpu.mfma %233 * %152#25 + %267 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %272 = amdgpu.mfma %152#51 * %152#13 + %268 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %273 = vector.extract_strided_slice %272 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %274 = vector.load %154[%160] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %275 = arith.addf %273, %274 : vector<1xf32>
        %276 = arith.truncf %275 : vector<1xf32> to vector<1xf16>
        %277 = vector.load %155[%158, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %278 = arith.addf %276, %277 : vector<1xf16>
        vector.store %278, %153[%158, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %279 = vector.extract_strided_slice %272 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %280 = arith.addf %279, %274 : vector<1xf32>
        %281 = arith.truncf %280 : vector<1xf32> to vector<1xf16>
        %282 = vector.load %155[%169, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %283 = arith.addf %281, %282 : vector<1xf16>
        vector.store %283, %153[%169, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %284 = vector.extract_strided_slice %272 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %285 = arith.addf %284, %274 : vector<1xf32>
        %286 = arith.truncf %285 : vector<1xf32> to vector<1xf16>
        %287 = vector.load %155[%175, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %288 = arith.addf %286, %287 : vector<1xf16>
        vector.store %288, %153[%175, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %289 = vector.extract_strided_slice %272 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %290 = arith.addf %289, %274 : vector<1xf32>
        %291 = arith.truncf %290 : vector<1xf32> to vector<1xf16>
        %292 = vector.load %155[%181, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %293 = arith.addf %291, %292 : vector<1xf16>
        vector.store %293, %153[%181, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %294 = amdgpu.mfma %232 * %152#44 + %269 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %295 = amdgpu.mfma %232 * %152#35 + %270 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %296 = amdgpu.mfma %232 * %152#26 + %271 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %297 = amdgpu.mfma %152#2 * %152#45 + %294 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %298 = amdgpu.mfma %152#2 * %152#36 + %295 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %299 = amdgpu.mfma %152#2 * %152#27 + %296 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %300 = amdgpu.mfma %259 * %152#14 + %152#53 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %301 = amdgpu.mfma %152#3 * %152#46 + %297 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %302 = amdgpu.mfma %152#3 * %152#37 + %298 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %303 = amdgpu.mfma %152#3 * %152#28 + %299 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %304 = amdgpu.mfma %258 * %152#15 + %300 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %305 = amdgpu.mfma %152#4 * %152#47 + %301 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %306 = amdgpu.mfma %152#4 * %152#38 + %302 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %307 = amdgpu.mfma %152#4 * %152#29 + %303 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %308 = amdgpu.mfma %233 * %152#16 + %304 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %309 = amdgpu.mfma %152#5 * %152#48 + %305 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %310 = vector.extract_strided_slice %309 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %311 = vector.load %154[%208] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %312 = arith.addf %310, %311 : vector<1xf32>
        %313 = arith.truncf %312 : vector<1xf32> to vector<1xf16>
        %314 = vector.load %155[%157, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %315 = arith.addf %313, %314 : vector<1xf16>
        vector.store %315, %153[%157, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %316 = vector.extract_strided_slice %309 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %317 = arith.addi %157, %c1 : index
        %318 = arith.addf %316, %311 : vector<1xf32>
        %319 = arith.truncf %318 : vector<1xf32> to vector<1xf16>
        %320 = vector.load %155[%317, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %321 = arith.addf %319, %320 : vector<1xf16>
        vector.store %321, %153[%317, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %322 = vector.extract_strided_slice %309 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %323 = arith.addi %157, %c2 : index
        %324 = arith.addf %322, %311 : vector<1xf32>
        %325 = arith.truncf %324 : vector<1xf32> to vector<1xf16>
        %326 = vector.load %155[%323, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %327 = arith.addf %325, %326 : vector<1xf16>
        vector.store %327, %153[%323, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %328 = vector.extract_strided_slice %309 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %329 = arith.addi %157, %c3 : index
        %330 = arith.addf %328, %311 : vector<1xf32>
        %331 = arith.truncf %330 : vector<1xf32> to vector<1xf16>
        %332 = vector.load %155[%329, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %333 = arith.addf %331, %332 : vector<1xf16>
        vector.store %333, %153[%329, %208] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %334 = amdgpu.mfma %152#5 * %152#39 + %306 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %335 = vector.extract_strided_slice %334 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %336 = vector.load %154[%186] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %337 = arith.addf %335, %336 : vector<1xf32>
        %338 = arith.truncf %337 : vector<1xf32> to vector<1xf16>
        %339 = vector.load %155[%157, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %340 = arith.addf %338, %339 : vector<1xf16>
        vector.store %340, %153[%157, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %341 = vector.extract_strided_slice %334 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %342 = arith.addf %341, %336 : vector<1xf32>
        %343 = arith.truncf %342 : vector<1xf32> to vector<1xf16>
        %344 = vector.load %155[%317, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %345 = arith.addf %343, %344 : vector<1xf16>
        vector.store %345, %153[%317, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %346 = vector.extract_strided_slice %334 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %347 = arith.addf %346, %336 : vector<1xf32>
        %348 = arith.truncf %347 : vector<1xf32> to vector<1xf16>
        %349 = vector.load %155[%323, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %350 = arith.addf %348, %349 : vector<1xf16>
        vector.store %350, %153[%323, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %351 = vector.extract_strided_slice %334 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %352 = arith.addf %351, %336 : vector<1xf32>
        %353 = arith.truncf %352 : vector<1xf32> to vector<1xf16>
        %354 = vector.load %155[%329, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %355 = arith.addf %353, %354 : vector<1xf16>
        vector.store %355, %153[%329, %186] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %356 = amdgpu.mfma %152#5 * %152#30 + %307 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %357 = vector.extract_strided_slice %356 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %358 = vector.load %154[%161] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %359 = arith.addf %357, %358 : vector<1xf32>
        %360 = arith.truncf %359 : vector<1xf32> to vector<1xf16>
        %361 = vector.load %155[%157, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %362 = arith.addf %360, %361 : vector<1xf16>
        vector.store %362, %153[%157, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %363 = vector.extract_strided_slice %356 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %364 = arith.addf %363, %358 : vector<1xf32>
        %365 = arith.truncf %364 : vector<1xf32> to vector<1xf16>
        %366 = vector.load %155[%317, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %367 = arith.addf %365, %366 : vector<1xf16>
        vector.store %367, %153[%317, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %368 = vector.extract_strided_slice %356 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %369 = arith.addf %368, %358 : vector<1xf32>
        %370 = arith.truncf %369 : vector<1xf32> to vector<1xf16>
        %371 = vector.load %155[%323, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %372 = arith.addf %370, %371 : vector<1xf16>
        vector.store %372, %153[%323, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %373 = vector.extract_strided_slice %356 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %374 = arith.addf %373, %358 : vector<1xf32>
        %375 = arith.truncf %374 : vector<1xf32> to vector<1xf16>
        %376 = vector.load %155[%329, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %377 = arith.addf %375, %376 : vector<1xf16>
        vector.store %377, %153[%329, %161] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %378 = amdgpu.mfma %232 * %152#17 + %308 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %379 = amdgpu.mfma %152#2 * %152#18 + %378 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %380 = amdgpu.mfma %259 * %152#6 + %152#52 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %381 = amdgpu.mfma %152#3 * %152#19 + %379 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %382 = amdgpu.mfma %258 * %152#7 + %380 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %383 = amdgpu.mfma %152#4 * %152#20 + %381 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %384 = amdgpu.mfma %233 * %152#8 + %382 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %385 = amdgpu.mfma %152#5 * %152#21 + %383 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %386 = vector.extract_strided_slice %385 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %387 = vector.load %154[%235] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %388 = arith.addf %386, %387 : vector<1xf32>
        %389 = arith.truncf %388 : vector<1xf32> to vector<1xf16>
        %390 = vector.load %155[%157, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %391 = arith.addf %389, %390 : vector<1xf16>
        vector.store %391, %153[%157, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %392 = vector.extract_strided_slice %385 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %393 = arith.addf %392, %387 : vector<1xf32>
        %394 = arith.truncf %393 : vector<1xf32> to vector<1xf16>
        %395 = vector.load %155[%317, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %396 = arith.addf %394, %395 : vector<1xf16>
        vector.store %396, %153[%317, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %397 = vector.extract_strided_slice %385 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %398 = arith.addf %397, %387 : vector<1xf32>
        %399 = arith.truncf %398 : vector<1xf32> to vector<1xf16>
        %400 = vector.load %155[%323, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %401 = arith.addf %399, %400 : vector<1xf16>
        vector.store %401, %153[%323, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %402 = vector.extract_strided_slice %385 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %403 = arith.addf %402, %387 : vector<1xf32>
        %404 = arith.truncf %403 : vector<1xf32> to vector<1xf16>
        %405 = vector.load %155[%329, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %406 = arith.addf %404, %405 : vector<1xf16>
        vector.store %406, %153[%329, %235] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %407 = amdgpu.mfma %232 * %152#9 + %384 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %408 = amdgpu.mfma %152#2 * %152#10 + %407 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %409 = amdgpu.mfma %152#3 * %152#11 + %408 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %410 = amdgpu.mfma %152#4 * %152#12 + %409 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %411 = amdgpu.mfma %152#5 * %152#13 + %410 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %412 = vector.extract_strided_slice %411 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %413 = vector.load %154[%160] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %414 = arith.addf %412, %413 : vector<1xf32>
        %415 = arith.truncf %414 : vector<1xf32> to vector<1xf16>
        %416 = vector.load %155[%157, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %417 = arith.addf %415, %416 : vector<1xf16>
        vector.store %417, %153[%157, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %418 = vector.extract_strided_slice %411 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %419 = arith.addf %418, %413 : vector<1xf32>
        %420 = arith.truncf %419 : vector<1xf32> to vector<1xf16>
        %421 = vector.load %155[%317, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %422 = arith.addf %420, %421 : vector<1xf16>
        vector.store %422, %153[%317, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %423 = vector.extract_strided_slice %411 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %424 = arith.addf %423, %413 : vector<1xf32>
        %425 = arith.truncf %424 : vector<1xf32> to vector<1xf16>
        %426 = vector.load %155[%323, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %427 = arith.addf %425, %426 : vector<1xf16>
        vector.store %427, %153[%323, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %428 = vector.extract_strided_slice %411 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %429 = arith.addf %428, %413 : vector<1xf32>
        %430 = arith.truncf %429 : vector<1xf32> to vector<1xf16>
        %431 = vector.load %155[%329, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %432 = arith.addf %430, %431 : vector<1xf16>
        vector.store %432, %153[%329, %160] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

