#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
module {
  flow.executable private @tk_gemm_fused_2048x1280x1280_bias0 {
    flow.executable.export public @tk_gemm_fused_2048x1280x1280_bias0 workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c20 = arith.constant 20 : index
      %c1 = arith.constant 1 : index
      flow.return %c32, %c20, %c1 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2048x1280x1280_bias0(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c20 = arith.constant 20 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c48 = arith.constant 48 : index
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %alloc = memref.alloc() : memref<64x68xf16, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<64x68xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %1 = arith.muli %thread_id_y, %c16 : index
        %2 = arith.muli %thread_id_z, %c32 : index
        %3 = arith.muli %workgroup_id_0, %c64 : index
        %4 = arith.divsi %thread_id_x, %c8 : index
        %5 = arith.addi %4, %3 : index
        %6 = arith.addi %5, %2 : index
        %7 = arith.addi %6, %1 : index
        %8 = arith.remsi %thread_id_x, %c8 : index
        %9 = arith.muli %8, %c8 : index
        %10 = vector.load %0[%7, %9] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %11 = arith.addi %7, %c32 : index
        %12 = vector.load %0[%11, %9] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %13 = arith.addi %4, %2 : index
        %14 = arith.addi %13, %1 : index
        vector.store %10, %alloc_0[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %15 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<1280x1280xf16, strided<[1280, 1], offset: ?>>
        %16 = arith.muli %workgroup_id_1, %c64 : index
        %17 = arith.addi %4, %16 : index
        %18 = arith.addi %17, %2 : index
        %19 = arith.addi %18, %1 : index
        %20 = vector.load %15[%19, %9] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %21 = arith.addi %14, %c32 : index
        vector.store %12, %alloc_0[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %22 = arith.addi %19, %c32 : index
        %23 = vector.load %15[%22, %9] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        vector.store %20, %alloc[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %23, %alloc[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        amdgpu.lds_barrier
        %24 = arith.divsi %thread_id_x, %c64 : index
        %25 = arith.muli %24, %c32 : index
        %26 = arith.remsi %thread_id_x, %c16 : index
        %27 = arith.addi %26, %25 : index
        %28 = arith.addi %27, %c16 : index
        %29 = arith.remsi %thread_id_x, %c64 : index
        %30 = arith.divsi %29, %c16 : index
        %31 = arith.muli %30, %c4 : index
        %32 = arith.addi %31, %c48 : index
        %33 = vector.load %alloc_0[%28, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %34 = arith.muli %thread_id_y, %c32 : index
        %35 = arith.addi %26, %34 : index
        %36 = arith.addi %35, %c16 : index
        %37 = vector.load %alloc[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %38 = arith.addi %31, %c32 : index
        %39 = vector.load %alloc_0[%28, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %40 = vector.load %alloc[%36, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %41 = arith.addi %31, %c16 : index
        %42 = vector.load %alloc_0[%28, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %43 = vector.load %alloc[%36, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %44 = vector.load %alloc_0[%28, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %45 = vector.load %alloc[%36, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %46 = amdgpu.mfma %44 * %45 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %47 = vector.load %alloc[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %48 = amdgpu.mfma %42 * %43 + %46 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %49 = vector.load %alloc[%35, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %50 = amdgpu.mfma %39 * %40 + %48 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %51 = vector.load %alloc[%35, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %52 = amdgpu.mfma %33 * %37 + %50 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %53 = vector.load %alloc[%35, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %54 = amdgpu.mfma %44 * %53 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %55 = vector.load %alloc_0[%27, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %56 = amdgpu.mfma %42 * %51 + %54 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %57 = vector.load %alloc_0[%27, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %58:16 = scf.for %arg3 = %c1 to %c20 step %c1 iter_args(%arg4 = %56, %arg5 = %57, %arg6 = %55, %arg7 = %53, %arg8 = %51, %arg9 = %49, %arg10 = %47, %arg11 = %45, %arg12 = %43, %arg13 = %40, %arg14 = %37, %arg15 = %39, %arg16 = %33, %arg17 = %cst, %arg18 = %cst, %arg19 = %52) -> (vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %100 = arith.muli %arg3, %c64 : index
          %101 = arith.addi %100, %9 : index
          %102 = vector.load %0[%7, %101] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %103 = amdgpu.mfma %arg15 * %arg9 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %104 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %105 = vector.load %0[%11, %101] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %106 = amdgpu.mfma %arg16 * %arg10 + %103 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %107 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          vector.store %102, %alloc_0[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %108 = vector.load %15[%19, %101] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %109 = amdgpu.mfma %107 * %arg11 + %arg18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %110 = amdgpu.mfma %107 * %arg7 + %arg17 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %105, %alloc_0[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %111 = vector.load %15[%22, %101] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %112 = amdgpu.mfma %104 * %arg12 + %109 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %113 = amdgpu.mfma %104 * %arg8 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %108, %alloc[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %114 = amdgpu.mfma %arg5 * %arg13 + %112 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %115 = amdgpu.mfma %arg5 * %arg9 + %113 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %111, %alloc[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %116 = amdgpu.mfma %arg6 * %arg14 + %114 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %117 = amdgpu.mfma %arg6 * %arg10 + %115 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %118 = vector.load %alloc_0[%28, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %119 = vector.load %alloc[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %120 = vector.load %alloc_0[%28, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %121 = vector.load %alloc[%36, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %122 = vector.load %alloc_0[%28, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %123 = vector.load %alloc[%36, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %124 = vector.load %alloc_0[%28, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %125 = vector.load %alloc[%36, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %126 = amdgpu.mfma %124 * %125 + %arg19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %127 = vector.load %alloc[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %128 = amdgpu.mfma %122 * %123 + %126 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %129 = vector.load %alloc[%35, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %130 = amdgpu.mfma %120 * %121 + %128 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %131 = vector.load %alloc[%35, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %132 = amdgpu.mfma %118 * %119 + %130 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %133 = vector.load %alloc[%35, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %134 = amdgpu.mfma %124 * %133 + %106 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %135 = vector.load %alloc_0[%27, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %136 = amdgpu.mfma %122 * %131 + %134 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %137 = vector.load %alloc_0[%27, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          scf.yield %136, %137, %135, %133, %131, %129, %127, %125, %123, %121, %119, %120, %118, %117, %116, %132 : vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %59 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<2048x1280xf32, strided<[1280, 1], offset: ?>>
        %60 = arith.addi %3, %25 : index
        %61 = arith.addi %60, %31 : index
        %62 = arith.addi %61, %c16 : index
        %63 = arith.addi %26, %16 : index
        %64 = arith.addi %63, %34 : index
        %65 = arith.addi %64, %c16 : index
        %66 = vector.extract_strided_slice %58#15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %66, %59[%62, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %67 = vector.extract_strided_slice %58#15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %68 = arith.addi %61, %c17 : index
        vector.store %67, %59[%68, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %69 = vector.extract_strided_slice %58#15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %70 = arith.addi %61, %c18 : index
        vector.store %69, %59[%70, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %71 = vector.extract_strided_slice %58#15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %72 = arith.addi %61, %c19 : index
        vector.store %71, %59[%72, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %73 = amdgpu.mfma %58#11 * %58#5 + %58#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %74 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %75 = amdgpu.mfma %58#12 * %58#6 + %73 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %76 = vector.extract_strided_slice %75 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %76, %59[%62, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %77 = vector.extract_strided_slice %75 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %77, %59[%68, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %78 = vector.extract_strided_slice %75 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %78, %59[%70, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %79 = vector.extract_strided_slice %75 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %79, %59[%72, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %80 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %81 = amdgpu.mfma %80 * %58#7 + %58#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %82 = amdgpu.mfma %80 * %58#3 + %58#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %83 = amdgpu.mfma %74 * %58#8 + %81 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %84 = amdgpu.mfma %74 * %58#4 + %82 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %85 = amdgpu.mfma %58#1 * %58#9 + %83 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %86 = amdgpu.mfma %58#1 * %58#5 + %84 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %87 = amdgpu.mfma %58#2 * %58#10 + %85 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %88 = vector.extract_strided_slice %87 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %88, %59[%61, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %89 = vector.extract_strided_slice %87 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %90 = arith.addi %61, %c1 : index
        vector.store %89, %59[%90, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %91 = vector.extract_strided_slice %87 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %92 = arith.addi %61, %c2 : index
        vector.store %91, %59[%92, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %93 = vector.extract_strided_slice %87 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %94 = arith.addi %61, %c3 : index
        vector.store %93, %59[%94, %65] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %95 = amdgpu.mfma %58#2 * %58#6 + %86 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %96 = vector.extract_strided_slice %95 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %96, %59[%61, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %97 = vector.extract_strided_slice %95 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %97, %59[%90, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %98 = vector.extract_strided_slice %95 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %98, %59[%92, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %99 = vector.extract_strided_slice %95 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %99, %59[%94, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
}

