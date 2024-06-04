#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
module {
  flow.executable private @tk_gemm_fused_2048x1280x1280 {
    flow.executable.export public @tk_gemm_fused_2048x1280x1280 workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c20 = arith.constant 20 : index
      %c1 = arith.constant 1 : index
      flow.return %c32, %c20, %c1 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2048x1280x1280(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
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
        %13 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<1280x1280xf16, strided<[1280, 1], offset: ?>>
        %14 = arith.muli %workgroup_id_1, %c64 : index
        %15 = arith.addi %4, %14 : index
        %16 = arith.addi %15, %2 : index
        %17 = arith.addi %16, %1 : index
        %18 = vector.load %13[%17, %9] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %19 = arith.addi %17, %c32 : index
        %20 = vector.load %13[%19, %9] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %21 = arith.addi %4, %2 : index
        %22 = arith.addi %21, %1 : index
        vector.store %10, %alloc_0[%22, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %23 = arith.addi %22, %c32 : index
        vector.store %12, %alloc_0[%23, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %18, %alloc[%22, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %20, %alloc[%23, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
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
        %48 = vector.load %alloc[%35, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %49 = amdgpu.mfma %42 * %43 + %46 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %50 = vector.load %alloc[%35, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %51 = vector.load %alloc[%35, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %52:16 = scf.for %arg3 = %c1 to %c20 step %c1 iter_args(%arg4 = %49, %arg5 = %44, %arg6 = %51, %arg7 = %50, %arg8 = %48, %arg9 = %47, %arg10 = %42, %arg11 = %45, %arg12 = %43, %arg13 = %40, %arg14 = %37, %arg15 = %39, %arg16 = %33, %arg17 = %cst, %arg18 = %cst, %arg19 = %cst) -> (vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %100 = arith.muli %arg3, %c64 : index
          %101 = arith.addi %100, %9 : index
          %102 = vector.load %0[%7, %101] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %103 = vector.load %0[%11, %101] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %104 = amdgpu.mfma %arg15 * %arg13 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %105 = amdgpu.mfma %arg5 * %arg6 + %arg19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %106 = vector.load %alloc_0[%27, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %107 = vector.load %alloc_0[%27, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %108 = vector.load %13[%17, %101] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %109 = vector.load %13[%19, %101] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %110 = amdgpu.mfma %arg16 * %arg14 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %111 = amdgpu.mfma %arg10 * %arg7 + %105 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %112 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %113 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          vector.store %102, %alloc_0[%22, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %103, %alloc_0[%23, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %114 = amdgpu.mfma %arg15 * %arg8 + %111 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %115 = amdgpu.mfma %113 * %arg11 + %arg18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %116 = amdgpu.mfma %113 * %arg6 + %arg17 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %108, %alloc[%22, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %109, %alloc[%23, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %117 = amdgpu.mfma %arg16 * %arg9 + %114 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %118 = amdgpu.mfma %112 * %arg12 + %115 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %119 = amdgpu.mfma %112 * %arg7 + %116 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %120 = vector.load %alloc_0[%28, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %121 = vector.load %alloc[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %122 = amdgpu.mfma %107 * %arg13 + %118 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %123 = amdgpu.mfma %107 * %arg8 + %119 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %124 = vector.load %alloc_0[%28, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %125 = vector.load %alloc[%36, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %126 = amdgpu.mfma %106 * %arg14 + %122 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %127 = amdgpu.mfma %106 * %arg9 + %123 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %128 = vector.load %alloc_0[%28, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %129 = vector.load %alloc[%36, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %130 = vector.load %alloc_0[%28, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %131 = vector.load %alloc[%36, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %132 = amdgpu.mfma %130 * %131 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %133 = vector.load %alloc[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %134 = vector.load %alloc[%35, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %135 = amdgpu.mfma %128 * %129 + %132 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %136 = vector.load %alloc[%35, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %137 = vector.load %alloc[%35, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          scf.yield %135, %130, %137, %136, %134, %133, %128, %131, %129, %125, %121, %124, %120, %127, %126, %117 : vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %53 = amdgpu.mfma %52#11 * %52#9 + %52#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %54 = amdgpu.mfma %52#1 * %52#2 + %52#15 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %55 = vector.load %alloc_0[%27, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %56 = vector.load %alloc_0[%27, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %57 = amdgpu.mfma %52#12 * %52#10 + %53 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %58 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<2048x1280xf32, strided<[1280, 1], offset: ?>>
        %59 = arith.addi %3, %25 : index
        %60 = arith.addi %59, %31 : index
        %61 = arith.addi %60, %c16 : index
        %62 = arith.addi %26, %14 : index
        %63 = arith.addi %62, %34 : index
        %64 = arith.addi %63, %c16 : index
        %65 = vector.extract_strided_slice %57 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %65, %58[%61, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %66 = vector.extract_strided_slice %57 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %67 = arith.addi %60, %c17 : index
        vector.store %66, %58[%67, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %68 = vector.extract_strided_slice %57 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %69 = arith.addi %60, %c18 : index
        vector.store %68, %58[%69, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %70 = vector.extract_strided_slice %57 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %71 = arith.addi %60, %c19 : index
        vector.store %70, %58[%71, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %72 = amdgpu.mfma %52#6 * %52#3 + %54 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %73 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %74 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %75 = amdgpu.mfma %52#11 * %52#4 + %72 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %76 = amdgpu.mfma %74 * %52#7 + %52#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %77 = amdgpu.mfma %74 * %52#2 + %52#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %78 = amdgpu.mfma %52#12 * %52#5 + %75 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %79 = vector.extract_strided_slice %78 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %79, %58[%61, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %80 = vector.extract_strided_slice %78 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %80, %58[%67, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %81 = vector.extract_strided_slice %78 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %81, %58[%69, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %82 = vector.extract_strided_slice %78 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %82, %58[%71, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %83 = amdgpu.mfma %73 * %52#8 + %76 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %84 = amdgpu.mfma %73 * %52#3 + %77 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %85 = amdgpu.mfma %56 * %52#9 + %83 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %86 = amdgpu.mfma %56 * %52#4 + %84 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %87 = amdgpu.mfma %55 * %52#10 + %85 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %88 = vector.extract_strided_slice %87 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %88, %58[%60, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %89 = vector.extract_strided_slice %87 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %90 = arith.addi %60, %c1 : index
        vector.store %89, %58[%90, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %91 = vector.extract_strided_slice %87 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %92 = arith.addi %60, %c2 : index
        vector.store %91, %58[%92, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %93 = vector.extract_strided_slice %87 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %94 = arith.addi %60, %c3 : index
        vector.store %93, %58[%94, %64] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %95 = amdgpu.mfma %55 * %52#5 + %86 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %96 = vector.extract_strided_slice %95 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %96, %58[%60, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %97 = vector.extract_strided_slice %95 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %97, %58[%90, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %98 = vector.extract_strided_slice %95 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %98, %58[%92, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %99 = vector.extract_strided_slice %95 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %99, %58[%94, %63] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
}

