#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
module {
  flow.executable private @tk_gemm_fused_2048x1280x1280_bias1 {
    flow.executable.export public @tk_gemm_fused_2048x1280x1280_bias1 workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c20 = arith.constant 20 : index
      %c1 = arith.constant 1 : index
      flow.return %c32, %c20, %c1 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2048x1280x1280_bias1(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
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
        %58:16 = scf.for %arg4 = %c1 to %c20 step %c1 iter_args(%arg5 = %56, %arg6 = %57, %arg7 = %55, %arg8 = %53, %arg9 = %51, %arg10 = %49, %arg11 = %47, %arg12 = %45, %arg13 = %43, %arg14 = %40, %arg15 = %37, %arg16 = %39, %arg17 = %33, %arg18 = %cst, %arg19 = %cst, %arg20 = %52) -> (vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %137 = arith.muli %arg4, %c64 : index
          %138 = arith.addi %137, %9 : index
          %139 = vector.load %0[%7, %138] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %140 = amdgpu.mfma %arg16 * %arg10 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %141 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %142 = vector.load %0[%11, %138] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %143 = amdgpu.mfma %arg17 * %arg11 + %140 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %144 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          vector.store %139, %alloc_0[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %145 = vector.load %15[%19, %138] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %146 = amdgpu.mfma %144 * %arg12 + %arg19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %147 = amdgpu.mfma %144 * %arg8 + %arg18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %142, %alloc_0[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %148 = vector.load %15[%22, %138] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %149 = amdgpu.mfma %141 * %arg13 + %146 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %150 = amdgpu.mfma %141 * %arg9 + %147 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %145, %alloc[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %151 = amdgpu.mfma %arg6 * %arg14 + %149 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %152 = amdgpu.mfma %arg6 * %arg10 + %150 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %148, %alloc[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %153 = amdgpu.mfma %arg7 * %arg15 + %151 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %154 = amdgpu.mfma %arg7 * %arg11 + %152 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %155 = vector.load %alloc_0[%28, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %156 = vector.load %alloc[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %157 = vector.load %alloc_0[%28, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %158 = vector.load %alloc[%36, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %159 = vector.load %alloc_0[%28, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %160 = vector.load %alloc[%36, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %161 = vector.load %alloc_0[%28, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %162 = vector.load %alloc[%36, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %163 = amdgpu.mfma %161 * %162 + %arg20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %164 = vector.load %alloc[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %165 = amdgpu.mfma %159 * %160 + %163 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %166 = vector.load %alloc[%35, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %167 = amdgpu.mfma %157 * %158 + %165 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %168 = vector.load %alloc[%35, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %169 = amdgpu.mfma %155 * %156 + %167 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %170 = vector.load %alloc[%35, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %171 = amdgpu.mfma %161 * %170 + %143 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %172 = vector.load %alloc_0[%27, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %173 = amdgpu.mfma %159 * %168 + %171 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %174 = vector.load %alloc_0[%27, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          scf.yield %173, %174, %172, %170, %168, %166, %164, %162, %160, %158, %156, %157, %155, %154, %153, %169 : vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %59 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %60 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1280xf16, strided<[1], offset: ?>>
        %61 = arith.addi %3, %25 : index
        %62 = arith.addi %61, %31 : index
        %63 = arith.addi %62, %c16 : index
        %64 = arith.addi %26, %16 : index
        %65 = arith.addi %64, %34 : index
        %66 = arith.addi %65, %c16 : index
        %67 = vector.extract_strided_slice %58#15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %68 = vector.load %60[%66] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %69 = arith.truncf %67 : vector<1xf32> to vector<1xf16>
        %70 = arith.addf %69, %68 : vector<1xf16>
        vector.store %70, %59[%63, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %71 = vector.extract_strided_slice %58#15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %72 = arith.addi %62, %c17 : index
        %73 = arith.truncf %71 : vector<1xf32> to vector<1xf16>
        %74 = arith.addf %73, %68 : vector<1xf16>
        vector.store %74, %59[%72, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %75 = vector.extract_strided_slice %58#15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %76 = arith.addi %62, %c18 : index
        %77 = arith.truncf %75 : vector<1xf32> to vector<1xf16>
        %78 = arith.addf %77, %68 : vector<1xf16>
        vector.store %78, %59[%76, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %79 = vector.extract_strided_slice %58#15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %80 = arith.addi %62, %c19 : index
        %81 = arith.truncf %79 : vector<1xf32> to vector<1xf16>
        %82 = arith.addf %81, %68 : vector<1xf16>
        vector.store %82, %59[%80, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %83 = amdgpu.mfma %58#11 * %58#5 + %58#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %84 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %85 = amdgpu.mfma %58#12 * %58#6 + %83 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %86 = vector.extract_strided_slice %85 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %87 = vector.load %60[%65] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %88 = arith.truncf %86 : vector<1xf32> to vector<1xf16>
        %89 = arith.addf %88, %87 : vector<1xf16>
        vector.store %89, %59[%63, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %90 = vector.extract_strided_slice %85 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %91 = arith.truncf %90 : vector<1xf32> to vector<1xf16>
        %92 = arith.addf %91, %87 : vector<1xf16>
        vector.store %92, %59[%72, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %93 = vector.extract_strided_slice %85 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %94 = arith.truncf %93 : vector<1xf32> to vector<1xf16>
        %95 = arith.addf %94, %87 : vector<1xf16>
        vector.store %95, %59[%76, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %96 = vector.extract_strided_slice %85 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %97 = arith.truncf %96 : vector<1xf32> to vector<1xf16>
        %98 = arith.addf %97, %87 : vector<1xf16>
        vector.store %98, %59[%80, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %99 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %100 = amdgpu.mfma %99 * %58#7 + %58#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %101 = amdgpu.mfma %99 * %58#3 + %58#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %102 = amdgpu.mfma %84 * %58#8 + %100 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %103 = amdgpu.mfma %84 * %58#4 + %101 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %104 = amdgpu.mfma %58#1 * %58#9 + %102 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %105 = amdgpu.mfma %58#1 * %58#5 + %103 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %106 = amdgpu.mfma %58#2 * %58#10 + %104 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %107 = vector.extract_strided_slice %106 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %108 = vector.load %60[%66] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %109 = arith.truncf %107 : vector<1xf32> to vector<1xf16>
        %110 = arith.addf %109, %108 : vector<1xf16>
        vector.store %110, %59[%62, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %111 = vector.extract_strided_slice %106 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %112 = arith.addi %62, %c1 : index
        %113 = arith.truncf %111 : vector<1xf32> to vector<1xf16>
        %114 = arith.addf %113, %108 : vector<1xf16>
        vector.store %114, %59[%112, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %115 = vector.extract_strided_slice %106 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %116 = arith.addi %62, %c2 : index
        %117 = arith.truncf %115 : vector<1xf32> to vector<1xf16>
        %118 = arith.addf %117, %108 : vector<1xf16>
        vector.store %118, %59[%116, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %119 = vector.extract_strided_slice %106 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %120 = arith.addi %62, %c3 : index
        %121 = arith.truncf %119 : vector<1xf32> to vector<1xf16>
        %122 = arith.addf %121, %108 : vector<1xf16>
        vector.store %122, %59[%120, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %123 = amdgpu.mfma %58#2 * %58#6 + %105 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %124 = vector.extract_strided_slice %123 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %125 = vector.load %60[%65] : memref<1280xf16, strided<[1], offset: ?>>, vector<1xf16>
        %126 = arith.truncf %124 : vector<1xf32> to vector<1xf16>
        %127 = arith.addf %126, %125 : vector<1xf16>
        vector.store %127, %59[%62, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %128 = vector.extract_strided_slice %123 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %129 = arith.truncf %128 : vector<1xf32> to vector<1xf16>
        %130 = arith.addf %129, %125 : vector<1xf16>
        vector.store %130, %59[%112, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %131 = vector.extract_strided_slice %123 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %132 = arith.truncf %131 : vector<1xf32> to vector<1xf16>
        %133 = arith.addf %132, %125 : vector<1xf16>
        vector.store %133, %59[%116, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %134 = vector.extract_strided_slice %123 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %135 = arith.truncf %134 : vector<1xf32> to vector<1xf16>
        %136 = arith.addf %135, %125 : vector<1xf16>
        vector.store %136, %59[%120, %65] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

