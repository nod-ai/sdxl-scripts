#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}}>
module {
  flow.executable private @tk_gemm_fused_2048x1280x1280_bias2 {
    flow.executable.export public @tk_gemm_fused_2048x1280x1280_bias2 workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c20 = arith.constant 20 : index
      %c1 = arith.constant 1 : index
      flow.return %c32, %c20, %c1 : index, index, index
    }
    builtin.module {
      func.func @tk_gemm_fused_2048x1280x1280_bias2(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
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
        %58:16 = scf.for %arg5 = %c1 to %c20 step %c1 iter_args(%arg6 = %56, %arg7 = %57, %arg8 = %55, %arg9 = %53, %arg10 = %51, %arg11 = %49, %arg12 = %47, %arg13 = %45, %arg14 = %43, %arg15 = %40, %arg16 = %37, %arg17 = %39, %arg18 = %33, %arg19 = %cst, %arg20 = %cst, %arg21 = %52) -> (vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %170 = arith.muli %arg5, %c64 : index
          %171 = arith.addi %170, %9 : index
          %172 = vector.load %0[%7, %171] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %173 = amdgpu.mfma %arg17 * %arg11 + %arg6 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %174 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %175 = vector.load %0[%11, %171] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %176 = amdgpu.mfma %arg18 * %arg12 + %173 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %177 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          vector.store %172, %alloc_0[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %178 = vector.load %15[%19, %171] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %179 = amdgpu.mfma %177 * %arg13 + %arg20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %180 = amdgpu.mfma %177 * %arg9 + %arg19 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %175, %alloc_0[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %181 = vector.load %15[%22, %171] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %182 = amdgpu.mfma %174 * %arg14 + %179 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %183 = amdgpu.mfma %174 * %arg10 + %180 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %178, %alloc[%14, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %184 = amdgpu.mfma %arg7 * %arg15 + %182 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %185 = amdgpu.mfma %arg7 * %arg11 + %183 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          vector.store %181, %alloc[%21, %9] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %186 = amdgpu.mfma %arg8 * %arg16 + %184 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %187 = amdgpu.mfma %arg8 * %arg12 + %185 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          amdgpu.lds_barrier
          amdgpu.lds_barrier
          %188 = vector.load %alloc_0[%28, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %189 = vector.load %alloc[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %190 = vector.load %alloc_0[%28, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %191 = vector.load %alloc[%36, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %192 = vector.load %alloc_0[%28, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %193 = vector.load %alloc[%36, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %194 = vector.load %alloc_0[%28, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %195 = vector.load %alloc[%36, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %196 = amdgpu.mfma %194 * %195 + %arg21 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %197 = vector.load %alloc[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %198 = amdgpu.mfma %192 * %193 + %196 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %199 = vector.load %alloc[%35, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %200 = amdgpu.mfma %190 * %191 + %198 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %201 = vector.load %alloc[%35, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %202 = amdgpu.mfma %188 * %189 + %200 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %203 = vector.load %alloc[%35, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %204 = amdgpu.mfma %194 * %203 + %176 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %205 = vector.load %alloc_0[%27, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %206 = amdgpu.mfma %192 * %201 + %204 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %207 = vector.load %alloc_0[%27, %38] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          scf.yield %206, %207, %205, %203, %201, %199, %197, %195, %193, %191, %189, %190, %188, %187, %186, %202 : vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %59 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %60 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<1280xf32, strided<[1], offset: ?>>
        %61 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %62 = arith.addi %3, %25 : index
        %63 = arith.addi %62, %31 : index
        %64 = arith.addi %63, %c16 : index
        %65 = arith.addi %26, %16 : index
        %66 = arith.addi %65, %34 : index
        %67 = arith.addi %66, %c16 : index
        %68 = vector.extract_strided_slice %58#15 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %69 = vector.load %60[%67] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %70 = arith.addf %68, %69 : vector<1xf32>
        %71 = arith.truncf %70 : vector<1xf32> to vector<1xf16>
        %72 = vector.load %61[%64, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %73 = arith.addf %71, %72 : vector<1xf16>
        vector.store %73, %59[%64, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %74 = vector.extract_strided_slice %58#15 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %75 = arith.addi %63, %c17 : index
        %76 = arith.addf %74, %69 : vector<1xf32>
        %77 = arith.truncf %76 : vector<1xf32> to vector<1xf16>
        %78 = vector.load %61[%75, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %79 = arith.addf %77, %78 : vector<1xf16>
        vector.store %79, %59[%75, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %80 = vector.extract_strided_slice %58#15 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %81 = arith.addi %63, %c18 : index
        %82 = arith.addf %80, %69 : vector<1xf32>
        %83 = arith.truncf %82 : vector<1xf32> to vector<1xf16>
        %84 = vector.load %61[%81, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %85 = arith.addf %83, %84 : vector<1xf16>
        vector.store %85, %59[%81, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %86 = vector.extract_strided_slice %58#15 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %87 = arith.addi %63, %c19 : index
        %88 = arith.addf %86, %69 : vector<1xf32>
        %89 = arith.truncf %88 : vector<1xf32> to vector<1xf16>
        %90 = vector.load %61[%87, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %91 = arith.addf %89, %90 : vector<1xf16>
        vector.store %91, %59[%87, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %92 = amdgpu.mfma %58#11 * %58#5 + %58#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %93 = vector.load %alloc_0[%27, %41] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %94 = amdgpu.mfma %58#12 * %58#6 + %92 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %95 = vector.extract_strided_slice %94 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %96 = vector.load %60[%66] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %97 = arith.addf %95, %96 : vector<1xf32>
        %98 = arith.truncf %97 : vector<1xf32> to vector<1xf16>
        %99 = vector.load %61[%64, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %100 = arith.addf %98, %99 : vector<1xf16>
        vector.store %100, %59[%64, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %101 = vector.extract_strided_slice %94 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %102 = arith.addf %101, %96 : vector<1xf32>
        %103 = arith.truncf %102 : vector<1xf32> to vector<1xf16>
        %104 = vector.load %61[%75, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %105 = arith.addf %103, %104 : vector<1xf16>
        vector.store %105, %59[%75, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %106 = vector.extract_strided_slice %94 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %107 = arith.addf %106, %96 : vector<1xf32>
        %108 = arith.truncf %107 : vector<1xf32> to vector<1xf16>
        %109 = vector.load %61[%81, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %110 = arith.addf %108, %109 : vector<1xf16>
        vector.store %110, %59[%81, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %111 = vector.extract_strided_slice %94 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %112 = arith.addf %111, %96 : vector<1xf32>
        %113 = arith.truncf %112 : vector<1xf32> to vector<1xf16>
        %114 = vector.load %61[%87, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %115 = arith.addf %113, %114 : vector<1xf16>
        vector.store %115, %59[%87, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %116 = vector.load %alloc_0[%27, %31] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %117 = amdgpu.mfma %116 * %58#7 + %58#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %118 = amdgpu.mfma %116 * %58#3 + %58#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %119 = amdgpu.mfma %93 * %58#8 + %117 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %120 = amdgpu.mfma %93 * %58#4 + %118 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %121 = amdgpu.mfma %58#1 * %58#9 + %119 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %122 = amdgpu.mfma %58#1 * %58#5 + %120 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %123 = amdgpu.mfma %58#2 * %58#10 + %121 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %124 = vector.extract_strided_slice %123 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %125 = vector.load %60[%67] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %126 = arith.addf %124, %125 : vector<1xf32>
        %127 = arith.truncf %126 : vector<1xf32> to vector<1xf16>
        %128 = vector.load %61[%63, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %129 = arith.addf %127, %128 : vector<1xf16>
        vector.store %129, %59[%63, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %130 = vector.extract_strided_slice %123 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %131 = arith.addi %63, %c1 : index
        %132 = arith.addf %130, %125 : vector<1xf32>
        %133 = arith.truncf %132 : vector<1xf32> to vector<1xf16>
        %134 = vector.load %61[%131, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %135 = arith.addf %133, %134 : vector<1xf16>
        vector.store %135, %59[%131, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %136 = vector.extract_strided_slice %123 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %137 = arith.addi %63, %c2 : index
        %138 = arith.addf %136, %125 : vector<1xf32>
        %139 = arith.truncf %138 : vector<1xf32> to vector<1xf16>
        %140 = vector.load %61[%137, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %141 = arith.addf %139, %140 : vector<1xf16>
        vector.store %141, %59[%137, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %142 = vector.extract_strided_slice %123 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %143 = arith.addi %63, %c3 : index
        %144 = arith.addf %142, %125 : vector<1xf32>
        %145 = arith.truncf %144 : vector<1xf32> to vector<1xf16>
        %146 = vector.load %61[%143, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %147 = arith.addf %145, %146 : vector<1xf16>
        vector.store %147, %59[%143, %67] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %148 = amdgpu.mfma %58#2 * %58#6 + %122 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %149 = vector.extract_strided_slice %148 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %150 = vector.load %60[%66] : memref<1280xf32, strided<[1], offset: ?>>, vector<1xf32>
        %151 = arith.addf %149, %150 : vector<1xf32>
        %152 = arith.truncf %151 : vector<1xf32> to vector<1xf16>
        %153 = vector.load %61[%63, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %154 = arith.addf %152, %153 : vector<1xf16>
        vector.store %154, %59[%63, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %155 = vector.extract_strided_slice %148 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %156 = arith.addf %155, %150 : vector<1xf32>
        %157 = arith.truncf %156 : vector<1xf32> to vector<1xf16>
        %158 = vector.load %61[%131, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %159 = arith.addf %157, %158 : vector<1xf16>
        vector.store %159, %59[%131, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %160 = vector.extract_strided_slice %148 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %161 = arith.addf %160, %150 : vector<1xf32>
        %162 = arith.truncf %161 : vector<1xf32> to vector<1xf16>
        %163 = vector.load %61[%137, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %164 = arith.addf %162, %163 : vector<1xf16>
        vector.store %164, %59[%137, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %165 = vector.extract_strided_slice %148 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %166 = arith.addf %165, %150 : vector<1xf32>
        %167 = arith.truncf %166 : vector<1xf32> to vector<1xf16>
        %168 = vector.load %61[%143, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        %169 = arith.addf %167, %168 : vector<1xf16>
        vector.store %169, %59[%143, %66] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}

