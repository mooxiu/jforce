module {
  func.func @kernel(%arg0: !fir.ref<f64>, %arg1: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg2: !fir.ref<f64> {jit.literal_val = 140725787249680 : i64}, %arg3: !fir.ref<!fir.array<?xf64>>, %arg4: !fir.ref<f64> {jit.literal_val = 4 : i64}, %arg5: !fir.ref<!fir.array<?xf64>>, %arg6: !fir.ref<f64> {jit.literal_val = 1 : i64}, %arg7: !fir.ref<!fir.array<?x?xf64>>, %arg8: !fir.ref<!fir.array<?x?xf64>>, %arg9: !fir.ref<!fir.array<?x?xf64>>, %arg10: !fir.ref<f64> {jit.literal_val = 4427486594234968593 : i64}, %arg11: !fir.ref<f64> {jit.literal_val = 140725787249512 : i64}, %arg12: !fir.ref<f64> {jit.literal_val = 4602678819172646912 : i64}, %arg13: !fir.ref<f64> {jit.literal_val = 4294967295 : i64}, %arg14: !fir.ref<f64> {jit.literal_val = 0 : i64}, %arg15: !fir.ref<!fir.array<?x?xf64>>, %arg16: !fir.ref<!fir.array<?x?xf64>>, %arg17: !fir.ref<f64> {jit.literal_val = 140724603453441 : i64}, %arg18: !fir.ref<f64> {jit.literal_val = 136931361587936 : i64}, %arg19: !fir.ref<!fir.array<?x?xf64>>, %arg20: !fir.ref<f64> {jit.literal_val = 4602678819172646912 : i64}, %arg21: !fir.ref<!fir.array<?x?xf64>>, %arg22: !fir.ref<!fir.array<?x?xf64>>, %arg23: !fir.ref<f64> {jit.literal_val = 136931361318304 : i64}, %arg24: !fir.ref<f64> {jit.literal_val = 4602678819172646912 : i64}, %arg25: !fir.ref<f64> {jit.literal_val = 4906019910204099648 : i64}, %arg26: !fir.ref<f64> {jit.literal_val = 140725787249712 : i64}, %arg27: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg28: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg29: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg30: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg31: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg32: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg33: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg34: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg35: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg36: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg37: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg38: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg39: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg40: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg41: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg42: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg43: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg44: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg45: !fir.ref<i32> {jit.literal_val = 100 : i64}, %arg46: !fir.ref<i32> {jit.literal_val = 100 : i64}) {
    %0 = fir.load %arg46 : !fir.ref<i32>
    %1 = fir.load %arg45 : !fir.ref<i32>
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c1_i32_2 = arith.constant 1 : i32
    %c1_i32_3 = arith.constant 1 : i32
    %2 = fir.load %arg44 : !fir.ref<i32>
    %c1_i32_4 = arith.constant 1 : i32
    %3 = fir.load %arg43 : !fir.ref<i32>
    %c1_i32_5 = arith.constant 1 : i32
    %4 = fir.load %arg42 : !fir.ref<i32>
    %c1_i32_6 = arith.constant 1 : i32
    %5 = fir.load %arg41 : !fir.ref<i32>
    %c1_i32_7 = arith.constant 1 : i32
    %6 = fir.load %arg40 : !fir.ref<i32>
    %c1_i32_8 = arith.constant 1 : i32
    %7 = fir.load %arg39 : !fir.ref<i32>
    %8 = fir.load %arg38 : !fir.ref<i32>
    %9 = fir.load %arg37 : !fir.ref<i32>
    %10 = fir.load %arg36 : !fir.ref<i32>
    %11 = fir.load %arg35 : !fir.ref<i32>
    %12 = fir.load %arg34 : !fir.ref<i32>
    %13 = fir.load %arg33 : !fir.ref<i32>
    %14 = fir.load %arg32 : !fir.ref<i32>
    %15 = fir.load %arg31 : !fir.ref<i32>
    %16 = arith.addi %7, %c1_i32_8 : i32
    %17 = arith.addi %6, %c1_i32_7 : i32
    %18 = arith.addi %5, %c1_i32_6 : i32
    %19 = fir.load %arg30 : !fir.ref<i32>
    %20 = fir.load %arg29 : !fir.ref<i32>
    %21 = fir.load %arg28 : !fir.ref<i32>
    %22 = arith.addi %4, %c1_i32_5 : i32
    %23 = arith.addi %3, %c1_i32_4 : i32
    %24 = fir.load %arg27 : !fir.ref<i32>
    %25 = arith.addi %2, %c1_i32_3 : i32
    %26 = fir.convert %25 : (i32) -> i64
    %27 = fir.convert %24 : (i32) -> i64
    %28 = fir.convert %23 : (i32) -> i64
    %29 = fir.convert %22 : (i32) -> i64
    %30 = fir.convert %21 : (i32) -> i64
    %31 = fir.convert %20 : (i32) -> i64
    %32 = fir.convert %19 : (i32) -> i64
    %33 = fir.convert %18 : (i32) -> i64
    %34 = fir.convert %17 : (i32) -> i64
    %35 = fir.convert %16 : (i32) -> i64
    %36 = fir.convert %15 : (i32) -> i64
    %37 = fir.convert %14 : (i32) -> i64
    %38 = fir.convert %13 : (i32) -> i64
    %39 = fir.convert %12 : (i32) -> i64
    %40 = fir.convert %11 : (i32) -> i64
    %41 = fir.convert %10 : (i32) -> i64
    %42 = fir.convert %9 : (i32) -> i64
    %43 = fir.convert %8 : (i32) -> i64
    %c0 = arith.constant 0 : index
    %44 = fir.convert %43 : (i64) -> index
    %45 = arith.cmpi sgt, %44, %c0 : index
    %c0_9 = arith.constant 0 : index
    %46 = fir.convert %42 : (i64) -> index
    %47 = arith.cmpi sgt, %46, %c0_9 : index
    %c0_10 = arith.constant 0 : index
    %48 = fir.convert %41 : (i64) -> index
    %49 = arith.cmpi sgt, %48, %c0_10 : index
    %c0_11 = arith.constant 0 : index
    %50 = fir.convert %40 : (i64) -> index
    %51 = arith.cmpi sgt, %50, %c0_11 : index
    %c0_12 = arith.constant 0 : index
    %52 = fir.convert %39 : (i64) -> index
    %53 = arith.cmpi sgt, %52, %c0_12 : index
    %c0_13 = arith.constant 0 : index
    %54 = fir.convert %38 : (i64) -> index
    %55 = arith.cmpi sgt, %54, %c0_13 : index
    %c0_14 = arith.constant 0 : index
    %56 = fir.convert %37 : (i64) -> index
    %57 = arith.cmpi sgt, %56, %c0_14 : index
    %c0_15 = arith.constant 0 : index
    %58 = fir.convert %36 : (i64) -> index
    %59 = arith.cmpi sgt, %58, %c0_15 : index
    %c0_16 = arith.constant 0 : index
    %60 = fir.convert %35 : (i64) -> index
    %61 = arith.cmpi sgt, %60, %c0_16 : index
    %c0_17 = arith.constant 0 : index
    %62 = fir.convert %34 : (i64) -> index
    %63 = arith.cmpi sgt, %62, %c0_17 : index
    %c0_18 = arith.constant 0 : index
    %64 = fir.convert %33 : (i64) -> index
    %65 = arith.cmpi sgt, %64, %c0_18 : index
    %c0_19 = arith.constant 0 : index
    %66 = fir.convert %32 : (i64) -> index
    %67 = arith.cmpi sgt, %66, %c0_19 : index
    %c0_20 = arith.constant 0 : index
    %68 = fir.convert %31 : (i64) -> index
    %69 = arith.cmpi sgt, %68, %c0_20 : index
    %c0_21 = arith.constant 0 : index
    %70 = fir.convert %30 : (i64) -> index
    %71 = arith.cmpi sgt, %70, %c0_21 : index
    %c0_22 = arith.constant 0 : index
    %72 = fir.convert %29 : (i64) -> index
    %73 = arith.cmpi sgt, %72, %c0_22 : index
    %c0_23 = arith.constant 0 : index
    %74 = fir.convert %28 : (i64) -> index
    %75 = arith.cmpi sgt, %74, %c0_23 : index
    %c0_24 = arith.constant 0 : index
    %76 = fir.convert %27 : (i64) -> index
    %77 = arith.cmpi sgt, %76, %c0_24 : index
    %c0_25 = arith.constant 0 : index
    %78 = fir.convert %26 : (i64) -> index
    %79 = arith.cmpi sgt, %78, %c0_25 : index
    %80 = arith.select %79, %78, %c0_25 : index
    %81 = arith.select %77, %76, %c0_24 : index
    %82 = arith.select %75, %74, %c0_23 : index
    %83 = arith.select %73, %72, %c0_22 : index
    %84 = arith.select %71, %70, %c0_21 : index
    %85 = arith.select %69, %68, %c0_20 : index
    %86 = arith.select %67, %66, %c0_19 : index
    %87 = arith.select %65, %64, %c0_18 : index
    %88 = arith.select %63, %62, %c0_17 : index
    %89 = arith.select %61, %60, %c0_16 : index
    %90 = arith.select %59, %58, %c0_15 : index
    %91 = arith.select %57, %56, %c0_14 : index
    %92 = arith.select %55, %54, %c0_13 : index
    %93 = arith.select %53, %52, %c0_12 : index
    %94 = arith.select %51, %50, %c0_11 : index
    %95 = arith.select %49, %48, %c0_10 : index
    %96 = arith.select %47, %46, %c0_9 : index
    %97 = arith.select %45, %44, %c0 : index
    %98:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFFrun_benchmarkEdt_min_val"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %99:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %100:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEdsx"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %101 = fir.shape %97 : (index) -> !fir.shape<1>
    %102:2 = hlfir.declare %arg3(%101) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEcelldx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %103:2 = hlfir.declare %arg4 {uniq_name = "_QFFrun_benchmarkEdsy"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %104 = fir.shape %96 : (index) -> !fir.shape<1>
    %105:2 = hlfir.declare %arg5(%104) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEcelldy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %106:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEcc"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %107 = fir.shape %95, %94 : (index, index) -> !fir.shape<2>
    %108:2 = hlfir.declare %arg7(%107) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEsoundspeed"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %109 = fir.shape %93, %92 : (index, index) -> !fir.shape<2>
    %110:2 = hlfir.declare %arg8(%109) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEviscosity_a"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %111 = fir.shape %91, %90 : (index, index) -> !fir.shape<2>
    %112:2 = hlfir.declare %arg9(%111) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEdensity0"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %113:2 = hlfir.declare %arg10 {uniq_name = "_QFFrun_benchmarkEg_small"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %114:2 = hlfir.declare %arg11 {uniq_name = "_QFFrun_benchmarkEdtct"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %115:2 = hlfir.declare %arg12 {uniq_name = "_QFFrun_benchmarkEdtc_safe"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %116:2 = hlfir.declare %arg13 {uniq_name = "_QFFrun_benchmarkEdiv"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %117:2 = hlfir.declare %arg14 {uniq_name = "_QFFrun_benchmarkEdv1"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %118 = fir.shape %89, %88 : (index, index) -> !fir.shape<2>
    %119:2 = hlfir.declare %arg15(%118) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkExvel0"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %120 = fir.shape %87, %86 : (index, index) -> !fir.shape<2>
    %121:2 = hlfir.declare %arg16(%120) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkExarea"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %122:2 = hlfir.declare %arg17 {uniq_name = "_QFFrun_benchmarkEdv2"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %123:2 = hlfir.declare %arg18 {uniq_name = "_QFFrun_benchmarkEdtut"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %124 = fir.shape %85, %84 : (index, index) -> !fir.shape<2>
    %125:2 = hlfir.declare %arg19(%124) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEvolume"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %126:2 = hlfir.declare %arg20 {uniq_name = "_QFFrun_benchmarkEdtu_safe"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %127 = fir.shape %83, %82 : (index, index) -> !fir.shape<2>
    %128:2 = hlfir.declare %arg21(%127) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEyvel0"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %129 = fir.shape %81, %80 : (index, index) -> !fir.shape<2>
    %130:2 = hlfir.declare %arg22(%129) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFFrun_benchmarkEyarea"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %131:2 = hlfir.declare %arg23 {uniq_name = "_QFFrun_benchmarkEdtdivt"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %132:2 = hlfir.declare %arg24 {uniq_name = "_QFFrun_benchmarkEdtdiv_safe"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %133:2 = hlfir.declare %arg25 {uniq_name = "_QFFrun_benchmarkEg_big"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %134:2 = hlfir.declare %arg26 {uniq_name = "_QFFrun_benchmarkEdtvt"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %135 = fir.convert %c1_i32 : (i32) -> index
    %136 = fir.convert %1 : (i32) -> index
    %137 = fir.convert %c1_i32_1 : (i32) -> index
    fir.do_loop %arg47 = %135 to %136 step %137 {
      %138 = fir.convert %c1_i32_0 : (i32) -> index
      %139 = fir.convert %0 : (i32) -> index
      %140 = fir.convert %c1_i32_2 : (i32) -> index
      fir.do_loop %arg48 = %138 to %139 step %140 {
        %141 = fir.convert %arg48 : (index) -> i32
        %142 = fir.convert %arg47 : (index) -> i32
        %143 = fir.alloca i32
        %144:2 = hlfir.declare %143 {uniq_name = "_QFFrun_benchmarkEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        %145 = fir.alloca i32
        %146:2 = hlfir.declare %145 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        hlfir.assign %142 to %144#0 : i32, !fir.ref<i32>
        hlfir.assign %141 to %146#0 : i32, !fir.ref<i32>
        %147 = fir.load %146#0 : !fir.ref<i32>
        %148 = fir.convert %147 : (i32) -> i64
        %149 = hlfir.designate %102#0 (%148)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
        %150 = fir.load %149 : !fir.ref<f64>
        hlfir.assign %150 to %100#0 : f64, !fir.ref<f64>
        %151 = fir.load %144#0 : !fir.ref<i32>
        %152 = fir.convert %151 : (i32) -> i64
        %153 = hlfir.designate %105#0 (%152)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
        %154 = fir.load %153 : !fir.ref<f64>
        hlfir.assign %154 to %103#0 : f64, !fir.ref<f64>
        %155 = fir.load %146#0 : !fir.ref<i32>
        %156 = fir.convert %155 : (i32) -> i64
        %157 = fir.load %144#0 : !fir.ref<i32>
        %158 = fir.convert %157 : (i32) -> i64
        %159 = hlfir.designate %108#0 (%156, %158)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %160 = fir.load %159 : !fir.ref<f64>
        %161 = fir.load %146#0 : !fir.ref<i32>
        %162 = fir.convert %161 : (i32) -> i64
        %163 = fir.load %144#0 : !fir.ref<i32>
        %164 = fir.convert %163 : (i32) -> i64
        %165 = hlfir.designate %108#0 (%162, %164)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %166 = fir.load %165 : !fir.ref<f64>
        %167 = arith.mulf %160, %166 fastmath<contract> : f64
        hlfir.assign %167 to %106#0 : f64, !fir.ref<f64>
        %168 = fir.load %106#0 : !fir.ref<f64>
        %cst = arith.constant 2.000000e+00 : f64
        %169 = fir.load %146#0 : !fir.ref<i32>
        %170 = fir.convert %169 : (i32) -> i64
        %171 = fir.load %144#0 : !fir.ref<i32>
        %172 = fir.convert %171 : (i32) -> i64
        %173 = hlfir.designate %110#0 (%170, %172)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %174 = fir.load %173 : !fir.ref<f64>
        %175 = arith.mulf %cst, %174 fastmath<contract> : f64
        %176 = fir.load %146#0 : !fir.ref<i32>
        %177 = fir.convert %176 : (i32) -> i64
        %178 = fir.load %144#0 : !fir.ref<i32>
        %179 = fir.convert %178 : (i32) -> i64
        %180 = hlfir.designate %112#0 (%177, %179)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %181 = fir.load %180 : !fir.ref<f64>
        %182 = arith.divf %175, %181 fastmath<contract> : f64
        %183 = arith.addf %168, %182 fastmath<contract> : f64
        hlfir.assign %183 to %106#0 : f64, !fir.ref<f64>
        %184 = fir.load %106#0 : !fir.ref<f64>
        %185 = math.sqrt %184 fastmath<contract> : f64
        %186 = fir.load %113#0 : !fir.ref<f64>
        %187 = arith.cmpf ogt, %185, %186 fastmath<contract> : f64
        %188 = arith.select %187, %185, %186 : f64
        hlfir.assign %188 to %106#0 : f64, !fir.ref<f64>
        %189 = fir.load %115#0 : !fir.ref<f64>
        %190 = fir.load %100#0 : !fir.ref<f64>
        %191 = fir.load %103#0 : !fir.ref<f64>
        %192 = arith.cmpf olt, %190, %191 fastmath<contract> : f64
        %193 = arith.select %192, %190, %191 : f64
        %194 = arith.mulf %189, %193 fastmath<contract> : f64
        %195 = fir.load %106#0 : !fir.ref<f64>
        %196 = arith.divf %194, %195 fastmath<contract> : f64
        hlfir.assign %196 to %114#0 : f64, !fir.ref<f64>
        %cst_26 = arith.constant 0.000000e+00 : f64
        hlfir.assign %cst_26 to %116#0 : f64, !fir.ref<f64>
        %197 = fir.load %146#0 : !fir.ref<i32>
        %198 = fir.convert %197 : (i32) -> i64
        %199 = fir.load %144#0 : !fir.ref<i32>
        %200 = fir.convert %199 : (i32) -> i64
        %201 = hlfir.designate %119#0 (%198, %200)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %202 = fir.load %201 : !fir.ref<f64>
        %203 = fir.load %146#0 : !fir.ref<i32>
        %204 = fir.convert %203 : (i32) -> i64
        %205 = fir.load %144#0 : !fir.ref<i32>
        %c1_i32_27 = arith.constant 1 : i32
        %206 = arith.addi %205, %c1_i32_27 overflow<nsw> : i32
        %207 = fir.convert %206 : (i32) -> i64
        %208 = hlfir.designate %119#0 (%204, %207)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %209 = fir.load %208 : !fir.ref<f64>
        %210 = arith.addf %202, %209 fastmath<contract> : f64
        %211 = hlfir.no_reassoc %210 : f64
        %212 = fir.load %146#0 : !fir.ref<i32>
        %213 = fir.convert %212 : (i32) -> i64
        %214 = fir.load %144#0 : !fir.ref<i32>
        %215 = fir.convert %214 : (i32) -> i64
        %216 = hlfir.designate %121#0 (%213, %215)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %217 = fir.load %216 : !fir.ref<f64>
        %218 = arith.mulf %211, %217 fastmath<contract> : f64
        hlfir.assign %218 to %117#0 : f64, !fir.ref<f64>
        %219 = fir.load %146#0 : !fir.ref<i32>
        %c1_i32_28 = arith.constant 1 : i32
        %220 = arith.addi %219, %c1_i32_28 overflow<nsw> : i32
        %221 = fir.convert %220 : (i32) -> i64
        %222 = fir.load %144#0 : !fir.ref<i32>
        %223 = fir.convert %222 : (i32) -> i64
        %224 = hlfir.designate %119#0 (%221, %223)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %225 = fir.load %224 : !fir.ref<f64>
        %226 = fir.load %146#0 : !fir.ref<i32>
        %c1_i32_29 = arith.constant 1 : i32
        %227 = arith.addi %226, %c1_i32_29 overflow<nsw> : i32
        %228 = fir.convert %227 : (i32) -> i64
        %229 = fir.load %144#0 : !fir.ref<i32>
        %c1_i32_30 = arith.constant 1 : i32
        %230 = arith.addi %229, %c1_i32_30 overflow<nsw> : i32
        %231 = fir.convert %230 : (i32) -> i64
        %232 = hlfir.designate %119#0 (%228, %231)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %233 = fir.load %232 : !fir.ref<f64>
        %234 = arith.addf %225, %233 fastmath<contract> : f64
        %235 = hlfir.no_reassoc %234 : f64
        %236 = fir.load %146#0 : !fir.ref<i32>
        %c1_i32_31 = arith.constant 1 : i32
        %237 = arith.addi %236, %c1_i32_31 overflow<nsw> : i32
        %238 = fir.convert %237 : (i32) -> i64
        %239 = fir.load %144#0 : !fir.ref<i32>
        %240 = fir.convert %239 : (i32) -> i64
        %241 = hlfir.designate %121#0 (%238, %240)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %242 = fir.load %241 : !fir.ref<f64>
        %243 = arith.mulf %235, %242 fastmath<contract> : f64
        hlfir.assign %243 to %122#0 : f64, !fir.ref<f64>
        %244 = fir.load %116#0 : !fir.ref<f64>
        %245 = fir.load %122#0 : !fir.ref<f64>
        %246 = arith.addf %244, %245 fastmath<contract> : f64
        %247 = fir.load %117#0 : !fir.ref<f64>
        %248 = arith.subf %246, %247 fastmath<contract> : f64
        hlfir.assign %248 to %116#0 : f64, !fir.ref<f64>
        %249 = fir.load %126#0 : !fir.ref<f64>
        %cst_32 = arith.constant 2.000000e+00 : f64
        %250 = arith.mulf %249, %cst_32 fastmath<contract> : f64
        %251 = fir.load %146#0 : !fir.ref<i32>
        %252 = fir.convert %251 : (i32) -> i64
        %253 = fir.load %144#0 : !fir.ref<i32>
        %254 = fir.convert %253 : (i32) -> i64
        %255 = hlfir.designate %125#0 (%252, %254)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %256 = fir.load %255 : !fir.ref<f64>
        %257 = arith.mulf %250, %256 fastmath<contract> : f64
        %258 = fir.load %117#0 : !fir.ref<f64>
        %259 = math.absf %258 fastmath<contract> : f64
        %260 = fir.load %122#0 : !fir.ref<f64>
        %261 = math.absf %260 fastmath<contract> : f64
        %262 = fir.load %113#0 : !fir.ref<f64>
        %263 = fir.load %146#0 : !fir.ref<i32>
        %264 = fir.convert %263 : (i32) -> i64
        %265 = fir.load %144#0 : !fir.ref<i32>
        %266 = fir.convert %265 : (i32) -> i64
        %267 = hlfir.designate %125#0 (%264, %266)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %268 = fir.load %267 : !fir.ref<f64>
        %269 = arith.mulf %262, %268 fastmath<contract> : f64
        %270 = arith.cmpf ogt, %259, %261 fastmath<contract> : f64
        %271 = arith.select %270, %259, %261 : f64
        %272 = arith.cmpf ogt, %271, %269 fastmath<contract> : f64
        %273 = arith.select %272, %271, %269 : f64
        %274 = arith.divf %257, %273 fastmath<contract> : f64
        hlfir.assign %274 to %123#0 : f64, !fir.ref<f64>
        %275 = fir.load %146#0 : !fir.ref<i32>
        %276 = fir.convert %275 : (i32) -> i64
        %277 = fir.load %144#0 : !fir.ref<i32>
        %278 = fir.convert %277 : (i32) -> i64
        %279 = hlfir.designate %128#0 (%276, %278)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %280 = fir.load %279 : !fir.ref<f64>
        %281 = fir.load %146#0 : !fir.ref<i32>
        %c1_i32_33 = arith.constant 1 : i32
        %282 = arith.addi %281, %c1_i32_33 overflow<nsw> : i32
        %283 = fir.convert %282 : (i32) -> i64
        %284 = fir.load %144#0 : !fir.ref<i32>
        %285 = fir.convert %284 : (i32) -> i64
        %286 = hlfir.designate %128#0 (%283, %285)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %287 = fir.load %286 : !fir.ref<f64>
        %288 = arith.addf %280, %287 fastmath<contract> : f64
        %289 = hlfir.no_reassoc %288 : f64
        %290 = fir.load %146#0 : !fir.ref<i32>
        %291 = fir.convert %290 : (i32) -> i64
        %292 = fir.load %144#0 : !fir.ref<i32>
        %293 = fir.convert %292 : (i32) -> i64
        %294 = hlfir.designate %130#0 (%291, %293)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %295 = fir.load %294 : !fir.ref<f64>
        %296 = arith.mulf %289, %295 fastmath<contract> : f64
        hlfir.assign %296 to %117#0 : f64, !fir.ref<f64>
        %297 = fir.load %146#0 : !fir.ref<i32>
        %298 = fir.convert %297 : (i32) -> i64
        %299 = fir.load %144#0 : !fir.ref<i32>
        %c1_i32_34 = arith.constant 1 : i32
        %300 = arith.addi %299, %c1_i32_34 overflow<nsw> : i32
        %301 = fir.convert %300 : (i32) -> i64
        %302 = hlfir.designate %128#0 (%298, %301)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %303 = fir.load %302 : !fir.ref<f64>
        %304 = fir.load %146#0 : !fir.ref<i32>
        %c1_i32_35 = arith.constant 1 : i32
        %305 = arith.addi %304, %c1_i32_35 overflow<nsw> : i32
        %306 = fir.convert %305 : (i32) -> i64
        %307 = fir.load %144#0 : !fir.ref<i32>
        %c1_i32_36 = arith.constant 1 : i32
        %308 = arith.addi %307, %c1_i32_36 overflow<nsw> : i32
        %309 = fir.convert %308 : (i32) -> i64
        %310 = hlfir.designate %128#0 (%306, %309)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %311 = fir.load %310 : !fir.ref<f64>
        %312 = arith.addf %303, %311 fastmath<contract> : f64
        %313 = hlfir.no_reassoc %312 : f64
        %314 = fir.load %146#0 : !fir.ref<i32>
        %315 = fir.convert %314 : (i32) -> i64
        %316 = fir.load %144#0 : !fir.ref<i32>
        %c1_i32_37 = arith.constant 1 : i32
        %317 = arith.addi %316, %c1_i32_37 overflow<nsw> : i32
        %318 = fir.convert %317 : (i32) -> i64
        %319 = hlfir.designate %130#0 (%315, %318)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %320 = fir.load %319 : !fir.ref<f64>
        %321 = arith.mulf %313, %320 fastmath<contract> : f64
        hlfir.assign %321 to %122#0 : f64, !fir.ref<f64>
        %322 = fir.load %116#0 : !fir.ref<f64>
        %323 = fir.load %122#0 : !fir.ref<f64>
        %324 = arith.addf %322, %323 fastmath<contract> : f64
        %325 = fir.load %117#0 : !fir.ref<f64>
        %326 = arith.subf %324, %325 fastmath<contract> : f64
        hlfir.assign %326 to %116#0 : f64, !fir.ref<f64>
        %327 = fir.load %116#0 : !fir.ref<f64>
        %cst_38 = arith.constant 2.000000e+00 : f64
        %328 = fir.load %146#0 : !fir.ref<i32>
        %329 = fir.convert %328 : (i32) -> i64
        %330 = fir.load %144#0 : !fir.ref<i32>
        %331 = fir.convert %330 : (i32) -> i64
        %332 = hlfir.designate %125#0 (%329, %331)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %333 = fir.load %332 : !fir.ref<f64>
        %334 = arith.mulf %cst_38, %333 fastmath<contract> : f64
        %335 = hlfir.no_reassoc %334 : f64
        %336 = arith.divf %327, %335 fastmath<contract> : f64
        hlfir.assign %336 to %116#0 : f64, !fir.ref<f64>
        %337 = fir.load %116#0 : !fir.ref<f64>
        %338 = fir.load %113#0 : !fir.ref<f64>
        %339 = arith.negf %338 fastmath<contract> : f64
        %340 = arith.cmpf olt, %337, %339 fastmath<contract> : f64
        fir.if %340 {
          %367 = fir.load %132#0 : !fir.ref<f64>
          %cst_42 = arith.constant 1.000000e+00 : f64
          %368 = fir.load %116#0 : !fir.ref<f64>
          %369 = arith.divf %cst_42, %368 fastmath<contract> : f64
          %370 = arith.negf %369 fastmath<contract> : f64
          %371 = hlfir.no_reassoc %370 : f64
          %372 = arith.mulf %367, %371 fastmath<contract> : f64
          hlfir.assign %372 to %131#0 : f64, !fir.ref<f64>
        } else {
          %367 = fir.load %133#0 : !fir.ref<f64>
          hlfir.assign %367 to %131#0 : f64, !fir.ref<f64>
        }
        %341 = fir.load %98#0 : !fir.ref<f64>
        %342 = fir.load %114#0 : !fir.ref<f64>
        %343 = fir.load %123#0 : !fir.ref<f64>
        %344 = fir.load %134#0 : !fir.ref<f64>
        %345 = fir.load %131#0 : !fir.ref<f64>
        %346 = arith.cmpf olt, %341, %342 fastmath<contract> : f64
        %347 = arith.select %346, %341, %342 : f64
        %348 = arith.cmpf olt, %347, %343 fastmath<contract> : f64
        %349 = arith.select %348, %347, %343 : f64
        %350 = arith.cmpf olt, %349, %344 fastmath<contract> : f64
        %351 = arith.select %350, %349, %344 : f64
        %352 = arith.cmpf olt, %351, %345 fastmath<contract> : f64
        %353 = arith.select %352, %351, %345 : f64
        hlfir.assign %353 to %98#0 : f64, !fir.ref<f64>
        %354 = fir.load %99#0 : !fir.ref<i32>
        %c1_i32_39 = arith.constant 1 : i32
        %355 = fir.load %99#0 : !fir.ref<i32>
        %c1_i32_40 = arith.constant 1 : i32
        %356 = arith.addi %142, %c1_i32_39 : i32
        %c0_i32 = arith.constant 0 : i32
        %357 = arith.cmpi slt, %c1_i32_39, %c0_i32 : i32
        %358 = arith.cmpi slt, %356, %354 : i32
        %359 = arith.cmpi sgt, %356, %354 : i32
        %360 = arith.select %357, %358, %359 : i1
        %361 = arith.addi %141, %c1_i32_40 : i32
        %c0_i32_41 = arith.constant 0 : i32
        %362 = arith.cmpi slt, %c1_i32_40, %c0_i32_41 : i32
        %363 = arith.cmpi slt, %361, %355 : i32
        %364 = arith.cmpi sgt, %361, %355 : i32
        %365 = arith.select %362, %363, %364 : i1
        %366 = arith.andi %360, %365 : i1
        fir.if %366 {
          hlfir.assign %356 to %144#0 : i32, !fir.ref<i32>
          hlfir.assign %361 to %146#0 : i32, !fir.ref<i32>
          %367 = fir.load %144#0 : !fir.ref<i32>
          hlfir.assign %367 to %144#0 : i32, !fir.ref<i32>
          %368 = fir.load %146#0 : !fir.ref<i32>
          hlfir.assign %368 to %146#0 : i32, !fir.ref<i32>
        }
      }
    }
    omp.terminator
  }
}

