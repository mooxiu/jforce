# Test Transform with Jforce-opt

## TODO
Use LLVM Test Suite


## Manual test
```sh
./build/src/tool/jforce-opt ./tests/{name}.mlir \
   --jforce-annotate \
   --jforce-propagate-constants \
   --canonicalize --sccp --cse --canonicalize \
   --jforce-shape-infer --canonicalize \

   --convert-hlfir-to-fir --fir-to-memref --canonicalize \
   --loop-invariant-code-motion --cse --canonicalize \
   --jforce-clean-fir-loop --canonicalize --jforce-clean-fir-op  --promote-to-affine --affine-loop-normalize --canonicalize \
   --jforce-optimize-mem-ops --canonicalize \
   --jforce-loop-sink \
   --enzyme-affinecfg --jforce-outline-affine \
   --enzyme-affine-to-stablehlo --enzyme-arith-raise --canonicalize \
   --jforce-switch-outline-function --jforce-translate --inline --canonicalize \
   --stablehlo-aggressive-simplification \
   --jforce-aliasing \
   --jforce-trim-args \
   --mlir-print-ir-after-all
```

## Available Passes In Jforce
```sh
--jfroce-annotate
--jforce-shape-infer"
--jforce-aliasing
--jforce-translate
--jforce-trim-args
```

