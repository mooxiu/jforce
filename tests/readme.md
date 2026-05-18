# Test Transform with Jforce-opt

## TODO
Use LLVM Test Suite


## Manual test
```sh
./build/src/tool/jforce-opt ./tests/axpy-wd.mlir --jforce-shape-infer
```

## Available Passes In Jforce
```sh
--jfroce-annotate
--jforce-shape-infer"
--jforce-aliasing
--jforce-translate
--jforce-trim-args
```

