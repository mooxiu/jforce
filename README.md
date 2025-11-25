# XLA_GLUE

## Dependencies
- compile time
```sh
cp ${XLA}/xla/pjrt/c/*.h ./third_party/headers/
```
- library

seems we have build by ourseleves:
```sh
# under xla project
bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so
```


## Compile

```sh
mkdir build && cd build
cmake -G Ninja ./..
ninja
```

## Runtime

Runtime will do following this:
- Compile the StableHLO to executable (PJRT)
- Put the argumnt vectors in a buffer (PJRT)
- Get back the result (PJRT)

Referencing: https://github.com/EnzymeAD/Reactant.jl/blob/1e62b5988e90deb790532c54823dbf41893dbc25/deps/ReactantExtra/API.cpp#L1243