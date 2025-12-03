# XLA_GLUE

## Dependencies
## compile time
```sh
cp ${XLA}/xla/pjrt/c/*.h ./third_party/headers/
```
But I think most header files are not needed, only `pjrt_c_api.h` is the single most important one.

### Compile Option 
To compile MLIR, we need to set compile option.
But in the `pjrt_c_api.h`, we have to set a "Serialized CompileOptionsProto". 

CompileOptionsProto is a `struct` produced by `xla/pjrt/proto/compile_options.proto`.
Which means we have to copy it from XLA project and configure protobuf to compile it to `pb.cc` files and `pb.h` files.

To make it worse, the `.proto` file also depends on others,
to make it work, we have to copy and compile the whole tree of `.proto` files.

This is done in `third_party/protos/`.

Need to install protoc, or the cmake won't work.
- Install using package manager is the simplest, just follow: https://protobuf.dev/installation/
- If don't have the permission, need to install both the compiler and the runtime: https://github.com/protocolbuffers/protobuf/tree/main/src

## library
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