# XLA_GLUE

## Dependencies
## compile time
```sh
export XLA_PATH="/home/muyao/projects/xla"
cp ${XLA_PATH}/xla/pjrt/c/pjrt_c_api.h ./third_party/headers/
```
But I think most header files are not needed, only `pjrt_c_api.h` is the single most important one.

### Compile Option 
To compile MLIR, we need to set compile option.
But in the `pjrt_c_api.h`, we have to set a "Serialized CompileOptionsProto". 

CompileOptionsProto is a `struct` produced by `xla/pjrt/proto/compile_options.proto`.
Which means we have to copy it from XLA project and configure protobuf to compile it to `pb.cc` files and `pb.h` files.

To make it worse, the `.proto` file also depends on others,
to make it work, we have to copy and compile the whole tree of `.proto` files.

In total, here are the protos I copied to `./third_party/protos/xla`:
```
./pjrt/proto/compile_options.proto
./service/hlo.proto
./service/metrics.proto
./tsl/protobuf/dnn.proto
./stream_executor/cuda/cuda_compute_capability.proto
./stream_executor/device_description.proto
./autotune_results.proto
./autotuning.proto
./xla.proto
./xla_data.proto
```

Need to install protoc, or the cmake won't work.
- Install using package manager is the simplest, just follow: https://protobuf.dev/installation/
- If don't have the permission, need to install both the compiler and the runtime: https://github.com/protocolbuffers/protobuf/tree/main/src

## library
seems we have build by ourseleves:
```sh
# under xla project
bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so

# if build for GPU
python3.11 ./configure.py --backend CUDA
# output be something like:
# INFO:root:Trying to find path to nvidia-smi...
# INFO:root:Found path to nvidia-smi at /usr/bin/nvidia-smi
# INFO:root:Found CUDA compute capabilities: ['8.0']
# INFO:root:Writing bazelrc to /home/muyao/projects/xla/xla_configure.bazelrc...

bazel build --config=cuda -c opt //xla/pjrt/c:pjrt_c_api_gpu_plugin.so
```

## Compile

```sh
mkdir build && cd build

# this may fail if protoc not found
cmake -G Ninja ./..
# replace the path with the place protobuf installed
cmake -G Ninja ./.. -DCMAKE_PREFIX_PATH=$HOME/opt/protobuf

ninja
```

## Runtime

When running in GPU, there might be some libraries could not be found.
Need to find those in the `external` library of `bazel-out` of XLA when compiling plugin shared library.

Note that this is also because of lack of library:
```sh
[muyao@memkf03 xla_glue]$ ./build/test_dot_product
error loading plugin: /home/muyao/projects/xla_glue/third_party/pjrt_plugins/pjrt_c_api_gpu_plugin.so: undefined symbol: ncclMemAlloc
```
In my case, I need `libnccl.so.2`.

In total, here are the libs I manually copied:
```
libnvshmem_host.so.3
nvshmem_bootstrap_uid.so.3
nvshmem_transport_ibrc.so.3
libcudnn_engines_precompiled.so.9
libcudnn_ops.so.9
libcudnn_graph.so.9
libcudnn_cnn.so.9
libcudnn_adv.so.9
libcudnn_engines_runtime_compiled.so.9
libcudnn_heuristic.so.9
libnvrtc-builtins.so.12.9
libcudnn.so.9
libnccl.so.2
```



Runtime will do following this:
- Compile the StableHLO to executable (PJRT)
- Put the argumnt vectors in a buffer (PJRT)
- Get back the result (PJRT)

Referencing: https://github.com/EnzymeAD/Reactant.jl/blob/1e62b5988e90deb790532c54823dbf41893dbc25/deps/ReactantExtra/API.cpp#L1243