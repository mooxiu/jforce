set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(XLA_PROTO_FILES 
    "${PROTO_SRC_DIR}/xla/tsl/protobuf/dnn.proto"
    "${PROTO_SRC_DIR}/xla/autotuning.proto"
    "${PROTO_SRC_DIR}/xla/autotune_results.proto"
    "${PROTO_SRC_DIR}/xla/stream_executor/cuda/cuda_compute_capability.proto"
    "${PROTO_SRC_DIR}/xla/stream_executor/device_description.proto"
    "${PROTO_SRC_DIR}/xla/service/metrics.proto"
    "${PROTO_SRC_DIR}/xla/service/hlo.proto"
    "${PROTO_SRC_DIR}/xla/xla_data.proto"
    "${PROTO_SRC_DIR}/xla/xla.proto"
    "${PROTO_SRC_DIR}/xla/pjrt/proto/compile_options.proto"
)

set(PROTO_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

add_library(xla_protos STATIC "")
set_target_properties(xla_protos PROPERTIES POSITION_INDEPENDENT_CODE ON)

# https://cmake.org/cmake/help/latest/module/FindProtobuf.html#command:protobuf_generate
# protobuf_generate_cpp will be deprecated: https://cmake.org/cmake/help/latest/module/FindProtobuf.html#command:protobuf_generate_cpp
protobuf_generate(
    TARGET        xla_protos
    LANGUAGE      cpp
    PROTOC_OUT_DIR "${PROTO_TARGET_DIR}"
    PROTOS        "${XLA_PROTO_FILES}"
    IMPORT_DIRS   
      "${PROTO_SRC_DIR}"   
      "${protobuf_SOURCE_DIR}/src"
)

target_link_libraries(xla_protos PUBLIC libprotobuf)

target_include_directories(xla_protos PUBLIC 
  "${PROTO_TARGET_DIR}"
)
