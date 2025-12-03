# Should install protobuf first: https://protobuf.dev/installation/
message(STATUS "Found Protobuf: ${PROTOBUF_LIBRARY} (Version ${PROTOBUF_VERSION})")

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

# https://cmake.org/cmake/help/latest/module/FindProtobuf.html#command:protobuf_generate
# protobuf_generate_cpp will be deprecated: https://cmake.org/cmake/help/latest/module/FindProtobuf.html#command:protobuf_generate_cpp
protobuf_generate(
    TARGET        xla_protos
    LANGUAGE      cpp
    PROTOC_OUT_DIR "${PROTO_TARGET_DIR}"
    PROTOS        "${XLA_PROTO_FILES}"
    IMPORT_DIRS   "${PROTO_SRC_DIR}"  "${Protobuf_INCLUDE_DIRS}" # <--- 添加这个
)

target_link_libraries(xla_protos PUBLIC protobuf::libprotobuf)

target_include_directories(xla_protos PUBLIC 
  "${PROTO_TARGET_DIR}"
)
