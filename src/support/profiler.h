#ifndef PROFILER_H
#define PROFILER_H

#ifdef ENABLE_PROFILING
#include <chrono>
#include <mutex>
#include <string>
#include <vector>
enum class Phase {
  TOTAL, // total jit execution time

  LOWERING_SHAPE_INFER,
  LOWERING_TO_STABLEHLO,
  LOWERING_EXTRA,

  JITCOMPILE,

  EXECUTION_BUFFER_PREPARE,
  EXECUTION_RUN,
  EXECUTION_BUFFER_CLEARUP,

  // Nested in TOTAL: getOrCompileKernel (lowering + compile on a cache miss).
  KERNEL_LOOKUP,
  // Nested in EXECUTION_BUFFER_PREPARE: PJRT BufferFromHostBuffer calls.
  H2D_CALL,
  // Called by the offload plugin, outside TOTAL.
  DATA_INVALIDATE, // InvalidateBuffer, from dataSubmit
  DATA_RETRIEVE,   // RetrieveData, from dataRetrieve
  D2H_TRANSFER,    // nested in DATA_RETRIEVE: ToHostBuffer + await
  D2H_SCATTER,     // nested in DATA_RETRIEVE: host copy into the full tensor
  BUFFER_DESTROY,  // DestroyPjrtBuffer, from free
};

struct ProfilerRecord {
  std::string name;
  Phase phase;
  double durationInMicroSec;
  std::string serialize() const;
};

class Profiler {
private:
  std::vector<ProfilerRecord> records;
  std::mutex mtx;

public:
  // Should be initialized by JitManager
  Profiler() {};
  // When destroy, dumping everything
  ~Profiler();
  // Should be a singleton, so we do not allow copy constructor
  Profiler(const Profiler &) = delete;
  Profiler &operator=(const Profiler &) = delete;

  static Profiler &getInstance();

  void appendRecord(std::string kernelName, Phase phase,
                    double durationInMicroSec);
};

class ProfilerRecorder {
private:
  std::string name;
  Phase phase;
  std::chrono::high_resolution_clock::time_point start;

public:
  ProfilerRecorder(std::string kernelName, Phase phase);
  // Add recorder to profiler's records vector
  ~ProfilerRecorder();
};

#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)

#define PROFILE_SCOPE(kernelName, phase)                                       \
  ProfilerRecorder MACRO_CONCAT(rec_, __LINE__)(kernelName, phase)

#else

// If ENABLE_PROFILING not defined, will expand to nothing, so no impact on
// performance
#define PROFILE_SCOPE(kernelName, phase)

#endif
#endif
