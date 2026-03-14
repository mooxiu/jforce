#include <unordered_map>
#ifdef ENABLE_PROFILING
#include "profiler.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static std::string serializePhase(Phase phase) {
  switch (phase) {
    case Phase::TOTAL:
      return "Total";
    case Phase::LOWERING_SHAPE_INFER:

      return "LOWERING_ShapeInference";
    case Phase::LOWERING_TO_STABLEHLO:
      return "LOWERING_TranslateToStableHLO";
    case Phase::LOWERING_EXTRA:
      return "LOWERING_Extra";

    case Phase::JITCOMPILE: 
      return "JITCOMPILE";

    case Phase::EXECUTION_BUFFER_PREPARE:
      return "Execution_BufferPrepare";
    case Phase::EXECUTION_RUN:
      return "Execution_Run";
    case Phase::EXECUTION_BUFFER_CLEARUP:
      return "Execution_BufferClearUp";
  } 
};

/// ProfilerRecord
///
///
std::string ProfilerRecord::serialize() const {
  std::stringstream ss;
  ss << "Phase: " << serializePhase(this->phase) << "; "; 
  ss << "KernalName: " << this->name << "; ";
  ss << "Duration(microsecond): " << this->durationInMicroSec; 
  return ss.str();
}

/// Profiler
///
///
Profiler& Profiler::getInstance(){
  static Profiler profiler;
  return profiler;
}

void Profiler::appendRecord(
  std::string kernelName, 
  Phase phase, 
  double durationInMicroSec
){
  std::unique_lock<std::mutex> wLock(mtx);
  ProfilerRecord pr = {
    .name = kernelName,
    .phase = phase,
    .durationInMicroSec = durationInMicroSec,
  };
  records.push_back(std::move(pr));
  return;
}

Profiler::~Profiler() {
  std::cout << "\n --- print profiling info --- \n";
  std::for_each(this->records.begin(), this->records.end(), [](const ProfilerRecord& r){
    std::cout << r.serialize() << "\n";
    if (r.phase == Phase::TOTAL) {
      std::cout << "\n";
    }
  });

  std::cout << "\n --- print profiling summary ---\n";
  std::unordered_map<Phase, double> perPhaseTimes;
  for (int i = 0; i < this->records.size(); i++) {
    perPhaseTimes[this->records[i].phase] += this->records[i].durationInMicroSec;
  }

  auto timeTotal = perPhaseTimes[Phase::TOTAL];
  auto printHLPhase = [timeTotal](double time, std::string desc){
    std::cout << "> " << desc 
      << ": " << time << " mus"
      << ", Ratio: " << time/timeTotal * 100 << "%\n";
  };
  auto printLLPhase = [timeTotal](double time, std::string desc){
    std::cout << "  >> " << desc 
      << ": " << time << " mus"
      << ", Ratio: " << time/timeTotal * 100 << "%\n";
  };


  auto timeLoweringShapeInfer= perPhaseTimes[Phase::LOWERING_SHAPE_INFER];
  auto timeLoweringToStableHLO = perPhaseTimes[Phase::LOWERING_TO_STABLEHLO];
  auto timeLoweringExtra = perPhaseTimes[Phase::LOWERING_EXTRA];
  auto totalLowering = timeLoweringShapeInfer + timeLoweringToStableHLO + timeLoweringExtra;
  printHLPhase(totalLowering, "Lowering: Shape Inference and translate");
  printLLPhase(timeLoweringShapeInfer, "Shape Inference");
  printLLPhase(timeLoweringToStableHLO, "Translate To StableHLO");
  printLLPhase(timeLoweringExtra, "Lowering Extra");

  printHLPhase(perPhaseTimes[Phase::JITCOMPILE], "JIT COMPILE");

  auto timeExecutionBufferPrepare = perPhaseTimes[Phase::EXECUTION_BUFFER_PREPARE];
  auto timeExecutionRun = perPhaseTimes[Phase::EXECUTION_RUN];
  auto timeExecutionBufferClearUp = perPhaseTimes[Phase::EXECUTION_BUFFER_CLEARUP];
  auto totalExecution = timeExecutionBufferPrepare + timeExecutionRun + timeExecutionBufferClearUp;
  printHLPhase(totalExecution, "Execution");
  printLLPhase(timeExecutionBufferPrepare, "Buffer Preparation");
  printLLPhase(timeExecutionRun, "Pure Execution");
  printLLPhase(timeExecutionBufferClearUp, "Buffer ClearUp");


  auto timeLeft = timeTotal - totalLowering - perPhaseTimes[Phase::JITCOMPILE] - totalExecution;
  printHLPhase(timeLeft, "Data Preparation");
  std::cout << "TOTAL: " << perPhaseTimes[Phase::TOTAL] << " mus\n";
};


/// ProfilerRecorder
///
///
ProfilerRecorder::ProfilerRecorder(
  std::string kernelName, 
  Phase phase 
) {
  this->start = std::chrono::high_resolution_clock::now();
  this->name = kernelName;
  this->phase = phase;
}

ProfilerRecorder::~ProfilerRecorder() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::micro>(now - this->start).count();
  Profiler::getInstance().appendRecord(name, phase, duration);
}
#endif
