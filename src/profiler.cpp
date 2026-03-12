#include "profiler.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <istream>
#include <mutex>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static std::string serializePhase(Phase phase) {
  switch (phase) {
  case Phase::JITCOMPILE: 
    return "JITCOMPILEPhase";
  case Phase::LOWERING:
    return "MLIRLOWERINGPhase";
  } 
};

/// ProfilerRecord
///
///
std::string ProfilerRecord::serialize() const {
  std::stringstream ss;
  ss << "Phase: " << serializePhase(this->phase) << "; "; 
  ss << "KernalName: " << this->name << "; ";
  ss << "IsHot: " << (this->isHot ? "True": "False")  << "; ";
  ss << "Duration(microsecond): " << this->durationInMicroSec; 
  return ss.str();
}

/// Profiler
///
///
void Profiler::appendRecord(
  std::string kernelName, 
  Phase phase, 
  bool isHot, 
  double durationInMicroSec
){
  std::unique_lock<std::mutex> wLock(mtx);
  ProfilerRecord pr = {
    .name = kernelName,
    .phase = phase,
    .durationInMicroSec = durationInMicroSec,
    .isHot = isHot
  };
  records.push_back(std::move(pr));
  return;
}

Profiler::~Profiler() {
  std::for_each(this->records.begin(), this->records.end(), [](const ProfilerRecord& r){
    std::cout << r.serialize() << "\n";
  });
};


/// ProfilerRecorder
///
///
ProfilerRecorder::ProfilerRecorder() {
  this->start = std::chrono::high_resolution_clock::now();
}

ProfilerRecorder::~ProfilerRecorder() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::micro>(now - this->start).count();
  Profiler::getInstance().appendRecord(name, phase, isHot, duration);
}
