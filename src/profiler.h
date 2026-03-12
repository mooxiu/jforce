#ifndef PROFILER_H
#define PROFILER_H

#include <mutex>
#include <string>
#include <vector>
enum class Phase {
  JITCOMPILE,
  LOWERING,
};

struct ProfilerRecord {
  std::string name;
  Phase phase;
  double durationInMicroSec;
  bool isHot;

  std::string serialize() const;
};

class Profiler {
private:
  std::vector<ProfilerRecord> records;  
  std::mutex mtx;
                                        
public:                                 
  // Should be initialized by JitManager
  Profiler();
  // When destroy, dumping everything
  ~Profiler();
  // Should be a singleton, so we do not allow copy constructor
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete; 
    
  static Profiler& getInstance(); 

  void appendRecord(std::string kernelName, Phase phase, bool isHot, double durationInMicroSec);
};

class ProfilerRecorder {
private:
  std::string name;
  Phase phase;
  bool isHot;
  std::chrono::high_resolution_clock::time_point start;

public:
  ProfilerRecorder();
  // Add recorder to profiler's records vector
  ~ProfilerRecorder();
};

#endif
