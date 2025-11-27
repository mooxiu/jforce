#include <string>

enum class logLevel {
  DEBUG, 
  INFO, 
  ERROR
};

class logger {
public:
  static void Log(std::string msg, logLevel level);
};


std::string GetVectorAdditionOp(int size);