#include <string>

enum class logLevel {
  DEBUG, 
  ERROR
};

class logger {
public:
  static void Log(std::string msg, logLevel level);
};


std::string GetVectorAdditionOp(int size);