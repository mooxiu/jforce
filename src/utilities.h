#include <cstdint>
#include <string>
#include <vector>

enum class logLevel { DEBUG, ERROR };

class logger {
public:
  static void Log(std::string msg, logLevel level);
};

