#include "utilities.h"
#include <cstdlib>
#include <iostream>

void logger::Log(std::string msg, logLevel level) {
  switch (level) {
  case logLevel::DEBUG:
    std::cout << "[DEBUG] " << msg << std::endl;
    break;
  case logLevel::ERROR:
    std::cerr << "[ERROR] " << msg << std::endl;
    break;
  }
}

