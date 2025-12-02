#include <cstdint>
#include <string>
#include <vector>

enum class logLevel { DEBUG, ERROR };

class logger {
public:
  static void Log(std::string msg, logLevel level);
};

std::string GetVectorAdditionOp(int64_t size);
std::string GetDotProductOp(int64_t size);
std::string GetTransposeOp(std::vector<int64_t> shape);
std::string GetMatrixMultiplicationOp(std::vector<int64_t> matrix1Shape,
                                      std::vector<int64_t> matrix2Shape);