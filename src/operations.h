#include <string>
#include <vector>

enum class logLevel { DEBUG, ERROR };

class logger {
public:
  static void Log(std::string msg, logLevel level);
};

std::string GetVectorAdditionOp(int size);
std::string GetDotProductOp(int size);
std::string GetTransposeOp(std::vector<int> shape);
std::string GetMatrixMultiplicationOp(std::vector<int> matrix1Shape,
                                      std::vector<int> matrix2Shape);