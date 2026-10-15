#ifndef SHARDING_EVALUATION_H
#define SHARDING_EVALUATION_H

#include "sharding-types.h"

class ShardingEvaluator {
public:
  virtual ~ShardingEvaluator() = default;

  virtual double evaluate(const ShardingDecision &sd) = 0;
};

#endif
