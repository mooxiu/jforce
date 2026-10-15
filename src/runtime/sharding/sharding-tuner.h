#ifndef SHARDING_TUNER_H
#define SHARDING_TUNER_H

#include "sharding-evaluator.h"
#include "sharding-types.h"


class ShardingTuner {
public:
  virtual ~ShardingTuner() = default;

  virtual ShardingDecision tune(const ShardingDecision &lastSD,
                                ShardingEvaluator &se) = 0;
};

class DFSTuner : public ShardingTuner {
public:
  ShardingDecision tune(const ShardingDecision &initial,
                        ShardingEvaluator &evaluator) override;
};

#endif
