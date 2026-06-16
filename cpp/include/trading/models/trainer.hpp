// === FILE: cpp/include/trading/models/trainer.hpp ===
#pragma once

#include <string>
#include <vector>

#include <torch/torch.h>

#include "trading/models/dataset.hpp"
#include "trading/models/metrics.hpp"
#include "trading/models/return_predictor.hpp"
#include "trading/util/result.hpp"

namespace trading::models {

// Result of a full training + evaluation run.
struct TrainingResult {
    Metrics metrics;
    std::vector<double> predictions;  // test-set MC means (raw return units)
    std::vector<double> actuals;      // test-set actual returns
    std::vector<double> uncertainties;
    std::vector<std::string> dates;   // aligned to test predictions
    int best_epoch{0};
    double best_val_loss{0.0};
};

// Train the model with AdamW, cosine-annealing LR, gradient clipping, the
// hybrid loss, and early stopping; then evaluate on the test split with
// Monte-Carlo dropout. Returns metrics + aligned predictions/actuals.
Result<TrainingResult> train_and_evaluate(ReturnPredictor& model,
                                          const PreparedData& data,
                                          const torch::Device& device);

}  // namespace trading::models
