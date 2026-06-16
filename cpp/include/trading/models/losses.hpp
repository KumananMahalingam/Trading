// === FILE: cpp/include/trading/models/losses.hpp ===
#pragma once

#include <torch/torch.h>

namespace trading::models {

// HybridLoss = alpha*MSE + beta*direction_loss + gamma*large_move_MAE.
//
// direction_loss   : mean fraction of sign mismatches between prediction/target.
// large_move_MAE   : mean absolute error restricted to |target| > threshold;
//                    zero when no large moves are present in the batch.
class HybridLoss {
public:
    HybridLoss(double alpha, double beta, double gamma, double large_move_threshold);

    torch::Tensor operator()(const torch::Tensor& predictions,
                             const torch::Tensor& targets) const;

private:
    double alpha_;
    double beta_;
    double gamma_;
    double large_move_threshold_;
};

}  // namespace trading::models
