// === FILE: cpp/src/models/losses.cpp ===
#include "trading/models/losses.hpp"

namespace trading::models {

HybridLoss::HybridLoss(double alpha, double beta, double gamma,
                       double large_move_threshold)
    : alpha_(alpha),
      beta_(beta),
      gamma_(gamma),
      large_move_threshold_(large_move_threshold) {}

torch::Tensor HybridLoss::operator()(const torch::Tensor& predictions,
                                     const torch::Tensor& targets) const {
    auto mse_loss = torch::mse_loss(predictions, targets);

    // Direction loss: fraction of sign mismatches.
    auto pred_sign = torch::sign(predictions);
    auto true_sign = torch::sign(targets);
    auto direction_loss = (pred_sign != true_sign).to(torch::kFloat).mean();

    // Large-move emphasis: MAE on |target| > threshold.
    auto large_mask = targets.abs() > large_move_threshold_;
    torch::Tensor large_move_loss;
    if (large_mask.any().item<bool>()) {
        auto pred_large = predictions.masked_select(large_mask);
        auto target_large = targets.masked_select(large_mask);
        large_move_loss = (pred_large - target_large).abs().mean();
    } else {
        large_move_loss = torch::zeros({}, predictions.options());
    }

    return alpha_ * mse_loss + beta_ * direction_loss + gamma_ * large_move_loss;
}

}  // namespace trading::models
