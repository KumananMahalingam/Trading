// === FILE: cpp/include/trading/models/dataset.hpp ===
#pragma once

#include <torch/torch.h>

#include "trading/features/scaler.hpp"
#include "trading/features/technical_indicators.hpp"
#include "trading/util/result.hpp"

namespace trading::models {

// A windowed tensor split ready for training/evaluation.
//   x : [num_samples, window, num_features]
//   y : [num_samples, 1]
struct TensorSplit {
    torch::Tensor x;
    torch::Tensor y;

    int64_t size() const { return x.size(0); }
    bool empty() const { return x.size(0) == 0; }
};

// All three splits plus the fitted feature scaler.
struct PreparedData {
    TensorSplit train;
    TensorSplit val;
    TensorSplit test;
    features::StandardScaler feature_scaler;

    // Raw (unscaled) targets and dates aligned to the test windows, for
    // reporting and plotting.
    std::vector<double> test_target_raw;
    std::vector<std::string> test_dates;
};

// Build sliding windows over a FeatureMatrix and split it temporally into
// 60/20/20 train/val/test. The feature scaler is fit on the training rows only
// (zero mean, unit variance) and applied to all splits. Targets are kept in raw
// return units.
Result<PreparedData> prepare_data(const features::FeatureMatrix& fm,
                                  int window_size,
                                  const torch::Device& device);

}  // namespace trading::models
