// === FILE: cpp/include/trading/models/metrics.hpp ===
#pragma once

#include <vector>

namespace trading::models {

// Test-set performance metrics (mirrors metrics.py, reduced to the requested
// set plus a couple of useful extras).
struct Metrics {
    double rmse{0.0};
    double mae{0.0};
    double directional_accuracy{0.0};  // percent
    double sharpe_ratio{0.0};          // annualized
    double win_rate{0.0};              // percent
    double avg_uncertainty{0.0};
};

// Compute metrics from aligned predictions/actuals (raw return units) and the
// per-sample MC-dropout uncertainties.
Metrics compute_metrics(const std::vector<double>& predictions,
                        const std::vector<double>& actuals,
                        const std::vector<double>& uncertainties);

}  // namespace trading::models
