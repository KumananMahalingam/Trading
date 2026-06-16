// === FILE: cpp/src/models/metrics.cpp ===
#include "trading/models/metrics.hpp"

#include <cmath>

#include "trading/config/settings.hpp"

namespace trading::models {

namespace {
double sign(double x) { return (x > 0.0) ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
}  // namespace

Metrics compute_metrics(const std::vector<double>& predictions,
                        const std::vector<double>& actuals,
                        const std::vector<double>& uncertainties) {
    Metrics m;
    const std::size_t n = predictions.size();
    if (n == 0 || actuals.size() != n) {
        return m;
    }

    // RMSE / MAE.
    double sq_sum = 0.0;
    double abs_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double err = predictions[i] - actuals[i];
        sq_sum += err * err;
        abs_sum += std::abs(err);
    }
    m.rmse = std::sqrt(sq_sum / static_cast<double>(n));
    m.mae = abs_sum / static_cast<double>(n);

    // Directional accuracy.
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (sign(predictions[i]) == sign(actuals[i])) {
            ++correct;
        }
    }
    m.directional_accuracy = static_cast<double>(correct) / static_cast<double>(n) * 100.0;

    // Signal returns: +|actual| when direction matches, else -|actual|.
    std::vector<double> signal_returns(n, 0.0);
    double sr_mean = 0.0;
    std::size_t wins = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double sr = (sign(predictions[i]) == sign(actuals[i]))
                              ? std::abs(actuals[i])
                              : -std::abs(actuals[i]);
        signal_returns[i] = sr;
        sr_mean += sr;
        if (sr > 0.0) {
            ++wins;
        }
    }
    sr_mean /= static_cast<double>(n);

    double sr_var = 0.0;
    for (double sr : signal_returns) {
        const double d = sr - sr_mean;
        sr_var += d * d;
    }
    const double sr_std = std::sqrt(sr_var / static_cast<double>(n));

    m.sharpe_ratio = (sr_std > 0.0)
                         ? (sr_mean / sr_std) * std::sqrt(config::kAnnualizationDays)
                         : 0.0;
    m.win_rate = static_cast<double>(wins) / static_cast<double>(n) * 100.0;

    // Average uncertainty.
    if (!uncertainties.empty()) {
        double u_sum = 0.0;
        for (double u : uncertainties) {
            u_sum += u;
        }
        m.avg_uncertainty = u_sum / static_cast<double>(uncertainties.size());
    }

    return m;
}

}  // namespace trading::models
