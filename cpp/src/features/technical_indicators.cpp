// === FILE: cpp/src/features/technical_indicators.cpp ===
#include "trading/features/technical_indicators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace trading::features {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

std::vector<double> sma(const std::vector<double>& close, int window) {
    const std::size_t n = close.size();
    std::vector<double> out(n, kNaN);
    if (window <= 0 || n < static_cast<std::size_t>(window)) {
        return out;
    }
    double running = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        running += close[i];
        if (i >= static_cast<std::size_t>(window)) {
            running -= close[i - static_cast<std::size_t>(window)];
        }
        if (i + 1 >= static_cast<std::size_t>(window)) {
            out[i] = running / static_cast<double>(window);
        }
    }
    return out;
}

std::vector<double> rsi(const std::vector<double>& close, int period) {
    const std::size_t n = close.size();
    std::vector<double> out(n, kNaN);
    if (period <= 0 || n <= static_cast<std::size_t>(period)) {
        return out;
    }

    // delta = close.diff(); gain = positive deltas, loss = -negative deltas.
    std::vector<double> gain(n, 0.0);
    std::vector<double> loss(n, 0.0);
    for (std::size_t i = 1; i < n; ++i) {
        const double delta = close[i] - close[i - 1];
        gain[i] = delta > 0.0 ? delta : 0.0;
        loss[i] = delta < 0.0 ? -delta : 0.0;
    }

    // Rolling mean of gains/losses over `period`. The diff at index 0 is NaN in
    // pandas, so the first defined RSI value lands at index `period`.
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        gain_sum += gain[i];
        loss_sum += loss[i];
        if (i > static_cast<std::size_t>(period)) {
            gain_sum -= gain[i - static_cast<std::size_t>(period)];
            loss_sum -= loss[i - static_cast<std::size_t>(period)];
        }
        if (i >= static_cast<std::size_t>(period)) {
            const double avg_gain = gain_sum / static_cast<double>(period);
            const double avg_loss = loss_sum / static_cast<double>(period);
            if (avg_loss == 0.0) {
                // rs -> inf, RSI -> 100.
                out[i] = 100.0;
            } else {
                const double rs = avg_gain / avg_loss;
                out[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
    }
    return out;
}

std::vector<double> atr(const std::vector<double>& high,
                        const std::vector<double>& low,
                        const std::vector<double>& close,
                        int period) {
    const std::size_t n = close.size();
    std::vector<double> out(n, kNaN);
    if (period <= 0 || n == 0) {
        return out;
    }

    // True range. At index 0 the previous close is undefined; pandas leaves
    // high-close/low-close as NaN there, so true_range[0] = high-low.
    std::vector<double> tr(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        const double hl = high[i] - low[i];
        if (i == 0) {
            tr[i] = hl;
        } else {
            const double hc = std::abs(high[i] - close[i - 1]);
            const double lc = std::abs(low[i] - close[i - 1]);
            tr[i] = std::max(hl, std::max(hc, lc));
        }
    }

    // Rolling mean over `period`.
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += tr[i];
        if (i >= static_cast<std::size_t>(period)) {
            sum -= tr[i - static_cast<std::size_t>(period)];
        }
        if (i + 1 >= static_cast<std::size_t>(period)) {
            out[i] = sum / static_cast<double>(period);
        }
    }
    return out;
}

std::vector<double> obv(const std::vector<double>& close,
                        const std::vector<double>& volume) {
    const std::size_t n = close.size();
    std::vector<double> out(n, 0.0);
    if (n == 0) {
        return out;
    }
    double running = 0.0;
    out[0] = 0.0;  // diff at index 0 is NaN -> sign 0 -> contributes 0.
    for (std::size_t i = 1; i < n; ++i) {
        const double diff = close[i] - close[i - 1];
        const double sign = (diff > 0.0) ? 1.0 : (diff < 0.0 ? -1.0 : 0.0);
        running += sign * volume[i];
        out[i] = running;
    }
    return out;
}

Result<FeatureMatrix> build_feature_matrix(const data::OhlcvSeries& series) {
    const std::size_t n = series.size();
    const int warmup = config::kWarmupRows;

    // Need at least warmup + 2 rows: warmup for SMA20, +1 row to drop for the
    // undefined final target, +1 usable row.
    if (n < static_cast<std::size_t>(warmup) + 2) {
        return Result<FeatureMatrix>::err(
            Error(Error::Code::InsufficientData,
                  "Not enough bars to compute features (have " +
                      std::to_string(n) + ")"));
    }

    const auto sma5 = sma(series.close, 5);
    const auto sma20 = sma(series.close, 20);
    const auto rsi14 = rsi(series.close, 14);
    const auto atr14 = atr(series.high, series.low, series.close, 14);
    const auto obv_v = obv(series.close, series.volume);

    // Next-day return target: (close[t+1] - close[t]) / close[t]; the final
    // row's target is undefined, so it is excluded by iterating to n-1.
    FeatureMatrix fm;
    fm.feature_names = {"SMA5", "SMA20", "RSI", "ATR", "OBV"};
    fm.features.reserve(n);
    fm.target.reserve(n);
    fm.close.reserve(n);
    fm.dates.reserve(n);

    for (std::size_t t = static_cast<std::size_t>(warmup); t + 1 < n; ++t) {
        // Guard against any residual NaN in the 5 features.
        if (std::isnan(sma5[t]) || std::isnan(sma20[t]) || std::isnan(rsi14[t]) ||
            std::isnan(atr14[t]) || std::isnan(obv_v[t])) {
            continue;
        }
        if (series.close[t] == 0.0) {
            continue;
        }
        const double next_return =
            (series.close[t + 1] - series.close[t]) / series.close[t];

        fm.features.push_back({sma5[t], sma20[t], rsi14[t], atr14[t], obv_v[t]});
        fm.target.push_back(next_return);
        fm.close.push_back(series.close[t]);
        fm.dates.push_back(series.dates[t]);
    }

    if (fm.size() < static_cast<std::size_t>(config::kWindowSize) + 10) {
        return Result<FeatureMatrix>::err(
            Error(Error::Code::InsufficientData,
                  "Too few usable rows after feature computation: " +
                      std::to_string(fm.size())));
    }

    return Result<FeatureMatrix>::ok(std::move(fm));
}

}  // namespace trading::features
