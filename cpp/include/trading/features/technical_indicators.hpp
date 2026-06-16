// === FILE: cpp/include/trading/features/technical_indicators.hpp ===
#pragma once

#include <vector>

#include "trading/config/settings.hpp"
#include "trading/data/ohlcv.hpp"
#include "trading/util/result.hpp"

namespace trading::features {

// A fully prepared feature matrix:
//   - `features[t]` holds the feature values for row t. Initially these are the
//      5 indicators (SMA5, SMA20, RSI, ATR, OBV); alpha columns may be appended
//      later (see alpha_formula.hpp), so the width is variable.
//   - `feature_names` labels each column, in order.
//   - `target[t]`   holds the next-day return for row t.
//   - `dates[t]`    holds the calendar date for row t.
//   - `close[t]`    holds the (raw) close used for reference/plots.
//
// Rows are aligned: the last raw bar is dropped because its next-day return is
// undefined, and the first kWarmupRows rows are dropped because SMA20 is NaN.
struct FeatureMatrix {
    std::vector<std::vector<double>> features;  // size() rows x num_features() cols
    std::vector<double> target;
    std::vector<double> close;
    std::vector<std::string> dates;
    std::vector<std::string> feature_names;

    std::size_t size() const { return features.size(); }
    bool empty() const { return features.empty(); }
    int num_features() const { return static_cast<int>(feature_names.size()); }
};

// Individual indicator helpers. Each returns a vector the same length as the
// input series, using NaN for positions where the indicator is undefined.

// 5-day simple moving average of close.
std::vector<double> sma(const std::vector<double>& close, int window);

// 14-period RSI using a rolling mean of gains/losses (matches the Python
// implementation in technical_indicators.py).
std::vector<double> rsi(const std::vector<double>& close, int period = 14);

// 14-period Average True Range: rolling mean of
//   max(high-low, |high-prev_close|, |low-prev_close|).
std::vector<double> atr(const std::vector<double>& high,
                        const std::vector<double>& low,
                        const std::vector<double>& close,
                        int period = 14);

// On-Balance Volume: cumulative sum of sign(close_change) * volume.
std::vector<double> obv(const std::vector<double>& close,
                        const std::vector<double>& volume);

// Compute the 5 features and next-day-return target from raw OHLCV, dropping
// warm-up rows (first kWarmupRows) and the final row (undefined target).
Result<FeatureMatrix> build_feature_matrix(const data::OhlcvSeries& series);

}  // namespace trading::features
