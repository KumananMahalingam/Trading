// === FILE: cpp/include/trading/data/ohlcv.hpp ===
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trading::data {

// A single OHLCV bar. `date` is an ISO-8601 calendar date (YYYY-MM-DD).
struct Bar {
    std::string date;
    double open{0.0};
    double high{0.0};
    double low{0.0};
    double close{0.0};
    double volume{0.0};
};

// Column-oriented OHLCV series. All vectors share the same length.
struct OhlcvSeries {
    std::vector<std::string> dates;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    std::size_t size() const { return close.size(); }
    bool empty() const { return close.empty(); }

    void push_back(const Bar& bar) {
        dates.push_back(bar.date);
        open.push_back(bar.open);
        high.push_back(bar.high);
        low.push_back(bar.low);
        close.push_back(bar.close);
        volume.push_back(bar.volume);
    }
};

}  // namespace trading::data
