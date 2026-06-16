// === FILE: cpp/include/trading/data/yahoo_client.hpp ===
#pragma once

#include <string>
#include <string_view>

#include "trading/data/ohlcv.hpp"
#include "trading/util/result.hpp"

namespace trading::data {

// Fetches daily OHLCV bars from the public Yahoo Finance chart endpoint using
// libcurl, and parses the JSON response with nlohmann/json.
//
// Dates are ISO calendar dates (YYYY-MM-DD). The range is half-open in spirit
// but Yahoo treats it inclusively at day granularity.
class YahooClient {
public:
    YahooClient();
    ~YahooClient();

    YahooClient(const YahooClient&) = delete;
    YahooClient& operator=(const YahooClient&) = delete;

    // Download daily bars for `ticker` between `start_date` and `end_date`.
    Result<OhlcvSeries> fetch_daily(std::string_view ticker,
                                    std::string_view start_date,
                                    std::string_view end_date);

private:
    // Perform an HTTP GET and return the response body.
    Result<std::string> http_get(const std::string& url);

    // Opaque CURL handle (void* to keep curl out of the public header).
    void* curl_{nullptr};
};

// Convert an ISO date (YYYY-MM-DD) to a UTC Unix timestamp (seconds).
Result<std::int64_t> iso_date_to_unix(std::string_view iso_date);

// Convert a UTC Unix timestamp (seconds) to an ISO date (YYYY-MM-DD).
std::string unix_to_iso_date(std::int64_t unix_seconds);

}  // namespace trading::data
