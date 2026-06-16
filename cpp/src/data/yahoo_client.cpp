#include "trading/util/logging.hpp"
// === FILE: cpp/src/data/yahoo_client.cpp ===
#include "trading/data/yahoo_client.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <sstream>
#include <string>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "trading/config/secret_key.hpp"

namespace trading::data {

namespace {

using json = nlohmann::json;

// libcurl write callback that appends received bytes to a std::string.
std::size_t write_callback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    const std::size_t total = size * nmemb;
    out->append(ptr, total);
    return total;
}

bool is_leap(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

}  // namespace

Result<std::int64_t> iso_date_to_unix(std::string_view iso_date) {
    // Expect exactly "YYYY-MM-DD".
    if (iso_date.size() != 10 || iso_date[4] != '-' || iso_date[7] != '-') {
        return Result<std::int64_t>::err(
            Error(Error::Code::InvalidArgument,
                  "Invalid ISO date (expected YYYY-MM-DD): " + std::string(iso_date)));
    }

    int year = 0, month = 0, day = 0;
    try {
        year = std::stoi(std::string(iso_date.substr(0, 4)));
        month = std::stoi(std::string(iso_date.substr(5, 2)));
        day = std::stoi(std::string(iso_date.substr(8, 2)));
    } catch (...) {
        return Result<std::int64_t>::err(
            Error(Error::Code::InvalidArgument, "Non-numeric ISO date: " + std::string(iso_date)));
    }

    if (month < 1 || month > 12 || day < 1 || day > 31) {
        return Result<std::int64_t>::err(
            Error(Error::Code::InvalidArgument, "Out-of-range ISO date: " + std::string(iso_date)));
    }

    // Days from epoch (1970-01-01), computed directly to avoid timezone issues.
    static constexpr std::array<int, 12> kCumDays = {0, 31, 59, 90, 120, 151,
                                                      181, 212, 243, 273, 304, 334};
    std::int64_t days = 0;
    for (int y = 1970; y < year; ++y) {
        days += is_leap(y) ? 366 : 365;
    }
    days += kCumDays[static_cast<std::size_t>(month - 1)];
    if (month > 2 && is_leap(year)) {
        days += 1;
    }
    days += (day - 1);

    return Result<std::int64_t>::ok(days * 86400LL);
}

std::string unix_to_iso_date(std::int64_t unix_seconds) {
    const std::time_t t = static_cast<std::time_t>(unix_seconds);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d",
                  tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday);
    return std::string(buffer);
}

YahooClient::YahooClient() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();
}

YahooClient::~YahooClient() {
    if (curl_ != nullptr) {
        curl_easy_cleanup(static_cast<CURL*>(curl_));
        curl_ = nullptr;
    }
    curl_global_cleanup();
}

Result<std::string> YahooClient::http_get(const std::string& url) {
    if (curl_ == nullptr) {
        return Result<std::string>::err(
            Error(Error::Code::Network, "CURL handle was not initialized"));
    }

    auto* curl = static_cast<CURL*>(curl_);
    std::string body;

    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT,
                     "Mozilla/5.0 (compatible; trading-cpp/1.0)");

    // Attach an optional bearer token if provided via the environment.
    struct curl_slist* headers = nullptr;
    if (auto key = config::yahoo_api_key()) {
        const std::string auth = "Authorization: Bearer " + *key;
        headers = curl_slist_append(headers, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    const CURLcode rc = curl_easy_perform(curl);

    if (headers != nullptr) {
        curl_slist_free_all(headers);
    }

    if (rc != CURLE_OK) {
        return Result<std::string>::err(
            Error(Error::Code::Network,
                  std::string("HTTP request failed: ") + curl_easy_strerror(rc)));
    }

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (status < 200 || status >= 300) {
        return Result<std::string>::err(
            Error(Error::Code::Network,
                  "Unexpected HTTP status " + std::to_string(status)));
    }

    return Result<std::string>::ok(std::move(body));
}

Result<OhlcvSeries> YahooClient::fetch_daily(std::string_view ticker,
                                             std::string_view start_date,
                                             std::string_view end_date) {
    auto start_unix = iso_date_to_unix(start_date);
    if (start_unix.is_err()) {
        return Result<OhlcvSeries>::err(start_unix.error());
    }
    auto end_unix = iso_date_to_unix(end_date);
    if (end_unix.is_err()) {
        return Result<OhlcvSeries>::err(end_unix.error());
    }

    std::ostringstream url;
    url << "https://query1.finance.yahoo.com/v8/finance/chart/"
        << std::string(ticker)
        << "?period1=" << start_unix.value()
        << "&period2=" << end_unix.value()
        << "&interval=1d&events=div%2Csplit";

    log::info(log::cat("Fetching ", std::string(ticker), " from ",
                          std::string(start_date), " to ", std::string(end_date)));

    auto body = http_get(url.str());
    if (body.is_err()) {
        return Result<OhlcvSeries>::err(body.error());
    }

    json root;
    try {
        root = json::parse(body.value());
    } catch (const json::exception& e) {
        return Result<OhlcvSeries>::err(
            Error(Error::Code::Parse, std::string("JSON parse error: ") + e.what()));
    }

    // Navigate: chart.result[0].{timestamp, indicators.quote[0].*}
    if (!root.contains("chart") || root["chart"].is_null()) {
        return Result<OhlcvSeries>::err(Error(Error::Code::Parse, "Missing 'chart' node"));
    }
    const auto& chart = root["chart"];
    if (chart.contains("error") && !chart["error"].is_null()) {
        return Result<OhlcvSeries>::err(
            Error(Error::Code::Network, "Yahoo returned error: " + chart["error"].dump()));
    }
    if (!chart.contains("result") || chart["result"].empty()) {
        return Result<OhlcvSeries>::err(Error(Error::Code::Parse, "Empty 'result' array"));
    }

    const auto& result = chart["result"][0];
    if (!result.contains("timestamp") || !result.contains("indicators")) {
        return Result<OhlcvSeries>::err(
            Error(Error::Code::Parse, "Missing timestamp/indicators"));
    }

    const auto& indicators = result["indicators"];
    if (!indicators.contains("quote") || indicators["quote"].empty()) {
        return Result<OhlcvSeries>::err(
            Error(Error::Code::Parse, "Missing indicators.quote"));
    }

    const auto& timestamps = result["timestamp"];
    const auto& quote = indicators["quote"][0];

    auto column = [&quote](const char* key) -> const json& {
        static const json kEmpty = json::array();
        return quote.contains(key) ? quote[key] : kEmpty;
    };

    const auto& opens = column("open");
    const auto& highs = column("high");
    const auto& lows = column("low");
    const auto& closes = column("close");
    const auto& volumes = column("volume");

    const std::size_t n = timestamps.size();
    if (closes.size() != n) {
        return Result<OhlcvSeries>::err(
            Error(Error::Code::Parse, "Timestamp/close length mismatch"));
    }

    OhlcvSeries series;
    series.dates.reserve(n);
    series.open.reserve(n);
    series.high.reserve(n);
    series.low.reserve(n);
    series.close.reserve(n);
    series.volume.reserve(n);

    auto get_or_nan = [](const json& arr, std::size_t i) -> double {
        if (i >= arr.size() || arr[i].is_null()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return arr[i].get<double>();
    };

    for (std::size_t i = 0; i < n; ++i) {
        const double c = get_or_nan(closes, i);
        // Yahoo sometimes includes rows with all-null values; skip them.
        if (std::isnan(c)) {
            continue;
        }
        Bar bar;
        bar.date = unix_to_iso_date(timestamps[i].get<std::int64_t>());
        bar.open = get_or_nan(opens, i);
        bar.high = get_or_nan(highs, i);
        bar.low = get_or_nan(lows, i);
        bar.close = c;
        bar.volume = get_or_nan(volumes, i);
        series.push_back(bar);
    }

    if (series.empty()) {
        return Result<OhlcvSeries>::err(
            Error(Error::Code::InsufficientData, "No valid bars returned"));
    }

    log::info(log::cat("Fetched ", series.size(), " bars for ", std::string(ticker)));
    return Result<OhlcvSeries>::ok(std::move(series));
}

}  // namespace trading::data
