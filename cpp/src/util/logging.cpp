// === FILE: cpp/src/util/logging.cpp ===
//
// This is the ONLY translation unit that includes spdlog. It is compiled into
// the standalone `trading_logging` target, which does not link LibTorch and so
// never sees Torch's bundled fmt headers. That keeps spdlog's fmt dependency
// (vcpkg fmt) cleanly separated from Torch's (newer) bundled fmt.

#include "trading/util/logging.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace trading::log {

void init() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    auto logger = spdlog::stdout_color_mt("trading");
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    initialized = true;
}

// Pass pre-built strings as a single argument so spdlog/fmt never parses
// replacement fields or formats typed arguments.
void info(const std::string& message) {
    spdlog::default_logger_raw()->info("{}", message);
}

void warn(const std::string& message) {
    spdlog::default_logger_raw()->warn("{}", message);
}

void error(const std::string& message) {
    spdlog::default_logger_raw()->error("{}", message);
}

}  // namespace trading::log
