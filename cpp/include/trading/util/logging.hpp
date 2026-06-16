// === FILE: cpp/include/trading/util/logging.hpp ===
#pragma once

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

// NOTE: This header intentionally does NOT include <spdlog/spdlog.h>.
//
// LibTorch bundles its own (newer) copy of fmt under <libtorch>/include/fmt,
// which is incompatible with the fmt version the precompiled vcpkg spdlog was
// built against. Any translation unit that includes BOTH <torch/torch.h> and
// spdlog headers fails to compile. To avoid this, spdlog is confined to a
// single Torch-free translation unit (logging.cpp / the trading_logging
// target); everywhere else we only use these plain, spdlog-free declarations.

namespace trading::log {

// Initialize the global logger once at program start.
void init();

// Emit a fully formatted message at the given level. Messages are pre-built by
// callers (see cat/fixed below), so spdlog never has to format typed args.
void info(const std::string& message);
void warn(const std::string& message);
void error(const std::string& message);

// Concatenate arbitrary values into a single std::string via ostream insertion.
template <typename... Args>
std::string cat(Args&&... args) {
    std::ostringstream os;
    (os << ... << std::forward<Args>(args));
    return os.str();
}

// Fixed-precision double -> string helper for metric reporting.
inline std::string fixed(double value, int precision) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

}  // namespace trading::log
