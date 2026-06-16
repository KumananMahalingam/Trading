// === FILE: cpp/src/config/secret_key.cpp ===
#include "trading/config/secret_key.hpp"

#include <cstdlib>

namespace trading::config {

std::optional<std::string> get_env(std::string_view name) {
    // std::getenv requires a null-terminated string.
    const std::string key(name);
#if defined(_WIN32)
    // _dupenv_s avoids the MSVC deprecation warning for getenv.
    char* buffer = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buffer, &len, key.c_str()) != 0 || buffer == nullptr) {
        return std::nullopt;
    }
    std::string value(buffer);
    std::free(buffer);
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
#else
    const char* value = std::getenv(key.c_str());
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

std::optional<std::string> yahoo_api_key() {
    return get_env("YAHOO_API_KEY");
}

}  // namespace trading::config
