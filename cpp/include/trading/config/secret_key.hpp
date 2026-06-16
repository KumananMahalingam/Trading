// === FILE: cpp/include/trading/config/secret_key.hpp ===
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace trading::config {

// Read a secret/configuration value from an environment variable only.
// API keys are never hard-coded or read from disk. Returns std::nullopt when
// the variable is unset or empty.
std::optional<std::string> get_env(std::string_view name);

// Convenience accessors for the keys this project recognizes. All optional:
// the public Yahoo Finance chart endpoint requires no key, but a key may be
// supplied via the environment for rate-limit headroom or proxies.
std::optional<std::string> yahoo_api_key();

}  // namespace trading::config
