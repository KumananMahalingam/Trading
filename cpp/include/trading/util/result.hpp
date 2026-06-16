// === FILE: cpp/include/trading/util/result.hpp ===
#pragma once

#include <string>
#include <utility>
#include <variant>

namespace trading {

// Simple error type carrying a human-readable message and a category code.
struct Error {
    enum class Code {
        Network,
        Parse,
        Io,
        InvalidArgument,
        InsufficientData,
        Model,
        Unknown
    };

    Code code{Code::Unknown};
    std::string message;

    Error() = default;
    Error(Code c, std::string msg) : code(c), message(std::move(msg)) {}
};

inline const char* to_string(Error::Code c) {
    switch (c) {
        case Error::Code::Network:          return "Network";
        case Error::Code::Parse:            return "Parse";
        case Error::Code::Io:               return "Io";
        case Error::Code::InvalidArgument:  return "InvalidArgument";
        case Error::Code::InsufficientData: return "InsufficientData";
        case Error::Code::Model:            return "Model";
        case Error::Code::Unknown:          return "Unknown";
    }
    return "Unknown";
}

// A minimal Result<T, E> type used instead of exceptions in hot/data paths.
// E defaults to trading::Error.
template <typename T, typename E = Error>
class Result {
public:
    // Construct a success value.
    static Result ok(T value) { return Result(std::move(value), Tag::Ok); }

    // Construct an error value.
    static Result err(E error) { return Result(std::move(error), Tag::Err); }

    bool is_ok() const noexcept { return std::holds_alternative<T>(storage_); }
    bool is_err() const noexcept { return std::holds_alternative<E>(storage_); }

    explicit operator bool() const noexcept { return is_ok(); }

    // Access the success value. Caller must ensure is_ok().
    T& value() & { return std::get<T>(storage_); }
    const T& value() const& { return std::get<T>(storage_); }
    T&& value() && { return std::get<T>(std::move(storage_)); }

    // Access the error. Caller must ensure is_err().
    E& error() & { return std::get<E>(storage_); }
    const E& error() const& { return std::get<E>(storage_); }

    // Return the value or a provided fallback.
    T value_or(T fallback) const {
        return is_ok() ? std::get<T>(storage_) : std::move(fallback);
    }

private:
    enum class Tag { Ok, Err };

    Result(T value, Tag) : storage_(std::move(value)) {}
    Result(E error, Tag) : storage_(std::move(error)) {}

    std::variant<T, E> storage_;
};

// Specialization helper for void-like success.
struct Unit {};

template <typename E = Error>
using Status = Result<Unit, E>;

template <typename E = Error>
inline Status<E> ok_status() {
    return Status<E>::ok(Unit{});
}

}  // namespace trading
