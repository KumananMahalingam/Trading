// === FILE: cpp/src/features/alpha_generator.cpp ===
#include "trading/features/alpha_generator.hpp"

#include <string>

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include "trading/config/secret_key.hpp"
#include "trading/features/alpha_formula.hpp"
#include "trading/util/logging.hpp"

namespace trading::features {

namespace {

using json = nlohmann::json;

std::size_t write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    const std::size_t total = size * nmemb;
    out->append(ptr, total);
    return total;
}

constexpr char kGroqUrl[] = "https://api.groq.com/openai/v1/chat/completions";
constexpr char kGroqModel[] = "llama-3.3-70b-versatile";

std::string build_prompt(std::string_view ticker) {
    return std::string(
               "You are a quantitative analyst. Propose exactly 5 alpha signal "
               "formulas to predict the next-day return of ") +
           std::string(ticker) +
           ".\n"
           "You may ONLY use these variables: SMA5, SMA20, RSI, ATR, OBV.\n"
           "Allowed operators: + - * / ^ and parentheses.\n"
           "Allowed functions: abs(x), log(x), sqrt(x), exp(x), sign(x), "
           "delay(x, n), delta(x, n), ts_mean(x, n), ts_std(x, n), "
           "ts_min(x, n), ts_max(x, n) where n is an integer.\n"
           "Output EXACTLY 5 lines and nothing else, each of the form:\n"
           "alpha_1 = <expression>\n"
           "alpha_2 = <expression>\n"
           "alpha_3 = <expression>\n"
           "alpha_4 = <expression>\n"
           "alpha_5 = <expression>\n";
}

}  // namespace

std::vector<std::string> default_alpha_formulas() {
    return {
        "(SMA5 - SMA20) / SMA20",            // trend / moving-average spread
        "(RSI - 50) / 50",                   // normalized momentum oscillator
        "delta(OBV, 5)",                     // 5-day change in volume flow
        "ATR / SMA20",                       // normalized volatility
        "(SMA5 - delay(SMA5, 5)) / ATR",     // volatility-scaled short momentum
    };
}

Result<std::vector<std::string>> generate_alpha_formulas_llm(std::string_view ticker) {
    auto key = config::get_env("GROQ_API_KEY");
    if (!key) {
        return Result<std::vector<std::string>>::err(
            Error(Error::Code::InvalidArgument, "GROQ_API_KEY not set"));
    }

    json body;
    body["model"] = kGroqModel;
    body["temperature"] = 0.2;
    body["messages"] = json::array(
        {{{"role", "system"}, {"content", "You output only alpha formulas, one per line."}},
         {{"role", "user"}, {"content", build_prompt(ticker)}}});
    const std::string payload = body.dump();

    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        return Result<std::vector<std::string>>::err(
            Error(Error::Code::Network, "Failed to init CURL"));
    }

    std::string response;
    struct curl_slist* headers = nullptr;
    const std::string auth = "Authorization: Bearer " + *key;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, kGroqUrl);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

    const CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        return Result<std::vector<std::string>>::err(
            Error(Error::Code::Network,
                  std::string("Groq request failed: ") + curl_easy_strerror(rc)));
    }
    if (status < 200 || status >= 300) {
        return Result<std::vector<std::string>>::err(
            Error(Error::Code::Network, "Groq HTTP status " + std::to_string(status)));
    }

    std::string content;
    try {
        const json root = json::parse(response);
        content = root.at("choices").at(0).at("message").at("content").get<std::string>();
    } catch (const json::exception& e) {
        return Result<std::vector<std::string>>::err(
            Error(Error::Code::Parse, std::string("Groq JSON parse error: ") + e.what()));
    }

    auto formulas = parse_formula_lines(content);
    if (formulas.empty()) {
        return Result<std::vector<std::string>>::err(
            Error(Error::Code::Parse, "No formulas parsed from LLM response"));
    }
    return Result<std::vector<std::string>>::ok(std::move(formulas));
}

std::vector<std::string> obtain_alpha_formulas(std::string_view ticker) {
    if (config::get_env("GROQ_API_KEY")) {
        auto llm = generate_alpha_formulas_llm(ticker);
        if (llm.is_ok()) {
            log::info(log::cat("Using ", llm.value().size(),
                               " LLM-generated alpha formulas"));
            return std::move(llm.value());
        }
        log::warn(log::cat("LLM alpha generation failed (", llm.error().message,
                           "); using static fallback"));
    } else {
        log::info("GROQ_API_KEY not set; using static fallback alpha formulas");
    }
    return default_alpha_formulas();
}

}  // namespace trading::features
