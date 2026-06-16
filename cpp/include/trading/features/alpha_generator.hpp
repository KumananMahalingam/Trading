// === FILE: cpp/include/trading/features/alpha_generator.hpp ===
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "trading/util/result.hpp"

namespace trading::features {

// Static fallback alpha formulas expressed purely over the 5 base indicators
// (SMA5, SMA20, RSI, ATR, OBV). Returned as bare right-hand-side expressions.
// Analogous to generate_simple_alphas() in the Python project.
std::vector<std::string> default_alpha_formulas();

// Ask the Groq LLM (OpenAI-compatible chat completions) to author alpha
// formulas using ONLY the 5 indicators and the evaluator's supported
// operators/functions. Requires the GROQ_API_KEY environment variable.
Result<std::vector<std::string>> generate_alpha_formulas_llm(std::string_view ticker);

// High-level entry point: use the LLM when GROQ_API_KEY is set and the call
// succeeds; otherwise fall back to the static set. Always returns a usable,
// non-empty list of formulas.
std::vector<std::string> obtain_alpha_formulas(std::string_view ticker);

}  // namespace trading::features
