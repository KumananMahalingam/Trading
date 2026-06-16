// === FILE: cpp/include/trading/features/alpha_formula.hpp ===
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "trading/features/technical_indicators.hpp"
#include "trading/util/result.hpp"

namespace trading::features {

// A vectorized alpha-formula evaluator.
//
// Grammar (all operations are evaluated element-wise across rows):
//   expr    := term (('+'|'-') term)*
//   term    := power (('*'|'/') power)*
//   power   := unary ('^' power)?            (right associative)
//   unary   := ('-'|'+') unary | primary
//   primary := number
//            | ident                         (a variable series)
//            | ident '(' args ')'            (a function call)
//            | '(' expr ')'
//
// Element-wise functions : abs(x), log(x), sqrt(x), exp(x), sign(x)
// Time-series functions  : delay(x, n), delta(x, n), ts_mean(x, n),
//                          ts_std(x, n), ts_min(x, n), ts_max(x, n)
//   where n is an integer literal window/lag (edge-replicated at the start).
//
// Variables resolve against the provided series map (e.g. "SMA5", "RSI", ...).

// Evaluate a single formula over the named series of length `n`.
Result<std::vector<double>> evaluate_formula(
    std::string_view formula,
    const std::unordered_map<std::string, const std::vector<double>*>& variables,
    std::size_t n);

// Normalize raw LLM/text formula lines into bare right-hand-side expressions.
// Handles leading labels ("alpha_1 =", "α1:", "α1 ->") and unicode operators.
std::vector<std::string> parse_formula_lines(std::string_view text);

// Compute alpha columns from a FeatureMatrix's existing indicator columns and
// append them as new feature columns (named alpha_1, alpha_2, ...). At most
// `max_alphas` are added. Each alpha is sanitized (non-finite -> 0) and clipped
// to mean +/- 3*std, matching the Python EnhancedAlphaComputer. If no formula
// evaluates successfully, the static default set is used as a fallback.
Status<> append_alphas(FeatureMatrix& fm,
                       const std::vector<std::string>& formulas,
                       int max_alphas);

}  // namespace trading::features
