// === FILE: cpp/include/trading/config/settings.hpp ===
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace trading::config {

// ---------------------------------------------------------------------------
// Feature definition: EXACTLY 5 input features, no more, no less.
// ---------------------------------------------------------------------------
inline constexpr int kNumFeatures = 5;

inline constexpr std::array<std::string_view, kNumFeatures> kFeatureNames = {
    "SMA5", "SMA20", "RSI", "ATR", "OBV"};

// Number of initial rows that contain NaN after computing the 5 indicators.
// SMA20 needs 20 observations, so the first 20 rows are dropped.
inline constexpr int kWarmupRows = 20;

// ---------------------------------------------------------------------------
// Alpha signals: LLM-derived (or static fallback) formulas evaluated over the
// 5 base indicators and appended as extra input features to the single-stream
// model. Set kUseAlphas=false to run with the 5 base indicators only.
// ---------------------------------------------------------------------------
inline constexpr bool kUseAlphas = true;
inline constexpr int kNumAlphas = 5;

// ---------------------------------------------------------------------------
// Model architecture (mirrors config/settings.py).
// ---------------------------------------------------------------------------
inline constexpr int kInputSize   = kNumFeatures;  // 5
inline constexpr int kHiddenSize  = 192;
inline constexpr int kNumLayers   = 3;
inline constexpr int kNumHeads     = 6;     // (kHiddenSize * 2) must be divisible by this
inline constexpr double kDropout  = 0.35;

// LSTM is bidirectional, so the effective feature width is 2 * hidden.
inline constexpr int kLstmOutputSize = kHiddenSize * 2;
static_assert(kLstmOutputSize % kNumHeads == 0,
              "kHiddenSize*2 must be divisible by kNumHeads for attention heads");

// ---------------------------------------------------------------------------
// Training hyper-parameters.
// ---------------------------------------------------------------------------
inline constexpr int kWindowSize       = 30;
inline constexpr int kNumEpochs        = 100;
inline constexpr double kLearningRate  = 2e-4;
inline constexpr double kWeightDecay   = 1e-3;
inline constexpr int kBatchSizeMin     = 16;
inline constexpr int kBatchSizeMax     = 64;
inline constexpr int kPatience         = 25;
inline constexpr double kGradientClipNorm = 1.0;
inline constexpr double kCosineEtaMin  = 1e-6;

// Monte-Carlo dropout samples at inference.
inline constexpr int kMcSamples = 10;

// ---------------------------------------------------------------------------
// Hybrid loss weights:  loss = alpha*MSE + beta*direction + gamma*large_move_MAE
// ---------------------------------------------------------------------------
inline constexpr double kLossAlpha = 0.7;
inline constexpr double kLossBeta  = 0.3;
inline constexpr double kLossGamma = 0.1;

// Threshold (absolute return) above which a move is considered "large".
inline constexpr double kLargeMoveThreshold = 0.02;

// ---------------------------------------------------------------------------
// Data split fractions (train / val / test = 60 / 20 / 20), temporal order.
// ---------------------------------------------------------------------------
inline constexpr double kTrainFraction = 0.60;
inline constexpr double kValFraction   = 0.20;
inline constexpr double kTestFraction  = 0.20;

// Trading-day annualization factor for the Sharpe ratio.
inline constexpr double kAnnualizationDays = 252.0;

// ---------------------------------------------------------------------------
// Default run configuration.
// ---------------------------------------------------------------------------
inline constexpr std::string_view kDefaultTicker = "AAPL";
inline constexpr std::string_view kDefaultStartDate = "2023-09-01";
inline constexpr std::string_view kDefaultEndDate   = "2025-09-01";

}  // namespace trading::config
