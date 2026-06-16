// === FILE: cpp/src/models/return_predictor.cpp ===
#include "trading/models/return_predictor.hpp"

#include <vector>

namespace trading::models {

namespace {
constexpr int kFusionHidden1 = 128;
constexpr int kFusionHidden2 = 64;
}  // namespace

ReturnPredictorImpl::ReturnPredictorImpl(int input_size, int hidden_size,
                                         int num_layers, double dropout,
                                         int num_heads)
    : dropout_p_(dropout) {
    lstm_ = register_module(
        "lstm", ResidualLSTM(input_size, hidden_size, num_layers, dropout, true));
    const int feature_width = lstm_->output_size();  // 2 * hidden_size

    attention_ = register_module(
        "attention", MultiHeadAttention(feature_width, num_heads));

    // 3-layer MLP fusion head -> scalar.
    fc1_ = register_module("fc1", torch::nn::Linear(feature_width, kFusionHidden1));
    ln1_ = register_module("ln1", torch::nn::LayerNorm(
                                      torch::nn::LayerNormOptions({kFusionHidden1})));
    drop1_ = register_module("drop1", torch::nn::Dropout(dropout));

    fc2_ = register_module("fc2", torch::nn::Linear(kFusionHidden1, kFusionHidden2));
    ln2_ = register_module("ln2", torch::nn::LayerNorm(
                                      torch::nn::LayerNormOptions({kFusionHidden2})));
    drop2_ = register_module("drop2", torch::nn::Dropout(dropout));

    mc_dropout_ = register_module("mc_dropout", torch::nn::Dropout(dropout));
    fc_out_ = register_module("fc_out", torch::nn::Linear(kFusionHidden2, 1));
}

torch::Tensor ReturnPredictorImpl::extract_features(const torch::Tensor& x) {
    auto lstm_out = lstm_->forward(x);               // [b, s, 2h]
    auto attn_out = attention_->forward(lstm_out);   // [b, s, 2h]
    auto combined = lstm_out + attn_out;             // residual add
    auto context = combined.mean(/*dim=*/1);         // [b, 2h] global avg pool

    auto h = torch::relu(ln1_->forward(fc1_->forward(context)));
    h = drop1_->forward(h);
    h = torch::relu(ln2_->forward(fc2_->forward(h)));
    h = drop2_->forward(h);
    return h;                                         // [b, 64]
}

torch::Tensor ReturnPredictorImpl::forward(const torch::Tensor& x) {
    auto h = extract_features(x);
    h = mc_dropout_->forward(h);
    return fc_out_->forward(h);                       // [b, 1]
}

std::pair<torch::Tensor, torch::Tensor> ReturnPredictorImpl::predict_mc(
    const torch::Tensor& x, int n_samples) {
    torch::NoGradGuard no_grad;

    // extract_features respects module train/eval; callers set eval() first so
    // drop1_/drop2_ are inactive here. Only the MC dropout is sampled.
    auto h = extract_features(x);

    if (n_samples <= 1) {
        auto mean = fc_out_->forward(h);
        auto std = torch::zeros_like(mean);
        return {mean, std};
    }

    std::vector<torch::Tensor> preds;
    preds.reserve(static_cast<std::size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        // Force-active dropout regardless of module mode for Bayesian sampling.
        auto sampled = torch::dropout(h, dropout_p_, /*train=*/true);
        preds.push_back(fc_out_->forward(sampled));
    }

    auto stacked = torch::stack(preds, /*dim=*/0);    // [n, b, 1]
    auto mean_pred = stacked.mean(0);                 // [b, 1]
    auto std_pred = stacked.std(/*dim=*/0, /*unbiased=*/true, /*keepdim=*/false) + 1e-6;
    return {mean_pred, std_pred};
}

}  // namespace trading::models
