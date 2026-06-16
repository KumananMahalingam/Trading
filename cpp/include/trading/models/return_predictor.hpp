// === FILE: cpp/include/trading/models/return_predictor.hpp ===
#pragma once

#include <utility>

#include <torch/torch.h>

#include "trading/models/lstm_modules.hpp"

namespace trading::models {

// Single-stream LSTM return predictor for the 5-feature setup.
//
// Pipeline:
//   x [batch, seq, 5]
//     -> ResidualLSTM (bidirectional)        -> [batch, seq, 2*hidden]
//     -> + MultiHeadAttention (residual add)  -> [batch, seq, 2*hidden]
//     -> mean over seq (global avg pool)      -> [batch, 2*hidden]
//     -> 3-layer MLP fusion head              -> scalar return
//
// Monte-Carlo dropout is applied at the head for uncertainty at inference time.
class ReturnPredictorImpl : public torch::nn::Module {
public:
    ReturnPredictorImpl(int input_size, int hidden_size, int num_layers,
                        double dropout, int num_heads);

    // Training forward pass: returns predicted return [batch, 1].
    torch::Tensor forward(const torch::Tensor& x);

    // Inference with Monte-Carlo dropout. Returns {mean_pred, std_pred}, each
    // [batch, 1]. With n_samples <= 1, std is all zeros.
    std::pair<torch::Tensor, torch::Tensor> predict_mc(const torch::Tensor& x,
                                                       int n_samples);

private:
    // Shared feature extraction up to (but excluding) the MC dropout + final fc.
    torch::Tensor extract_features(const torch::Tensor& x);

    ResidualLSTM lstm_{nullptr};
    MultiHeadAttention attention_{nullptr};

    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc_out_{nullptr};
    torch::nn::LayerNorm ln1_{nullptr};
    torch::nn::LayerNorm ln2_{nullptr};
    torch::nn::Dropout drop1_{nullptr};
    torch::nn::Dropout drop2_{nullptr};
    torch::nn::Dropout mc_dropout_{nullptr};

    double dropout_p_{0.0};
};
TORCH_MODULE(ReturnPredictor);

}  // namespace trading::models
