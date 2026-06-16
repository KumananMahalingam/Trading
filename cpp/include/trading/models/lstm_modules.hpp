// === FILE: cpp/include/trading/models/lstm_modules.hpp ===
#pragma once

#include <torch/torch.h>

namespace trading::models {

// LSTM with a residual connection, mirroring ResidualLSTM in lstm_modules.py.
// Input  : [batch, seq, input_size]
// Output : [batch, seq, output_size] where
//          output_size = hidden_size * (bidirectional ? 2 : 1)
class ResidualLSTMImpl : public torch::nn::Module {
public:
    ResidualLSTMImpl(int input_size, int hidden_size, int num_layers,
                     double dropout, bool bidirectional);

    torch::Tensor forward(const torch::Tensor& x);

    int output_size() const { return output_size_; }

private:
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear residual_fc_{nullptr};
    bool identity_{false};
    int output_size_{0};
};
TORCH_MODULE(ResidualLSTM);

// Multi-head self-attention for time series, mirroring MultiHeadAttention in
// lstm_modules.py.
// Input/Output : [batch, seq, hidden_size]
class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int hidden_size, int num_heads);

    torch::Tensor forward(const torch::Tensor& x);

private:
    int num_heads_{0};
    int head_dim_{0};
    torch::nn::Linear query_{nullptr};
    torch::nn::Linear key_{nullptr};
    torch::nn::Linear value_{nullptr};
    torch::nn::Linear fc_out_{nullptr};
};
TORCH_MODULE(MultiHeadAttention);

}  // namespace trading::models
