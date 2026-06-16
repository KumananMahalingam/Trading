// === FILE: cpp/src/models/lstm_modules.cpp ===
#include "trading/models/lstm_modules.hpp"

#include <cmath>

namespace trading::models {

ResidualLSTMImpl::ResidualLSTMImpl(int input_size, int hidden_size, int num_layers,
                                   double dropout, bool bidirectional) {
    auto options = torch::nn::LSTMOptions(input_size, hidden_size)
                       .num_layers(num_layers)
                       .batch_first(true)
                       .bidirectional(bidirectional)
                       .dropout(num_layers > 1 ? dropout : 0.0);
    lstm_ = register_module("lstm", torch::nn::LSTM(options));

    output_size_ = bidirectional ? hidden_size * 2 : hidden_size;
    identity_ = (input_size == output_size_);
    if (!identity_) {
        residual_fc_ = register_module(
            "residual_fc", torch::nn::Linear(input_size, output_size_));
    }
}

torch::Tensor ResidualLSTMImpl::forward(const torch::Tensor& x) {
    // lstm returns (output, (h_n, c_n)); we only need the output sequence.
    auto lstm_out = std::get<0>(lstm_->forward(x));
    torch::Tensor residual = identity_ ? x : residual_fc_->forward(x);
    return lstm_out + residual;
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int hidden_size, int num_heads)
    : num_heads_(num_heads), head_dim_(hidden_size / num_heads) {
    TORCH_CHECK(head_dim_ * num_heads == hidden_size,
                "hidden_size must be divisible by num_heads");
    query_ = register_module("query", torch::nn::Linear(hidden_size, hidden_size));
    key_ = register_module("key", torch::nn::Linear(hidden_size, hidden_size));
    value_ = register_module("value", torch::nn::Linear(hidden_size, hidden_size));
    fc_out_ = register_module("fc_out", torch::nn::Linear(hidden_size, hidden_size));
}

torch::Tensor MultiHeadAttentionImpl::forward(const torch::Tensor& x) {
    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);

    auto reshape_heads = [&](const torch::Tensor& t) {
        // [b, s, hidden] -> [b, heads, s, head_dim]
        return t.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
    };

    auto Q = reshape_heads(query_->forward(x));
    auto K = reshape_heads(key_->forward(x));
    auto V = reshape_heads(value_->forward(x));

    const double scale = std::sqrt(static_cast<double>(head_dim_));
    auto attention = torch::matmul(Q, K.transpose(-2, -1)) / scale;
    attention = torch::softmax(attention, -1);

    auto out = torch::matmul(attention, V);             // [b, heads, s, head_dim]
    out = out.transpose(1, 2).contiguous().view(        // [b, s, hidden]
        {batch_size, seq_len, num_heads_ * head_dim_});
    return fc_out_->forward(out);
}

}  // namespace trading::models
