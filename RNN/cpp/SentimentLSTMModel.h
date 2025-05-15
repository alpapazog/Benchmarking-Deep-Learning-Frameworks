#pragma once
#include <torch/torch.h>

struct SentimentLSTMImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};

    SentimentLSTMImpl(int vocab_size, int embed_dim, int hidden_dim, int output_dim)
        : embedding(torch::nn::Embedding(vocab_size, embed_dim)),
          lstm(torch::nn::LSTM(torch::nn::LSTMOptions(embed_dim, hidden_dim).batch_first(true))),
          fc(torch::nn::Linear(hidden_dim, output_dim)) {
        register_module("embedding", embedding);
        register_module("lstm", lstm);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = embedding(x);
        auto lstm_out = lstm->forward(x);
        auto last_hidden = std::get<0>(lstm_out).select(1, -1);  // shape: [batch, hidden]
        return fc(last_hidden);
    }
};
TORCH_MODULE(SentimentLSTM);
