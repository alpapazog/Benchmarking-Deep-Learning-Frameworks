#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <vector>

torch::Tensor positional_encoding(int seq_len, int embed_dim, torch::Device device) {
    torch::Tensor pe = torch::zeros({seq_len, embed_dim}, torch::TensorOptions().device(device));
    auto position = torch::arange(0, seq_len, torch::TensorOptions().device(device)).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, embed_dim, 2, torch::TensorOptions().device(device)) * -(std::log(10000.0) / embed_dim));
    pe.slice(1, 0, embed_dim, 2) = torch::sin(position * div_term);
    pe.slice(1, 1, embed_dim, 2) = torch::cos(position * div_term);
    return pe.unsqueeze(1); // [seq_len, 1, embed_dim]
}

struct TransformerEncoderImpl : torch::nn::Module {
    torch::nn::MultiheadAttention self_attn{nullptr};
    torch::nn::Linear feedforward1{nullptr}, feedforward2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};

    TransformerEncoderImpl(int embed_dim, int num_heads, int ff_hidden_dim)
        : self_attn(torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(embed_dim, num_heads))),
          feedforward1(torch::nn::Linear(embed_dim, ff_hidden_dim)),
          feedforward2(torch::nn::Linear(ff_hidden_dim, embed_dim)),
          norm1(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}))),
          norm2(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}))) {
        register_module("self_attn", self_attn);
        register_module("feedforward1", feedforward1);
        register_module("feedforward2", feedforward2);
        register_module("norm1", norm1);
        register_module("norm2", norm2);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto attn_output = std::get<0>(self_attn(x, x, x));
        x = norm1(x + attn_output);
        auto ff_output = feedforward2(torch::relu(feedforward1(x)));
        x = norm2(x + ff_output);
        return x;
    }
};
TORCH_MODULE(TransformerEncoder);

struct TransformerClassifierImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Linear classifier{nullptr};
    TransformerEncoder encoder{nullptr};
    int seq_len;
    int embed_dim;
    torch::Device device;

    TransformerClassifierImpl(int vocab_size, int embed_dim_, int seq_len_, int num_heads, int ff_hidden_dim, int num_classes, torch::Device device_)
        : embedding(torch::nn::Embedding(vocab_size, embed_dim_)),
          encoder(embed_dim_, num_heads, ff_hidden_dim),
          classifier(torch::nn::Linear(embed_dim_, num_classes)),
          seq_len(seq_len_),
          embed_dim(embed_dim_),
          device(device_) {
        register_module("embedding", embedding);
        register_module("encoder", encoder);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor input_ids) {
        auto x = embedding->forward(input_ids); // [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1); // [seq_len, batch_size, embed_dim]
        auto pe = positional_encoding(seq_len, embed_dim, device);
        x = x + pe;
        x = encoder->forward(x);
        x = x.mean(0); // [batch_size, embed_dim]
        auto logits = classifier->forward(x);
        return logits;
    }
};
TORCH_MODULE(TransformerClassifier);

std::string generate_log_filename(const std::string& prefix = "log") {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm;
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    local_tm = *std::localtime(&now_time);
#endif
    std::ostringstream oss;
    oss << prefix << "_" << std::put_time(&local_tm, "%Y-%m-%d_%H%M") << ".txt";
    return oss.str();
}

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    // Load IMDB dataset from .pt file
    std::cout << "test1 \n";
    torch::Tensor inputs, labels;
    std::ifstream file("./data/dataset.pt", std::ios::binary);
    std::cout << "test2 \n";
    if (!file) throw std::runtime_error("Could not open dataset.pt");
    std::cout << "test3 \n";
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    
    c10::IValue ivalue = torch::pickle_load(buffer);
    auto dict = ivalue.toGenericDict();
    inputs = dict.at("inputs").toTensor().to(device);
    labels = dict.at("labels").toTensor().to(device);

    const int vocab_size = 100684;  // Match your vocab size from dataset
    const int seq_len = inputs.size(1);
    const int embed_dim = 512;
    const int num_heads = 8;
    const int ff_hidden_dim = 2048;
    const int num_classes = 2;
    const int batch_size = 64;
    const int num_epochs = 5;
    const int num_samples = inputs.size(0);
    const int num_batches = (num_samples + batch_size - 1) / batch_size;

    // Build Transformer model
    TransformerClassifier model(vocab_size, embed_dim, seq_len, num_heads, ff_hidden_dim, num_classes, device);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));
    torch::nn::CrossEntropyLoss criterion;

    auto training_start = std::chrono::high_resolution_clock::now();
    std::string log_filename = generate_log_filename("train_log");
    std::ofstream log_file(log_filename);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        model->train();
        double total_loss = 0.0;

        for (int batch = 0; batch < num_batches; ++batch) {
            int start = batch * batch_size;
            int end = std::min(start + batch_size, num_samples);

            auto batch_inputs = inputs.slice(0, start, end);
            auto batch_labels = labels.slice(0, start, end);

            optimizer.zero_grad();
            auto logits = model->forward(batch_inputs);
            auto loss = criterion(logits, batch_labels);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<double>();

            if ((batch + 1) % 10 == 0) {
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Batch [" << batch + 1 << "/" << num_batches
                          << "], Loss: " << loss.item<double>() << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
        std::cout << "[Epoch " << epoch + 1 << "] Average Loss: " << (total_loss / num_batches) << " duration: " << epoch_duration.count() << " seconds\n";
        log_file << "Epoch " << epoch + 1 << ": " << epoch_duration.count() << " sec\n";
    }

    auto training_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_training_time = training_end - training_start;
    std::cout << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Average epoch duration: " << total_training_time.count() / num_epochs << " seconds\n";
    log_file.close();

    std::cout << "Training log saved to " << log_filename << "\n";
    torch::save(model, "transformer_imdb.pt");
    std::cout << "Training complete. Model saved to transformer_imdb.pt\n";
}