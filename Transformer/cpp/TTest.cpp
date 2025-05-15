#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <cuda_runtime.h>

// Utility to generate timestamped log filename
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

torch::Tensor positional_encoding(int seq_len, int embed_dim, torch::Device device) {
    torch::Tensor pe = torch::zeros({seq_len, embed_dim}, torch::TensorOptions().device(device));
    auto position = torch::arange(0, seq_len, torch::TensorOptions().device(device)).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, embed_dim, 2, torch::TensorOptions().device(device)) * -(std::log(10000.0) / embed_dim));
    pe.slice(1, 0, embed_dim, 2) = torch::sin(position * div_term);
    pe.slice(1, 1, embed_dim, 2) = torch::cos(position * div_term);
    return pe.unsqueeze(1);
}

struct TransformerEncoderImpl : torch::nn::Module {
    torch::nn::MultiheadAttention self_attn{nullptr};
    torch::nn::Linear feedforward1{nullptr}, feedforward2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};

    TransformerEncoderImpl(int embed_dim, int num_heads, int ff_hidden_dim)
        : self_attn(torch::nn::MultiheadAttention(embed_dim, num_heads)),
        feedforward1(torch::nn::Linear(embed_dim, ff_hidden_dim)),
        feedforward2(torch::nn::Linear(ff_hidden_dim, embed_dim)),
        norm1(std::vector<int64_t>{embed_dim}),
        norm2(std::vector<int64_t>{embed_dim}) {
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
    TransformerEncoder encoder{nullptr};
    torch::nn::Linear classifier{nullptr};
    int seq_len;
    int embed_dim;
    torch::Device device;

    TransformerClassifierImpl(int vocab_size, int embed_dim_, int seq_len_, int num_heads, int ff_hidden_dim, int num_classes, torch::Device device_)
        : embedding(vocab_size, embed_dim_),
          encoder(embed_dim_, num_heads, ff_hidden_dim),
          classifier(embed_dim_, num_classes),
          seq_len(seq_len_),
          embed_dim(embed_dim_),
          device(device_) {
        register_module("embedding", embedding);
        register_module("encoder", encoder);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor input_ids) {
        auto x = embedding->forward(input_ids);
        x = x.transpose(0, 1);
        auto pe = positional_encoding(seq_len, embed_dim, device);
        x = x + pe;
        x = encoder->forward(x);
        x = x.mean(0);
        return classifier->forward(x);
    }
};
TORCH_MODULE(TransformerClassifier);

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    size_t free_bytes_before = 0, total_bytes = 0;
    if (device.is_cuda()) {
        // Ensure clean device state
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_bytes_before, &total_bytes);
    }
    // Load dataset
    torch::Tensor inputs;
    std::ifstream file("./data/dataset.pt", std::ios::binary);
    if (!file) throw std::runtime_error("Could not open dataset.pt");
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    c10::IValue ivalue = torch::pickle_load(buffer);
    auto dict = ivalue.toGenericDict();
    inputs = dict.at("inputs").toTensor().to(device);

    const int vocab_size = 100684;
    const int seq_len = inputs.size(1);
    const int embed_dim = 512;
    const int num_heads = 8;
    const int ff_hidden_dim = 2048;
    const int num_classes = 2;
    const int batch_size = 64;
    const int num_samples = inputs.size(0);
    const int num_batches = (num_samples + batch_size - 1) / batch_size;

    // Load model
    TransformerClassifier model(vocab_size, embed_dim, seq_len, num_heads, ff_hidden_dim, num_classes, device);
    torch::load(model, "transformer_imdb.pt");
    model->to(device);
    model->eval();

    if (device.is_cuda()) torch::cuda::synchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < num_batches; ++batch) {
        int start_idx = batch * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_samples);
        auto batch_inputs = inputs.slice(0, start_idx, end_idx);
        auto logits = model->forward(batch_inputs);
    }

    if (device.is_cuda()) torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double throughput = static_cast<double>(num_samples) / duration.count();
    double avg_batch_time = (duration.count() / num_batches) * 1000.0;  // ms per batch

    size_t param_count = 0;
    for (const auto& p : model->parameters()) param_count += p.numel();

    double model_size_mb = static_cast<double>(std::filesystem::file_size("transformer_imdb.pt")) / (1024.0 * 1024.0);

    std::cout << "Total Inference Time: " << duration.count() << " seconds\n";
    std::cout << "Throughput: " << throughput << " samples/sec\n";
    std::cout << "Average Inference Time per Batch: " << avg_batch_time << " ms/batch\n";
    std::cout << "Model size: " << model_size_mb << " MB\n";
    std::cout << "Parameter count: " << param_count << "\n";

    std::string log_filename = generate_log_filename("transformer_inference_log");
    std::ofstream log_file(log_filename);
    log_file << "Total Inference Time: " << duration.count() << " seconds\n";
    log_file << "Throughput: " << throughput << " samples/sec\n";
    log_file << "Average Inference Time per Batch: " << avg_batch_time << " ms/batch\n";
    log_file << "Model size: " << model_size_mb << " MB\n";
    log_file << "Parameter count: " << param_count << "\n";

    if (device.is_cuda()) {
        cudaDeviceSynchronize(); 

        size_t free_bytes_after = 0;
        cudaMemGetInfo(&free_bytes_after, &total_bytes);

        double used_mb_during_inference = (free_bytes_before - free_bytes_after) / (1024.0 * 1024.0);

        std::cout << "Approx. Additional Allocated GPU Mem During Inference (MB): " << used_mb_during_inference << "\n";
        log_file << "Approx. Additional Allocated GPU Mem During Inference (MB): " << used_mb_during_inference << "\n";

        }


    log_file.close();

    std::cout << "Inference log saved to " << log_filename << "\n";
    return 0;
}
