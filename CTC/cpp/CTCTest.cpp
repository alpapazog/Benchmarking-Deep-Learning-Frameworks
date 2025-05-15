#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <vector>

// Generate timestamped log filename
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
    torch::Device device(!torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    // Dataset specs
    const int N = 1519;
    const int T = 1383;
    const int n_mels = 80;
    const int batch_size = 8;
    const int vocab_size = 28;

    // Load dataset from TorchScript extra_files
    std::unordered_map<std::string, std::string> extra_files = {
        {"features", ""},
        {"targets", ""},
        {"input_lengths", ""},
        {"target_lengths", ""}
    };

    auto module = torch::jit::load("data/librispeech_20min_dataset_model.pt", device, extra_files);

    auto features = torch::from_blob((void*)extra_files.at("features").data(), {N, T, n_mels}, torch::kFloat).clone().to(device);

    // Load model
    auto acoustic_model = torch::nn::Sequential(
        torch::nn::Linear(n_mels, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, vocab_size + 1)
    );
    torch::load(acoustic_model, "ctc.pt");
    acoustic_model->to(device);
    acoustic_model->eval();

    // Inference only, measure time
    size_t num_batches = (N + batch_size - 1) / batch_size;

    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto batch_features = features.slice(0, batch_idx * batch_size, std::min(static_cast<int64_t>((batch_idx + 1) * batch_size), static_cast<int64_t>(N)));
        auto logits = acoustic_model->forward(batch_features);
        logits = logits.transpose(0, 1).log_softmax(2);
        // No loss or accuracy, just forward pass
    }

    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Total Inference Time: " << duration.count() << " seconds\n";

    std::string log_filename = generate_log_filename("ctc_inference_log");
    std::ofstream log_file(log_filename);
    log_file << "Total Inference Time: " << duration.count() << " seconds\n";
    log_file.close();

    std::cout << "Inference log saved to " << log_filename << "\n";
    return 0;
}
