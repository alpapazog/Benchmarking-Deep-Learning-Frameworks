#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

#include "IMDBDataset.h"
#include "SentimentLSTMModel.h"

// Timestamped log file name
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

    // Load model
    const int64_t vocab_size = 50000, embed_dim = 100, hidden_dim = 128, output_dim = 2;
    SentimentLSTM model(vocab_size, embed_dim, hidden_dim, output_dim);
    torch::load(model, "sentiment_lstmcuda.pt");
    model->to(device);
    model->eval();

    // Load dataset
    torch::Tensor inputs, labels;
    std::ifstream file("data/dataset_test.pt", std::ios::binary);
    if (!file) throw std::runtime_error("Could not open dataset_test.pt");
    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    c10::IValue ivalue = torch::pickle_load(buffer);
    auto dict = ivalue.toGenericDict();
    inputs = dict.at("inputs").toTensor();
    labels = dict.at("labels").toTensor();

    // Create TensorDataset and DataLoader
    auto test_dataset = IMDBDataset("./data/dataset_test.pt").map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader(
    std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(64).drop_last(false)
    );


    torch::NoGradGuard no_grad;
    int total_correct = 0;
    size_t total_samples = 0;

    std::cout << "Starting inference...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& batch : *loader) {
        auto batch_inputs = batch.data.to(device);
        auto batch_labels = batch.target.to(device);
        auto outputs = model->forward(batch_inputs);
        auto preds = outputs.argmax(1);
        total_correct += preds.eq(batch_labels).sum().item<int>();
        total_samples += batch_inputs.size(0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    float accuracy = static_cast<float>(total_correct) / total_samples;

    std::cout << "Accuracy: " << accuracy << "\n";
    std::cout << "Inference time: " << duration.count() << " seconds\n";

    std::string log_filename = generate_log_filename("test_log");
    std::ofstream log_file(log_filename);
    log_file << "Accuracy: " << accuracy << "\n";
    log_file << "Total Inference Time: " << duration.count() << " seconds\n";
    log_file.close();

    std::cout << "Test log saved to " << log_filename << "\n";
    return 0;
}
