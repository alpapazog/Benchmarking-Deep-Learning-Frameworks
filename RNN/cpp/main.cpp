#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <vector>

#include "IMDBDataset.h"
#include "SentimentLSTMModel.h"

// Returns timestamped filename like "log_2025-05-10_0314.txt"
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

    const int64_t vocab_size = 100684;  // match your vocab size
    const int64_t embed_dim = 100;
    const int64_t hidden_dim = 128;
    const int64_t output_dim = 2;

    SentimentLSTM model(vocab_size, embed_dim, hidden_dim, output_dim);
    model->to(device);

    std::cout << "Loading dataset...\n";
    torch::Tensor inputs, labels;

    std::ifstream file("./data/dataset.pt", std::ios::binary);
    if (!file) throw std::runtime_error("Could not open dataset.pt");

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    c10::IValue ivalue = torch::pickle_load(buffer);
    auto dict = ivalue.toGenericDict();

    inputs = dict.at("inputs").toTensor();
    labels = dict.at("labels").toTensor();

    auto train_dataset = IMDBDataset("./data/dataset.pt").map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(train_dataset, torch::data::DataLoaderOptions().batch_size(64));

    std::cout << "Training model...\n";
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;

    const int num_epochs = 5;
    
    auto training_start = std::chrono::high_resolution_clock::now();
    
    std::string log_filename = generate_log_filename("train_log");
    std::ofstream log_file(log_filename);

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        size_t batch_idx = 0;
        double epoch_loss = 0.0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            optimizer.zero_grad();
            auto outputs = model->forward(data);
            auto loss = criterion(outputs, targets);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            if (++batch_idx % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Batch " << batch_idx
                          << ", Loss: " << loss.item<float>() << "\n";
            }
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
        std::cout << "Epoch " << epoch << " average loss: " << epoch_loss / batch_idx << " duration: " << epoch_duration.count() << " seconds\n";
        log_file << "Epoch " << epoch << ": " << epoch_duration.count() << " sec\n";
    }
    auto training_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_training_time = training_end - training_start;
    std::cout << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Average epoch duration: " << total_training_time.count() / num_epochs << " seconds\n";
    log_file.close();

    std::cout << "Training log saved to " << log_filename << "\n";
    
    torch::save(model, "sentiment_lstm.pt");
    std::cout << "Training complete. Model saved to sentiment_lstm.pt\n";

    return 0;
}
