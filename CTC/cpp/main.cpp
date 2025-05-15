#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <vector>

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
    torch::Device device(torch::kCUDA); //device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // --------------------------
    // Known dataset sizes
    // --------------------------
    const int N = 1519;
    const int T = 1383;
    const int n_mels = 80;
    const int target_seq_len = 320;
    const int vocab_size = 28;
    const int batch_size = 8;
    const int num_epochs = 5;

    // --------------------------
    // Load dataset from TorchScript extra_files
    // --------------------------
    std::unordered_map<std::string, std::string> extra_files = {
        {"features", ""},
        {"targets", ""},
        {"input_lengths", ""},
        {"target_lengths", ""}
    };

    auto module = torch::jit::load("data/librispeech_20min_dataset_model.pt", device, extra_files);

    auto features = torch::from_blob((void*)extra_files.at("features").data(), {N, T, n_mels}, torch::kFloat).clone().to(device);
    auto targets = torch::from_blob((void*)extra_files.at("targets").data(), {N, target_seq_len}, torch::kLong).clone().to(device);
    auto input_lengths = torch::from_blob((void*)extra_files.at("input_lengths").data(), {N}, torch::kLong).clone().to(device);
    auto target_lengths = torch::from_blob((void*)extra_files.at("target_lengths").data(), {N}, torch::kLong).clone().to(device);

    std::cout << "Test" << "\n";
    // --------------------------
    // Define acoustic model + optimizer + loss
    // --------------------------
    auto acoustic_model = torch::nn::Sequential(
        torch::nn::Linear(n_mels, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, vocab_size + 1) // +1 for CTC blank
    );
    acoustic_model->to(device);
    torch::optim::Adam optimizer(acoustic_model->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CTCLoss ctc_loss(torch::nn::CTCLossOptions().zero_infinity(true));

    // --------------------------
    // Training loop
    // --------------------------
     
    auto training_start = std::chrono::high_resolution_clock::now();
    
    std::string log_filename = generate_log_filename("train_log");
    std::ofstream log_file(log_filename);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        acoustic_model->train();
        float epoch_loss = 0.0f;
        int num_batches = (N + batch_size - 1) / batch_size;

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int start = batch_idx * batch_size;
            int end = std::min(start + batch_size, N);
            int current_batch_size = end - start;

            auto batch_features = features.slice(0, start, end);
            auto batch_targets = targets.slice(0, start, end);
            auto batch_input_lengths = input_lengths.slice(0, start, end);
            auto batch_target_lengths = target_lengths.slice(0, start, end);

            auto logits = acoustic_model->forward(batch_features);
            logits = logits.transpose(0, 1).log_softmax(2);

            auto loss = ctc_loss(logits, batch_targets, batch_input_lengths, batch_target_lengths);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();

            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], "
                      << "Batch [" << (batch_idx + 1) << "/" << num_batches << "], "
                      << "Loss: " << loss.item<float>() << "\n";
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

        std::cout << "Epoch " << (epoch + 1) << " average loss: " << epoch_loss / num_batches << " duration: " << epoch_duration.count() << " seconds\n";
        log_file << "Epoch " << (epoch + 1) << ": " << epoch_duration.count() << " sec\n";
    }
    auto training_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_training_time = training_end - training_start;
    std::cout << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Average epoch duration: " << total_training_time.count() / num_epochs << " seconds\n";
    log_file.close();
    std::cout << "Training log saved to " << log_filename << "\n";
    
    torch::save(acoustic_model, "ctc.pt");
    std::cout << "Training complete. Model saved to ctc.pt\n";
    return 0;
}
