#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>

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

// Must match the Net structure used in training
struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 10, 5));
        conv2 = register_module("conv2", torch::nn::Conv2d(10, 20, 5));
        fc1 = register_module("fc1", torch::nn::Linear(320, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<Net>();

    try {
        torch::load(model, "./mnist_cnn.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return -1;
    }

    double total_inference_time = 0.0;
    model->eval();

    // Load MNIST test set
    auto test_dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 64);

    int64_t correct = 0, total = 0;
    auto inference_start = std::chrono::high_resolution_clock::now();
    for (const auto& batch : *test_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto start = std::chrono::high_resolution_clock::now();
        auto output = model->forward(data);
        auto end = std::chrono::high_resolution_clock::now();
        auto predicted = output.argmax(1);

        correct += predicted.eq(targets).sum().item<int64_t>();
        total += targets.size(0);

        std::chrono::duration<double> inference_duration = end - start;
        total_inference_time += inference_duration.count();
    }
    auto inference_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(inference_end - inference_start).count();
    double accuracy = 100.0 * correct / total;
    double avg_inference_time = total_inference_time / total;

    std::cout << "Test Accuracy: " << accuracy << "%\n";
    std::cout << "Total Inference Time: " << duration << " seconds\n";
    std::cout << "Avg Inference Time per Sample: " << avg_inference_time << " seconds\n";
    
    std::string log_filename = generate_log_filename("test_log");
    std::ofstream log_file(log_filename);

    log_file << "Test Accuracy: " << accuracy << "%\n";
    log_file << "Total Inference Time: " << duration << " seconds\n";
    log_file << "Avg inference time per sample: " << avg_inference_time << " seconds\n";

    log_file.close();
    std::cout << "Test log saved to " << log_filename << "\n";

    return 0;
}
