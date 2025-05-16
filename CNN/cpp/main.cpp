#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>

//Returns timestamped filename like "log_2025-05-10_0314.txt"
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
    torch::manual_seed(1);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    //torch::Device device(torch::kCUDA);
    torch::globalContext().setBenchmarkCuDNN(true);
    torch::globalContext().setDeterministicCuDNN(false);
    // Load MNIST dataset
    std::cout<<"test1";
    auto dataset = torch::data::datasets::MNIST("./data/MNIST/raw")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    
    std::cout<<"test2";
    std::cout << "Dataset loaded. Sample count: " << dataset.size().value() << "\n";

    // Data loader
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(64));

    // Define model and optimizer
    auto model = std::make_shared<Net>();
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(), 0.01);

    // Training loop
    const int num_epochs = 5;
    auto training_start = std::chrono::high_resolution_clock::now();
    
    std::string log_filename = generate_log_filename("train_log");
    std::ofstream log_file(log_filename);
        
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        size_t batch_idx = 0;
        for (auto& batch : *data_loader) {
            model->train();
            optimizer.zero_grad();
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            auto output = model->forward(data);
            auto loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();

            if (batch_idx++ % 100 == 0) {
                std::cout << "Epoch " << epoch << " | Batch " << batch_idx
                          << " | Loss: " << loss.item<float>() << "\n";
            }
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
        std::cout << "Epoch " << epoch << " duration: " << epoch_duration.count() << " seconds\n";
        log_file << "Epoch " << epoch << ": " << epoch_duration.count() << " sec\n";
    }
    auto training_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_training_time = training_end - training_start;
    std::cout << "Total training time: " << total_training_time.count() << " seconds\n";

    log_file << "Total training time: " << total_training_time.count() << " seconds\n";
    log_file << "Average epoch duration: " << total_training_time.count() / num_epochs << " seconds\n";
    log_file.close();

    std::cout << "Training log saved to " << log_filename << "\n";


    std::cout << "Saving model to mnist_cnn.pt...\n";
    torch::save(model, "./mnist_cnn.pt");
    std::cout << "Model saved to mnist_cnn.pt...\n";

    return 0;
}
