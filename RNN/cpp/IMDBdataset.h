#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

class IMDBDataset : public torch::data::datasets::Dataset<IMDBDataset> {
public:
    IMDBDataset(const std::string& path) {
        try {
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Could not open " + path);

            std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();

            c10::IValue ivalue = torch::pickle_load(buffer);
            auto dict = ivalue.toGenericDict();
            inputs_ = dict.at("inputs").toTensor();
            labels_ = dict.at("labels").toTensor();
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load dataset: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    torch::data::Example<> get(size_t index) override {
        return {inputs_[index], labels_[index]};
    }

    torch::optional<size_t> size() const override {
        return inputs_.size(0);
    }

private:
    torch::Tensor inputs_, labels_;
};
