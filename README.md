# Benchmarking-Deep-Learning-Frameworks# Deep Learning Benchmarking Suite

This project provides a comparative benchmarking framework of key deep learning models—**CNN, RNN, CTC, and Transformer**—across **LibTorch C++**, **PyTorch**, and **TensorFlow**, targeting both **CPU and GPU executions**.

---

## Project Structure

Each model is organized under its own directory containing implementations in different frameworks:

```

├── CNN
│   ├── cpp/           # LibTorch C++ implementation
│   ├── pytorch/       # PyTorch Python implementation
│   └── tensorflow/    # TensorFlow Python implementation
├── RNN
│   ├── cpp/
│   ├── pytorch/
│   └── tensorflow/
├── CTC
│   ├── cpp/
│   ├── pytorch/
│   └── tensorflow/
└── Transformer
├── cpp/
├── pytorch/
└── tensorflow/

---

## Running on CPU or GPU

All implementations are designed to run on **both CPU and GPU** by modifying a **single line of code in each script** (device setup line):
- **LibTorch C++:** Toggle `torch::Device` between `"cuda"` and `"cpu"`.
- **PyTorch Python:** Change `torch.device("cuda")` to `torch.device("cpu")`.
- **TensorFlow Python:** Use `"/GPU:0"` or `"/CPU:0"` device contexts.

---

## Environments and Versions

| Component              | Version                                                    |
|------------------------|------------------------------------------------------------|
| **Python**              | 3.10                                                       |
| **CUDA**                | 12.9                                                       |
| **PyTorch (Python)**    | 2.2.2                                                      |
| **LibTorch C++**        | 2.2.2 (CUDA 12.1 build)                                    |
| **TensorFlow (Python)** | 2.15.0 (CPU: Windows, GPU: Ubuntu 24.04 only)              |
| **MSVC Compiler (C++)** | 19.43.34810 (Visual Studio 2022)                           |

---

## Requirements

### Python (PyTorch, TensorFlow)
Install via `requirements.txt` or manually:
```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install tensorflow==2.15.0
````

### C++ (LibTorch)

* Download **LibTorch 2.2.2 GPU build (CUDA 12.1)** from [PyTorch official](https://pytorch.org/get-started/locally/).
* Ensure **CUDA Toolkit 12.9** installed.
* **Visual Studio 2022 Community (MSVC v143, 64-bit)**.
* CMake 3.27+.

### Additional (Optional)

* nvToolsExt (Windows only, used optionally for GPU profiling in some builds).

---

## Notes

* TensorFlow GPU runs were only tested under **Ubuntu 24.04 + CUDA 12.9**, since TensorFlow 2.15.0 lacks official CUDA 12.9 support on Windows.
* All datasets are loaded locally or via provided scripts for IMDB, MNIST, and custom datasets.

---

## License

MIT License.

## Authors

Alexios Chrysostomos Papazoglou

---

