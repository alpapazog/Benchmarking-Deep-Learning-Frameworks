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
```
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
```
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

## Datasets Used and Preparation

| Model           | Dataset                            | Download / Preparation                                                                                                                            |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CNN**         | MNIST Handwritten Digits           | PyTorch and TensorFlow download automatically; C++ uses `getdata.py` if needed.                                                                   |
| **RNN**         | IMDB Movie Review Sentiment        | PyTorch and TensorFlow download automatically; C++ uses `getdata.py` if needed.                                                                   |
| **CTC**         | LibriSpeech `train-clean-5` subset | **Manually download** [train-clean-5.tar.gz](https://www.openslr.org/31/)<br>Place it in `CTC/data/` and run `getdata.py` to prepare the dataset. |
| **Transformer** | IMDB Movie Review Sentiment        | PyTorch and TensorFlow download automatically; C++ uses `getdata.py` if needed.                                                                   |

### Notes on Dataset Handling:

* **CTC model requires manual dataset download and preprocessing.**
* For **CTC**, only **`train-clean-5.tar.gz`** from LibriSpeech [OpenSLR 31](https://www.openslr.org/31/) is required.
* After placing the file in the `CTC/data/` folder, run:

  ```bash
  python getdata.py
  ```
* Most other models (PyTorch and TensorFlow implementations) automatically download the datasets using framework APIs.
* **LibTorch C++ implementations provide `getdata.py` scripts for dataset preparation where needed.**

---

