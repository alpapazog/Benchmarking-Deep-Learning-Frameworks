# Benchmarking-Deep-Learning-Frameworks

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
Each implementation has a training file (or script marked as main.py or main.cpp) and an inference file (or script marked as test.py or test.cpp). Build with cmake for cpp files, and just run the scripts for python files. Some dataset handling is required, and instructions are provided below.

## Build and run libtorch c++ projects.
CMakeFiles (CMakeLists.txt) have paths for libtorch and CUDA, make sure you fix them to the correct path depending on your system installations.

The following commands can be run in Windows Powershell (individually) to build and run (train and test) the CNN model with c++ using libtorch.
```
cd \path\to\CNN\cpp\
mkdir build
cd build
$env:CudaToolkitDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" 
cmake .. -DCMAKE_PREFIX_PATH="C:/libtorchgpu/libtorch" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..
python getdata.py # download and process dataset
.\build\Release\CNNGpu.exe # train and log
.\build\Release\TestCNN.exe # inference and log 

````
The following commands can be run in Windows Powershell (individually) to build and run (train and test) the CNN model with python using pytorch (or tensorflow when in a pytorch directory).

```
cd \path\to\CNN\pytorch\
python main.py # train and log
python test.py # inference and log

````

Windows environment variables need to be set up correctly for these to succed. Path variable must include CUDA 12.8, and if possible the libtorch dll's. If libtorch dll's are included in path they can be excluded from the cmake files, cause copying the entire library helps deployment but backtracks development.

## Running on CPU or GPU

All implementations are designed to run on **both CPU and GPU** by modifying a **single line of code in each script** (device setup line):
- **LibTorch C++:** Toggle `torch::Device` between `"cuda"` and `"cpu"`.
- **PyTorch Python:** Change `torch.device("cuda")` to `torch.device("cpu")`.
- **TensorFlow Python:** Use `"/GPU:0"` or `"/CPU:0"` device contexts.

---

## Environments and Versions

| Component              | Version                                                    |
|------------------------|------------------------------------------------------------|
| **Python**              | 3.13 (Windows), 3.10 (Linux)                               |
| **CUDA**                | 12.8 (Windows), 12.2 (Linux)                               |
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

