cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CNNGpu)

set(CMAKE_PREFIX_PATH "C:/libtorchgpu/libtorch")
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# Needed to find nvToolsExt manually
link_directories("C:/Program Files/NVIDIA Corporation/nvToolsExt/lib/x64")

add_executable(CNNGpu main.cpp)
target_link_libraries(CNNGpu "${TORCH_LIBRARIES}" nvToolsExt64_1.lib)

add_executable(TestCNN test.cpp)
target_link_libraries(TestCNN "${TORCH_LIBRARIES}" nvToolsExt64_1.lib)
add_custom_command(TARGET CNNGpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "C:/libtorchgpu/libtorch/lib"
    $<TARGET_FILE_DIR:CNNGpu>
)