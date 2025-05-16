cmake_minimum_required(VERSION 3.18)
project(RNN)

set(CMAKE_PREFIX_PATH "C:/libtorchgpu/libtorch")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(RNNTrain main.cpp IMDBDataset.h SentimentLSTMModel.h)
target_link_libraries(RNNTrain "${TORCH_LIBRARIES}")

add_executable(RNNTest RNNTest.cpp IMDBDataset.h SentimentLSTMModel.h)
target_link_libraries(RNNTest "${TORCH_LIBRARIES}")

add_custom_command(TARGET RNNTrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "C:/libtorchgpu/libtorch/lib"
    $<TARGET_FILE_DIR:RNNTrain>
)