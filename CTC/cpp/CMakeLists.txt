cmake_minimum_required(VERSION 3.18)
project(CTC)

set(CMAKE_PREFIX_PATH "C:/libtorchgpu/libtorch")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(CTCTrain main.cpp)
target_link_libraries(CTCTrain "${TORCH_LIBRARIES}")

add_executable(CTCTest CTCTest.cpp)
target_link_libraries(CTCTest "${TORCH_LIBRARIES}")

add_custom_command(TARGET CTCTrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "C:/libtorchgpu/libtorch/lib"
    $<TARGET_FILE_DIR:CTCTrain>
)