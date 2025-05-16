cmake_minimum_required(VERSION 3.18)
project(TRANSFORMER)

set(CMAKE_PREFIX_PATH "C:/libtorchgpu/libtorch")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(TRANSFORMERTrain main.cpp)
target_link_libraries(TRANSFORMERTrain "${TORCH_LIBRARIES}")

add_executable(TRANSFORMERTest TTest.cpp)
target_link_libraries(TRANSFORMERTest "${TORCH_LIBRARIES}")

add_custom_command(TARGET TRANSFORMERTrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "C:/libtorchgpu/libtorch/lib"
    $<TARGET_FILE_DIR:TRANSFORMERTrain>
)