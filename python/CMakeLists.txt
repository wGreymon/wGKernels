cmake_minimum_required(VERSION 3.24)
project(wgkernel_python LANGUAGES CXX CUDA)

# Build the pybind module against the already-compiled CUDA static libraries.
# This directory is included from the top-level CMakeLists.txt.
# Usage (from top-level build directory):
#   cmake --build . --target wgkernel_cuda -j

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# pybind11 cmake files from pip-installed package.
set(pybind11_DIR "/home/wgreymon/miniconda3/envs/jnz_env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 CONFIG REQUIRED)

# Collect all wgkernel CUDA static libraries (must be built first).
set(WGKERNEL_CUDA_LIBS
    wgkernel_cuda_reduce
    wgkernel_cuda_activation_elementwise
    wgkernel_cuda_embedding_indexing
    wgkernel_cuda_transpose_layout_transform
    wgkernel_cuda_norm
    wgkernel_cuda_convolution
    wgkernel_cuda_pooling
    wgkernel_cuda_resize
)

pybind11_add_module(wgkernel_cuda
    bindings/pybind_module.cpp
)

target_link_libraries(wgkernel_cuda PRIVATE
    ${pybind11_LIBRARIES}
    ${CUDAToolkit_LIBRARY}
    CUDA::cudart
    ${WGKERNEL_CUDA_LIBS}
)

target_include_directories(wgkernel_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${pybind11_INCLUDE_DIRS}
)
