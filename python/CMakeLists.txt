cmake_minimum_required(VERSION 3.24)
project(wgkernel_python LANGUAGES CXX)

# Build per-device pybind modules against the already-compiled backend libraries.
# This directory is included from the top-level CMakeLists.txt.
# Usage (from top-level build directory):
#   cmake --build . --target wgkernel_cuda -j

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# pybind11 cmake files from pip-installed package.
set(pybind11_DIR "/home/wgreymon/miniconda3/envs/jnz_env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 CONFIG REQUIRED)

if(WGKERNEL_ENABLE_CPU)
    pybind11_add_module(wgkernel_cpu
        bindings/cpu_module.cpp
    )

    target_link_libraries(wgkernel_cpu PRIVATE
        ${pybind11_LIBRARIES}
        wgkernel_cpu_activation
        wgkernel_cpu_attention
        wgkernel_cpu_convolution
        wgkernel_cpu_elementwise
        wgkernel_cpu_embedding_indexing
        wgkernel_cpu_gemm
        wgkernel_cpu_norm
        wgkernel_cpu_pooling
        wgkernel_cpu_reduce
    )

    target_include_directories(wgkernel_cpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${pybind11_INCLUDE_DIRS}
    )
endif()

# CUDA backend module: wgkernel_cuda.
# Additional device modules should follow the same Python API names so tests can
# load wgkernel_<device> based on --device.
if(WGKERNEL_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(WGKERNEL_CUDA_LIBS
        wgkernel_cuda_reduce
        wgkernel_cuda_activation
        wgkernel_cuda_elementwise
        wgkernel_cuda_embedding_indexing
        wgkernel_cuda_layout
        wgkernel_cuda_norm
        wgkernel_cuda_convolution
        wgkernel_cuda_pooling
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
endif()
