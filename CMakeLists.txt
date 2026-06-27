cmake_minimum_required(VERSION 3.24)

project(wGKernel LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Warnings)
include(CUDA)
include(Utils)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

option(WGKERNEL_BUILD_BENCHMARKS "Build benchmark targets" ON)
option(WGKERNEL_ENABLE_CPU "Enable CPU baseline kernels and CPU-specific targets" ON)
option(WGKERNEL_ENABLE_CUDA "Enable CUDA kernels and CUDA-specific targets" ON)

add_library(wgkernel_project_options INTERFACE)
wgkernel_enable_warnings(wgkernel_project_options)

if(WGKERNEL_ENABLE_CPU)
    add_subdirectory(cpu)
endif()

if(WGKERNEL_ENABLE_CUDA)
    enable_language(CUDA)
    wgkernel_configure_cuda()
    add_subdirectory(cuda)
endif()

add_subdirectory(python)

if(WGKERNEL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

wgkernel_print_configuration()
