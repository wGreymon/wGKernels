include_guard(GLOBAL)

function(wgkernel_configure_cuda)
    find_package(CUDAToolkit REQUIRED)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for" FORCE)
    endif()

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    message(STATUS "Using CUDA toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endfunction()
