add_library(wgkernel_cuda_common INTERFACE)

target_link_libraries(wgkernel_cuda_common
    INTERFACE
        wgkernel_project_options
        CUDA::cudart
)

target_include_directories(wgkernel_cuda_common
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
