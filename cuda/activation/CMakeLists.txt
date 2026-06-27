add_library(wgkernel_cuda_activation STATIC
    src/activation.cu
)

target_link_libraries(wgkernel_cuda_activation
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_activation
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_activation)
