add_library(wgkernel_cuda_convolution STATIC
    src/conv2d.cu
)

target_link_libraries(wgkernel_cuda_convolution
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_convolution
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_convolution)
