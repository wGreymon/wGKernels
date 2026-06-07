add_library(wgkernel_cuda_norm STATIC
    src/batchnorm.cu
)

target_link_libraries(wgkernel_cuda_norm
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_norm
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_norm)
