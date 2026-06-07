add_library(wgkernel_cuda_pooling STATIC
    src/maxpool.cu
)

target_link_libraries(wgkernel_cuda_pooling
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_pooling
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_pooling)
