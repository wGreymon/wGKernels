add_library(wgkernel_cuda_transpose_layout_transform STATIC
    src/tensor_transform.cu
)

target_link_libraries(wgkernel_cuda_transpose_layout_transform
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_transpose_layout_transform
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_transpose_layout_transform)
