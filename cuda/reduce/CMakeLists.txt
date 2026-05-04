add_library(wgkernel_cuda_reduce STATIC
    src/argmax.cu
    src/max.cu
    src/sum.cu
)

target_link_libraries(wgkernel_cuda_reduce
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_reduce
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_reduce)
