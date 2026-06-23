add_executable(wgkernel_reduce_benchmark
    reduce_benchmark.cu
)

target_link_libraries(wgkernel_reduce_benchmark
    PRIVATE
        wgkernel_cuda_reduce
)

wgkernel_configure_target(wgkernel_reduce_benchmark)
