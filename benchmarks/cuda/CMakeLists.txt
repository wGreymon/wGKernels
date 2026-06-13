add_executable(wgkernel_reduce_benchmark
    reduce_benchmark.cu
)

target_link_libraries(wgkernel_reduce_benchmark
    PRIVATE
        wgkernel_cuda_reduce
)

wgkernel_configure_target(wgkernel_reduce_benchmark)

# The conv2d benchmark is a Python script that drives the
# `wgkernel_conv2d_tool` binary from the correctness suite. It is wired into
# CTest as an opt-in target so users can run `ctest -R conv2d_bench` after
# building the conv2d tool.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(TARGET wgkernel_conv2d_tool)
    add_test(
        NAME conv2d_bench_torch
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/bench_conv2d_torch.py
            --tool
            $<TARGET_FILE:wgkernel_conv2d_tool>
    )
endif()
