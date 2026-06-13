add_executable(wgkernel_reduce_tool
    reduce_tool.cu
)

target_link_libraries(wgkernel_reduce_tool
    PRIVATE
        wgkernel_cuda_reduce
)

wgkernel_configure_target(wgkernel_reduce_tool)

add_executable(wgkernel_conv2d_tool
    conv2d_tool.cu
)

target_link_libraries(wgkernel_conv2d_tool
    PRIVATE
        wgkernel_cuda_convolution
)

wgkernel_configure_target(wgkernel_conv2d_tool)

# The tool binaries above are invoked by the Python test harness. C++ no
# longer owns correctness or performance checks; the Python scripts under
# this directory drive both.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME reduce_pytorch_correctness
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_vs_pytorch.py
        --tool
        $<TARGET_FILE:wgkernel_reduce_tool>
)

add_test(
    NAME conv2d_pytorch_correctness
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_conv2d_vs_pytorch.py
        --tool
        $<TARGET_FILE:wgkernel_conv2d_tool>
)
