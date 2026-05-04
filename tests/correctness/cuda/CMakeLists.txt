add_executable(wgkernel_reduce_tool
    reduce_tool.cu
)

target_link_libraries(wgkernel_reduce_tool
    PRIVATE
        wgkernel_cuda_reduce
)

wgkernel_configure_target(wgkernel_reduce_tool)

add_test(
    NAME reduce_cpp_correctness
    COMMAND wgkernel_reduce_tool
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME reduce_pytorch_correctness
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_vs_pytorch.py
        --tool
        $<TARGET_FILE:wgkernel_reduce_tool>
)
