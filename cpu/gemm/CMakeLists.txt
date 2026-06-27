add_library(wgkernel_cpu_gemm STATIC
    sgemm.cc
)

target_include_directories(wgkernel_cpu_gemm
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_gemm)
