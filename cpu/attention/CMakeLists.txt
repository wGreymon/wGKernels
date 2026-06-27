add_library(wgkernel_cpu_attention STATIC
    self_attention.cc
)

target_include_directories(wgkernel_cpu_attention
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_attention)
