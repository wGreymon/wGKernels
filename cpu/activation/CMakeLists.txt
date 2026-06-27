add_library(wgkernel_cpu_activation STATIC
    exp.cc
    sigmoid.cc
    silu.cc
)

target_include_directories(wgkernel_cpu_activation
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_activation)
