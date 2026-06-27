add_library(wgkernel_cpu_convolution STATIC
    conv2d.cc
)

target_include_directories(wgkernel_cpu_convolution
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_convolution)
