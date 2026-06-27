add_library(wgkernel_cpu_elementwise STATIC
    add.cc
)

target_include_directories(wgkernel_cpu_elementwise
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_elementwise)
