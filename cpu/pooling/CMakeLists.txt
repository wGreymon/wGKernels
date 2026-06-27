add_library(wgkernel_cpu_pooling STATIC
    maxpooling.cc
)

target_include_directories(wgkernel_cpu_pooling
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_pooling)
