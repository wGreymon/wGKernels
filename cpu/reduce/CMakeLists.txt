add_library(wgkernel_cpu_reduce STATIC
    argmax.cc
    max.cc
    softmax.cc
    sum.cc
)

target_include_directories(wgkernel_cpu_reduce
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_reduce)
