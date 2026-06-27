add_library(wgkernel_cpu_embedding_indexing STATIC
    embedding.cc
)

target_include_directories(wgkernel_cpu_embedding_indexing
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cpu_embedding_indexing)
