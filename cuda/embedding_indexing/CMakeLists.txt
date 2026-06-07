add_library(wgkernel_cuda_embedding_indexing STATIC
    src/indexing.cu
)

target_link_libraries(wgkernel_cuda_embedding_indexing
    PUBLIC
        wgkernel_cuda_common
)

target_include_directories(wgkernel_cuda_embedding_indexing
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

wgkernel_configure_target(wgkernel_cuda_embedding_indexing)
