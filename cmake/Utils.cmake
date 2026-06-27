include_guard(GLOBAL)

function(wgkernel_configure_target target)
    target_link_libraries(${target} PRIVATE wgkernel_project_options)
    target_include_directories(${target}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/utils
            ${PROJECT_SOURCE_DIR}/cuda/common/include
    )

    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    if(CMAKE_CUDA_COMPILER)
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endfunction()

function(wgkernel_print_configuration)
    message(STATUS "")
    message(STATUS "wGKernel configuration")
    message(STATUS "  CMAKE_BUILD_TYPE      : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  WGKERNEL_ENABLE_CPU   : ${WGKERNEL_ENABLE_CPU}")
    message(STATUS "  WGKERNEL_ENABLE_CUDA  : ${WGKERNEL_ENABLE_CUDA}")
    message(STATUS "  WGKERNEL_BUILD_BENCHMARKS : ${WGKERNEL_BUILD_BENCHMARKS}")
    message(STATUS "")
endfunction()
