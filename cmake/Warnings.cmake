include_guard(GLOBAL)

function(wgkernel_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(${target} INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
        )
    endif()
endfunction()
