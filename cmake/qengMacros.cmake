## set up RPATH for macos
include(setupMacosRPath)

# links qeng to an executable
macro(qeng_target_link_libraries target)
    if(DEFINED QENG_VERSION)
        # link qeng
        target_link_libraries(${target} qeng)
    else()
        # link qeng and gtest/gmock
        conan_target_link_libraries(${target} ${CONAN_LIBS} ${CONAN_LIBS_QENG_RELEASE_SHARED} ${CONAN_LIBS_QENG_RELEASE_STATIC} ${CONAN_LIBS_QENG_DEBUG_SHARED} ${CONAN_LIBS_QENG_DEBUG_STATIC})
    endif()

    # fix macos dynamic library linking issues
    fix_install_name(${target})
endmacro()

# links qeng and gtest to an executable
macro(qeng_target_link_libraries_test target)
    if(DEFINED QENG_VERSION)
        # link qeng and gtest/gmock
        target_link_libraries(${target} qeng ${CONAN_LIBS})
    else()
        # link qeng and gtest/gmock
        conan_target_link_libraries(${target} ${CONAN_LIBS})
    endif()
    
    # fix macos dynamic library linking issues
    fix_install_name(${target})
    fix_install_name_gtest(${target})
endmacro()

# links the target qeng lib
macro(qeng_target_link_libraries_libqeng target)
    # link conan except for GTest
    set(CONAL_LIBS_NO_GTEST ${CONAN_LIBS})
    if(CONAN_LIBS_GTEST)
        list(REMOVE_ITEM CONAL_LIBS_NO_GTEST ${CONAN_LIBS_GTEST})
    endif()
    target_link_libraries(${target} ${CONAL_LIBS_NO_GTEST})
    
     # use pthread if on UNIX
    if(UNIX)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${target} Threads::Threads)
    endif()   
    
    # link proper static libraries for Visual Studio
    if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT BUILD_SHARED_LIBS)
        target_link_libraries(${target} debug MSVCRTD.lib optimized MSVCRT.lib)
    endif()
    
    # fix macos dynamic library linking issues
    fix_install_name(${target})
endmacro()