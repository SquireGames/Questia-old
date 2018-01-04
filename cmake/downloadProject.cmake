
include(CMakeParseArguments)

function(downloadProject)
    cmake_parse_arguments(
        DP_ARG  # prefix of input arguments
        ""      # list of boolean argument names
        "PROJECT;REPOSITORY;COMMIT;VERSION_MAJOR;VERSION_MINOR" # regular arguments
        ""      # list argumentd
        ${ARGN} # arguments to parse
    )
    
    set(DP_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/${DP_ARG_PROJECT}-download")
    set(DP_SOURCE_DIR   "${CMAKE_BINARY_DIR}/${DP_ARG_PROJECT}-src")
    set(DP_BINARY_DIR   "${CMAKE_BINARY_DIR}/${DP_ARG_PROJECT}-build")
    
    # CLion compatibility
    file(REMOVE "${DP_DOWNLOAD_DIR}/CMakeCache.txt")
    
    # add source and binary path locations to variable
    set(${DP_ARG_PROJECT}_SOURCE_DIR "${DP_SOURCE_DIR}" PARENT_SCOPE)
    set(${DP_ARG_PROJECT}_BINARY_DIR "${DP_BINARY_DIR}" PARENT_SCOPE)
    
    # generate CMakeLists for subproject
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/downloadProject.CMakeLists.cmake.in"
        "${DP_DOWNLOAD_DIR}/CMakeLists.txt")
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} 
            -G "${CMAKE_GENERATOR}"
            -D "CMAKE_MAKE_PROGRAM:FILE=${CMAKE_MAKE_PROGRAM}"
            .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${DP_DOWNLOAD_DIR})
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${DP_DOWNLOAD_DIR})
    
    add_subdirectory("${DP_SOURCE_DIR}" "${DP_BINARY_DIR}")
    
    # TODO: ensure the version is correct
    
endfunction(downloadProject)