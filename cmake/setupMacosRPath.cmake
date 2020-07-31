
# configure rpath
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_SKIP_BUILD_RPATH ON)
    set(CMAKE_SKIP_RPATH OFF)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
        
    endif()
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# enforcing RPath install_name for a particular executable
macro(fix_install_name executable)
    if(APPLE AND BUILD_SHARED_LIBS)
        add_custom_command(TARGET ${executable} POST_BUILD 
                           COMMAND for f in `ls ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/` \;
                                   do ${CMAKE_INSTALL_NAME_TOOL}
                                         -change $$f @rpath/$$f
                                        $<TARGET_FILE:${executable}> || : \; 
                                   done \; )
    endif()
endmacro()

# enforcing RPath install_name for libgmock_main
macro(fix_install_name_gtest executable)
    if(APPLE AND BUILD_SHARED_LIBS)
        add_custom_command(TARGET ${executable} POST_BUILD 
                           COMMAND for f in `ls ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/` \;
                                   do ${CMAKE_INSTALL_NAME_TOOL}
                                         -change $$f @rpath/$$f
                                        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libgmock.dylib || : \;
                                   done \; )
        add_custom_command(TARGET ${executable} POST_BUILD 
                           COMMAND for f in `ls ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/` \;
                                   do ${CMAKE_INSTALL_NAME_TOOL}
                                         -change $$f @rpath/$$f
                                        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libgmock_main.dylib || : \;
                                   done \; )
    endif()
endmacro()
