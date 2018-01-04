#----------------------------------------
#			Define source files				 
#----------------------------------------

## define header files
set(QUESTIA_HEADERS
	"${CMAKE_CURRENT_BINARY_DIR}/../include/Questia/config.h"
)

## define source files to compile
set(QUESTIA_SOURCES
    "Main.cpp"
)

# allow other to get src and header files
set(QUESTIA_HEADERS ${QUESTIA_HEADERS} PARENT_SCOPE)
set(QUESTIA_SOURCES ${QUESTIA_SOURCES} PARENT_SCOPE)

# create executable
set(EXE_NAME questia)
add_executable(${EXE_NAME} ${QUESTIA_SOURCES} ${QUESTIA_HEADERS})


if(CMAKE_GENERATOR MATCHES "Visual Studio" AND BUILD_SHARED_LIBS)
    set_target_properties(${EXE_NAME} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT BUILD_SHARED_LIBS)
    target_link_libraries(${EXE_NAME} debug MSVCRTD.lib optimized MSVCRT.lib)
endif()

set_target_properties(${EXE_NAME} PROPERTIES VERSION ${QUESTIA_VERSION})
 
# link dependencies
target_link_libraries(${EXE_NAME} qeng)

# install rule
install(
    TARGETS ${EXE_NAME}
    RUNTIME DESTINATION bin COMPONENT bin
    LIBRARY DESTINATION lib${LIB_SUFFIX} COMPONENT bin
    ARCHIVE DESTINATION lib${LIB_SUFFIX} COMPONENT devel
)

 
