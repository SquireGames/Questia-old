cmake_minimum_required (VERSION 3.8.2)
enable_language(C CXX)

# cmake module directory
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

## c++ specification
set(CMAKE_CXX_STANDARD 17)

## define directory structure of output 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

## define project properties
project(Questia)
set(QUESTIA_VERSION_MAJOR "0")
set(QUESTIA_VERSION_MINOR "0")
set(QUESTIA_VERSION_PATCH "1")
set(QUESTIA_VERSION ${QUESTIA_VERSION_MAJOR}.${QUESTIA_VERSION_MINOR}.${QUESTIA_VERSION_PATCH})
set(QUESTIA_SOVERSION ${QUESTIA_VERSION_MAJOR}.${QUESTIA_VERSION_MINOR})

## set static/shared and release/debug options
include(buildConfiguration)

## get dependencies
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(KEEP_RPATHS)

## define helper functions to help with linking
include(qengMacros)

## define project include directories
include_directories(
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_BINARY_DIR}/include
)

## generate all necessary header files
add_subdirectory(include)
## compile library
add_subdirectory(src)
## generate documentation
add_subdirectory(doc)

## testing
option(QST_BUILD_TESTS OFF)
if(QST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()