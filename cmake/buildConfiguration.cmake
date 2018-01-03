
#----------------------------------------
#	      Compiler options	 
#----------------------------------------

# ensure the BUILD_SHARED_LIBS option exists
option(BUILD_SHARED_LIBS OFF)

## force visual studio to link with /MT and MTd when linking statically
if(NOT BUILD_SHARED_LIBS AND CMAKE_GENERATOR MATCHES "Visual Studio")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

