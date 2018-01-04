
# define an option to generate documentation, on by default
option(BUILD_DOC "Build Documentation" OFF)

# generate documentation
if(BUILD_DOC)
    # see if Doxygen exists
    find_package(Doxygen)

	if(NOT DOXYGEN_FOUND)
		message(STATUS "Doxygen must be intalled to generate documentation")
	else()
		# set input and output files
		set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/../doc/Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
		
		# configure DOXYGEN_IN
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		
		# add generating doxygen as a target
		add_custom_target(
			doc ALL
			COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			COMMENT "Generating Doxygen Documentation"
			VERBATIM
		)
	endif()
endif()
