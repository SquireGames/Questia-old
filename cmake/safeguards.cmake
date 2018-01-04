## dont allow in-source builds
if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
	message(FATAL_ERROR "In-source builds not allowed")
endif()
