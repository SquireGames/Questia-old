
#----------------------------------------
#				Add files				 
#----------------------------------------

include_directories("include")

## append header files
set(QUESTIA_TEST_HEADERS
    "include/Questia/Test/Test.h"
)

set(QUESTIA_TEST_SOURCES
    "Test.cpp"
)


#----------------------------------------
#				Create test			 
#----------------------------------------

add_executable(test_g ${QUESTIA_HEADERS} ${QUESTIA_TEST_SOURCES} ${QUESTIA_TEST_HEADERS})
target_link_libraries(test_g qeng gtest gmock_main)
add_test(NAME unit_tests_g COMMAND test_g)
