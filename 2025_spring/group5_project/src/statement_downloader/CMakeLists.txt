project(statement_downloader)

add_library(${PROJECT_NAME} STATIC src/statement_downloader.cpp)
add_library(sub::statement_downloader ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PUBLIC CURL::libcurl)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

find_package(Catch2 3 REQUIRED)
add_executable(${PROJECT_NAME}_unit_tests test/test_parse_mixed_number.cpp test/test_statement_downloader.cpp)
target_link_libraries(${PROJECT_NAME}_unit_tests PRIVATE Catch2::Catch2WithMain sub::statement_downloader)

include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME}_unit_tests)