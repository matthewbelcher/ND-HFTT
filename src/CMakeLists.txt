cmake_minimum_required(VERSION 3.10)

project(fed-market-impact VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()
message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "-gdwarf-4" )

OPTION(SET_DEBUG "IS DEBUG" ON)
IF(SET_DEBUG)
    ADD_DEFINITIONS(-DIS_DEBUG=1)
ELSE()
    ADD_DEFINITIONS(-DIS_DEBUG=0)
ENDIF(SET_DEBUG)


enable_testing()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

find_package( CURL REQUIRED )

add_subdirectory(statement_downloader)

add_executable(executor main.cpp)
target_link_libraries(executor PUBLIC sub::statement_downloader)