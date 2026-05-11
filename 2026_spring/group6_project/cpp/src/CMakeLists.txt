add_library(ghostbook_matching_engine matching_engine/matching_main.cpp)
add_executable(matching_engine matching_engine/main.cpp)
add_executable(gateway gateway/main.cpp gateway/tcp_gateway.cpp)
add_executable(matching_engine_bench ../tests/matching_engine_bench.cpp)

target_include_directories(ghostbook_matching_engine PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(matching_engine PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(gateway PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(matching_engine_bench PRIVATE ${CMAKE_SOURCE_DIR}/src/matching_engine)

target_link_libraries(matching_engine PRIVATE ghostbook_matching_engine)
target_link_libraries(gateway PRIVATE ghostbook_matching_engine Boost::system pthread)
target_link_libraries(matching_engine_bench PRIVATE ghostbook_matching_engine)

add_library(ghostbook_replay_engine replay_engine/replay_engine.cpp)
target_include_directories(ghostbook_replay_engine
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src/replay_engine
    PRIVATE ${CMAKE_SOURCE_DIR}/src/matching_engine)
target_link_libraries(ghostbook_replay_engine PUBLIC ghostbook_matching_engine)

add_library(ghostbook_ndfex_gateway gateway/ndfex_gateway.cpp)
target_include_directories(ghostbook_ndfex_gateway
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src/gateway
    PRIVATE ${CMAKE_SOURCE_DIR}/src/replay_engine
            ${CMAKE_SOURCE_DIR}/src/matching_engine)
target_link_libraries(ghostbook_ndfex_gateway
    PUBLIC ghostbook_matching_engine ghostbook_replay_engine Boost::system pthread)

add_executable(ndfex_gateway gateway/main_ndfex.cpp)
target_link_libraries(ndfex_gateway PRIVATE ghostbook_ndfex_gateway)

add_executable(replay_engine replay_engine/main.cpp)
target_include_directories(replay_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/replay_engine
    ${CMAKE_SOURCE_DIR}/src/gateway)
target_link_libraries(replay_engine PRIVATE ghostbook_ndfex_gateway ghostbook_replay_engine)
