cmake_minimum_required(VERSION 3.10)
project(PQ_Recommendation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 假设Eigen头文件位于系统包含路径

add_executable(pq_recommendation
    DataLoader.cpp
    SimilarityComputer.cpp
    PQ.cpp
    main.cpp
)

target_include_directories(pq_recommendation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})