cmake_minimum_required(VERSION 3.24.4)

project(main)
set(CMAKE_CXX_STANDARD 23)

add_executable(${PROJECT_NAME} main.cpp)

project(generator)
set(CMAKE_CXX_STANDARD 23)

add_executable(${PROJECT_NAME} generator.cpp)

project(stress)
set(CMAKE_CXX_STANDARD 23)

add_executable(${PROJECT_NAME} stress.cpp
)

