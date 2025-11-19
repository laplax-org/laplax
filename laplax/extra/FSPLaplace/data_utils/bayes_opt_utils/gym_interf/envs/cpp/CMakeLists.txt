cmake_minimum_required(VERSION 3.9)

project(interf VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)

if (${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    add_definitions(-DBUILDING_FOR_WINDOWS -D_USE_MATH_DEFINES)
endif()

add_library(interf SHARED src/utils.cpp src/interflib.cpp)

set_target_properties(interf PROPERTIES PUBLIC_HEADER include/interflib.h)

target_include_directories(interf PRIVATE include)

install(TARGETS interf
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    PUBLIC_HEADER DESTINATION include)
