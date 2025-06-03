cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if (MSVC)
    set_source_files_properties(src/main.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2" "/O2" "/fp:fast")
else()
    set_source_files_properties(src/main.cpp PROPERTIES COMPILE_OPTIONS "-mavx2" "-O3" "-ffast-math")
endif()

pybind11_add_module(_core MODULE src/main.cpp)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
