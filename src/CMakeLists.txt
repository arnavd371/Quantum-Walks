cmake_minimum_required(VERSION 3.14)
project(quantum_walk CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Eigen ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(quantum_walk quantum_walk.cpp)
target_link_libraries(quantum_walk Eigen3::Eigen)
target_compile_options(quantum_walk PRIVATE -O2 -Wall -Wextra)
