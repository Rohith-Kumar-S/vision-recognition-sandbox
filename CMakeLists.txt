# CMakeLists.txt for a simple OpenCV project
cmake_minimum_required(VERSION 3.16)

# Define the project name
project(OBJRECSystem)

# Set ONNX Runtime path relative to project directory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

# Add executable
add_executable(OBJRECSystem 
    vidDisplay.cpp
    projectutils.h
    csv_util/csv_util.cpp
)

# Link libraries
target_link_libraries(OBJRECSystem 
    ${OpenCV_LIBS}
)