
# 项目名称
PROJECT(bayers_matting)

# cmake 版本 >= 3.5
cmake_minimum_required(VERSION 3.5)

# 屏蔽警告
add_definitions(-w)

# 设置可执行文件目录 CMAKE_RUNTIME_OUTPUT_DIRECTORY
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O1")

# 添加我自己的头文件
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加 eigen3 库
include_directories(SYSTEM /usr/include/eigen3)

# 编译 .cpp 到可执行文件
add_executable(executor src/bayers_matting.cpp)

# 寻找 opencv 的库
set(OpenCV_DIR /usr/local/opencv/opencv-4.5.2/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS})
# 设置库文件
target_link_libraries(executor ${OpenCV_LIBS})