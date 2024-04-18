
# 项目名称
PROJECT(opencv_dnn_demo)

# cmake 版本 >= 3.5
cmake_minimum_required(VERSION 3.5)

# 屏蔽警告
add_definitions(-w)

# 设置可执行文件目录 CMAKE_RUNTIME_OUTPUT_DIRECTORY
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# set(CMAKE_CXX_STANDARD 14)
# add_compile_options(-std=c++11)
# add_compile_options(-O3)
SET(CMAKE_CXX_FLAGS "-std=c++11 -O1")


# 添加我自己的头文件
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)






# # CUDA
# find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS -G;-g)
# # (可执行文件, 源码)
# cuda_add_executable(executor src/bilateral_filter.cu)

add_executable(executor src/bilateral_filter.cpp)


# 寻找 opencv 的库
set(OpenCV_DIR /usr/local/opencv/opencv_4.5.2/share/opencv4)
find_package(OpenCV REQUIRED)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS})
# 设置库文件
target_link_libraries(executor ${OpenCV_LIBS})


