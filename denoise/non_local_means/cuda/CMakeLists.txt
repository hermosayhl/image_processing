cmake_minimum_required(VERSION 3.12)
project(non_local_means)

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++11 -O1")



# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


# include 目录
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include/)



find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -G;-g)
# (可执行文件, 源码)
cuda_add_executable(non_local_means src/non_local_means_demo.cpp src/non_local_means.cu)

# 添加可执行文件
# add_executable(non_local_means src/non_local_means_demo.cu src/non_local_means.cu)



# Boost
find_package(Boost REQUIRED COMPONENTS system filesystem)
if(NOT Boost_FOUND)
    message("Not found Boost")
endif()
include_directories(${Boost_INCLUDE_DIRS})
message("${Boost_INCLUDE_DIRS}")
message("${Boost_LIBRARIES}")


# 寻找 opencv 的库
set(OpenCV_DIR /usr/local/opencv/opencv-4.5.2/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS})

# 设置库文件
target_link_libraries(non_local_means ${OpenCV_LIBS} ${Boost_LIBRARIES})



