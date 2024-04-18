cmake_minimum_required(VERSION 3.12)
project(guided_filter)

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++14 -O2")


set(PROJECT_SOURCE_DIR ..)

# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


# include 目录
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include/)

# 添加可执行文件
add_executable(guided_filter src/guided_demo.cpp src/guided_filter_padded.cpp src/bilateral_filter.cpp)


# 设置  opencv
set(OpenCV_DIR "D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/")
# 寻找 opencv 的库
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs xphoto)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS} )
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS} )


# 设置 Eigen3
include_directories(D:/environments/C++/3rdparty/Eigen3/eigen-3.3.9/installed/include/eigen3)

# 设置库文件
target_link_libraries(guided_filter ${OpenCV_LIBS})

