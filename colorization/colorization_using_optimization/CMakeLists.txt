


cmake_minimum_required(VERSION 3.19)
project(colorize)


# 指定这个还是会调用 visual studio 的 cl, 得在命令行 -G "MinGW Makefiles" 才能正常使用 gcc, g++
# SET(CMAKE_C_COMPILER "gcc")
# SET(CMAKE_CXX_COMPILER "g++")

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2")
set(PROJECT_SOURCE_DIR ./)

# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# include 目录
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(colorize
	src/colorize.cpp
)

set(OpenCV_DIR "D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/")
# 寻找 opencv 的库
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS} )
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS} )

# 设置 Eigen3
include_directories(D:/environments/C++/3rdparty/Eigen3/eigen-3.3.9/installed/include/eigen3)

## 设置库文件
target_link_libraries(colorize ${OpenCV_LIBS})