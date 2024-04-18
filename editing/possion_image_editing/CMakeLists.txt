cmake_minimum_required(VERSION 3.12)
# set (CMAKE_C_COMPILER "D:/environments/C++/MinGW_posix_sjlj/mingw64/bin/gcc.exe")
# set (CMAKE_CXX_COMPILER "D:/environments/C++/MinGW_posix_sjlj/mingw64/bin/g++.exe")

# 设置 ccache 加快编译速度
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)


project(possion_image_editing)

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++14 -O2")

set(PROJECT_SOURCE_DIR ..)

# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# include 目录
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include/)

# 添加可执行文件
add_executable(possion_image_editing src/possion_image_editing_demo.cpp)

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
target_link_libraries(possion_image_editing ${OpenCV_LIBS})

