
target("hello_world")
    set_kind("binary")
    add_files("$(projectdir)/src/run.cpp")
    set_toolset("cxx", "g++")
    set_toolset("ld", "g++")



target("possion_editing")
	set_kind("binary")
	-- 添加跟 possion_editing 有关的 cpp 源文件
	add_files("$(projectdir)/src/*possion_editing*.cpp")
	-- 设置编译期跟链接器
	set_toolset("cxx", "g++")
    set_toolset("ld", "g++")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 开启警告
    set_warnings("all")
    -- 开启优化
    set_optimize("faster")
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")
    -- 设置第三方库
    -- 	1. 添加 OpenCV
    local opencv_root = "D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/"
    add_includedirs(opencv_root .. "include")
    add_linkdirs(opencv_root .. "x64/mingw/bin")
    add_links("libopencv_core452", "libopencv_highgui452", "libopencv_imgproc452", "libopencv_imgcodecs452")
    -- 	2. 添加 Eigen3
	add_includedirs("D:/environments/C++/3rdparty/Eigen3/eigen-3.3.9/installed/include/eigen3")
    -- 设置目标 possion_editing 工作目录
    set_rundir("$(projectdir)/")
