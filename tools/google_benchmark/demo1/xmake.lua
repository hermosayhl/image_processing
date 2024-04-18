set_policy("check.auto_ignore_flags", false)

add_syslinks("pthread")

target("googlebenckmark")
	set_kind("binary")
	-- 添加跟 googlebenckmark 有关的 cpp 源文件
	add_files("$(projectdir)/src/test.cpp")
	-- 设置编译期跟链接器
	set_toolset("cxx", "g++")
    set_toolset("ld", "g++")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 开启警告
    set_warnings("all")
    -- 开启优化
    -- set_optimize("faster")
    -- 添加 include 自己的代码
    -- 设置第三方库
    local benchmark_dir = "D:/work/crane/algorithm/image_processing/tools/google_benchmark/source/install/"
    add_includedirs(benchmark_dir .. "include")
    add_ldflags("-L/D:/work/crane/algorithm/image_processing/tools/google_benchmark/source/install/lib", "-llibbenchmark", "-llibbenchmark_main")
    -- 设置目标 googlebenckmark 工作目录
    set_rundir("$(projectdir)/")
