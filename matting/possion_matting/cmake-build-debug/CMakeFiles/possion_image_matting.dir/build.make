# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\software\editor\CLion\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = D:\software\editor\CLion\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\work\crane\algorithm\image_processing\matting\possion_matting

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\work\crane\algorithm\image_processing\matting\possion_matting\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/possion_image_matting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/possion_image_matting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/possion_image_matting.dir/flags.make

CMakeFiles/possion_image_matting.dir/src/others.cpp.obj: CMakeFiles/possion_image_matting.dir/flags.make
CMakeFiles/possion_image_matting.dir/src/others.cpp.obj: CMakeFiles/possion_image_matting.dir/includes_CXX.rsp
CMakeFiles/possion_image_matting.dir/src/others.cpp.obj: ../src/others.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\work\crane\algorithm\image_processing\matting\possion_matting\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/possion_image_matting.dir/src/others.cpp.obj"
	ccache D:\environments\C++\MinGW_posix_sjlj\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\possion_image_matting.dir\src\others.cpp.obj -c D:\work\crane\algorithm\image_processing\matting\possion_matting\src\others.cpp

CMakeFiles/possion_image_matting.dir/src/others.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/possion_image_matting.dir/src/others.cpp.i"
	D:\environments\C++\MinGW_posix_sjlj\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\work\crane\algorithm\image_processing\matting\possion_matting\src\others.cpp > CMakeFiles\possion_image_matting.dir\src\others.cpp.i

CMakeFiles/possion_image_matting.dir/src/others.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/possion_image_matting.dir/src/others.cpp.s"
	D:\environments\C++\MinGW_posix_sjlj\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\work\crane\algorithm\image_processing\matting\possion_matting\src\others.cpp -o CMakeFiles\possion_image_matting.dir\src\others.cpp.s

# Object files for target possion_image_matting
possion_image_matting_OBJECTS = \
"CMakeFiles/possion_image_matting.dir/src/others.cpp.obj"

# External object files for target possion_image_matting
possion_image_matting_EXTERNAL_OBJECTS =

bin/possion_image_matting.exe: CMakeFiles/possion_image_matting.dir/src/others.cpp.obj
bin/possion_image_matting.exe: CMakeFiles/possion_image_matting.dir/build.make
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_gapi452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_stitching452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_alphamat452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_aruco452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_bgsegm452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_bioinspired452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_ccalib452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_dnn_objdetect452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_dnn_superres452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_dpm452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_face452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_fuzzy452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_hfs452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_img_hash452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_intensity_transform452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_line_descriptor452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_mcc452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_quality452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_rapid452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_reg452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_rgbd452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_saliency452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_stereo452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_structured_light452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_superres452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_surface_matching452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_tracking452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_videostab452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_xfeatures2d452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_xobjdetect452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_xphoto452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_shape452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_highgui452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_datasets452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_plot452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_text452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_ml452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_phase_unwrapping452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_optflow452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_ximgproc452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_video452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_dnn452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_videoio452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_imgcodecs452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_objdetect452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_calib3d452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_features2d452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_flann452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_photo452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_imgproc452.dll.a
bin/possion_image_matting.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/lib/libopencv_core452.dll.a
bin/possion_image_matting.exe: CMakeFiles/possion_image_matting.dir/linklibs.rsp
bin/possion_image_matting.exe: CMakeFiles/possion_image_matting.dir/objects1.rsp
bin/possion_image_matting.exe: CMakeFiles/possion_image_matting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\work\crane\algorithm\image_processing\matting\possion_matting\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin\possion_image_matting.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\possion_image_matting.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/possion_image_matting.dir/build: bin/possion_image_matting.exe

.PHONY : CMakeFiles/possion_image_matting.dir/build

CMakeFiles/possion_image_matting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\possion_image_matting.dir\cmake_clean.cmake
.PHONY : CMakeFiles/possion_image_matting.dir/clean

CMakeFiles/possion_image_matting.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\work\crane\algorithm\image_processing\matting\possion_matting D:\work\crane\algorithm\image_processing\matting\possion_matting D:\work\crane\algorithm\image_processing\matting\possion_matting\cmake-build-debug D:\work\crane\algorithm\image_processing\matting\possion_matting\cmake-build-debug D:\work\crane\algorithm\image_processing\matting\possion_matting\cmake-build-debug\CMakeFiles\possion_image_matting.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/possion_image_matting.dir/depend

