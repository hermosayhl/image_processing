chcp 65001
set exe_file=run.exe
set INCLUDE=D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/include
set LIBRAIY=D:/environments/C++/OpenCV/opencv-4.5.2/build_no_qt/install/x64/mingw/bin
set DLL=-llibopencv_highgui452 -llibopencv_core452 -llibopencv_imgcodecs452  -llibopencv_imgproc452 -llibopencv_dnn452
set ARGS=-std=c++14 -lpthread -O1
del %exe_file%
g++ %ARGS%  -I%INCLUDE% -I../include/ -L %LIBRAIY% gaussi_demo.cpp faster_gaussi_filter.cpp gaussi_filter.cpp  %DLL%  -o %exe_file%
%exe_file%
pause