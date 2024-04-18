chcp 65001
set exe_file=run.exe
set INCLUDE=F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/include
set LIBRAIY=F:/liuchang/environments/OpenCV/4.5.5/opencv-4.5.5/build/install/x64/mingw/bin
set DLL=-llibopencv_highgui455 -llibopencv_core455 -llibopencv_imgcodecs455  -llibopencv_imgproc455 -llibopencv_dnn455
set ARGS=-std=c++14 -lpthread -O1
del %exe_file%
g++ %ARGS%  -I%INCLUDE% -I../include/ -L %LIBRAIY% gaussi_demo.cpp faster_gaussi_filter.cpp gaussi_filter.cpp  %DLL%  -o %exe_file%
%exe_file%
pause