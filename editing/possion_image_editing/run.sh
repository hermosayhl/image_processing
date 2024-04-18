cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
./bin/possion_image_editing.exe