cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
ls
./bin/bayers_matting.exe