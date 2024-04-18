cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
./bin/non_local_means.exe