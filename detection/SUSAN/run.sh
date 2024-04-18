cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
./bin/susan_detector.exe