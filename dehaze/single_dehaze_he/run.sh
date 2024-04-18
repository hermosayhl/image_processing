cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
ls
./bin/dark_channel_dehaze.exe