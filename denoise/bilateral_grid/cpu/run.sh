# executor=./bin/executor
# rm -rf ./bin/*
# rm -rf ./build/*
# mkdir -p bin
# mkdir -p build
# mkdir -p include
# mkdir -p src
# mkdir -p images
# cd build
# cmake ..
# make
# cd ..
# $executor
# echo "Done !"

cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
./bin/fast_bilateral.exe
