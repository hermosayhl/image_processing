executor=./bin/non_local_means
rm -rf ./images/output/*
rm -rf ./bin/*
# rm -rf ./build/*
mkdir -p bin
mkdir -p build
mkdir -p include
mkdir -p src
mkdir -p images
cd build
cmake ..
make
cd ..
$executor
echo "Done !"

