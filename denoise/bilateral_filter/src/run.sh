executor=./executor
rm -rf $executor
INCLUDE="-I /usr/local/include"
LIBRARY="-L /usr/local/lib"
LIBS="-lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread"
g++ -std=c++14 -Wall $INCLUDE $LIBRARY bilateral_filter.cpp $LIBS -o $executor
cd ..
./src/$executor
echo "Done !"

