cd build
cmake .. -G "MinGW Makefiles"
mingw32-make -j4
cd ..
./bin/anisotropic_diffusion_filter.exe