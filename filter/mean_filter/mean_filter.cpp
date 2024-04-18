// C & C++
#include <list>
#include <iostream>
#include <random>
#include <assert.h>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>


void run(const std::function<void()>& work=[]{}, const std::string message="") {
    auto start = std::chrono::high_resolution_clock::now();
    work();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);
    std::cout << message << " " << duration.count() * 1.0 / 1e3 << " us" <<  std::endl;
}


template<typename T>
void display(const T* const matrix_ptr, const int height, const int width) {
	for (int i = 0; i < height; ++i) {
		const T* const row_ptr = matrix_ptr + i * width;
		for (int j = 0; j < width; ++j) 
			std::cout << row_ptr[j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


template<typename T>
bool verify_is_same(const T* const lhs, const T* const rhs, const int length) {
	for (int i = 0; i < length; ++i) 
		if (std::abs(lhs[i] - rhs[i]) > 1e-5)
			return false;
	return true;
}




template<typename T>
void make_padding(T* const des_ptr, const T* const src_ptr, const int H, const int W, const int radius) {
	// 获取目标图像一行的像素个数， 一列的像素个数
	const int W2 = W + 2 * radius;
	const int H2 = H + 2 * radius;
	// 拷贝一行的代价
	const int bytes_in_line = sizeof(T) * W;
	// 首先拷贝中间的内容
	for (int i = 0; i < H; ++i)
		std::memcpy(des_ptr + (radius + i) * W2 + radius, src_ptr + i * W, bytes_in_line);
	// 拷贝上面 padding 的内容, 镜像
	for (int i = 0; i < radius; ++i) 
		std::memcpy(des_ptr + (radius - 1 - i) * W2 + radius, src_ptr + i * W, bytes_in_line);
	// 拷贝下面 padding 的内容, 镜像
	for (int i = 0; i < radius; ++i) 
		std::memcpy(des_ptr + (H + radius + i) * W2 + radius, src_ptr + (H - 1 - i) * W, bytes_in_line);
	// 拷贝左边 padding 的内容
	for (int i = 0; i < H2; ++i) {
		int start = i * W2 + radius + radius - 1;
		for (int j = 0; j < radius; ++j) des_ptr[i * W2 + j] = des_ptr[start - j];
		start = i * W2 + W + radius;
		for (int j = 0; j < radius; ++j) des_ptr[start + j] = des_ptr[start - 1 - j];
	}
}


template<typename T>
inline T square(const T x) {
	return x * x;
}




template<typename T, typename S>
void plain_mean_filtering(S* const des_ptr, const T* const src_ptr, const int H, const int W, const int radius) {
	// 计算 padding 之后的图像尺寸
	const int W2 = W + 2 * radius;

	// 先求出求解每个点需要访问的邻域偏移量
	int neighbor_count = -1;
	std::vector<int> offset(square(2 * radius + 1));
	for (int i = -radius; i <= radius; ++i) {
		for (int j = -radius; j <= radius; ++j) {
			if (i == 0 and j == 0)
				continue;
			offset[++neighbor_count] = i * W2 + j;
		}
	}

	// 一共需要求解 H * W 个点
	for (int i = 0; i < H; ++i) {
		// 获取第 i 行数据的起始地址
		const T* const row_ptr = src_ptr + (radius + i) * W2 + radius;
		for (int j = 0; j < W; ++j) {
			// 累加当前点
			T sum_value = row_ptr[j];
			// 累加邻域
			for (int k = 0; k < neighbor_count; ++k)
				sum_value += row_ptr[j + offset[k]]; // (这里有溢出风险)
			// 求均值, 写到结果中
			des_ptr[i * W + j] = sum_value / static_cast<S>((neighbor_count + 1));
		}
	}
}



int main() {

	std::vector<int> image({
		25, 43, 79, 24,  0, 48, 99, 76, 20,
        75,  7, 41, 31, 57, 41, 86, 24, 91,
        28, 43,  6, 20, 94,  0, 42, 19, 94,
        32, 13, 99, 59, 10, 64, 46, 66, 38,
        92,  8, 90, 88, 82, 69,  0, 27, 70
	});

	const int height = 5;
	const int width  = 9;
	assert(height * width == static_cast<int>(image.size()));


	// 决定滤波半径
	constexpr int radius = 2;

	// 先写一个 padding 函数
	std::vector<int> padded_image((height + 2 * radius) * (width + 2 * radius));
	make_padding(padded_image.data(), image.data(), height, width, radius);
	display(padded_image.data(), height + 2 * radius, width + 2 * radius);

	// 暴力均值滤波
	{
		std::vector<int> result(height * width);
		plain_mean_filtering<int>(result.data(), padded_image.data(), height, width, radius);
		display(result.data(), height, width);
	}


	

	return 0;
}


