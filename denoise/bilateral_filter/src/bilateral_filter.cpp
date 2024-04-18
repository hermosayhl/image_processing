// C++
#include <assert.h>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <functional>
#include <algorithm>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



namespace {
	void show(const cv::Mat& image, const std::string& title) {
		cv::imshow(title, image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	std::string string_replace(const std::string& origin_path, const char* src, const char* des) {
		std::string result(origin_path.c_str());
		const int pos = result.find(src);	
		if(pos != -1)
			result.replace(pos, std::strlen(src), des);	
		return result;
	}

	// 计算运行时间
	void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    // 依据 typeid 输出变量数据类型
    template<typename T>
	void print_datatype(T var_) {
		std::cout << typeid(var_).name() << std::endl;
	}

	// 做填充
	cv::Mat make_pad(const cv::Mat& source, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(source, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}
}


// 顺便写一下高斯滤波
// 双边滤波
cv::Mat gaussi_filter(
		const cv::Mat& noise_image, const int window_size, const double space_variance) {
	// 检查合理性
	const int H = noise_image.rows;
	const int W = noise_image.cols;
	const int C = noise_image.channels();
	const int pad_size = (window_size - 1) >> 1;
	// std::cout << H << ", " << W << ", " << C << ", " << pad_size << "\n";
	// 对边缘填充
	const auto padded_image = make_pad(noise_image, pad_size, pad_size);
	// std::cout << "step  " << padded_image.step << std::endl;

	// 结果图像
	auto result = noise_image.clone();

	// 获取 distance filter 模板
	const int half_size = (window_size - 1) >> 1;
	const int space_size = window_size * window_size;
	std::vector<double> space_table(space_size);
	std::vector<int> space_offset(space_size);
	int maxk = 0;
	const int space_variance_2 = - 0.5 / (space_variance * space_variance);
	// (以最中心的点为参照点, 计算其他点的空间权重, 以及在图像中的像素偏移值)
	for(int i = -half_size;i <= half_size; ++i) {
		for(int j = -half_size;j <= half_size; ++j) {
			space_table[maxk] = std::exp(space_variance_2 * (i * i + j * j));
			space_offset[maxk] = i * padded_image.step + j * C;
			++maxk;
		}
	}

	// 只是把 vector 换成指针的形式, 就这样 ? iterater 真的慢
	double* space_ptr = &space_table[0];
	int *offset_ptr = &space_offset[0];

	// 遍历图像中每一个点, 然后滤波
		for(int i = 0;i < H; ++i) {
		// 从有效的滤波中心开始算, pad_size 行空的, 加上现在是有效图像第 i 行的数据, 当前第 i 行处在 pad_size 位置
		const uchar* row_ptr = padded_image.data + (i + pad_size) * padded_image.step + pad_size * C;
		uchar* result_row_ptr = result.data + i * result.step;
		for(int j = 0;j < W * 3; j += 3) {
			double sum_b = 0, sum_g = 0, sum_r = 0;
			double norm_b = 0, norm_g = 0, norm_r = 0;
			// 遍历窗口(滤波核) 主要计算量就在这里
			for(int k = 0;k < maxk; ++k) {
				const uchar* cur_ptr = row_ptr + j + offset_ptr[k];
				
				const int __b = cur_ptr[0];
				const double w_b = space_ptr[k];
				sum_b += __b * w_b;
				norm_b += w_b;

				const int __g = cur_ptr[1];
				const double w_g = space_ptr[k];
				sum_g += __g * w_g;
				norm_g += w_g;
				
				const int __r = cur_ptr[2];
				const double w_r = space_ptr[k];
				sum_r += __r * w_r;
				norm_r += w_r;
			}
			// 更新到结果图像上
			result_row_ptr[j] = (uchar)cvRound(sum_b / norm_b);
			result_row_ptr[j + 1] = (uchar)cvRound(sum_g / norm_g);
			result_row_ptr[j + 2] = (uchar)cvRound(sum_r / norm_r);
		}
	}

	return result;
}




uchar cv_round(const double data) {
    int temp = int(data);
    if(temp < 0) temp = 0;
    else if(temp > 255) temp = 255;
    return (uchar)temp;
}



// 双边滤波
cv::Mat bilateral_filter(
		const cv::Mat& noise_image, const int window_size, 
		const double value_variance, const double space_variance) {
	// 检查合理性
	const int H = noise_image.rows;
	const int W = noise_image.cols;
	const int C = noise_image.channels();
	const int pad_size = (window_size - 1) >> 1;
	// std::cout << H << ", " << W << ", " << C << ", " << pad_size << "\n";
	// 对边缘填充
	const auto padded_image = make_pad(noise_image, pad_size, pad_size);
	// std::cout << "step  " << padded_image.step << std::endl;

	// 结果图像
	auto result = noise_image.clone();

	// 获取 range filter 模板
	const int value_range = C * 256;
	std::vector<double> value_table(value_range);
	const double value_variance_2 = 1.0 / (value_variance * value_variance);
	for(int i = 0;i < value_range; ++i)
		value_table[i] = std::exp(- 0.5 * value_variance_2 * i * i);

	// 获取 distance filter 模板
	const int half_size = (window_size - 1) >> 1;
	const int space_size = window_size * window_size;
	std::vector<double> space_table(space_size);
	std::vector<int> space_offset(space_size);
	int maxk = 0;
	const double space_variance_2 = - 0.5 / (space_variance * space_variance);
	// (以最中心的点为参照点, 计算其他点的空间权重, 以及在图像中的像素偏移值)
	for(int i = -half_size;i <= half_size; ++i) {
		for(int j = -half_size;j <= half_size; ++j) {
			space_table[maxk] = std::exp(space_variance_2 * (i * i + j * j));
			space_offset[maxk] = i * padded_image.step + j * C;
			++maxk;
		}
	}


	// 只是把 vector 换成指针的形式, 就这样 ? iterater 真的慢
	double* value_ptr = &value_table[0];
	double* space_ptr = &space_table[0];
	int *offset_ptr = &space_offset[0];

	// 遍历图像中每一个点, 然后滤波
	for(int i = 0;i < H; ++i) {
		// 从有效的滤波中心开始算, pad_size 行空的, 加上现在是有效图像第 i 行的数据, 当前第 i 行处在 pad_size 位置
		const uchar* row_ptr = padded_image.data + (i + pad_size) * padded_image.step + pad_size * C;
		uchar* result_row_ptr = result.data + i * result.step;
		for(int j = 0;j < W; ++j) {
			double sum_b = 0, sum_g = 0, sum_r = 0;
			double norm_b = 0, norm_g = 0, norm_r = 0;
			// 中心像素的下标是 j
			const int J = j * 3;
			const int b = row_ptr[J];
			const int g = row_ptr[J + 1];
			const int r = row_ptr[J + 2];
			// 遍历窗口(滤波核) 主要计算量就在这里
			for(int k = 0;k < maxk; ++k) {
				const uchar* cur_ptr = row_ptr + J + offset_ptr[k];
				
				const int __b = cur_ptr[0];
				const double w_b = space_ptr[k] * value_ptr[std::abs(b - __b)];
				sum_b += __b * w_b;
				norm_b += w_b;

				const int __g = cur_ptr[1];
				const double w_g = space_ptr[k] * value_ptr[std::abs(g - __g)];
				sum_g += __g * w_g;
				norm_g += w_g;
				
				const int __r = cur_ptr[2];
				const double w_r = space_ptr[k] * value_ptr[std::abs(r - __r)];
				sum_r += __r * w_r;
				norm_r += w_r;
			}
			// 更新到结果图像上
			result_row_ptr[J] = cv::saturate_cast<uchar>(sum_b / norm_b);
			result_row_ptr[J + 1] = cv::saturate_cast<uchar>(sum_g / norm_g);
			result_row_ptr[J + 2] = cv::saturate_cast<uchar>(sum_r / norm_r);
		}
	}
	return result;
}






int main() {
	// 读取原始图像
	const std::string image_path("./images/woman_2.png");
	auto noise_image = cv::imread(image_path);
	// Resize 
	// cv::resize(noise_image, noise_image, {224, 224});

	// 保存结果
	cv::Mat opencv_result, bilateral_result, gaussi_result;

	// opencv 的双边滤波
	run([&](){
		cv::bilateralFilter(noise_image, opencv_result, 23, 10, 30);
	}, "opencv  :  ");


	run([&](){
		gaussi_result = gaussi_filter(noise_image, 23, 30);
	}, "gaussi  :  ");

	// 准备双边滤波的参数
	const int window_size = 23;
	const double space_variance = 30;
	const double value_variance = 10;
	// 双边滤波
	run([&](){
		bilateral_result = bilateral_filter(noise_image, window_size, value_variance, space_variance);
	}, "self  :  ");

	// 后续处理
	const std::string save_path = string_replace(image_path, ".png", "_bilateral_filter.png");
	// show(bilateral_result, save_path);
	cv::imwrite(save_path, bilateral_result);
	return 0;
}
