// C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "faster_gaussi_filter.h"


std::vector<double> get_filter_table(const int kernel_size, const double variance) {
    std::vector<double> space_table(kernel_size, 0.0);
    const double variance_2 = -0.5 / (variance * variance); // 常量
    const int radius = (kernel_size - 1) >> 1; // 半径
    space_table[radius] = 1; // 中心的值固定
    double weight_sum = space_table[radius]; // 归一化需要, 这里第一次给我漏掉了, 所以图片偏亮, 坑爹
    for(int i = 1;i <= radius; ++i) {
        space_table[radius - i] = space_table[radius + i] = std::exp(variance_2 * i * i);
        weight_sum += space_table[radius - i] * 2;
    }
    // 归一化
    for(int i = 0;i < kernel_size; ++i)
        space_table[i] /= weight_sum;
    return space_table;
}

// 两个分离的一维高斯卷积实现, 我这里
cv::Mat faster_gaussi_filter_channel(const cv::Mat& noise_channel, const int kernel_size, const double variance_x, const double variance_y) {
    // 获取图像边界信息
    const int H = noise_channel.rows;
    const int W = noise_channel.cols;
    // 首先求 x 方向和 y 方向的模板
    const auto x_space_table = get_filter_table(kernel_size, variance_x);
    const auto y_space_table = get_filter_table(kernel_size, variance_y);
    const double* const x_table_ptr = x_space_table.data();
    const double* const y_table_ptr = y_space_table.data();
    // 半径
    const int radius = (kernel_size - 1) >> 1;
    // 找一个 double 数组存储第一次滤波的结果, 一开始不要变成 uchar, 虽然差不多, 但我不喜欢
    std::vector<double> temp(H * W, 0.0);
    double* const temp_ptr = temp.data();
	// 先做 X 方向上的高斯滤波, 注意, 以 W 为边界
	const int x_step = noise_channel.step;
	for (int i = 0; i < H; ++i) {
	    const uchar* const row_ptr = noise_channel.data + i * x_step;
        double* const row_ptr_denose = temp_ptr + i * x_step;
		for (int j = 0; j < W; ++j) {
            double sum_value = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                if (j + k < 0 || j + k >= W)
                    continue;
                sum_value += x_table_ptr[k + radius] * row_ptr[j + k];
            }
            row_ptr_denose[j] = sum_value;
		}
	}
	// 准备好最终结果
	auto denoise_channel = noise_channel.clone();
    // 再做 Y 方向上的一维高斯滤波, 以 H 为边界
    const int y_step = denoise_channel.step;
	for (int i = 0; i < H; ++i) {
	    uchar* const row_ptr_denoise = denoise_channel.data + i * y_step;
		for (int j = 0; j < W; ++j) {
            double sum_value = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                if (i + k < 0 or i + k >= H) // 这种做法, 边缘会变暗 ? 因为少算了一部分, 权重少了, 所以边缘会有一圈
                    continue;
                sum_value += y_table_ptr[k + radius] * temp_ptr[(i + k) * y_step + j];
            }
            row_ptr_denoise[j] = cv::saturate_cast<uchar>(sum_value);
		}
	}
	return denoise_channel;
}


namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}
}

// 去除边缘效应, 加一个 padding
// 而且, 速度比之前要快, 因为没有 if else 判断
// 但是消耗更大点, 空间换时间
cv::Mat faster_2_gaussi_filter_channel(const cv::Mat& noise_channel, const int kernel_size, const double variance_x, const double variance_y) {
    // 获取图像边界信息
    const int H = noise_channel.rows;
    const int W = noise_channel.cols;
    const int C = noise_channel.channels();
    if(C not_eq 1) {
        std::cout << "该函数只接受单通道图像的高斯滤波!" << std::endl;
        return noise_channel;
    }
    if(kernel_size % 2 == 0 or kernel_size <= 0) {
        std::cout << "滤波核的大小非法, 应当为正奇数!" << std::endl;
        return noise_channel;
    }
    // 首先求 x 方向和 y 方向的模板
    const auto x_space_table = get_filter_table(kernel_size, variance_x);
    const auto y_space_table = get_filter_table(kernel_size, variance_y);
    const double* const x_table_ptr = x_space_table.data();
    const double* const y_table_ptr = y_space_table.data();
    // 半径
    const int radius = (kernel_size - 1) >> 1;
    // 对图像做 padding, 以空间换时间
    const auto padded_image = make_pad(noise_channel, radius, radius);
    // 找一个 double 数组存储第一次滤波的结果, 一开始不要变成 uchar, 虽然差不多, 但我不喜欢
    std::vector<double> temp((H + 2 * radius) * (W + 2 * radius), 0.0);
    double* const temp_ptr = temp.data();
	// 先做 X 方向上的高斯滤波, 注意, 以 W 为边界
	for (int i = 0; i < H; ++i) {
	    const uchar* const row_ptr = padded_image.data + (radius + i) * padded_image.step + radius;
        double* const row_ptr_denose = temp_ptr + (radius + i) * padded_image.step + radius;
		for (int j = 0; j < W; ++j) {
            double sum_value = 0.0;
            for (int k = -radius; k <= radius; ++k)
                sum_value += x_table_ptr[k + radius] * row_ptr[j + k];
            row_ptr_denose[j] = sum_value;
		}
	}
	// 准备好最终结果
	auto denoise_channel = noise_channel.clone();
	// 第一次 X 之后, 上下是空的, 没有参与计算, 应该把值填补回去, 填第一次 X 之后的结果
	// 填上面
	for(int i = 0;i < radius; ++i) {
	    double* const row_ptr = temp_ptr + radius + i * padded_image.step;
	    double* const row_ptr_2 = temp_ptr + radius + (2 * radius - i) * padded_image.step;
	    for(int j = 0;j < W; ++j) row_ptr[j] = row_ptr_2[j];
	}
	// 填下面
	const int H2 = H + 2 * radius; // 注意, 高度已经不是 H 了
	for(int i = 0;i < radius; ++i) {
	    double* const row_ptr = temp_ptr + radius + (H2 - 1 - i) * padded_image.step;
	    double* const row_ptr_2 = temp_ptr + radius + (H2 - 1 - 2 * radius + i) * padded_image.step;
	    for(int j = 0;j < W; ++j) row_ptr[j] = row_ptr_2[j];
	}
	// 其实角落里也需要填, 不然四个顶角会发暗
    // 再做 Y 方向上的一维高斯滤波, 以 H 为边界
	for (int i = 0; i < H; ++i) {
	    uchar* const row_ptr_denoise = denoise_channel.data + i * denoise_channel.step;
	    const int inter = (radius + i) * padded_image.step + radius;
		for (int j = 0; j < W; ++j) {
            double sum_value = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                sum_value += y_table_ptr[k + radius] * temp_ptr[inter + k * padded_image.step + j];
            }
            row_ptr_denoise[j] = cv::saturate_cast<uchar>(sum_value);
		}
	}
	return denoise_channel;
}

cv::Mat faster_gaussi_filter(const cv::Mat& noise_image, const int kernel_size, const double variance) {
//    std::cout << "kernel  :  " << kernel_size << "\nvariance  :  " << variance << std::endl;
    // 或许直接用 .data 操作会快的多
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 用一个 vector 接收每个通道的去噪结果
    std::vector<cv::Mat> denoised_channels;
    for(const auto& channel : noise_channels)
        denoised_channels.emplace_back(faster_2_gaussi_filter_channel(channel, kernel_size, variance, variance));
    // 把去噪结果合并起来
    cv::Mat denoised_image;
    cv::merge(denoised_channels, denoised_image);
    return denoised_image;
}


