// C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "gaussi_filter.h"



namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
		cv::Mat padded_image;
		cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
		return padded_image;
	}
}


cv::Mat gaussi_filter_channel(const cv::Mat& noise_channel, const int kernel_size, const double variance) {
    // 准备一些信息
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
    // 算出半径大小
    const int radius = (kernel_size - 1) >> 1;
    // 给边缘做填充  // 256 x (288 + 5 x 2) x 1
    const auto padded_channel = make_pad(noise_channel, radius, radius);
    // 先计算一个滤波模板, 一维的加速计算
    const int window_size = (radius << 1) + 1;
    std::vector<double> space_table(window_size * window_size, 0.0);
    std::vector<int> space_offset(window_size * window_size, 0);
    // 用指针访问, 内存释放交给 C++ 了
    double* const table_ptr = space_table.data();
    int* const offset_ptr = space_offset.data();
    // 相对位置的偏移
    int offset = 0;
    // e^{} 里面那个常数, 1. / (2 * δ * δ)
    const double variance_2 = -0.5 / (variance * variance);
    // 模板里的权重之和也是常数, 因为只跟核有关, 与图像无关
    double kernel_weight_sum = 0.0;
    // 计算 space_offset, 第 i 行, 然后加上第 j 个
    for(int i = -radius; i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            space_table[offset] = std::exp(variance_2 * (i * i + j * j));
            kernel_weight_sum += space_table[offset];
            space_offset[offset] = i * padded_channel.step + j;
            ++offset;
        }
    }
    // 在这里就先把权重归一化
    for(int i = 0;i < offset; ++i) space_table[i] /= kernel_weight_sum;
    // 准备一个矩阵, 储存去噪的结果
    auto denoised_channel = noise_channel.clone();
    // 开始去噪
    for(int i = 0;i < H; ++i) {
        // 这里的这个 radius 行的 step 是一定要算的, 从 i = 0 开始算起, 随便打个草稿即可
        const uchar* const row_ptr = padded_channel.data + (radius + i) * padded_channel.step + radius;
        uchar* const row_ptr_denoise = denoised_channel.data + i * denoised_channel.step;
        for(int j = 0;j < W; ++j) {
            double value_sum = 0.0;
            // 遍历这个窗口
            for(int k = 0;k < offset; ++k) {
                const int pixel = row_ptr[j + offset_ptr[k]];
                const double weight = table_ptr[k];
                value_sum += weight * pixel;
            }
            // 这里特别占时间 ?
            row_ptr_denoise[j] = cv::saturate_cast<uchar>(value_sum);
        }
    }
    return denoised_channel;
}


// 现在只是单通道的高斯去噪, 会不会更慢? 感觉速度上会更慢点
cv::Mat gaussi_filter(const cv::Mat& noise_image, const int kernel_size, const double variance) {
//    std::cout << "kernel  :  " << kernel_size << "\nvariance  :  " << variance << std::endl;
    // 或许直接用 .data 操作会快的多
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 用一个 vector 接收每个通道的去噪结果
    std::vector<cv::Mat> denoised_channels;
    for(const auto& channel : noise_channels)
        denoised_channels.emplace_back(gaussi_filter_channel(channel, kernel_size, variance));
    // 把去噪结果合并起来
    cv::Mat denoised_image;
    cv::merge(denoised_channels, denoised_image);
    return denoised_image;
}

