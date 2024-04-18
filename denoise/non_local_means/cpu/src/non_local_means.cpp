// C++
#include <cmath>
#include <cstring>
#include <iostream>
// self
#include "non_local_means.h"


namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}




std::vector<double> get_kernel(const int window_size, const char* kernel_type) {
    // 权重模板是均值的话
    if(std::strcmp(kernel_type, "mean") == 0)
        return std::vector<double> (window_size, 1. / (window_size));
    // 高斯模板
    else if(std::strcmp(kernel_type, "gaussi") == 0) {
        std::vector<double> weight_kernel(window_size, 0);
        int offset = -1;
        double kernel_weight_sum = 0.0;
        const int radius = (int(std::sqrt(window_size)) - 1) >> 1;
        // 半径应该是 3 sigma 差不多了
        const double variance = int((2 * radius + 1) / 3);
        const double variance_2 = -0.5 / (variance * variance);
        for(int i = -radius; i <= radius; ++i)
            for(int j = -radius; j <= radius; ++j) {
                weight_kernel[++offset] = std::exp(variance_2 * (i * i + j * j));
                kernel_weight_sum += weight_kernel[offset];
            }
        for(int i = 0;i < window_size; ++i) weight_kernel[i] /= kernel_weight_sum;
        return weight_kernel;
    }
    // 没声明的话, 返回全 0 模板
    else return std::vector<double>(window_size, 0);
}


// 普通的
cv::Mat non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type) {
    // 先做一个计算领域相似性的权重模板, 先来最简单的均值模板
    const int window_len = (radius << 1) + 1;
    const int window_size = window_len * window_len;
    const auto weights_kernel = get_kernel(window_size, kernel_type);
    const double sigma_2_inv = 1. / (sigma * sigma);
    // 收集目标图像的信息
    cv::Mat denoised = noise_image.clone();
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    // 将图像 padding 一下
    const auto padded_image = make_pad(noise_image, radius, radius);
    const int H2 = padded_image.rows;
    const int W2 = padded_image.cols;
    const uchar* const noise_ptr = noise_image.data;
    const uchar* const padded_ptr = padded_image.data;
    // 现在开始滤波, 求目标图像中的每一点
    for(int i = 0;i < H; ++i) {
        const int up = std::max(radius, i - search_radius);
        const int down = std::min(H2, i + search_radius);
        for(int j = 0;j < W; ++j) {
            // 当前要去噪的点 (i, j), 以它为中心的区域的点, 我得收集起来
            uchar source[window_size];
            for(int t = 0;t < window_len; ++t)
                std::memcpy(source + t * window_len, padded_ptr + (i + t) * W2 + j, window_len * sizeof(uchar));
            // 累计值 和 权重总和
            double sum_value = 0;
            double weight_sum = 0;
            double weight_max = -1e3;
            // 每个点先确认它目前的搜索区域有多大, 为什么是 radius?
            const int left = std::max(radius, j - search_radius);
            const int right = std::min(W2, j + search_radius);
            // 在这个搜索区域搜索
            for(int x = up; x < down; ++x) {
                for(int y = left; y < right; ++y) {
                    // (i, j) 是相对于原图来说的位置, (x, y) 是相对于 padded 之后的图像来说的
                    // 如果碰到自己了, 不计算
                    if(x == i and y == j)
                        continue;
                    // 当前对比的区域是以 x, y 为中心, 半径为 radius 的区域
                    // 我得把这个区域的值都找出来, 收集起来
                    uchar target[window_size];
                    for(int t = 0;t < window_len; ++t)
                        std::memcpy(target + t * window_len, padded_ptr + (x - radius + t) * W2 + y - radius, window_len * sizeof(uchar));
                    // 然后计算两个区域的相似度
                    double distance = 0.0;
                    for(int k = 0;k < window_size; ++k) {
                        double res = static_cast<double>(target[k] - source[k]);
                        distance += weights_kernel[k] * (res * res);
                    }
                    const double cur_weight = std::exp(-distance * sigma_2_inv);
                    // 记录当前最大的权值
                    if(cur_weight > weight_max) weight_max = cur_weight;
                    // 累加值
                    sum_value += cur_weight * padded_image.at<uchar>(x, y);
                    weight_sum += cur_weight;
                }
            }
            // 搜索结束
            sum_value += weight_max * noise_image.at<uchar>(i, j);
            weight_sum += weight_max;
            denoised.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum_value / weight_sum);
        }
    }
    return denoised;
}







// 三通道一起考虑算 mse
cv::Mat non_local_means_color(const std::vector<cv::Mat>& noise_channels, const int search_radius=5, const int radius=2, const int sigma=1, const char* kernel_type="mean") {
    return cv::Mat();
}


// 搜索窗口大小 11x11, 邻域 5x5
cv::Mat non_local_means(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type, const bool fast, const bool multi_channel) {
    const int C = noise_image.channels();
    // 灰度图
    if(C == 1) {
        if(!fast) return non_local_means_gray(noise_image, search_radius, radius, sigma, kernel_type);
        else return fast_non_local_means_gray_2(noise_image, search_radius, radius, sigma);
    }
    // 彩色图的 non_local_means
    else if(C == 3) {
        std::vector<cv::Mat> bgr_channels;
        cv::split(noise_image, bgr_channels);
        if(!multi_channel) {
            std::vector<cv::Mat> denoised_channels;
            for(const auto & ch : bgr_channels)
                if(!fast)
                    denoised_channels.emplace_back(non_local_means_gray(ch, search_radius, radius, sigma, kernel_type));
                else
                    denoised_channels.emplace_back(fast_non_local_means_gray_2(ch, search_radius, radius, sigma));
            cv::Mat denoised;
            cv::merge(denoised_channels, denoised);
            return denoised;
        } else return non_local_means_color(bgr_channels, search_radius, radius, sigma, kernel_type);
    }
    return noise_image;
}
