// C++
#include <cmath>
#include <vector>
#include <iostream>
// self
#include "canny.h"
#include "faster_gaussi_filter.h"


namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }
    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
    template<typename T>
    cv::Mat double2uchar(const std::vector<T>& double_image, const int H, const int W) {
        cv::Mat uchar_image(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) uchar_image.data[i] = std::abs(double_image[i]);
        return uchar_image;
    }
}


std::pair< std::vector<double>, std::vector<float> > sobel_compute(const cv::Mat& source) {
    // 获取图像信息
    const int H = source.rows;
    const int W = source.cols;
    const int length = H * W;
    // 首先需要 padding
    const auto padded_image = make_pad(source, 1, 1);
    const int W2 = padded_image.cols;
    // 存放 x, y 方向的梯度
    std::vector<int> x_gradients(length, 0), y_gradients(length, 0);
    // 开始卷积
    for(int i = 0; i < H; ++i) {
        const uchar* const row_ptr = padded_image.data + (1 + i) * W2 + 1;
        int* const x_ptr = x_gradients.data() + i * W;
        int* const y_ptr = y_gradients.data() + i * W;
        for(int j = 0;j < W; ++j) {
            // x 方向
            x_ptr[j] = 2 * (row_ptr[j + 1] - row_ptr[j - 1]) + row_ptr[j + 1 - W2] + row_ptr[j + 1 + W2] - row_ptr[j - 1 - W2] - row_ptr[j - 1 + W2];
            // y 方向
            y_ptr[j] = 2 * (row_ptr[j - W2] - row_ptr[j + W2]) + row_ptr[j - W2 - 1] + row_ptr[j - W2 + 1] - row_ptr[j + W2 - 1] - row_ptr[j + W2 + 1];
        }
    }
    cv_show(double2uchar<int>(x_gradients, H, W));
    cv_show(double2uchar<int>(y_gradients, H, W));
    // 计算每个点的梯度大小
    std::vector<double> gradients(length, 0);
    for(int i = 0;i < length; ++i)
        gradients[i] = std::sqrt(x_gradients[i] * x_gradients[i] + y_gradients[i] * y_gradients[i]);
    cv_show(double2uchar<double>(gradients, H, W));
    // 计算每个点的梯度方向处在什么角度
    std::vector<float> directions(length, 0);
    for(int i = 0;i < length; ++i)
        directions[i] = y_gradients[i] * 1. / x_gradients[i];
    return std::make_pair(gradients, directions);
}


std::vector<double> non_maximization_suppress(const int H, const int W, const std::vector<double>& gradients, const std::vector<float>& directions) {
    // 先获取长度
    const int length = H * W;
    // 准备一个结果
    std::vector<double> nms_result;
    nms_result.resize(length);
    std::copy(gradients.begin(), gradients.end(), nms_result.begin());
    // 开始非极大化抑制
    for(int i = 1; i < H - 1; ++i) {
        const double* const row_ptr = gradients.data() + i * W;
        const float* const direct_ptr = directions.data() + i * W;
        double* const res_ptr = nms_result.data() + i * W;
        for(int j = 1; j < W - 1; ++j) {
            // 开始检查方向
            double lhs, rhs;
            const float ratio = direct_ptr[j];
            // 靠近 x 轴, 1 - 3 象限
            if(0 <= ratio and ratio < 1) {
                lhs = ratio * row_ptr[j - 1 + W] + (1 - ratio) * row_ptr[j - 1];
                rhs = ratio * row_ptr[j + 1 - W] + (1 - ratio) * row_ptr[j + 1];
            }
            // 靠近 y 轴, 1 - 3 象限
            else if(ratio >= 1) {
                const float ratio_inv = 1. / ratio;
                lhs = ratio_inv * row_ptr[j - 1 + W] * (1 - ratio_inv) * row_ptr[j + W];
                rhs = ratio_inv * row_ptr[j + 1 - W] * (1 - ratio_inv) * row_ptr[j - W];
            }
            // 靠近 x 轴, 2 - 4 象限
            else if(ratio > -1 and ratio < 0) {
                lhs = -ratio * row_ptr[j - 1 - W] + (1 + ratio) * row_ptr[j - 1];
                rhs = -ratio * row_ptr[j + 1 + W] + (1 + ratio) * row_ptr[j + 1];
            }
            // 靠近 y 轴, 2 - 4 象限
            else if(ratio <= -1) {
                const float ratio_inv = 1. / ratio;
                rhs = -ratio_inv * row_ptr[j - 1 - W] + (1 + ratio_inv) * row_ptr[j - W];
                lhs = -ratio_inv * row_ptr[j + 1 + W] + (1 + ratio_inv) * row_ptr[j + W];
            }
            // 判断是否是这个梯度方向上的局部极值
            if(row_ptr[j] < lhs or row_ptr[j] < rhs) {
                res_ptr[j] = 0;
            }
        }
    }
    cv_show(double2uchar<double>(nms_result, H, W));
    return nms_result;
}




cv::Mat double_threshold_filter(const int H, const int W, const std::vector<double>& nms_result, const double low_threshold, const double high_threshold) {
    // 准备一个结果
    cv::Mat refined = cv::Mat::zeros(H, W, CV_8UC1);
    // 准备过滤
    const int length = H * W;
    // 区分弱边缘和强边缘
    for(int i = 1; i < H - 1; ++i) {
        const double* const nms_ptr = nms_result.data() + i * W;
        uchar* const res_ptr = refined.data + i * W;
        for(int j = 1;j < W - 1; ++j) {
            if(nms_ptr[j] > high_threshold) res_ptr[j] = 255;
            else if(nms_ptr[j] > low_threshold) res_ptr[j] = 128;
        }
    }
    cv_show(refined);
    for(int i = 1; i < H - 1; ++i) {
        uchar* const res_ptr = refined.data + i * W;
        for(int j = 1;j < W - 1; ++j) {
            if(res_ptr[j] == 128) {
                if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                   res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                   res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                    res_ptr[j] = 255;
            }
        }
    }
    for(int i = 1; i < H - 1; ++i) {
        uchar* const res_ptr = refined.data + i * W;
        for(int j = W - 2;j > 0; --j) {
            if(res_ptr[j] == 128) {
                if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                   res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                   res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                    res_ptr[j] = 255;
            }
        }
    }
    for(int i = H - 2; i >= 1; --i) {
        uchar* const res_ptr = refined.data + i * W;
        for(int j = 1;j < W - 1; ++j) {
            if(res_ptr[j] == 128) {
                if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                   res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                   res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                    res_ptr[j] = 255;
            }
        }
    }
    for(int i = H - 2; i >= 1; --i) {
        uchar* const res_ptr = refined.data + i * W;
        for(int j = W - 2;j >= 1; --j) {
            if(res_ptr[j] == 128) {
                if(res_ptr[j - 1] == 255 or res_ptr[j + 1] == 255 or
                   res_ptr[j - 1 - W] == 255 or res_ptr[j - W] == 255 or res_ptr[j + 1 - W] == 255 or
                   res_ptr[j - 1 + W] == 255 or res_ptr[j + W] == 255 or res_ptr[j + 1 + W] == 255)
                    res_ptr[j] = 255;
                else res_ptr[j] = 0;
            }
        }
    }
    return refined;
}


cv::Mat canny(const cv::Mat& source, const double low_threshold, const double high_threshold) {
    // 首先获取图像信息
    const int C = source.channels();
    if(C != 1) {
        std::cout << "只支持单通道图像 !" << std::endl; return source;
    }
    // 首先高斯滤波
    const auto gaussi_result = faster_2_gaussi_filter_channel(source, 5, 1.2, 1.2);
    cv_show(cv_concat({source, gaussi_result}));
    // Sobel 算子计算每个点的梯度大小和方向
    const auto gradients_results = sobel_compute(source);
    // 非极大化抑制
    const auto nms_result = non_maximization_suppress(source.rows, source.cols, gradients_results.first, gradients_results.second);
    // 双阈值过滤, 连接边缘
    const auto refined = double_threshold_filter(source.rows, source.cols, nms_result, low_threshold, high_threshold);
    return refined;
}