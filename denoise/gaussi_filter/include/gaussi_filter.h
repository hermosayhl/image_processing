#ifndef GAUSSI_FILTER_GAUSSI_FILTER_H
#define GAUSSI_FILTER_GAUSSI_FILTER_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// 对单通道的去噪
cv::Mat gaussi_filter_channel(const cv::Mat& noise_channel, const int kernel_size, const double variance);

// 彩色图像的高斯去噪, 会不会更慢? 感觉速度上会更慢点
cv::Mat gaussi_filter(const cv::Mat& noise_image, const int kernel_size, const double variance);


#endif //GAUSSI_FILTER_GAUSSI_FILTER_H
