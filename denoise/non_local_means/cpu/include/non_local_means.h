#ifndef NON_LOCAL_MEANS_H
#define NON_LOCAL_MEANS_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



cv::Mat non_local_means(
        const cv::Mat& noise_image,
        const int search_radius=5,
        const int radius=2,
        const int sigma=1,
        const char* kernel_type="mean",
        const bool fast=false,
        const bool multi_channel=false);

// 用 box_filter
cv::Mat fast_non_local_means_gray_1(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const bool use_fast_exp=false);
// 用 integral image
cv::Mat fast_non_local_means_gray_2(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const bool use_fast_exp=false);
// 试试那个 SSE 指令
cv::Mat fast_non_local_means_gray_3(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const bool use_fast_exp=false);
#endif //NON_LOCAL_MEANS_H
