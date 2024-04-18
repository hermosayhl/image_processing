#ifndef NON_LOCAL_MEANS_H
#define NON_LOCAL_MEANS_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



cv::Mat non_local_means(const cv::Mat& noise_image, const int search_radius=5, const int radius=2, const int sigma=1, const char* kernel_type="mean", const bool fast=false);

#endif //NON_LOCAL_MEANS_H
