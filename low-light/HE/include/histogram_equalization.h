#ifndef HISTOGRAM_EQUALIZATION
#define HISTOGRAM_EQUALIZATION

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



cv::Mat plot_histogram(const cv::Mat& source);

cv::Mat histogram_equalization(const cv::Mat& source);


#endif
