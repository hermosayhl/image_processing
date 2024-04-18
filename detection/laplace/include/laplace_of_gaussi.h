
#ifndef GUIDED_FILTER_LAPLACE_OF_GAUSSI_H
#define GUIDED_FILTER_LAPLACE_OF_GAUSSI_H


// C++
#include <list>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


cv::Mat laplace_extract_edges(const cv::Mat&);


cv::Mat laplace_of_gaussi_edge_detection(const cv::Mat& source, const int radius=2, const double sigma=0.7, const double threshold=100);


using keypoints_type = std::list<std::pair<double, double> >;

std::pair< keypoints_type, keypoints_type > difference_of_gaussi_keypoints_detection(
        const cv::Mat& source,
        const int radius=2,
        const std::vector< double > sigma_list={0.3, 0.4, 0.5, 0.6},
        const double threshold=3);

// 分别记录  响应值, 对应的尺度 or sigma, 两个坐标
using blob_type = std::vector< std::tuple<double, double, int, int> >;

blob_type laplace_of_gaussi_blob_detection(
        const cv::Mat& source,
        const double init_sigma=2,
        const double k=1.414,
        const int scales=9,
        const double threshold=0.3,
        const int num_blobs=120);



void laplace_of_gaussi_blob_detection_single_scale(
        const cv::Mat& source,
        const double init_sigma=2,
        const double k=1.414,
        const int scales=9,
        const double threshold=0.3,
        const int num_blobs=120);
#endif //GUIDED_FILTER_LAPLACE_OF_GAUSSI_H

