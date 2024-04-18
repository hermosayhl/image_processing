// C++
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
// self
#include "laplace_of_gaussi.h"

namespace {
    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    cv::Mat double2uchar(const std::vector<double>& double_image, const int H, const int W) {
        cv::Mat origin = cv::Mat::zeros(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i) {
            origin.data[i] = cv::saturate_cast<uchar>(30 * std::abs(double_image[i]));
        }
        return origin;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

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


    std::vector<double> laplace_of_gaussi(const cv::Mat& source, const int radius, const double sigma, const bool norm=false) {
        // padding 处理边缘
        const auto padded_image = make_pad(source, radius, radius);
        const int W2 = padded_image.cols;
        // 准备一个 LOG 模板
        const int window_len = (radius << 1) + 1;
        const int window_size = window_len * window_len;
        const double sigma_2 = sigma * sigma;
        const double sigma_6 = norm ? sigma_2 * sigma_2 : sigma_2 * sigma_2 * sigma_2;
        double LOG[window_size];
        int LOG_offset[window_size];
        int offset = 0;
        for(int i = -radius; i <= radius; ++i) {
            for(int j = -radius; j <= radius; ++j) {
                const double distance = i * i + j * j;
                LOG[offset] = (distance - 2 * sigma_2) / sigma_6 * std::exp(-distance / (2 * sigma_2));
                LOG_offset[offset] = i * W2 + j;
                ++offset;
            }
        }
        // 平坦区域, LOG 响应为 0
        double sum_value = 0.0;
        for(int i = 0;i < offset; ++i) sum_value += LOG[i];
        for(int i = 0;i < offset; ++i) LOG[i] -= sum_value / offset;
        if(window_len < 10) {
            for(int i = 0;i < offset; ++i) {
                std::cout << std::setprecision(3) << LOG[i] << " ";
                if((i + 1) % window_len == 0) std::cout << "\n";
            }
        }
        /*
         *
         *
         0.001 0.011  0.044  0.067  0.044 0.011 0.001
         0.011 0.100  0.272  0.326  0.272 0.100 0.011
         0.044 0.272  0.088 -0.629  0.088 0.272 0.044
         0.067 0.326 -0.629 -2.460 -0.629 0.326 0.067
         0.043 0.272  0.088 -0.629  0.088 0.272 0.044
         0.011 0.100  0.272  0.326  0.272 0.100 0.011
         0.001 0.011  0.044  0.067  0.044 0.011 0.001
         */

        // 收集原始图像信息
        const int H = source.rows;
        const int W = source.cols;
        const int length = H * W;
        // LOG 模板扫过
        std::vector<double> LOG_result(length, 0);
        for(int i = 0;i < H; ++i) {
            const uchar* const row_ptr = padded_image.data + (radius + i) * W2 + radius;
            double* const res_ptr = LOG_result.data() + i * W;
            for(int j = 0;j < W; ++j) {
                // 开始卷积
                double conv_sum = 0;
                for(int k = 0;k < offset; ++k)
                    conv_sum += LOG[k] * row_ptr[j + LOG_offset[k]];
                res_ptr[j] = conv_sum;
            }
        }
        return LOG_result;
    }
}

blob_type laplace_of_gaussi_blob_detection(
        const cv::Mat& source,
        const double init_sigma,
        const double k,
        const int scales,
        const double threshold, const int num_blobs) {
    // 创建一个卷积核序列
    std::vector<std::vector<double> > LOG_octaves;
    std::vector<double> sigma_list;
    double cur_sigma = init_sigma;
    for(int i = 0;i < scales; ++i) {
        cur_sigma *= k;
        sigma_list.emplace_back(cur_sigma);
        // 确定卷积核大小
        const int radius = std::ceil(3 * cur_sigma);
        std::cout << "cur_sigma  " << cur_sigma << " - filt_size  " << (radius << 1) + 1 << std::endl;
        // 尺度规范化的 LOG
        LOG_octaves.emplace_back(laplace_of_gaussi(source, radius, cur_sigma, true));
    }
    const int H = source.rows;
    const int W = source.cols;
    blob_type result;
    // 遍历所有可以比较的尺度
    for(int level = 1; level < scales - 1; ++level) {
        for(int i = 1;i < H - 1; ++i) {
            double* const down = LOG_octaves[level - 1].data() + i * W;
            double* const mid = LOG_octaves[level].data() + i * W;
            double* const up = LOG_octaves[level + 1].data() + i * W;
            for(int j = 1;j < W - 1; ++j) {
                // 中间这个点的值, 和最近的 26 个点比较大小
                const auto center = mid[j];
                // 极大值
                if(center > mid[j - 1] and center > mid[j + 1] and
                   center > mid[j - 1 - W] and center > mid[j - W] and center > mid[j + 1 - W] and
                   center > mid[j - 1 + W] and center > mid[j + W] and center > mid[j + 1 + W] and

                   center > down[j - 1] and center > down[j] and center > down[j + 1] and
                   center > down[j - 1 - W] and center > down[j - W] and center > down[j + 1 - W] and
                   center > down[j - 1 + W] and center > down[j + W] and center > down[j + 1 + W] and

                   center > up[j - 1] and center > up[j] and center > up[j + 1] and
                   center > up[j - 1 - W] and center > up[j - W] and center > up[j + 1 - W] and
                   center > up[j - 1 + W] and center > up[j + W] and center > up[j + 1 + W]) {
                    // 记录这个点
                    result.emplace_back(std::make_tuple(std::abs(center), sigma_list[level], i, j));
                }
                // 极小值
                else if(center < mid[j - 1] and center < mid[j + 1] and
                   center < mid[j - 1 - W] and center < mid[j - W] and center < mid[j + 1 - W] and
                   center < mid[j - 1 + W] and center < mid[j + W] and center < mid[j + 1 + W] and

                   center < down[j - 1] and center < down[j] and center < down[j + 1] and
                   center < down[j - 1 - W] and center < down[j - W] and center < down[j + 1 - W] and
                   center < down[j - 1 + W] and center < down[j + W] and center < down[j + 1 + W] and

                   center < up[j - 1] and center < up[j] and center < up[j + 1] and
                   center < up[j - 1 - W] and center < up[j - W] and center < up[j + 1 - W] and
                   center < up[j - 1 + W] and center < up[j + W] and center < up[j + 1 + W]) {
                    result.emplace_back(std::make_tuple(std::abs(center), sigma_list[level], i, j));
                }
            }
        }
    }
    // 做排序
    std::sort(result.begin(), result.end());
    std::reverse(result.begin(), result.end());
    // 取响应值最大的若干个
    if(num_blobs < result.size()) result.erase(result.begin() + std::min(num_blobs, int(result.size())), result.end());
    // 做 2D 和 3D 的非极大值抑制
    // ...
    return result;
}




void laplace_of_gaussi_blob_detection_single_scale(
        const cv::Mat& source,
        const double init_sigma,
        const double k,
        const int scales,
        const double threshold,
        const int num_blobs) {

    // 转成灰度图
    cv::Mat origin_gray;
    cv::cvtColor(source, origin_gray, cv::COLOR_BGR2GRAY);

    const int H = origin_gray.rows;
    const int W = origin_gray.cols;
    const int length = H * W;



    double cur_sigma = init_sigma;
    for(int i = 0;i < scales; ++i) {
        auto display = source.clone();
        cur_sigma *= k;
        // 确定卷积核大小
        const int radius = std::ceil(3 * cur_sigma);
        std::cout << "cur_sigma  " << cur_sigma << " - filt_size  " << (radius << 1) + 1 << std::endl;
        // 尺度规范化的 LOG
        const auto this_scale_detection = laplace_of_gaussi(origin_gray, radius, cur_sigma, true);
        // 这个尺度下的结果
        std::vector< std::tuple<double, int, int> > temp;

        for(int i = 1;i < H - 1; ++i) {
            const double* const row_ptr = this_scale_detection.data() + i * W;
            for(int j = 1;j < W - 1; ++j) {
                // 判断这个点是不是周围点的极值
                const double center = row_ptr[j];
                if(center > row_ptr[j - 1] and center > row_ptr[j + 1] and
                   center > row_ptr[j - W - 1] and center > row_ptr[j - W] and center > row_ptr[j - W + 1] and
                   center > row_ptr[j + W - 1] and center > row_ptr[j + W] and center > row_ptr[j + W + 1])
                temp.emplace_back(std::make_tuple(std::abs(center), i, j));
            }
        }
        std::sort(temp.begin(), temp.end());
        std::reverse(temp.begin(), temp.end());
        for(int i = 0;i < 12; ++i) {
            cv::circle(display, cv::Point(std::get<2>(temp[i]), std::get<1>(temp[i])), int(cur_sigma * 1.414), CV_RGB(0, 255, 0), 1);
        }
        cv_show(display);
    }
}