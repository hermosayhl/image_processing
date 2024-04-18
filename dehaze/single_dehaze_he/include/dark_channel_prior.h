#ifndef GUIDED_FILTER_DARK_CHANNEL_PRIOR_H
#define GUIDED_FILTER_DARK_CHANNEL_PRIOR_H

// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



std::map<const std::string, cv::Mat> dark_channel_prior_dehaze(
        const cv::Mat& haze_image,
        // 局部窗口的直径 = 2 * radius + 1
        const int radius=3,
        // 求解全局大气光 A 时选取前多少的点(从暗通道中)
        const double top_percent=0.001,
        // 求解 J 时防止分母 t 太小
        const double t0=0.1,
        // 求解 t 时, 控制去雾程度, 0 完全不去雾, 1 不变
        const double omega=0.95,
        // 是否使用引导滤波修正
        const bool guided=false,
        // 是否每个通道都估计一个 t(wk)
        const bool multi_T=false,
        // 是否要返回一些中间的可视化结果
        const bool return_visuals=false);


cv::Mat get_dark_channel(const cv::Mat& I, const int radius, const bool accelerate=true);

#endif //GUIDED_FILTER_DARK_CHANNEL_PRIOR_H
