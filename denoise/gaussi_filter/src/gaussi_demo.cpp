//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "gaussi_filter.h"
#include "faster_gaussi_filter.h"

namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
        // 864 = 3 * 288
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat cv_concat(const cv::Mat& lhs, const cv::Mat& rhs) {
        cv::Mat result;
        cv::hconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        return result;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::hconcat(images, result);
        return result;
    }
}


void test_bgr_image(const cv::Mat& noise_image) {
    // 展示图像信息
    cv_info(noise_image);
    // 准备高斯滤波的参数, 滤波核大小与方差
    const int kernel_size = 21;
    const double variance = 5;
    // self
    cv::Mat gaussi_result, faster_gaussi_result, opencv_gaussi_result;
    run([&noise_image, &gaussi_result, kernel_size, variance]{
        gaussi_result = gaussi_filter(noise_image, kernel_size, variance);
    }, "第一次写  :  ");
    run([&noise_image, &faster_gaussi_result, kernel_size, variance]{
        faster_gaussi_result = faster_gaussi_filter(noise_image, kernel_size, variance);
    }, "改进之后  :  ");
    // OpenCV
    run([&noise_image, &opencv_gaussi_result, kernel_size, variance]{
        GaussianBlur(noise_image, opencv_gaussi_result, cv::Size(kernel_size, kernel_size), variance, variance);;
    }, "OpenCV   :  ");
    std::cout << "PSNR  " << cv::PSNR(gaussi_result, faster_gaussi_result) << std::endl;
    // 并排展示去噪结果
    cv::imwrite("../images/output/woman_1.png", faster_gaussi_result);
    cv_show(cv_concat(std::vector<cv::Mat>({noise_image, gaussi_result, faster_gaussi_result, opencv_gaussi_result})));
}



void test_gray_image(const cv::Mat& noise_image) {
    cv::Mat gray_noise_image;
    cv::cvtColor(noise_image, gray_noise_image, cv::COLOR_BGR2GRAY);
    cv::Mat gaussi_result, faster_gaussi_result, faster_2_gaussi_result, opencv_gaussi_result;
    run([&](){
        gaussi_result = gaussi_filter_channel(gray_noise_image, 11, 3);
    }, "第一次写     ");
    run([&]() {
        faster_gaussi_result = faster_gaussi_filter_channel(gray_noise_image, 11, 3, 3);
    }, "改进后的     ");
    run([&]() {
        faster_2_gaussi_result = faster_2_gaussi_filter_channel(gray_noise_image, 11, 3, 3);
    }, "边缘改进     ");
    run([&](){
        cv::GaussianBlur(gray_noise_image, opencv_gaussi_result, cv::Size(11, 11), 3, 3);
    }, "OpenCV 自带  ");
    // 这个 PSNR 不可能无穷大, 因为在四个边角, 两个方向的高斯滤波是没有计算那一块的
    std::cout << "PSNR  " << cv::PSNR(gaussi_result, faster_gaussi_result) << " ===>  ";
    std::cout << cv::PSNR(gaussi_result, faster_2_gaussi_result) << std::endl;
    cv_show(cv_concat(std::vector<cv::Mat>({gray_noise_image, gaussi_result, faster_gaussi_result, faster_2_gaussi_result, opencv_gaussi_result})));
}

int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // 根据图片路径读取图像
    const char* noise_path = "../images/input/woman_1.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return 0;
    }
    // noise_image = cv_resize(noise_image, 384, 384);
     test_bgr_image(noise_image);
//    test_gray_image(noise_image);
    return 0;
}
