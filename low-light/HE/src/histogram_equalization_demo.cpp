//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "histogram_equalization.h"

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

	cv::Mat cv_concat(const cv::Mat& lhs, const cv::Mat& rhs, const int dim=0) {
        cv::Mat result;
        if(dim == 0) cv::hconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        else cv::vconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        return result;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::hconcat(images, result);
        return result;
    }
}


void denoise_gray_demo_1() {
    // ---------- 【1】根据图片路径读取图像
    const char* noise_path = "./images/input/over_exposure.jpg";
    auto low_contrast_image = cv::imread(noise_path);
    if(low_contrast_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // ---------- 【2】灰度图的直方图均衡化
    low_contrast_image = cv_resize(low_contrast_image, 341, 512);
    cv::cvtColor(low_contrast_image, low_contrast_image, cv::COLOR_BGR2GRAY);
    const auto corrected_image = histogram_equalization(low_contrast_image);
    // ---------- 【2】画直方图展示等
    auto comparison_results = cv_concat(low_contrast_image, corrected_image);
    const auto hist_results = cv_concat(plot_histogram(low_contrast_image), plot_histogram(corrected_image));
    comparison_results = cv_concat(comparison_results, hist_results, 1);
    cv_show(comparison_results);
    // ---------- 【3】保存结果
    const std::string save_path("./images/output/comparison_overexposed_people2.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}



void denoise_gray_demo_2() {
    // ---------- 【1】根据图片路径读取图像
    const char* noise_path = "./images/input/people.png";
    auto low_contrast_image = cv::imread(noise_path);
    if(low_contrast_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // ---------- 【2】灰度图的直方图均衡化
    const auto corrected_image = histogram_equalization(low_contrast_image);
    // ---------- 【2】画直方图展示等
    auto comparison_results = cv_concat(low_contrast_image, corrected_image);
    const auto hist_results = cv_concat(plot_histogram(low_contrast_image), plot_histogram(corrected_image));
    comparison_results = cv_concat(comparison_results, hist_results, 1);
    cv_show(comparison_results);
    // ---------- 【3】保存结果
    const std::string save_path("./images/output/comparison_underexposed_people1.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}



void denoise_gray_demo_3() {
    // ---------- 【1】根据图片路径读取图像
    const char* noise_path = "./images/input/building.png";
    auto low_contrast_image = cv::imread(noise_path);
    if(low_contrast_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // ---------- 【2】灰度图的直方图均衡化
    const auto corrected_image = histogram_equalization(low_contrast_image);
    // ---------- 【2】画直方图展示等
    auto comparison_results = cv_concat(low_contrast_image, corrected_image);
    const auto hist_results = cv_concat(plot_histogram(low_contrast_image), plot_histogram(corrected_image));
    comparison_results = cv_concat(comparison_results, hist_results, 1);
    cv_show(comparison_results);
    // ---------- 【3】保存结果
    const std::string save_path("./images/output/comparison_underexposed_building.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // 灰度图如何根据引导滤波去噪
    denoise_gray_demo_1();
    denoise_gray_demo_2();
    denoise_gray_demo_3();
    return 0;
}
