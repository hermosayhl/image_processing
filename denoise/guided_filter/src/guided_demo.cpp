//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "guided_filter.h"

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


void denoise_gray_demo() {
    // 根据图片路径读取图像
    const char* noise_path = "./images/input/denoise/Kodak24/18.png";
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    cv::cvtColor(noise_image, noise_image, cv::COLOR_BGR2GRAY);
    auto& guided_image = noise_image;
    const auto guided_result = guided_filter_with_gray(noise_image, guided_image, 2, 2, 0.01);

    const auto comparison_resultss = cv_concat({noise_image, guided_result});
    cv_show(comparison_resultss);
    // 保存结果
    const std::string save_path("./results/comparison_gray.png");
    cv::imwrite(save_path, comparison_resultss, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_rgb_demo_1() {
    // 根据图片路径读取图像
    const char* noise_path = "./images/input/denoise/Kodak24/3.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // 自己对自己做引导滤波就是去噪, 那么此时, 将图像根据通道分离
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 用一个 vector 接收每个通道的去噪结果
    std::vector<cv::Mat> denoised_channels;
    for(const auto& channel : noise_channels)
        denoised_channels.emplace_back(guided_filter_with_gray(channel, channel, 2, 2, 0.01));
    // 把去噪结果合并起来
    cv::Mat denoised_image;
    cv::merge(denoised_channels, denoised_image);
    // 保存结果
    const auto comparison_resultss = cv_concat({noise_image, denoised_image});
    cv_show(comparison_resultss);
    const std::string save_path("./images/output/comparison_rgb.png");
    cv::imwrite(save_path, comparison_resultss, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_rgb_demo_2() {
    // 根据图片路径读取图像
    const char* noise_path = "./images/input/denoise/Kodak24/8.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // 自己对自己做引导滤波就是去噪, 那么此时, 将图像根据通道分离
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 存储不同 epsilon 的结果
    std::vector<cv::Mat> all_denoised_results({noise_image});
    std::vector<double> all_epsilons({0.0001, 0.001, 0.01, 0.1, 1});
    for(const auto epsilon : all_epsilons) {
        // 用一个 vector 接收每个通道的去噪结果
        std::vector<cv::Mat> denoised_channels;
        for(const auto& channel : noise_channels)
            denoised_channels.emplace_back(guided_filter_with_gray(channel, channel, 2, 2, epsilon));
        // 把去噪结果合并起来
        cv::Mat denoised_image;
        cv::merge(denoised_channels, denoised_image);
        all_denoised_results.emplace_back(denoised_image);
    }
    // 保存结果
    const auto comparison_resultss = cv_concat(all_denoised_results);
    cv_show(comparison_resultss);
    const std::string save_path("./images/output/comparison_rgb_epsilon.png");
    cv::imwrite(save_path, comparison_resultss, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_rgb_demo_3() {
    // 根据图片路径读取图像
    const char* noise_path = "./images/input/denoise/Kodak24/22.png";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // 自己对自己做引导滤波就是去噪, 那么此时, 将图像根据通道分离
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    // 存储不同滤波核半径的结果
    std::vector<cv::Mat> all_denoised_results({noise_image});
    std::vector<double> all_radius({1, 3, 5, 7, 9});
    for(const auto radius : all_radius) {
        // 用一个 vector 接收每个通道的去噪结果
        std::vector<cv::Mat> denoised_channels;
        for(const auto& channel : noise_channels)
            denoised_channels.emplace_back(guided_filter_with_gray(channel, channel, radius, radius, 0.01));
        // 把去噪结果合并起来
        cv::Mat denoised_image;
        cv::merge(denoised_channels, denoised_image);
        all_denoised_results.emplace_back(denoised_image);
    }
    // 保存结果
    const auto comparison_resultss = cv_concat(all_denoised_results);
    cv_show(comparison_resultss);
    const std::string save_path("./images/output/comparison_rgb_radius.png");
    cv::imwrite(save_path, comparison_resultss, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void denoise_flash_demo() {
    // 根据图片路径读取图像
    const char* flash_path = "./images/input/flash/cave-flash.bmp";
    const auto flash_image = cv::imread(flash_path);
    if(flash_image.empty()) {
        std::cout << "读取图片  " << flash_path << "  失败 !" << std::endl;
        return;
    }
    const char* noflash_path = "./images/input/flash/cave-noflash.bmp";
    const auto noflash_image = cv::imread(noflash_path);
    if(noflash_image.empty()) {
        std::cout << "读取图片  " << noflash_path << "  失败 !" << std::endl;
        return;
    }
    //  ---------- 【1】 flash 图像进行拆分
    std::vector<cv::Mat> flash_channels;
    cv::split(flash_image, flash_channels);
    std::vector<cv::Mat> noflash_channels;
    cv::split(noflash_image, noflash_channels);
    std::vector<cv::Mat> result;
    const int channels = (int)flash_channels.size();
    //  ---------- 【2】 对应通道做引导滤波
    for(int i = 0;i < channels; ++i)
        result.emplace_back(guided_filter_with_gray(noflash_channels[i], flash_channels[i], 8, 8, 4e-4));
    cv::Mat denoised_image;
    cv::merge(result, denoised_image);
    //  ---------- 【3】 展示
    const auto comparison_results = cv_concat({noflash_image, flash_image, denoised_image});
    cv_show(comparison_results);
    //  ---------- 【4】 保存结果
    const std::string save_path("./images/output/comparison_flash.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}



void detail_enhancement_demo() {
    // 根据图片路径读取图像
    const char* noise_path = "./images/input/detail/flower.bmp";
    const auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }

    // ---------- 【1】 引导滤波的结果
    std::vector<cv::Mat> noise_channels;
    cv::split(noise_image, noise_channels);
    std::vector<cv::Mat> denoised_channels;
    for(const auto& channel : noise_channels)
        denoised_channels.emplace_back(guided_filter_with_gray(channel, channel, 16, 16, 0.01));
    cv::Mat denoised_image;
    cv::merge(denoised_channels, denoised_image);

    // ---------- 【2】 双边滤波的结果
    auto bilateral_result = bilateral_filter(noise_image, 33, 16, 20);

    // ---------- 【3】 加减处理得到细节增强结果
    const int length = denoised_image.rows * denoised_image.cols * denoised_image.channels();
    for(int i = 0;i < length; ++i)
        denoised_image.data[i] = cv::saturate_cast<uchar>((noise_image.data[i] - denoised_image.data[i]) * 5 + denoised_image.data[i]);
    for(int i = 0;i < length; ++i)
        bilateral_result.data[i] = cv::saturate_cast<uchar>((noise_image.data[i] - bilateral_result.data[i]) * 5 + bilateral_result.data[i]);

    // ---------- 【4】 保存结果
    const auto comparison_resultss = cv_concat({noise_image, denoised_image, bilateral_result});
    cv_show(comparison_resultss);
    const std::string save_path("./images/output/comparison_detail_enhancement.png");
    cv::imwrite(save_path, comparison_resultss, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


void GEF_demo() {
    // Image Fusion with Guided Filtering 用于图像融合
}


void matting_demo() {
    // ---------- 【1】 读取 mask 图像(需要转成单通道) 和引导图像(彩色原图)
    const char* mask_path = "./images/input/matting/bird_mask.png";
    const auto mask_image = cv::imread(mask_path);
    if(mask_image.empty()) {
        std::cout << "读取图片  " << mask_path << "  失败 !" << std::endl;
        return;
    }
    cv::Mat mask_image_gray;
    cv::cvtColor(mask_image, mask_image_gray, cv::COLOR_BGR2GRAY);

    const char* detail_path = "./images/input/matting/bird_color.png";
    const auto detail_image = cv::imread(detail_path);
    if(detail_image.empty()) {
        std::cout << "读取图片  " << detail_path << "  失败 !" << std::endl;
        return;
    }

    // ---------- 【2】 彩色图像做指导灰度图过滤...
    auto mask_with_color = guided_filter_with_color(mask_image_gray, detail_image, 60, 60, 1e-6);

    // ---------- 【3】 灰度图像做指导灰度图过滤...
    cv::Mat detail_image_gray;
    cv::cvtColor(detail_image, detail_image_gray, cv::COLOR_BGR2GRAY);
    auto mask_with_gray = guided_filter_with_gray(mask_image_gray, detail_image_gray, 60, 60, 1e-6);

    // ---------- 【4】 展示结果
    // cv::merge(std::vector<cv::Mat>{mask_with_color, mask_with_color, mask_with_color}, mask_with_color);
    // cv::merge(std::vector<cv::Mat>{mask_with_gray, mask_with_gray, mask_with_gray}, mask_with_gray);
    const auto comparison_results = cv_concat({mask_image_gray, detail_image_gray, mask_with_color, mask_with_gray});
    cv_show(comparison_results);

    // ---------- 【5】 保存结果
    const std::string save_path("./images/output/comparison_matting_bird.png");
    cv::imwrite(save_path, comparison_results, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;
    // 灰度图如何根据引导滤波去噪
    denoise_gray_demo();
    // 彩色图如何根据引导滤波去噪
    denoise_rgb_demo_1();
    // 探究 epsilon 对滤波的影响
    denoise_rgb_demo_2();
    // 探究滤波核半径对滤波的影响
    denoise_rgb_demo_3();
    // flash 去噪实验
    denoise_flash_demo();
    // 细节增强实验
    detail_enhancement_demo();
    // 抠图实验
    matting_demo();
    // 还差个图像融合, 去雾
    return 0;
}
