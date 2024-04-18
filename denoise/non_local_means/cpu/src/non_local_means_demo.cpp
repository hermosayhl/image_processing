//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "non_local_means.h"

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
    const char* noise_path = "../images/input/denoise/Kodak24/20.png";
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    cv::cvtColor(noise_image, noise_image, cv::COLOR_BGR2GRAY);

    cv::Mat denoised, denoised_fast_1, denoised_fast_2, denoised_fast_3;
    // ---------- 【1】普通的 non_local_means
    run([&noise_image, &denoised](){
        denoised = non_local_means(noise_image, 5, 2, 10, "mean", false);
    }, "non_local_means_gray  :  ");
    // ---------- 【2】经 box_filter 加速的 non_local_means
    run([&noise_image, &denoised_fast_1](){
        denoised_fast_1 = fast_non_local_means_gray_1(noise_image, 5, 2, 10);
    }, "box filter 优化  :  ");
    // ---------- 【3】经 积分图 加速的 non_local_means
    run([&noise_image, &denoised_fast_2](){
        denoised_fast_2 = fast_non_local_means_gray_2(noise_image, 5, 2, 10);
    }, "积分图 优化  :  ");
    run([&noise_image, &denoised_fast_3](){
        denoised_fast_3 = fast_non_local_means_gray_3(noise_image, 5, 2, 10);
    }, "积分图 AVX 优化  :  ");
    const auto comparison_resultss = cv_concat({noise_image, denoised, denoised_fast_1, denoised_fast_2, denoised_fast_3});
    cv_show(comparison_resultss);
    // 保存结果
    const std::string save_path("../images/output/comparison_gray.png");
    cv::imwrite(save_path, comparison_resultss, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}



void denoise_rgb_demo_1() {
    // 根据图片路径读取图像
    const char* noise_path = "../images/input/denoise/Kodak24/20.png";
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图片  " << noise_path << "  失败 !" << std::endl;
        return;
    }
    // noise_image = cv_resize(noise_image, 128, 90);
    cv::Mat denoised;
    run([&noise_image, &denoised](){
        denoised = non_local_means(noise_image, 11, 5, 10, "mean", true, false);
    }, "color_split  :  ");
    const auto comparison_resultss = cv_concat({noise_image, denoised});
    cv_show(comparison_resultss);
    // 保存结果
    const std::string save_path("../images/output/Kodak24.png");
    cv::imwrite(save_path, denoised, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
}




void denoise_rgb_demo_2() {}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // 灰度图的 non_local_means
    denoise_gray_demo();

    // 彩色图的分通道 non_local_means
    denoise_rgb_demo_1();

    // 彩色图的三通道一起 non_local_means
    denoise_rgb_demo_2();
    return 0;
}
