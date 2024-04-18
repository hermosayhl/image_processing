//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// self
#include "canny.h"

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

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat cv_repeat(const cv::Mat& source) {
        cv::Mat result;
        cv::merge(std::vector<cv::Mat>({source, source, source}), result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }
}










void demo_1() {
    std::string noise_path("./images/input/a1058-_I2E8070.png");
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图像 " << noise_path << " 失败 !" << std::endl;
        return;
    }
    // 转成灰度图
    cv::Mat noise_gray;
    cv::cvtColor(noise_image, noise_gray, cv::COLOR_BGR2GRAY);
    // canny 算法
    auto details = canny(noise_gray, 60, 100);
    // 增强细节
    const auto comparison_results = cv_concat({noise_image, cv_repeat(details)}, false);
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/demo_1_canny.png");
}



int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // canny 边缘检测
    demo_1();

    return 0;
}
