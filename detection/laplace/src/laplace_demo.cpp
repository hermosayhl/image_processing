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
#include "faster_gaussi_filter.h"
#include "laplace_of_gaussi.h"


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
    std::string noise_path("./images/input/a0118-20051223_103622__MG_0617_noisy.png");
    auto origin_image = cv::imread(noise_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << noise_path << " 失败 !" << std::endl;
        return;
    }
    //origin_image = cv_resize(origin_image, 512, 341);
    // 转成灰度图
    cv::Mat noise_gray;
    cv::cvtColor(origin_image, noise_gray, cv::COLOR_BGR2GRAY);
    // 先过一遍高斯滤波
    // noise_gray = faster_2_gaussi_filter_channel(noise_gray, 3, 0.1, 0.1);
    // 利用拉普拉斯检测边缘
    auto details = laplace_extract_edges(noise_gray);
    // 增强细节
    const auto comparison_results = cv_concat({origin_image, cv_repeat(details)}, true);
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/demo_1_noisy.png");
}



void demo_2() {
    const std::string image_path("./images/input/a0015-DSC_0081.png"); // a0015-DSC_0081.png  a0025-kme_298.png
    auto origin_image = cv::imread(image_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !" << std::endl;
        return;
    }
    // BGR -> Gray
    cv::cvtColor(origin_image, origin_image, cv::COLOR_BGR2GRAY);

    // LOG 检测边缘
    cv::Mat detected_result;
    run([&](){
         detected_result = laplace_of_gaussi_edge_detection(origin_image, 3, 0.9, 2);
    });

    // 展示结果, 保存
    auto comparison_results = cv_concat({detected_result});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/LOG_edge_detection_2.png");
}


void demo_5() {
    const std::string image_path("./images/input/a0032-jmac_MG_0266.png"); // a0015-DSC_0081.png  a0025-kme_298.png
    auto origin_image = cv::imread(image_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !" << std::endl;
        return;
    }
    cv::Mat detected_result = origin_image.clone();
    // BGR -> Gray
    cv::cvtColor(origin_image, origin_image, cv::COLOR_BGR2GRAY);

    // LOG 检测边缘
    std::pair<keypoints_type, keypoints_type > key_points;
    run([&](){
         key_points = difference_of_gaussi_keypoints_detection(
                 origin_image, 3, {0.3, 0.4, 0.5, 0.6}, 6);
    });

    // 极大值
    for(const auto point : key_points.first)
        cv::circle(detected_result, cv::Point(point.second, point.first), 3, CV_RGB(255, 0, 0));

    // 极小值
    for(const auto point : key_points.second)
        cv::circle(detected_result, cv::Point(point.second, point.first), 3, CV_RGB(0, 255, 0));

    // 展示结果, 保存
    auto comparison_results = cv_concat({detected_result});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/LOG_keypoints_detection.png");
}




void demo_6() {
    std::string noise_path("./images/input/blob_2.png");
    auto origin_image = cv::imread(noise_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << noise_path << " 失败 !" << std::endl;
        return;
    }
    // 转成灰度图
    cv::Mat origin_gray;
    cv::cvtColor(origin_image, origin_gray, cv::COLOR_BGR2GRAY);

    // 利用  LOG  检测 blob
    auto blobs = laplace_of_gaussi_blob_detection(origin_gray, 1. / 1.414, 1.414, 10, 0.3, 400);

    for(const auto item : blobs) {
        const int cur_scale = std::get<1>(item);
        cv::circle(origin_image, cv::Point(std::get<3>(item), std::get<2>(item)), int(cur_scale * 1.414), CV_RGB(255, 0, 0), 1);
    }

    // 展示
    const auto comparison_results = cv_concat({origin_image});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/LOG_keypoints_3.png");
}


void demo_7() {
    std::string noise_path("./images/input/blob_2.png");
    auto origin_image = cv::imread(noise_path);
    if(origin_image.empty()) {
        std::cout << "读取图像 " << noise_path << " 失败 !" << std::endl;
        return;
    }

    // 利用  LOG  检测 blob
    laplace_of_gaussi_blob_detection_single_scale(origin_image, 1. / 1.414, 1.414, 12, 0.3, 400);
}


int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    // Laplace 检测边缘
    demo_1();

    // LOG
    demo_2();

    // LOG uint8 损失精度的优化

    // LOG 模板分离

    // 检测关键点
    demo_5();

    // LOG 检测关键点
    demo_6();

    // 验证单尺度下的
    demo_7();

    // DOB 近似 LOG  Difference of boxes
    return 0;
}
