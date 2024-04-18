//C++
#include <map>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// self
#include "dark_channel_prior.h"




void dark_prior_validation();
void dark_channel_prior_demo_1();
void dark_channel_prior_demo_2();
void dark_channel_prior_demo_3();
void dark_channel_prior_demo_4();
void dark_channel_prior_demo_5();
void dark_channel_prior_demo_6();


int main() {

    // 找几张图象验证一下暗通道先验
    dark_prior_validation();

    // 最简单的 demo, 只看去雾效果
    dark_channel_prior_demo_1();

    // 最开始的探索
    dark_channel_prior_demo_2();

    // 深度估计 + 错误的例子
    dark_channel_prior_demo_3();

    // 把估计 A 的那些点都抠出来
    dark_channel_prior_demo_4();

    // 是否使用 t0
    dark_channel_prior_demo_5();

    // 天安门那张图为什么我会错
    dark_channel_prior_demo_6();

    return 0;
}



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
        // cv::namedWindow("", cv::WindowFlags::WINDOW_NORMAL);
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat cv_resize(cv::Mat& one_image, const int height, const int width, const int _interpolation=cv::INTER_LINEAR) {
		cv::Mat result;
		cv::resize(one_image, result, cv::Size(width, height), 0, 0, _interpolation);
		return result;
	}

	cv::Mat cv_concat(const cv::Mat& lhs, const cv::Mat& rhs, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        else cv::vconcat(std::vector<cv::Mat>({lhs, rhs}), result);
        return result;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat cv_stack(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::merge(images, result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }
}




void dark_prior_validation() {
    const std::string dir_name("./images/output/dehaze/prior/origin/");
    const std::string save_dir("./images/output/dehaze/prior/");
    std::vector<std::string> images_list({
        "a4988-kme_0248.png",
        "a4928-Duggan_090127_4793.png",
        "a4822-DSC_0061.png",
        "a0843-IMG_0009.png",
        "a4776-DSC_0062-2.png",
        "a4548-Duggan_080130_5029.png",
        "a4534-DSC_0095.png",
        "a0626-20070618_190911__MG_8400.png",
        "a0955-IMG_0045.png",
        "a1003-_DSC0048-2.png",
        "a1070-_MG_6547.png",
        "a1346-20061213_142422__MG_3757.png",
        "a1557-IMG_0828.png",
        "0081_0.8_0.2.jpg",
        "0240_1_0.08.jpg",
        "0160_0.9_0.2.jpg",
        "0154_0.8_0.2.jpg",
        "0060_0.9_0.2.jpg",
        "1.jpg", "2.jpg", "3.jpg", "4.jpg"
    });
    for(const auto& image_name : images_list) {
        const auto image_path = dir_name + image_name;
        const auto image = cv::imread(image_path);
        if(image.empty())
            continue;
        const auto dark_channel = get_dark_channel(image, 5, true);
        const auto comparison_results = cv_concat({
            image,
            cv_stack({dark_channel, dark_channel, dark_channel})
        });
        cv_show(comparison_results);
        cv_write(comparison_results, save_dir + image_name);
    }
}




void dark_channel_prior_demo_1() {
    const std::string image_path("./images/input/girls.jpg");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    std::map<const std::string, cv::Mat> dehazed_result;
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.8, true, false, false);
    }, "t(x) = 1, without guide  ====>  ");
    // 展示
    cv_show(cv_concat(haze_image, dehazed_result["dehazed"]));
}





void dark_channel_prior_demo_2() {
    const std::string image_path("./images/input/tiananmen1.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;

    // ---------- 【1】三通道一个透射图 T, 没有 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, false, false, true);
    }, "t(x) = 1, without guide  ====>  ");
    // 计算深度图 ?
    comparison_results = cv_concat({
        dehazed_result["A_points"],
        cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
        cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
        dehazed_result["dehazed"]});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_1_without_guide.png");

    // ---------- 【2】三通道一个透射图 T, 经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, true, false, true);
    }, "t(x) = 1, with guide  ====>  ");
    comparison_results = cv_concat({
        dehazed_result["A_points"],
        cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
        cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
        cv_stack({dehazed_result["T_guided"], dehazed_result["T_guided"], dehazed_result["T_guided"]}),
        dehazed_result["dehazed"]});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_1_with_guide.png");

    // ---------- 【3】三通道分开计算透射图 T, 不经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 6, 0.001, 0.1, 0.95, false, true, true);
    }, "t(x) = 3, without guide  ====>  ");
    comparison_results = cv_concat(
        cv_concat({
                dehazed_result["A_points"],
                cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
                dehazed_result["dehazed"]}),
        cv_concat({
                cv_stack({dehazed_result["T_0"], dehazed_result["T_0"], dehazed_result["T_0"]}),
                cv_stack({dehazed_result["T_1"], dehazed_result["T_1"], dehazed_result["T_1"]}),
                cv_stack({dehazed_result["T_2"], dehazed_result["T_2"], dehazed_result["T_2"]})
        }),
        true
    );
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_3_without_guide.png");

    // ---------- 【4】三通道分开计算透射图 T, 经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, true, true, true);
    }, "t(x) = 3, with guide  ====>  ");
    comparison_results = cv_concat(
        {
            cv_concat({
                dehazed_result["A_points"],
                cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
                dehazed_result["dehazed"]}),
            cv_concat({
                    cv_stack({dehazed_result["T_0"], dehazed_result["T_0"], dehazed_result["T_0"]}),
                    cv_stack({dehazed_result["T_1"], dehazed_result["T_1"], dehazed_result["T_1"]}),
                    cv_stack({dehazed_result["T_2"], dehazed_result["T_2"], dehazed_result["T_2"]})}),
            cv_concat({
                    cv_stack({dehazed_result["T_0_guided"], dehazed_result["T_0_guided"], dehazed_result["T_0_guided"]}),
                    cv_stack({dehazed_result["T_1_guided"], dehazed_result["T_1_guided"], dehazed_result["T_1_guided"]}),
                    cv_stack({dehazed_result["T_2_guided"], dehazed_result["T_2_guided"], dehazed_result["T_2_guided"]})})
        },
        true
    );
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_3_with_guide.png");
}


void dark_channel_prior_demo_3() {
    const std::string image_path("./images/input/gugong.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;

    // ---------- 【1】直接跑 guided filter 精修之后的结果
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, true, false, true);
    }, "t(x) = 1, with guide  ====>  ");

    // ---------- 【2】根据 T 求深度图, 可视化
    const auto& T = dehazed_result["T_guided"]; // 随便选一个 T, 我选精修之后的, 原始的 T 也可以(注意这里取键值, 必须取有键值的, 不然程序崩溃)
    cv::Mat T_double;
    T.convertTo(T_double, CV_64FC1);
    cv::log(T_double, T_double);
    T_double = - 1. / 0.1 * T_double;
    cv::Mat hotmap_image = cv::Mat::zeros(T.rows, T.cols, CV_64FC1);
    cv::normalize(T_double, hotmap_image, 0, 255, cv::NORM_MINMAX);
    hotmap_image.convertTo(hotmap_image, CV_8UC1);
    cv::Mat temp;
    cv::applyColorMap(hotmap_image, temp, cv::COLORMAP_HOT);
    cv::cvtColor(hotmap_image, hotmap_image, cv::COLOR_BGR2RGB);
    cv::addWeighted(hotmap_image, 0.1, temp, 0.9, 0, hotmap_image);
    // ---------- 【3】展示
    comparison_results = cv_concat(
            {
        cv_concat({
            haze_image,
            cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
            hotmap_image
        }),
        cv_concat({
            cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
            cv_stack({T, T, T}),
            dehazed_result["dehazed"]
        }),
    }, true);
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_1_with_guide_hotmap.png");

    // ---------- 【4】跑一个粗糙的结果
    std::map<const std::string, cv::Mat> dehazed_result_2;
    run([&](){
        dehazed_result_2 = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.1, 0.95, false, false, true);
    }, "t(x) = 1, without guide  ====>  ");
    // ---------- 【5】对比下结果
    cv::Mat blank = cv::Mat(cv::Size(T.cols, 40), CV_8UC3);
    const int TOTAL = 3 * 40 * T.cols;
    for(int i = 0;i < TOTAL; ++i) blank.data[i] = 255;
    comparison_results = cv_concat({haze_image, blank, dehazed_result_2["dehazed"], blank, dehazed_result["dehazed"]}, true);
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_1_before_and_after_guide_fiilter.png");
}





void dark_channel_prior_demo_4() {
    const std::string image_path("./images/input/canon3.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;

    // ---------- 【1】直接跑 guided filter 精修之后的结果
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 4, 0.001, 0.1, 0.95, true, false, true);
    }, "t(x) = 1, with guide  ====>  ");

    // ---------- 【2】根据 T 求深度图, 可视化
    const auto& T = dehazed_result["T_guided"]; // 随便选一个 T, 我选精修之后的, 原始的 T 也可以(注意这里取键值, 必须取有键值的, 不然程序崩溃)
    cv::Mat T_double;
    T.convertTo(T_double, CV_64FC1);
    cv::log(T_double, T_double);
    T_double = - 1. / 0.1 * T_double;
    cv::Mat hotmap_image = cv::Mat::zeros(T.rows, T.cols, CV_64FC1);
    cv::normalize(T_double, hotmap_image, 0, 255, cv::NORM_MINMAX);
    hotmap_image.convertTo(hotmap_image, CV_8UC1);
    cv::Mat temp;
    cv::applyColorMap(hotmap_image, temp, cv::COLORMAP_HOT);
    cv::cvtColor(hotmap_image, hotmap_image, cv::COLOR_BGR2RGB);
    cv::addWeighted(hotmap_image, 0.1, temp, 0.9, 0, hotmap_image);
    // ---------- 【3】展示
    comparison_results = cv_concat(
            {
        cv_concat({
            dehazed_result["A_points"],
            cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
            hotmap_image
        }),
        cv_concat({
            cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
            cv_stack({T, T, T}),
            dehazed_result["dehazed"]
        }),
    }, true);
    cv_show(comparison_results);
    // cv_write(comparison_results, "./images/output/dehaze/t_1_A_points.png");
}




void dark_channel_prior_demo_5() {
    const std::string image_path("./images/input/canyon2.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;
    for(int radius = 3; radius <= 15; radius += 2) {
        run([&](){
            dehazed_result = dark_channel_prior_dehaze(haze_image, radius, 0.001, 0.2, 0.95, true, false, true);
        }, "t(x) = 1, with guide  ====>  ");
        comparison_results = cv_concat({
            dehazed_result["A_points"],
            cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
            cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
            cv_stack({dehazed_result["T_guided"], dehazed_result["T_guided"], dehazed_result["T_guided"]}),
            dehazed_result["dehazed"]});
        cv_show(comparison_results);
        cv_write(comparison_results, "./images/output/dehaze/t_1_with_radius_grows" + std::to_string(radius) + ".png");
    }
}



void dark_channel_prior_demo_6() {
    const std::string image_path("./images/input/tiananmen1.bmp");
    const auto haze_image = cv::imread(image_path);
    if(haze_image.empty()) {
        std::cout << "读取图像 " << image_path << " 失败 !\n";
        return;
    }
    cv::Mat comparison_results;
    std::map<const std::string, cv::Mat> dehazed_result;

    // ---------- 【2】三通道一个透射图 T, 经过 guided filter 精修
    run([&](){
        dehazed_result = dark_channel_prior_dehaze(haze_image, 3, 0.001, 0.3, 0.95, true, false, true);
    }, "t(x) = 1, with guide  ====>  ");
    comparison_results = cv_concat({
        dehazed_result["A_points"],
//        cv_stack({dehazed_result["dark_channel"], dehazed_result["dark_channel"], dehazed_result["dark_channel"]}),
//        cv_stack({dehazed_result["T"], dehazed_result["T"], dehazed_result["T"]}),
//        cv_stack({dehazed_result["T_guided"], dehazed_result["T_guided"], dehazed_result["T_guided"]}),
        dehazed_result["dehazed"]});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/dehaze/t_1_with_guide.png");
}