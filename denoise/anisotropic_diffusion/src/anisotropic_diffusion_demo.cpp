//C++
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
// opencv
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

    cv::Mat cv_concat(const std::vector<cv::Mat> images) {
        cv::Mat result;
        cv::hconcat(images, result);
        return result;
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }
}





cv::Mat anisotropic_diffusion_denoise_gray(
        const cv::Mat& noise_image,
        const int iterations=10,
        const double K=20,
        const double lambda=0.25) {
    // 获取图像信息
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int length = H * W;
    // 先对图像 padding 一个像素
    auto denoised = make_pad(noise_image, 1, 1);
    const int W2 = denoised.cols;
    // 准备一个结果
    const double K2_inv = 1. / (K * K);
    // 迭代
    for(int t = 0;t < iterations; ++t) {
        // 准备这次的结果
        cv::Mat temp = noise_image.clone();
        // 遍历每一个点
        for(int i = 0;i < H; ++i) {
            const uchar* const row_ptr = denoised.data + (1 + i) * W2 + 1;
            uchar* const res_ptr = temp.data + i * W;
            for(int j = 0;j < W; ++j) {
//                // 当前点求上下左右四个方向的散度
//                const double up = row_ptr[j - W2] - row_ptr[j];
//                const double down = row_ptr[j + W2] - row_ptr[j];
//                const double left = row_ptr[j - 1] - row_ptr[j];
//                const double right = row_ptr[j + 1] - row_ptr[j];
//                // 根据散度计算传导系数
//                const double up_coefficient = fast_exp(-up * up * K2_inv);
//                const double down_coefficient = fast_exp(-down * down * K2_inv);
//                const double left_coefficient = fast_exp(-left * left * K2_inv);
//                const double right_coefficient = fast_exp(-right * right * K2_inv);
//                // 执行这个像素上的扩散
//                res_ptr[j] = cv::saturate_cast<uchar>(row_ptr[j] + lambda * (
//                        up * up_coefficient
//                        + down * down_coefficient
//                        + left * left_coefficient
//                        + right * right_coefficient));
                // 求八个方向的
                const double up = row_ptr[j - W2] - row_ptr[j];
                const double up_1 = row_ptr[j - W2 - 1] - row_ptr[j];
                const double up_2 = row_ptr[j - W2 + 1] - row_ptr[j];
                const double down = row_ptr[j + W2] - row_ptr[j];
                const double down_1 = row_ptr[j + W2 - 1] - row_ptr[j];
                const double down_2 = row_ptr[j + W2 + 1] - row_ptr[j];
                const double left = row_ptr[j - 1] - row_ptr[j];
                const double right = row_ptr[j + 1] - row_ptr[j];
                // 根据散度计算传导系数
                const double up_coefficient = fast_exp(-up * up * K2_inv);
                const double up_1_coefficient = fast_exp(-up_1 * up_1 * K2_inv);
                const double up_2_coefficient = fast_exp(-up_2 * up_2 * K2_inv);
                const double down_coefficient = fast_exp(-down * down * K2_inv);
                const double down_1_coefficient = fast_exp(-down_1 * down_1 * K2_inv);
                const double down_2_coefficient = fast_exp(-down_2 * down_2 * K2_inv);
                const double left_coefficient = fast_exp(-left * left * K2_inv);
                const double right_coefficient = fast_exp(-right * right * K2_inv);
                // 执行这个像素上的扩散
                res_ptr[j] = cv::saturate_cast<uchar>(row_ptr[j] + lambda * (
                        up * up_coefficient
                        + down * down_coefficient
                        + left * left_coefficient
                        + right * right_coefficient
                        + up_1 * up_1_coefficient
                        + up_2 * up_2_coefficient
                        + down_1 * down_1_coefficient
                        + down_2 * down_2_coefficient));
            }
        }
        denoised = make_pad(temp, 1, 1);
    }
    return denoised(cv::Rect(1, 1, W, H));
}





cv::Mat anisotropic_diffusion_denoise_color(
        const cv::Mat& noise_image,
        const int iterations=10,
        const double K=20,
        const double lambda=0.25) {
    std::vector<cv::Mat> noise_channels, denoised_channels;
    cv::split(noise_image, noise_channels);
    for(int ch = 0;ch < 3; ++ch)
        denoised_channels.emplace_back(anisotropic_diffusion_denoise_gray(noise_channels[ch], iterations, K, lambda));
    cv::Mat denoised;
    cv::merge(denoised_channels, denoised);
    return denoised;
}





int main() {
    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    std::string noise_path("./images/input/woman_3.png");
    auto noise_image = cv::imread(noise_path);
    if(noise_image.empty()) {
        std::cout << "读取图像 " << noise_path << " 失败 !" << std::endl;
        return 0;
    }
    // cv::cvtColor(noise_image, noise_image, cv::COLOR_BGR2GRAY);
    cv::Mat denoised;
    run([&](){
        denoised = anisotropic_diffusion_denoise_color(noise_image, 10, 12, 0.125);
    });
    const auto comparison_results = cv_concat({denoised});
    cv_show(comparison_results);
    cv_write(comparison_results, "./images/output/demo_2.png");
    return 0;
}
