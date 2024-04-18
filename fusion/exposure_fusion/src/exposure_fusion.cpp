// C++
#include <map>
#include <vector>
#include <chrono>
#include <iostream>
#include <functional>
// OpenCV
#include <opencv2/opencv.hpp>


namespace {
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    void cv_info(const cv::Mat& one_image) {
        std::cout << "高  :  " << one_image.rows << "\n宽  :  " << one_image.cols << "\n通道 :  " << one_image.channels() << std::endl;
        std::cout << "步长 :  " << one_image.step << std::endl;
        std::cout << "是否连续" << std::boolalpha << one_image.isContinuous() << std::endl;
    }

    cv::Mat cv_concat(const std::vector<cv::Mat> images, const bool v=false) {
        cv::Mat result;
        if(not v) cv::hconcat(images, result);
        else cv::vconcat(images, result);
        return result;
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    template<typename T>
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type, const double times=2) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]) * times);
        return result;
    }
}



namespace {
    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

    inline double square(const double x) {
        return x * x;
    }
}


std::map<const std::string, cv::Mat> exposure_fusion(
        const std::vector<cv::Mat>& sequence,
        const std::tuple<float, float, float>& alphas={1.0, 1.0, 1.0},
        const bool use_lappyr=true,
        const int layers_num=5,
        const float best_illumination=0.5,
        const double sigma=0.2) {
    // 准备一些中间计算所需变量
    const double sigma_inv = 1. / (2 * sigma * sigma);
    const float norm = 1.f / 255;
    // 获取图像信息
    const int H = sequence.front().rows;
    const int W = sequence.front().cols;
    const int C = sequence.front().channels();
    assert(C == 3 and "该算法只支持 RGB24 图像 !");
    for(const auto& image : sequence)
        assert(image.rows == H and image.cols == W and image.channels() == C and "图像序列的大小必须都一致!");
    const int length = H * W;
    // 看有几张图像
    const int sequence_len = sequence.size();
    // 根据先验知识, 计算对比度、饱和度、亮度, 求权重
    std::vector< cv::Mat > weights;
    for(const auto& image : sequence) {
        // 【1】根据对比度求权重
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        const auto gray_padded = make_pad(gray, 1, 1);
        std::vector<float> contrast(length, 0);
        const int W_2 = W + 2;
        for(int i = 1; i < H + 1; ++i) {
            const uchar* const row_ptr = gray_padded.data + i * W_2;
            float* const res_ptr = contrast.data() + (i - 1) * W;
            for(int j = 1; j < W + 1; ++j)
                res_ptr[j - 1] = norm * std::abs(row_ptr[j - 1] + row_ptr[j + 1] + row_ptr[j - W_2] + row_ptr[j + W_2] - 4 * row_ptr[j]);
        }
        // 【2】根据饱和度求权重
        std::vector<float> saturation(length, 0);
        for(int i = 0; i < H; ++i) {
            const uchar* const row_ptr = image.data + i * W * 3;
            float* const res_ptr = saturation.data() + i * W;
            for(int j = 0; j < W; ++j) {
                const int p = 3 * j;
                const float mean = (row_ptr[p] + row_ptr[p + 1] + row_ptr[p + 2]) * norm / 3;
                res_ptr[j] = std::sqrt((square(row_ptr[p] * norm - mean) + square(row_ptr[p + 1] * norm - mean) + square(row_ptr[p + 2] * norm - mean)) * 1.f / 3);
            }
        }
        // 【3】根据亮度求权重
        std::vector<float> illumination(length, 0);
        for(int i = 0; i < H; ++i) {
            const uchar* const row_ptr = image.data + i * W * 3;
            float* const res_ptr = illumination.data() + i * W;
            for(int j = 0; j < W; ++j) {
                const int p = 3 * j;
                res_ptr[j] = fast_exp(-square(row_ptr[p] * norm - best_illumination) * sigma_inv)
                           * fast_exp(-square(row_ptr[p + 1] * norm - best_illumination) * sigma_inv)
                           * fast_exp(-square(row_ptr[p + 2] * norm - best_illumination) * sigma_inv);
            }
        }
        /*
        cv_show(image);
        cv_show(cv_concat({
            toint8(contrast, H, W, 1, CV_8UC1, 255.0),
            toint8(saturation, H, W, 1, CV_8UC1, 255.0),
            toint8(illumination, H, W, 1, CV_8UC1, 255.0)
        }));
        */
        // 三者结合, 求权重
        cv::Mat cur_weight(H, W, CV_32F);
        float* const weight_ptr = cur_weight.ptr<float>();
        for(int i = 0; i < length; ++i)
            weight_ptr[i] = std::pow(contrast[i], std::get<0>(alphas)) *
                            std::pow(saturation[i], std::get<1>(alphas)) *
                            std::pow(illumination[i], std::get<2>(alphas));
        weights.emplace_back(cur_weight);
    }
    std::vector<float*> weights_ptrs;
    for(int k = 0;k < sequence_len; ++k) weights_ptrs.emplace_back(weights[k].ptr<float>());
    // 求归一化的权重
    for(int i = 0;i < length; ++i) {
        float weight_sum = 1e-12;
        for(int k = 0;k < sequence_len; ++k) weight_sum += weights_ptrs[k][i];
        for(int k = 0;k < sequence_len; ++k) weights_ptrs[k][i] /= weight_sum;
    }
    // for(int k = 0;k < sequence_len; ++k) cv_show(toint8(weights[k], H, W, 1, CV_8UC1, 255));
    // 准备融合结果
    std::map<const std::string, cv::Mat> results;
    // 粗糙的融合
    auto weighted_fusion = [&](const bool use_gaussi=false)
            ->cv::Mat {
        cv::Mat fused = cv::Mat::zeros(H, W, sequence.front().type());
        uchar* fused_ptr = fused.data;
        // 每张图像对应一个权重图
        for(int k = 0;k < sequence_len; ++k) {
            // 对权重图做高斯模糊
            cv::Mat blurred;
            if(use_gaussi) cv::GaussianBlur(weights[k], blurred, cv::Size(49, 49), 8, 8);
            else blurred = std::ref(weights[k]);
            const float* const w_ptr = blurred.ptr<float>();
            const uchar* const cur_image = sequence[k].data;
            for(int i = 0;i < length; ++i) {
                const int p = 3 * i;
                fused_ptr[p] += w_ptr[i] * cur_image[p];
                fused_ptr[p + 1] += w_ptr[i] * cur_image[p + 1];
                fused_ptr[p + 2] += w_ptr[i] * cur_image[p + 2];
            }
        }
        const int total_length = 3 * length;
        for(int i = 0;i < total_length; ++i) fused_ptr[i] = cv::saturate_cast<uchar>(fused_ptr[i]);
        return fused;
    };
    results.emplace("naive", weighted_fusion());
    results.emplace("gaussi_smoothed", weighted_fusion(true));
    // 是否使用拉普拉斯金字塔融合
    if(use_lappyr) {
        // 检查 layers_num 不要太离谱
        assert((1 << layers_num) < sequence[0].cols and (1 << layers_num) < sequence[0].rows and "layers 太大, 图像分辨率太低不支持");
        // 从 high_res 开始构建层数为 layers_num 的高斯金字塔
        auto build_gaussi_pyramids = [](const cv::Mat& high_res, const int layers_num)
                -> std::vector<cv::Mat> {
            std::vector<cv::Mat> this_flash({high_res});
            for(int i = 1; i < layers_num; ++i) {
                cv::Mat blurred;
                cv::GaussianBlur(this_flash[i - 1], blurred, cv::Size(5, 5), 0.83, 0.83);
                cv::resize(blurred, blurred, cv::Size(this_flash[i - 1].cols / 2, this_flash[i - 1].rows / 2));
                this_flash.emplace_back(blurred);
            }
            return this_flash;
        };
        // 求每张图的权重的高斯金字塔
        std::vector< std::vector<cv::Mat> > sequence_weights_pyramids;
        sequence_weights_pyramids.reserve(sequence_len);
        cv::Mat high_res(H, W, CV_32FC1);
        for(int k = 0;k < sequence_len; ++k)
            sequence_weights_pyramids.emplace_back(build_gaussi_pyramids(weights[k], layers_num));
        // 提前释放 weights 空间
        for(int k = 0;k < sequence_len; ++k) weights[k].release();
        // 求每张图的高斯金字塔
        std::vector< std::vector<cv::Mat> > sequence_gaussi_pyramids;
        sequence_gaussi_pyramids.reserve(sequence_len);
        for(int k = 0;k < sequence_len; ++k)
            sequence_gaussi_pyramids.emplace_back(build_gaussi_pyramids(sequence[k], layers_num));
        // 转换数据类型 uchar -> float, 因为后面有减法
        for(int k = 0;k < sequence_len; ++k)
            for(int i = 0;i < layers_num; ++i)
                sequence_gaussi_pyramids[k][i].convertTo(sequence_gaussi_pyramids[k][i], CV_32F);
        // 求每张图的 laplace 金字塔
        std::vector< std::vector<cv::Mat> > sequence_laplace_pyramids;
        sequence_laplace_pyramids.reserve(sequence_len);
        for(int k = 0;k < sequence_len; ++k) {
            auto& gaussi_pyramid = sequence_gaussi_pyramids[k];
            cv::Mat upsampled = gaussi_pyramid[layers_num - 1].clone();
            std::vector<cv::Mat> pyramid({upsampled});
            for(int i = layers_num - 1; i > 0; --i) {
                cv::resize(gaussi_pyramid[i], upsampled, cv::Size(gaussi_pyramid[i - 1].cols, gaussi_pyramid[i - 1].rows));
                pyramid.emplace_back(gaussi_pyramid[i - 1] - upsampled);
            }
            std::reverse(pyramid.begin(), pyramid.end());
            sequence_laplace_pyramids.emplace_back(pyramid);
            for(int i = 0;i < layers_num; ++i) gaussi_pyramid[i].release(); // 释放 gaussi_pyramid 空间, 反正没用了
        }
        // 每一个尺度, 融合一系列图像的的 laplace 细节, 得到一个融合的 laplace 金字塔
        std::vector<cv::Mat> fused_laplace_pyramid;
        fused_laplace_pyramid.reserve(layers_num);
        for(int i = 0;i < layers_num; ++i) {
            cv::Mat weighted_laplace = cv::Mat::zeros(sequence_weights_pyramids[0][i].rows, sequence_weights_pyramids[0][i].cols, CV_32FC3);
            for(int k = 0; k < sequence_len; ++k) {
                cv::Mat new_weights;
                std::vector<cv::Mat> new_weights_vector({sequence_weights_pyramids[k][i], sequence_weights_pyramids[k][i], sequence_weights_pyramids[k][i]});
                cv::merge(new_weights_vector, new_weights);
                weighted_laplace += new_weights.mul(sequence_laplace_pyramids[k][i]); // 这里慢了, 拷贝的消耗大, 直接从指针赋值可能快一点
            }
            fused_laplace_pyramid.emplace_back(weighted_laplace);
        }
        // 从最底层开始, 每次上采样加上同等尺度的 laplace 细节
        cv::Mat fused = fused_laplace_pyramid[layers_num - 1];
        for(int i = layers_num - 2; i >= 0; --i) {
            cv::Mat upsampled;
            cv::resize(fused, upsampled, cv::Size(fused_laplace_pyramid[i].cols, fused_laplace_pyramid[i].rows));
            fused = upsampled + fused_laplace_pyramid[i];
        }
        fused.convertTo(fused, CV_8U);
        results.emplace("laplace_pyramid", fused);
    }
    return results;
}


 #include <ghc/filesystem.hpp>


int main() {

    // 获取图像列表
    std::vector<cv::Mat> sequence;
    const std::string sequence_dir("./images/input/5/");
    auto sequence_list = ghc::filesystem::directory_iterator(sequence_dir);
    for(const auto& it : sequence_list) {
        cv::Mat current = cv::imread(sequence_dir + it.path().filename().string());
        if(current.empty()) continue;
        sequence.emplace_back(current);
    }
    // 如果 layers_num 降低为 5, 就会有很严重的光晕
    const auto fusion_result = exposure_fusion(sequence, {1.0, 1.0, 1.0}, true, 7);
    for(const auto &item : fusion_result) {
        cv_show(item.second, item.first.c_str());
        cv_write(item.second, std::string("./images/output/5/C++_" + item.first + ".png"));
    }
    return 0;
}
