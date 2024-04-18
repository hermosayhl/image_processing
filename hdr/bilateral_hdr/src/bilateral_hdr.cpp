// C++
#include <cmath>
#include <assert.h>
#include <filesystem>
#include <unordered_set>
#include <unordered_map>
// self
#include "guided_filter.hpp"


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

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REFLECT);
        return padded_image;
    }

    inline float _min(const float* data, const int length) {
        float min_value = data[0];
        for(int i = 1;i < length; ++i)
            if(data[i] < min_value) min_value = data[i];
        return min_value;
    }

    inline float _max(const float* data, const int length) {
        float max_value = data[0];
        for(int i = 1;i < length; ++i)
            if(data[i] > max_value) max_value = data[i];
        return max_value;
    }

    inline float square(const float x) {
        return x * x;
    }

    inline float clip(float x, const float low, const float high) {
        if(x < low) x = low;
        else if(x > high) x = high;
        return x;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

}


// 高斯滤波
cv::Mat gaussi_filtering(const cv::Mat& origin, const float spatial_sigma=18) {
    // 收集图像信息
    const int H = origin.rows;
    const int W = origin.cols;
    const int C = origin.channels();
    assert(C == 1 and "only images of single channel is supported !");
    // 计算窗口半径等
    const int radius = int(3 * spatial_sigma);
    const int window_size = square(2 * radius + 1);
    // 对图像做 padding
    const auto padded_image = make_pad(origin, radius, radius);
    const int W2 = W + 2 * radius;
    // 准备一个空域模板
    int max_k = 0;
    std::vector<double> spatial_table(window_size);
    std::vector<int> offset(window_size, 0);
    const float sigma_inv = -0.5 / square(spatial_sigma);
    for(int i = -radius;i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            spatial_table[max_k] = fast_exp(double(sigma_inv * (i * i + j * j)));
            offset[max_k++] = i * W2 + j;
        }
    }
    // 准备一个结果
    cv::Mat result(H, W, CV_32F);
    float* const res_ptr = result.ptr<float>();
    int cnt = 0;
    // 求解每一个点
    for(int i = 0;i < H; ++i) {
        const float* const row_ptr = padded_image.ptr<float>() + (radius + i) * W2 + radius;
        for(int j = 0;j < W; ++j) {
            float sum_value = 0;
            float weight_sum = 0;
            for(int k = 0;k < max_k; ++k) {
                const float w = spatial_table[k];
                sum_value += w * row_ptr[j + offset[k]];
                weight_sum += w;
            }
            res_ptr[cnt++] = sum_value / weight_sum;
        }
    }
    return result;
}


// 双边滤波
cv::Mat bilateral_filtering(const cv::Mat& origin, const float range_sigma=0.4, const float spatial_sigma=18) {
    // 收集信息
    const int H = origin.rows;
    const int W = origin.cols;
    assert(origin.channels() == 1);
    // 求窗口大小
    const int radius = int(3 * spatial_sigma);
    const int window_size = radius * 2 + 1;
    // 对图像做 padding
    const auto padded_image = make_pad(origin, radius, radius);
    const int W2 = padded_image.cols;
    // 准备一个空域的模板(double 是因为 fast_exp 可以大大加快速度)
    std::vector<double> space_table(window_size * window_size);
	std::vector<int> space_offset(window_size * window_size);
	int max_k = 0;
	const double space_variance_2 = - 0.5 / (spatial_sigma * spatial_sigma);
	for(int i = -radius;i <= radius; ++i) {
		for(int j = -radius;j <= radius; ++j) {
			space_table[max_k] = fast_exp(double(space_variance_2 * (i * i + j * j)));
			space_offset[max_k] = i * W2 + j;
			++max_k;
		}
	}
	// 准备值域的
	const float sigma_inv = 0.5f / (range_sigma * range_sigma);
	// 准备一个结果
    cv::Mat result(H, W, CV_32F);
    float* const res_ptr = result.ptr<float>();
	// 开始滤波
	int cnt = 0; // 存放每次的加权结果
	for(int i = 0;i < H; ++i) {
	    // 取出当前滤波的这一行, 在 pad 图像中的行指针, 第 radiu + i 行, 偏移 radius 个像素
	    const float* const pad_ptr = padded_image.ptr<float>() + (radius + i) * W2 + radius;
	    for(int j = 0;j < W; ++j) {
	        const float center = pad_ptr[j];
	        // 遍历窗口
	        float intensity_sum = 0;
	        float weight_sum = 0;
	        for(int k = 0;k < max_k; ++k) {
	            const float neighbor = pad_ptr[j + space_offset[k]];
	            const float w = space_table[k] * fast_exp(double(-sigma_inv * square(neighbor - center)));;
	            intensity_sum += neighbor * w;
	            weight_sum += w;
	        }
            res_ptr[cnt++] = intensity_sum / weight_sum;
	    }
	}
    return result;
}


std::list<std::pair<std::string, cv::Mat> >
        bilateral_local_tonemapping(const cv::Mat& hdr_image, const float contrast_value=10) {
    // 收集中间结果
    std::list<std::pair<std::string, cv::Mat> > collections;
    // 获取图像信息
    const int H = hdr_image.rows;
    const int W = hdr_image.cols;
    const int C = hdr_image.channels();
    assert(C == 3 and "only BGR channels are supported!");
    const float hdr_min = std::max(_min(hdr_image.ptr<float>(), H * W * C), 1e-5f);
    const float hdr_max = _max(hdr_image.ptr<float>(), H * W * C);

    std::cout << "输入的高动态范围图像信息如下 : \n";
    std::cout << "\theight = " << hdr_image.rows << "\n\twidth = " << hdr_image.cols << "\n";
    std::cout << "\tdepth =  " << hdr_image.type() << std::endl;
    std::cout << "\tMax = " << hdr_max << "\n\tMin = " << hdr_min << std::endl;
    std::cout << "\t动态范围 = " << hdr_max / hdr_min << std::endl;

	// 获取 hdr 图像指针
	const float* const hdr_ptr = hdr_image.ptr<float>();

	// 先求亮度图 intensity = (20 * R + 40 * G + 1 * B) / 61;
	const int length = H * W;
	cv::Mat intensity(H, W, CV_32F);
	float* const intensity_ptr = intensity.ptr<float>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        intensity_ptr[i] = (20 * hdr_ptr[p + 2] + 40 * hdr_ptr[p + 1] + hdr_ptr[p]) / 61.f;
    }
    collections.emplace_back("intensity", intensity);

    // 计算 log10(intensity), 在 log 域很重要,
    cv::Mat log_intensity(H, W, CV_32F);
    float* const log_intensity_ptr = log_intensity.ptr<float>();
    for(int i = 0;i < length; ++i)
        log_intensity_ptr[i] = std::log10(intensity_ptr[i]);
    collections.emplace_back("intensity_log", log_intensity);

    // 对 log_intensity 做双边滤波, 得到更平滑的亮度图(base 层)
    const float range_sigma = 0.4;
    const float spatial_sigma = 0.02f * std::min(H, W);
    std::cout << "值域标准差 = " << range_sigma << "\n空域标准差 = " << spatial_sigma << std::endl;
    auto log_base = bilateral_filtering(log_intensity, range_sigma, spatial_sigma);
//    auto log_base = gaussi_filtering(log_intensity, spatial_sigma);
//    auto log_base = guided_filter_with_gray(log_intensity, log_intensity, 3 * spatial_sigma, 3 * spatial_sigma, 0.1);

    // 求 log_detail, 原 log 亮度图 - 平滑过后的亮度图(base 层) = 细节层(log)
    cv::Mat log_detail = log_intensity - log_base;
    collections.emplace_back("base", log_base);
    collections.emplace_back("detail", log_detail);

    // 压缩 base 层的对比度
    // 原来是 1.0 * base + 1.0 * detail
    // 现在假设变成 0.2 * base + 1.0 * detail
    const float log_base_max = _max(log_base.ptr<float>(), length);
    const float log_base_min = _min(log_base.ptr<float>(), length);
    const float factor = std::log10(contrast_value) / (log_base_max - log_base_min);
    std::cout << "Base 层的对比度缩放因子 = " << factor << std::endl;

    cv::Mat log_fusion = factor * log_base + log_detail;
    float* const fusion_ptr = log_fusion.ptr<float>();
    for(int i = 0;i < length; ++i)
        fusion_ptr[i] = std::pow(10.0, fusion_ptr[i]);

    // 准备一个结果, 三通道, 存放 float 数据
    cv::Mat result(H, W, CV_32FC3);
    float* const result_ptr = result.ptr<float>();

    // 计算每个点在亮度通道 intensity 压缩之后的比例大小, 然后 R, G, B 等比例缩放
    for(int i = 0;i < length; ++i) {
        const float ratio = fusion_ptr[i] / intensity_ptr[i];
        const int pos = 3 * i;  // 找到每个点的坐标
        result_ptr[pos + 2] = hdr_ptr[pos + 2] * ratio;
        result_ptr[pos + 1] = hdr_ptr[pos + 1] * ratio;
        result_ptr[pos] = hdr_ptr[pos] * ratio;
    }
    collections.emplace_back("compressed", result.clone());

    // 现在已经压缩了对比度, 尽可能保留了细节

    // 计算新的动态范围
    const float new_hdr_max = _max(result_ptr, length * 3);
    const float new_hdr_min = std::max(1e-5f, _min(result_ptr, length * 3));
    std::cout << "压缩之后的动态范围 = " << new_hdr_max / new_hdr_min << std::endl;

    // 在动态范围内, 将数据标准化到 0-1 或者 0-255, 方便显示器显示
    const float max_scale = std::pow(10.f, log_base_max * factor);
    // 截断函数
    auto normalize = [max_scale](float x) -> float {
        return clip(255 * x / max_scale, 0, 255);
    };
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        result_ptr[p + 2] = normalize(result_ptr[p + 2]);
        result_ptr[p + 1] = normalize(result_ptr[p + 1]);
        result_ptr[p] = normalize(result_ptr[p]);
    }
    // 数据从 float -> uchar,
    result.convertTo(result, CV_8UC3);
    collections.emplace_back("result", result);
    return collections;
}



cv::Mat final_normalize(cv::Mat& result, const int L, const int C, const int _type) {
    float* const res_ptr = result.ptr<float>();
    for(int c = 0;c < C; ++c) {
        // 找到这个通道的最大值
        float max_value = res_ptr[c];
        for(int i = 1;i < L; ++i)
            if(max_value < res_ptr[C * i + c])
                max_value = res_ptr[C * i + c];
        // 这个通道最大值找到了
        for(int i = 0;i < L; ++i)
            res_ptr[C * i + c] = (res_ptr[C * i + c] / max_value) * 255;
    }
    result.convertTo(result, _type);
    return result;
}


cv::Mat reinchard(const cv::Mat& origin) {
    const int H = origin.rows;
    const int W = origin.cols;
    const int C = origin.channels();
    const int length = H * W;
    cv::Mat result(H, W, origin.type());
    float* const res_ptr = result.ptr<float>();
    const float* const ori_ptr = origin.ptr<float>();
    for(int i = 0;i < length; ++i)
        res_ptr[i] = ori_ptr[i] / (ori_ptr[i] + 1);
    return final_normalize(result, H * W, C, CV_8UC3);
}


cv::Mat gamma_correct(const cv::Mat& origin, const float gamma=0.4, std::unordered_set<int> channels={0, 1, 2}) {
    const int H = origin.rows;
    const int W = origin.cols;
    const int C = origin.channels();
    const int length = H * W;
    cv::Mat result(H, W, origin.type());
    float* const res_ptr = result.ptr<float>();
    const float* const ori_ptr = origin.ptr<float>();
    for(int c = 0;c < C; ++c) {
        if(channels.count(c)) {
            for(int i = 0;i < length; ++i) {
                res_ptr[C * i + c] = std::pow(ori_ptr[C * i + c], gamma);
            }
        }
    }
    return final_normalize(result, H * W, C, CV_8UC3);
}




int main() {
	// 读取图像
	cv::Mat hdr_image = cv::imread("./images/input/memorial.hdr", cv::IMREAD_ANYDEPTH);

    cv_show(hdr_image);

	// 基于滤波分解 base  detail 的方法
	if(true) {
	    // 直接处理
        auto collections = bilateral_local_tonemapping(hdr_image, 10);

        // 保存
        std::string save_dir("./images/output/memorial_");
        for(const auto& item : collections) {
            cv_show(item.second, item.first.c_str());
            cv_write(item.second, save_dir + item.first + ".png");
        }
	}

	// 直接gamma校正
	if(true) {
	    auto corrected = gamma_correct(hdr_image, 0.5);
        cv_show(corrected);
        cv_write(corrected, "./images/output/gamma_correction.png");
        // 先转成 Ycbcr 通道再做
        cv::Mat hdr_ycrcb;
        cv::cvtColor(hdr_image, hdr_ycrcb, cv::COLOR_BGR2YCrCb);
        auto result = gamma_correct(hdr_ycrcb, 0.5, {0});
        cv_show(result);
	}

	// Reinchard
	if(false) {
	    auto corrected = reinchard(hdr_image);
        cv_show(corrected);
        cv_write(corrected, "./images/output/reinchard.png");
	}
    return 0;
}











