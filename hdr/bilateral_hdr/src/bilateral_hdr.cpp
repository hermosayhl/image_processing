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


// ????????????
cv::Mat gaussi_filtering(const cv::Mat& origin, const float spatial_sigma=18) {
    // ??????????????????
    const int H = origin.rows;
    const int W = origin.cols;
    const int C = origin.channels();
    assert(C == 1 and "only images of single channel is supported !");
    // ?????????????????????
    const int radius = int(3 * spatial_sigma);
    const int window_size = square(2 * radius + 1);
    // ???????????? padding
    const auto padded_image = make_pad(origin, radius, radius);
    const int W2 = W + 2 * radius;
    // ????????????????????????
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
    // ??????????????????
    cv::Mat result(H, W, CV_32F);
    float* const res_ptr = result.ptr<float>();
    int cnt = 0;
    // ??????????????????
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


// ????????????
cv::Mat bilateral_filtering(const cv::Mat& origin, const float range_sigma=0.4, const float spatial_sigma=18) {
    // ????????????
    const int H = origin.rows;
    const int W = origin.cols;
    assert(origin.channels() == 1);
    // ???????????????
    const int radius = int(3 * spatial_sigma);
    const int window_size = radius * 2 + 1;
    // ???????????? padding
    const auto padded_image = make_pad(origin, radius, radius);
    const int W2 = padded_image.cols;
    // ???????????????????????????(double ????????? fast_exp ????????????????????????)
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
	// ???????????????
	const float sigma_inv = 0.5f / (range_sigma * range_sigma);
	// ??????????????????
    cv::Mat result(H, W, CV_32F);
    float* const res_ptr = result.ptr<float>();
	// ????????????
	int cnt = 0; // ???????????????????????????
	for(int i = 0;i < H; ++i) {
	    // ??????????????????????????????, ??? pad ?????????????????????, ??? radiu + i ???, ?????? radius ?????????
	    const float* const pad_ptr = padded_image.ptr<float>() + (radius + i) * W2 + radius;
	    for(int j = 0;j < W; ++j) {
	        const float center = pad_ptr[j];
	        // ????????????
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
    // ??????????????????
    std::list<std::pair<std::string, cv::Mat> > collections;
    // ??????????????????
    const int H = hdr_image.rows;
    const int W = hdr_image.cols;
    const int C = hdr_image.channels();
    assert(C == 3 and "only BGR channels are supported!");
    const float hdr_min = std::max(_min(hdr_image.ptr<float>(), H * W * C), 1e-5f);
    const float hdr_max = _max(hdr_image.ptr<float>(), H * W * C);

    std::cout << "?????????????????????????????????????????? : \n";
    std::cout << "\theight = " << hdr_image.rows << "\n\twidth = " << hdr_image.cols << "\n";
    std::cout << "\tdepth =  " << hdr_image.type() << std::endl;
    std::cout << "\tMax = " << hdr_max << "\n\tMin = " << hdr_min << std::endl;
    std::cout << "\t???????????? = " << hdr_max / hdr_min << std::endl;

	// ?????? hdr ????????????
	const float* const hdr_ptr = hdr_image.ptr<float>();

	// ??????????????? intensity = (20 * R + 40 * G + 1 * B) / 61;
	const int length = H * W;
	cv::Mat intensity(H, W, CV_32F);
	float* const intensity_ptr = intensity.ptr<float>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        intensity_ptr[i] = (20 * hdr_ptr[p + 2] + 40 * hdr_ptr[p + 1] + hdr_ptr[p]) / 61.f;
    }
    collections.emplace_back("intensity", intensity);

    // ?????? log10(intensity), ??? log ????????????,
    cv::Mat log_intensity(H, W, CV_32F);
    float* const log_intensity_ptr = log_intensity.ptr<float>();
    for(int i = 0;i < length; ++i)
        log_intensity_ptr[i] = std::log10(intensity_ptr[i]);
    collections.emplace_back("intensity_log", log_intensity);

    // ??? log_intensity ???????????????, ???????????????????????????(base ???)
    const float range_sigma = 0.4;
    const float spatial_sigma = 0.02f * std::min(H, W);
    std::cout << "??????????????? = " << range_sigma << "\n??????????????? = " << spatial_sigma << std::endl;
    auto log_base = bilateral_filtering(log_intensity, range_sigma, spatial_sigma);
//    auto log_base = gaussi_filtering(log_intensity, spatial_sigma);
//    auto log_base = guided_filter_with_gray(log_intensity, log_intensity, 3 * spatial_sigma, 3 * spatial_sigma, 0.1);

    // ??? log_detail, ??? log ????????? - ????????????????????????(base ???) = ?????????(log)
    cv::Mat log_detail = log_intensity - log_base;
    collections.emplace_back("base", log_base);
    collections.emplace_back("detail", log_detail);

    // ?????? base ???????????????
    // ????????? 1.0 * base + 1.0 * detail
    // ?????????????????? 0.2 * base + 1.0 * detail
    const float log_base_max = _max(log_base.ptr<float>(), length);
    const float log_base_min = _min(log_base.ptr<float>(), length);
    const float factor = std::log10(contrast_value) / (log_base_max - log_base_min);
    std::cout << "Base ??????????????????????????? = " << factor << std::endl;

    cv::Mat log_fusion = factor * log_base + log_detail;
    float* const fusion_ptr = log_fusion.ptr<float>();
    for(int i = 0;i < length; ++i)
        fusion_ptr[i] = std::pow(10.0, fusion_ptr[i]);

    // ??????????????????, ?????????, ?????? float ??????
    cv::Mat result(H, W, CV_32FC3);
    float* const result_ptr = result.ptr<float>();

    // ?????????????????????????????? intensity ???????????????????????????, ?????? R, G, B ???????????????
    for(int i = 0;i < length; ++i) {
        const float ratio = fusion_ptr[i] / intensity_ptr[i];
        const int pos = 3 * i;  // ????????????????????????
        result_ptr[pos + 2] = hdr_ptr[pos + 2] * ratio;
        result_ptr[pos + 1] = hdr_ptr[pos + 1] * ratio;
        result_ptr[pos] = hdr_ptr[pos] * ratio;
    }
    collections.emplace_back("compressed", result.clone());

    // ??????????????????????????????, ????????????????????????

    // ????????????????????????
    const float new_hdr_max = _max(result_ptr, length * 3);
    const float new_hdr_min = std::max(1e-5f, _min(result_ptr, length * 3));
    std::cout << "??????????????????????????? = " << new_hdr_max / new_hdr_min << std::endl;

    // ??????????????????, ????????????????????? 0-1 ?????? 0-255, ?????????????????????
    const float max_scale = std::pow(10.f, log_base_max * factor);
    // ????????????
    auto normalize = [max_scale](float x) -> float {
        return clip(255 * x / max_scale, 0, 255);
    };
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        result_ptr[p + 2] = normalize(result_ptr[p + 2]);
        result_ptr[p + 1] = normalize(result_ptr[p + 1]);
        result_ptr[p] = normalize(result_ptr[p]);
    }
    // ????????? float -> uchar,
    result.convertTo(result, CV_8UC3);
    collections.emplace_back("result", result);
    return collections;
}



cv::Mat final_normalize(cv::Mat& result, const int L, const int C, const int _type) {
    float* const res_ptr = result.ptr<float>();
    for(int c = 0;c < C; ++c) {
        // ??????????????????????????????
        float max_value = res_ptr[c];
        for(int i = 1;i < L; ++i)
            if(max_value < res_ptr[C * i + c])
                max_value = res_ptr[C * i + c];
        // ??????????????????????????????
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
	// ????????????
	cv::Mat hdr_image = cv::imread("./images/input/vinesunset_2.hdr", cv::IMREAD_ANYDEPTH);

    cv_show(hdr_image);

	// ?????????????????? base  detail ?????????
	if(true) {
	    // ????????????
        auto collections = bilateral_local_tonemapping(hdr_image, 10);

        // ??????
        std::string save_dir("./images/output/vinesunset_");
        for(const auto& item : collections) {
            cv_show(item.second, item.first.c_str());
            cv_write(item.second, save_dir + item.first + ".png");
        }
	}

	// ??????gamma??????
	if(false) {
	    auto corrected = gamma_correct(hdr_image, 0.5);
        cv_show(corrected);
        cv_write(corrected, "./images/output/gamma_correction.png");
        // ????????? Ycbcr ????????????
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











