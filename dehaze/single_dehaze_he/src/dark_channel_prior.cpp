// C++
#include <assert.h>
#include <map>
#include <list>
#include <vector>
#include <iostream>
#include <algorithm>
// self
#include "dark_channel_prior.h"
#include "guided_filter.h"

namespace {
    cv::Mat make_pad(const cv::Mat &one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    void cv_show(const cv::Mat &one_image, const char *info = "") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat double2uchar(const std::vector<double>& double_image, const int H, const int W) {
        cv::Mat origin = cv::Mat::zeros(H, W, CV_8UC1);
        const int length = H * W;
        for(int i = 0;i < length; ++i)
            origin.data[i] = cv::saturate_cast<uchar>(255 * double_image[i]);
        return origin;
    }

    std::vector<double> uchar2double(const cv::Mat& origin) {
        const int length = origin.rows * origin.cols;
        std::vector<double> double_image(length, 0);
        for(int i = 0;i < length; ++i) double_image[i] = (double)origin.data[i] / 255;
        return double_image;
    }

    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }
}



// double 图找暗通道
std::vector<double> get_dark_channel_double(const cv::Mat& I, const std::vector<double>& A, const double t0, const int radius, const bool accelerate=false) {
    // 准备一些变量
    const int H = I.rows, W = I.cols;
    const int H2 = H + 2 * radius, W2 = W + 2 * radius;
    const int length = H * W;
    const int length2 = H2 * W2;
    const int kernel_size = (radius << 1) + 1;
    // 在这里就先把 I padded 了再说, 为找最小值方便
    const auto padded_I = make_pad(I, radius, radius);
    const uchar* const I_ptr = padded_I.data;
    // 暗通道
    std::vector<double> dark_channel(length2, 0);
    // 首先, 对于图像中每个像素, 我要比较一下 3 个通道, 记录每个像素在 3 个通道除以对应的 A 中的最小值
    for(int i = 0;i < length2; ++i) {
        const int _beg = 3 * i;
        double min_pixel = (double)I_ptr[_beg] / A[0];
        const double g_min = (double)I_ptr[_beg + 1] / A[1];
        const double r_min = (double)I_ptr[_beg + 2] / A[2];
        if(min_pixel > g_min) min_pixel = g_min;
        if(min_pixel > r_min) min_pixel = r_min;
        dark_channel[i] = min_pixel;
    }
    // 准备返回结果
    std::vector<double> min_dark_channel(length, 0);
    // 准备对 dark_channel 做最小值滤波 (这里还没写加速的)
    if(not accelerate) {
        for(int i = 0;i < H; ++i) {
            double* const min_row_ptr = min_dark_channel.data() + i * W;
            for(int j = 0;j < W; ++j) {
                double min_pixel = 1e7;
                // 每个点, 找出它所在的 kernel 所有元素, 然后逐一比较
                for(int x = 0;x < kernel_size; ++x) {
                    // 获取 padded_I 中 kernel 这一行的指针
                    double* const row_ptr = dark_channel.data() + (i + x) * W2 + j;
                    for(int y = 0;y < kernel_size; ++y) if(min_pixel > row_ptr[y]) min_pixel = row_ptr[y];
                }
                min_row_ptr[j] = std::max(1 - min_pixel, t0);
            }
        }
    } else {
        // 这里分两个方向, 有点麻烦, 还得我自己 padding, 以后有时间再做吧
    }
    return min_dark_channel;
}


//  uchar 图找暗通道
cv::Mat get_dark_channel(const cv::Mat& I, const int radius, const bool accelerate) {
    // 准备一些变量
    const uchar* const I_ptr = I.data;
    const int H = I.rows, W = I.cols;
    // 暗通道
    cv::Mat dark_channel = cv::Mat::zeros(H, W, CV_8UC1);
    // 首先, 对于图像中每个像素, 我要比较一下 3 个通道, 记录每个像素在 3 个通道中的最小值
    const int length = H * W;
    for(int i = 0;i < length; ++i) {
        const int _beg = 3 * i;
        uchar min_pixel = I_ptr[_beg];
        if(min_pixel > I_ptr[_beg + 1]) min_pixel = I_ptr[_beg + 1];
        if(min_pixel > I_ptr[_beg + 2]) min_pixel = I_ptr[_beg + 2];
        dark_channel.data[i] = min_pixel;
    }
    auto min_dark_channel = dark_channel.clone();
    cv::Mat temp;
    // 存储第一次横向最小值滤波的结果
    if(accelerate) temp = dark_channel.clone();
    // 然后, 对这个暗通道执行区域最小值滤波
    dark_channel = make_pad(dark_channel, radius, radius);
    const int W2 = dark_channel.cols;
    const int kernel_size = (radius << 1) + 1;
    // 如果最小值滤波采用加速的话(还有更更快的写法可以参考 http://blog.sina.com.cn/s/blog_7889f9830101ew9a.html)
    if(accelerate) {
        // 先横向扫一遍
        for(int i = 0;i < H; ++i) {
            uchar* const temp_ptr = temp.data + i * W;
            for(int j = 0;j < W; ++j) {
                // 每个点, 先找到最开始的指针(这里面加入 if else 会变得更慢, 不划算, 维护一个 list, 然后每次比较最小值和 Next_uchar)
                uchar* const row_ptr = dark_channel.data + (radius + i) * W2 + j;
                uchar min_pixel = row_ptr[0];
                for(int t = 1;t < kernel_size; ++t) if(min_pixel > row_ptr[t]) min_pixel = row_ptr[t];
                temp_ptr[j] = min_pixel;
            }
        }
        // 把空填好
        temp = make_pad(temp, radius, radius);
        // 在前面的的基础上再纵向扫一遍
        for(int j = 0;j < W; ++j) {
            for(int i = 0;i < H; ++i) {
                // 每个点, 开始扫一遍
                uchar min_pixel = 255;
                for(int t = 1, _beg = i * W2 + radius + j; t < kernel_size; ++t, _beg += W2)
                    if(min_pixel > temp.data[_beg]) min_pixel = temp.data[_beg];
                min_dark_channel.data[i * W + j] = min_pixel;
            }
        }
    } else { // 采用暴力的方法最小值滤波
        for(int i = 0;i < H; ++i) {
            uchar* const min_row_ptr = min_dark_channel.data + i * W;
            for(int j = 0;j < W; ++j) {
                uchar min_pixel = 255;
                // 每个点, 找出 kernel, 然后逐一比较
                for(int x = 0;x < kernel_size; ++x) {
                    // 获取 kernel 这一行的指针
                    uchar* const row_ptr = dark_channel.data + (i + x) * W2 + j;
                    for(int y = 0;y < kernel_size; ++y) if(min_pixel > row_ptr[y]) min_pixel = row_ptr[y];
                }
                min_row_ptr[j] = min_pixel;
            }
        }
    }
    return min_dark_channel;
}


std::pair< std::vector<double>, std::vector<int> > get_global_atmospheric_light(
        const cv::Mat& I, const cv::Mat& I_dark, const double top_percent,
        const std::string sort_type="bucket_sort") {
    // 从 dark_channel 中, 选像素值前 0.1%, 首先要把所有点看一遍 ? 排序一把, 但这种其实可以用桶排序加速
    const uchar* const dark_ptr = I_dark.data;
    const uchar* const I_ptr = I.data;
    const int H = I.rows, W = I.cols;
    const int length = H * W;
    // 前 top_percent 的点
    const int border = top_percent * length;
    std::vector<int> max_n(border, 0);
    // 现在 max_k 里保存了前 0.01 最大值的位置, 返回一个值还是三个值 ?
    std::vector<double> max_A(3, 0);
    // 如果用的桶排序
    if(sort_type == "bucket_sort") {
        // 遍历 dark_channel, 记录每一个像素出现的位置
        std::vector<int> book(256, 0);
        std::vector<int> pos[256];
        for(int i = 0;i < length; ++i) {
            const int pixel = dark_ptr[i];
            ++book[pixel];
            pos[pixel].emplace_back(i);
        }
        // 统计完毕, 现在筛选出前 0.01 的 dark channel 的点的位置
        int cnt = 0, ok = 0;
        for(int i = 255; i >= 0; --i) {
            const int frequency = book[i];
            for(int j = 0;j < frequency; ++j) {
                max_n[cnt++] = pos[i][j];
                if(cnt == border - 1) { ok = 1; break; }
            }
            if(ok) break;
        }
        // 求这 0.1% 的最大的点在原图中的像素平均值, 这里的加法可能会溢出, 换小数会好点, 但是慢
        for(int i = 0;i < border; ++i) {
            // 对应 color 图像中的点
            const int p = 3 * max_n[i];
            max_A[0] += I_ptr[p];
            max_A[1] += I_ptr[p + 1];
            max_A[2] += I_ptr[p + 2];
        }
    } else if(sort_type == "quick_sort") {
        // 如果用的是快速排序 + 堆排序 + 插入排序
        std::vector< std::pair<uchar, int> > pixels(length);
        for(int i = 0;i < length; ++i)
            pixels[i].first = dark_ptr[i], pixels[i].second = i;
        std::sort(pixels.begin(), pixels.end());
        // 现在选出前 border 个元素
        for(int i = 0;i < border; ++i) {
            const int p = 3 * pixels[length - 1 - i].second;
            max_A[0] += I_ptr[p];
            max_A[1] += I_ptr[p + 1];
            max_A[2] += I_ptr[p + 2];
        }
        // 记录前 border 个元素在的地方
        for(int i = 0;i < border; ++i) max_n[i] = pixels[length - 1 - i].second;
    }
    for(int i = 0;i < 3; ++i) max_A[i] /= border;
    return {max_A, max_n};
}


std::map<const std::string, cv::Mat> dark_channel_prior_dehaze(
        const cv::Mat& I, const int radius, const double top_percent, const double t0, const double omega, const bool guided, const bool multi_T, const bool return_visuals) {
    // 获取图像信息
    const int H = I.rows;
    const int W = I.cols;
    const int C = I.channels();
    assert(C == 3);
    const int length = H * W;
    // 首先求 I 的暗通道图 I_dark  8ms
    const auto I_dark = get_dark_channel(I, radius, false);
    // 然后求全局大气光 A, 一个数, 需要汇总前 top_percent 的像素, 然后求出来  12ms
    const auto A_results = get_global_atmospheric_light(I, I_dark, top_percent);
    const auto& A = A_results.first;
    // 准备返回结果
    std::map<const std::string, cv::Mat> packed_results{{"dark_channel", I_dark}};
    // 在这里画出来
    if(return_visuals) {
        auto I_circled = I.clone();
        for(const auto pos : A_results.second) {
            const int x = pos / W, y = pos % W;
            cv::circle(I_circled, cv::Point(y, x), 4, CV_RGB(255, 0, 0), 0);
        }
        packed_results.emplace("A_points", I_circled);
    }
    // 如果三通道的透射率分开计算
    if(not multi_T) {
        // 原图 I 三通道分别除以 A, 再一起求一个 dark_channel
        auto T = get_dark_channel_double(I, A, t0, radius);
        packed_results.emplace("T", double2uchar(T, H, W));
        // 是否要 guided_filter 精修下 ?
        if(guided) {
            cv::Mat gray_dehaze_image;
            cv::cvtColor(I, gray_dehaze_image, cv::COLOR_BGR2GRAY);
            const auto T_guided = guided_filter_with_gray(packed_results["T"], gray_dehaze_image, 12, 12, 1e-4);
            T = uchar2double(T_guided); // 这个不能删
            if(return_visuals) packed_results.emplace("T_guided", T_guided);
        }
        // A, T, I 都已经知道了, 现在开始求 J(x) = (I(x) - A) / T(x) + A
        auto J = I.clone();
        uchar* const dehazed_ptr = J.data;
        for(int i = 0;i < length; ++i) {
            const int _beg = 3 * i;
            dehazed_ptr[_beg] = cv::saturate_cast<uchar>(double(I.data[_beg] - A[0]) / T[i] + A[0]);
            dehazed_ptr[_beg + 1] = cv::saturate_cast<uchar>(double(I.data[_beg + 1] - A[1]) / T[i] + A[1]);
            dehazed_ptr[_beg + 2] = cv::saturate_cast<uchar>(double(I.data[_beg + 2] - A[2]) / T[i] + A[2]);
        }
        packed_results.emplace("dehazed", J);
    } else {
        // 现在我求出了三个通道的 A, 准备分别求三个通道的折射率 T = 1 - I_dark / A
        std::vector< std::vector<double> > T(3, std::vector<double>(length));
        const uchar* const I_dark_ptr = I_dark.data;
        for(int ch = 0;ch < 3; ++ch) {
            double* const cur_T = T[ch].data();
            for(int i = 0;i < length; ++i)
                cur_T[i] = 1 - omega * double(I_dark_ptr[i] / A[ch]);// T(x) = 1 - I_dark(x) / A
            for(int i = 0;i < length; ++i)
                if(cur_T[i] < t0) cur_T[i] = t0;// J(x) = (I(x) - A) / max[T(x), t0] + A
        }
        std::vector<cv::Mat> T_uchar;
        // 是否要返回 T 的可视化结果
        T_uchar = {double2uchar(T[0], H, W), double2uchar(T[1], H, W), double2uchar(T[2], H, W)};
        for(int i = 0;i < 3; ++i) packed_results.emplace("T_" + std::to_string(i), T_uchar[i]);
        // 是否要对投射图做精修
        if(guided) {
            cv::Mat gray_dehaze_image;
            cv::cvtColor(I, gray_dehaze_image, cv::COLOR_BGR2GRAY);
            std::vector<cv::Mat> T_guided;
            T_guided.emplace_back(guided_filter_with_gray(T_uchar[0], gray_dehaze_image, 12, 12, 1e-4));
            T_guided.emplace_back(guided_filter_with_gray(T_uchar[1], gray_dehaze_image, 12, 12, 1e-4));
            T_guided.emplace_back(guided_filter_with_gray(T_uchar[2], gray_dehaze_image, 12, 12, 1e-4));
            T = {uchar2double(T_guided[0]), uchar2double(T_guided[1]), uchar2double(T_guided[2])};
            // 添加一些中间结果
            if(return_visuals) for(int i = 0;i < 3; ++i) packed_results.emplace("T_" + std::to_string(i) + "_guided", T_guided[i]);
        }
        auto dehazed = I.clone();
        uchar* const dehazed_ptr = dehazed.data;
        for(int i = 0;i < length; ++i) {
            const int _beg = 3 * i;
            dehazed_ptr[_beg] = cv::saturate_cast<uchar>(double(I.data[_beg] - A[0]) / T[0][i] + A[0]);
            dehazed_ptr[_beg + 1] = cv::saturate_cast<uchar>(double(I.data[_beg + 1] - A[1]) / T[1][i] + A[1]);
            dehazed_ptr[_beg + 2] = cv::saturate_cast<uchar>(double(I.data[_beg + 2] - A[2]) / T[2][i] + A[2]);
        }
        packed_results.emplace("dehazed", dehazed);
    }
    return packed_results;
}
