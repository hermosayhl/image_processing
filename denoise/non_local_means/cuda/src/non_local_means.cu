// C++
#include <chrono>
#include <cstring>
#include <iostream>
// Boost
#include <boost/scope_exit.hpp>
// self
#include "non_local_means.h"
#include "cuda_helper.hpp"


namespace {
    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }

    // 测试时间
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }
}


std::vector<double> get_kernel(const int window_size, const char* kernel_type) {
    // 权重模板是均值的话
    if(std::strcmp(kernel_type, "mean") == 0)
        return std::vector<double> (window_size, 1. / (window_size));
    // 高斯模板
    else if(std::strcmp(kernel_type, "gaussi") == 0) {
        std::vector<double> weight_kernel(window_size, 0);
        int offset = -1;
        double kernel_weight_sum = 0.0;
        const int radius = (int(std::sqrt(window_size)) - 1) >> 1;
        // 半径应该是 3 sigma 差不多了
        const double variance = int((2 * radius + 1) / 3);
        const double variance_2 = -0.5 / (variance * variance);
        for(int i = -radius; i <= radius; ++i)
            for(int j = -radius; j <= radius; ++j) {
                weight_kernel[++offset] = std::exp(variance_2 * (i * i + j * j));
                kernel_weight_sum += weight_kernel[offset];
            }
        for(int i = 0;i < window_size; ++i) weight_kernel[i] /= kernel_weight_sum;
        return weight_kernel;
    }
    // 没声明的话, 返回全 0 模板
    else return std::vector<double>(window_size, 0);
}



// 在 GPU 上执行的程序, 因为核函数上没法用 opencv 和 STL 函数, 得自己写
// __device__ 
__device__ uchar cv_round(const double data) {
    int temp = int(data);
    if(temp < 0) temp = 0;
    else if(temp > 255) temp = 255;
    return (uchar)temp;
}



// 每个 kernel 上运行的程序
__global__ void non_local_means_cuda_kernel(
        const uchar* const cuda_in, 
        uchar* const cuda_out, 
        const int H, const int W, 
        const int H2, const int W2, 
        const int search_radius, const int radius, 
        const double sigma_2_inv,
        const double* const weights_kernel_ptr) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < H and j < W) {
        // 当前要去噪的点 (i, j), 以它为中心的区域的点, 我得收集起来
        const int window_len = 2 * radius + 1;
        // 累计值 和 权重总和
        double sum_value = 0;
        double weight_sum = 0;
        double weight_max = -1e3;
        // 确定搜索范围
        const int up = radius > i - search_radius ? radius : i - search_radius;
        const int down = H2 < i + search_radius ? H2: i + search_radius;
        const int left = radius > j - search_radius ? radius: j - search_radius;
        const int right = W2 < j + search_radius ? W2: j + search_radius;
        // 在范围内搜索
        for(int x = up; x < down; ++x) {
            const uchar* row_ptr = cuda_in + x * W2;
            for(int y = left; y < right; ++y) {
                // (i, j) 是相对于原图来说的位置, (x, y) 是相对于 padded 之后的图像来说的
                // 如果碰到自己了, 不计算
                if(x == i and y == j)
                    continue;
                // 然后计算两个区域的相似度
                double distance = 0.0;
                for(int t = 0;t < window_len; ++t) {
                    const uchar* const source = cuda_in + (i + t) * W2 + j;
                    const uchar* const target = cuda_in + (x - radius + t) * W2 + y - radius;
                    const double* const weight_row = weights_kernel_ptr + t * window_len;
                    for(int k = 0;k < window_len; ++k) {
                        double res = (double)(target[k] - source[k]);
                        distance += weight_row[k] * (res * res);
                    }
                }
                const double cur_weight = exp(-distance * sigma_2_inv);
                // 记录当前最大的权值
                if(cur_weight > weight_max) weight_max = cur_weight;
                // 累加值
                sum_value += cur_weight * row_ptr[y];
                weight_sum += cur_weight;
            }
        }
        // 搜索结束
        sum_value += weight_max * cuda_in[(i + radius) * W2 + j + radius];
        weight_sum += weight_max;
        cuda_out[i * W + j] = cv_round(sum_value / weight_sum);
    }
}



cv::Mat non_local_means_gray(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type) {
    
    // 先做一个计算领域相似性的权重模板, 先来最简单的均值模板
    const int window_len = (radius << 1) + 1;
    const int window_size = window_len * window_len;
    const auto weights_kernel = get_kernel(window_size, kernel_type);
    const double sigma_2_inv = 1. / (sigma * sigma);
    // 收集目标图像的信息
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    // 将图像 padding 一下
    const auto padded_image = make_pad(noise_image, radius, radius);
    const int H2 = padded_image.rows;
    const int W2 = padded_image.cols;
    const uchar* const padded_ptr = padded_image.data;

    // 定义在 GPU 上运算的数据, 一个权重模板, 一个输入图像指针, 一个输出图像指针
    double *weights_kernel_ptr;
    uchar *cuda_in, *cuda_out;

    // 分配显存
    const int origin_image_size = H * W * sizeof(uchar);
    const int padded_image_size = H2 * W2 * sizeof(uchar);
    crane::CudaSafeCall(cudaMalloc((void**)&cuda_out, origin_image_size));
    crane::CudaSafeCall(cudaMalloc((void**)&cuda_in, padded_image_size));
    crane::CudaSafeCall(cudaMalloc((void**)&weights_kernel_ptr, window_size * sizeof(double)));

    // 返回之前, RAII 释放 GPU 分配的内存
    BOOST_SCOPE_EXIT_ALL(&) {
        cudaFree(cuda_in);
        cudaFree(cuda_out);
        cudaFree(weights_kernel_ptr);
        cudaDeviceReset();
        std::cout << "GPU 上的申请的空间已 free!" << std::endl;
    };

    // 把 cpu 内存的数据拷贝到 cuda 对应内存中
    cudaMemcpy(cuda_in, padded_ptr, padded_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(weights_kernel_ptr, &weights_kernel[0], window_size * sizeof(double), cudaMemcpyHostToDevice);

    //每个线程处理一个像素(32 不行)
    constexpr int ts = 16;
    dim3 blockSize(ts, ts);
    dim3 gridSize((W + ts - 1) / ts, (H + ts - 1) / ts);

    run([&](){
        // 启动内核
        non_local_means_cuda_kernel<<<gridSize, blockSize>>>(
            cuda_in, cuda_out, H, W, H2, W2, search_radius, radius, sigma_2_inv, weights_kernel_ptr);

        // 执行内核是一个异步操作，因此需要同步以测量准确时间
        cudaDeviceSynchronize();
        crane::CudaCheckError();
    }, "cuda 上计算的时间");

    // 结果图像
    cv::Mat denoised = noise_image.clone();
    // 数据从 GPU 拷贝到 CPU, 填充图像
    cudaMemcpy(denoised.data, cuda_out, origin_image_size, cudaMemcpyDeviceToHost);

    return denoised;
}




// 搜索窗口大小 11x11, 邻域 5x5
cv::Mat non_local_means(const cv::Mat& noise_image, const int search_radius, const int radius, const int sigma, const char* kernel_type, const bool fast) {
    const int C = noise_image.channels();
    // 灰度图
    if(C == 1) return non_local_means_gray(noise_image, search_radius, radius, sigma, kernel_type);
    // 彩色图的 non_local_means
    else if(C == 3) {
        std::vector<cv::Mat> bgr_channels;
        cv::split(noise_image, bgr_channels);
        std::vector<cv::Mat> denoised_channels;
        for(const auto & ch : bgr_channels)
            denoised_channels.emplace_back(non_local_means_gray(ch, search_radius, radius, sigma, kernel_type));
        cv::Mat denoised;
        cv::merge(denoised_channels, denoised);
        return denoised;
    }
    return noise_image;
}
