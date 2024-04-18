// CUDA
#include "cuda_helper.hpp"
// C
#ifdef __unix__
    #include <unistd.h>
#elif defined(_WIN32) || defined(WIN32)
    #include <direct.h>
#endif
// C++
#include <assert.h>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <functional>
#include <algorithm>
// Boost
#include <boost/scope_exit.hpp>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>




namespace {

    inline void get_dir() {
        constexpr int buffer_size = 256;
        char buffer[buffer_size];
    #ifdef __unix__
        if(getcwd(buffer, buffer_size) != NULL) {
            std::cout << "Unix...当前工作路径 " << buffer << std::endl;
        } else std::cout << "获取工作路径失败 !" << std::endl;
    #elif defined(_WIN32) || defined(WIN32)
        _getcwd(buffer, buffer_size);
        std::cout << "Windows...当前工作路径 " << buffer << std::endl;
    #endif
    }

    void show(const cv::Mat& image, const std::string& window_name) {
        cv::imshow(window_name, image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::string string_replace(const std::string& origin_path, const char* src, const char* des) {
        std::string result(origin_path.c_str());
        const int pos = result.find(src);
        if(pos != -1)
            result.replace(pos, std::strlen(src), des);
        return result;
    }

    // 计算运行时间
    void run(const std::function<void()>& work=[]{}, const std::string message="") {
        auto start = std::chrono::steady_clock::now();
        work();
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
        std::cout << message << " " << duration.count() << " ms" <<  std::endl;
    }

    // 依据 typeid 输出变量数据类型
    template<typename T>
    void print_datatype(T var_) {
        std::cout << typeid(var_).name() << std::endl;
    }

    // 做填充
    cv::Mat make_pad(const cv::Mat& source, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(source, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }


    // 检查 GPU 情况
    void check_device() {
        int dev = 0;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, dev);
        std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    }
}



// 在 GPU 上执行的程序, 因为核函数上没法用 opencv 和 STL 函数, 得自己写
__device__ uchar cv_round(const double data) {
    int temp = int(data);
    if(temp < 0) temp = 0;
    else if(temp > 255) temp = 255;
    return (uchar)temp;
}




__global__ void bilateral_filter_cuda_kernel(
        uchar* padded_image, uchar* cuda_out,
        const int H, const int W,
        const int  C, const int pad_size,
        const int padding_step, const int result_step,
        const int maxk,
        double* value_ptr, double* space_ptr, int* offset_ptr) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < H and j < W) {

        // 从有效的滤波中心开始算, pad_size 行空的, 加上现在是有效图像第 i 行的数据, 当前第 i 行处在 pad_size 位置
        const uchar* row_ptr = padded_image + (i + pad_size) * padding_step + pad_size * C;
        uchar* result_row_ptr = cuda_out + i * result_step;

        double sum_b = 0, sum_g = 0, sum_r = 0;
        double norm_b = 0, norm_g = 0, norm_r = 0;
        // 中心像素的下标是 J
        const int J = j * 3;
        const int b = row_ptr[J];
        const int g = row_ptr[J + 1];
        const int r = row_ptr[J + 2];
        // 遍历窗口(滤波核) 主要计算量就在这里
        for(int k = 0;k < maxk; ++k) {
            const uchar* cur_ptr = row_ptr + J + offset_ptr[k];

            const int __b = cur_ptr[0];
            const double w_b = space_ptr[k] * value_ptr[std::abs(b - __b)];
            sum_b += __b * w_b;
            norm_b += w_b;

            const int __g = cur_ptr[1];
            const double w_g = space_ptr[k] * value_ptr[std::abs(g - __g)];
            sum_g += __g * w_g;
            norm_g += w_g;

            const int __r = cur_ptr[2];
            const double w_r = space_ptr[k] * value_ptr[std::abs(r - __r)];
            sum_r += __r * w_r;
            norm_r += w_r;
        }
        // 更新到结果图像上
         result_row_ptr[J] = cv_round(sum_b / norm_b);
         result_row_ptr[J + 1] = cv_round(sum_g / norm_g);
         result_row_ptr[J + 2] = cv_round(sum_r / norm_r);
    }
}






// 双边滤波 cuda 版
cv::Mat bilateral_filter_cuda(
        const cv::Mat& noise_image, const int window_size,
        const double value_variance, const double space_variance) {
    // 检查合理性
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int C = noise_image.channels();
    const int pad_size = (window_size - 1) >> 1;
    // 对边缘填充
    const auto padded_image = make_pad(noise_image, pad_size, pad_size);

    // 获取 range filter 模板
    const int value_range = C * 256;
    std::vector<double> value_table(value_range);
    const double value_variance_2 = 1.0 / (value_variance * value_variance);
    for(int i = 0;i < value_range; ++i)
        value_table[i] = std::exp(- 0.5 * value_variance_2 * i * i);

    // 获取 distance filter 模板
    const int half_size = (window_size - 1) >> 1;
    const int space_size = window_size * window_size;
    std::vector<double> space_table(space_size);
    std::vector<int> space_offset(space_size);
    int maxk = 0;
    const double space_variance_2 = - 0.5 / (space_variance * space_variance);
    // (以最中心的点为参照点, 计算其他点的空间权重, 以及在图像中的像素偏移值)
    for(int i = -half_size;i <= half_size; ++i) {
        for(int j = -half_size;j <= half_size; ++j) {
            space_table[maxk] = std::exp(space_variance_2 * (i * i + j * j));
            space_offset[maxk] = i * padded_image.step + j * C;
            ++maxk;
        }
    }

    // 查看 GPU
    check_device();

    // 定义在 GPU 上运算的数据
    double *value_ptr, *space_ptr;
    int * offset_ptr;
    uchar *cuda_in, *cuda_out;

    // 在 GPU 分配内存(我吐了, GPU 越界访问不报错, 就是不执行)
    // 这里最好做一下错误检查
    const int padded_image_size = padded_image.rows * padded_image.cols * C * sizeof(uchar);
    const int cuda_memory_size = H * W * C * sizeof(uchar);
    yhl::CudaSafeCall(cudaMalloc((void**)&cuda_in, padded_image_size));
    yhl::CudaSafeCall(cudaMalloc((void**)&cuda_out, cuda_memory_size));
    yhl::CudaSafeCall(cudaMalloc((void**)&value_ptr, value_range * sizeof(double)));
    yhl::CudaSafeCall(cudaMalloc((void**)&space_ptr, space_size * sizeof(double)));
    yhl::CudaSafeCall(cudaMalloc((void**)&offset_ptr, space_size * sizeof(int)));

    // 释放 GPU 分配的内存
    BOOST_SCOPE_EXIT_ALL(&) {
        cudaFree(cuda_in);
        cudaFree(cuda_out);
        cudaFree(value_ptr);
        cudaFree(space_ptr);
        cudaFree(offset_ptr);
        cudaDeviceReset();
        std::cout << "GPU 上的申请的空间已 free!" << std::endl;
    };

    // 把图像, 表数据拷贝到 GPU 对应的位置上
    // 这里尤其要注意, padded_image 的数据不是原始数据那么多了, 因为 padded 了, 所以一开始在 cuda_in 那里也要注意
    cudaMemcpy(cuda_in, padded_image.data, padded_image_size, cudaMemcpyHostToDevice);
    yhl::CudaCheckError();
    cudaMemcpy(value_ptr, &value_table[0], value_range * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(space_ptr, &space_table[0], space_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(offset_ptr, &space_offset[0], space_size * sizeof(int), cudaMemcpyHostToDevice);

    //每个线程处理一个像素(32 不行)
    constexpr int ts = 16;
    dim3 blockSize(ts, ts);
    dim3 gridSize((W + ts - 1) / ts, (H + ts - 1) / ts);

    // 启动内核
    bilateral_filter_cuda_kernel<<<gridSize, blockSize>>>(
        cuda_in, cuda_out, H, W, C, pad_size, padded_image.step, noise_image.step, maxk, value_ptr, space_ptr, offset_ptr);

    // 执行内核是一个异步操作，因此需要同步以测量准确时间
    // cudaDeviceSynchronize();
    yhl::CudaCheckError();

    // 结果图像
    auto result = noise_image.clone();
    // 数据从 GPU 拷贝到 CPU, 填充图像
    cudaMemcpy(result.data, cuda_out, cuda_memory_size, cudaMemcpyDeviceToHost);

    // 返回结果, 注意这里要改掉
    return result;
}













// 双边滤波
cv::Mat bilateral_filter_cpu(
        const cv::Mat& noise_image, const int window_size,
        const double value_variance, const double space_variance) {
    // 检查合理性
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int C = noise_image.channels();
    const int pad_size = (window_size - 1) >> 1;
    // 对边缘填充
    const auto padded_image = make_pad(noise_image, pad_size, pad_size);

    // 结果图像
    auto result = noise_image.clone();

    // 获取 range filter 模板
    const int value_range = C * 256;
    std::vector<double> value_table(value_range);
    const double value_variance_2 = 1.0 / (value_variance * value_variance);
    for(int i = 0;i < value_range; ++i)
        value_table[i] = std::exp(- 0.5 * value_variance_2 * i * i);

    // 获取 distance filter 模板
    const int half_size = (window_size - 1) >> 1;
    const int space_size = window_size * window_size;
    std::vector<double> space_table(space_size);
    std::vector<int> space_offset(space_size);
    int maxk = 0;
    const double space_variance_2 = - 0.5 / (space_variance * space_variance);
    // (以最中心的点为参照点, 计算其他点的空间权重, 以及在图像中的像素偏移值)
    for(int i = -half_size;i <= half_size; ++i) {
        for(int j = -half_size;j <= half_size; ++j) {
            space_table[maxk] = std::exp(space_variance_2 * (i * i + j * j));
            space_offset[maxk] = i * padded_image.step + j * C;
            ++maxk;
        }
    }

    // 只是把 vector 换成指针的形式, 就这样 ? iterater 真的慢
    double* value_ptr = &value_table[0];
    double* space_ptr = &space_table[0];
    int *offset_ptr = &space_offset[0];

    // 遍历图像中每一个点, 然后滤波
    for(int i = 0;i < H; ++i) {
        // 从有效的滤波中心开始算, pad_size 行空的, 加上现在是有效图像第 i 行的数据, 当前第 i 行处在 pad_size 位置
        const uchar* row_ptr = padded_image.data + (i + pad_size) * padded_image.step + pad_size * C;
        uchar* result_row_ptr = result.data + i * result.step;
        for(int j = 0;j < W; ++j) {
            double sum_b = 0, sum_g = 0, sum_r = 0;
            double norm_b = 0, norm_g = 0, norm_r = 0;
            // 中心像素的下标是 j
            const int J = j * 3;
            const int b = row_ptr[J];
            const int g = row_ptr[J + 1];
            const int r = row_ptr[J + 2];
            // 遍历窗口(滤波核) 主要计算量就在这里
            for(int k = 0;k < maxk; ++k) {
                const uchar* cur_ptr = row_ptr + J + offset_ptr[k];

                const int __b = cur_ptr[0];
                const double w_b = space_ptr[k] * value_ptr[std::abs(b - __b)];
                sum_b += __b * w_b;
                norm_b += w_b;

                const int __g = cur_ptr[1];
                const double w_g = space_ptr[k] * value_ptr[std::abs(g - __g)];
                sum_g += __g * w_g;
                norm_g += w_g;

                const int __r = cur_ptr[2];
                const double w_r = space_ptr[k] * value_ptr[std::abs(r - __r)];
                sum_r += __r * w_r;
                norm_r += w_r;
            }
            // 更新到结果图像上
            result_row_ptr[J] = cv::saturate_cast<uchar>(sum_b / norm_b);
            result_row_ptr[J + 1] = cv::saturate_cast<uchar>(sum_g / norm_g);
            result_row_ptr[J + 2] = cv::saturate_cast<uchar>(sum_r / norm_r);
        }
    }
    return result;
}





// 顺便写一下高斯滤波
// 双边滤波
cv::Mat gaussi_filter_cpu(
        const cv::Mat& noise_image, const int window_size, const double space_variance) {
    // 检查合理性
    const int H = noise_image.rows;
    const int W = noise_image.cols;
    const int C = noise_image.channels();
    const int pad_size = (window_size - 1) >> 1;
    // std::cout << H << ", " << W << ", " << C << ", " << pad_size << "\n";
    // 对边缘填充
    const auto padded_image = make_pad(noise_image, pad_size, pad_size);
    // std::cout << "step  " << padded_image.step << std::endl;

    // 结果图像
    auto result = noise_image.clone();

    // 获取 distance filter 模板
    const int half_size = (window_size - 1) >> 1;
    const int space_size = window_size * window_size;
    std::vector<double> space_table(space_size);
    std::vector<int> space_offset(space_size);
    int maxk = 0;
    const double space_variance_2 = - 0.5 / (space_variance * space_variance);
    // (以最中心的点为参照点, 计算其他点的空间权重, 以及在图像中的像素偏移值)
    for(int i = -half_size;i <= half_size; ++i) {
        for(int j = -half_size;j <= half_size; ++j) {
            space_table[maxk] = std::exp(space_variance_2 * (i * i + j * j));
            space_offset[maxk] = i * padded_image.step + j * C;
            ++maxk;
        }
    }

    // 只是把 vector 换成指针的形式, 就这样 ? iterater 真的慢
    double* space_ptr = &space_table[0];
    int *offset_ptr = &space_offset[0];

    // 遍历图像中每一个点, 然后滤波
        for(int i = 0;i < H; ++i) {
        // 从有效的滤波中心开始算, pad_size 行空的, 加上现在是有效图像第 i 行的数据, 当前第 i 行处在 pad_size 位置
        const uchar* row_ptr = padded_image.data + (i + pad_size) * padded_image.step + pad_size * C;
        uchar* result_row_ptr = result.data + i * result.step;
        for(int j = 0;j < W * 3; j += 3) {
            double sum_b = 0, sum_g = 0, sum_r = 0;
            double norm_b = 0, norm_g = 0, norm_r = 0;
            // 遍历窗口(滤波核) 主要计算量就在这里
            for(int k = 0;k < maxk; ++k) {
                const uchar* cur_ptr = row_ptr + j + offset_ptr[k];

                const int __b = cur_ptr[0];
                const double w_b = space_ptr[k];
                sum_b += __b * w_b;
                norm_b += w_b;

                const int __g = cur_ptr[1];
                const double w_g = space_ptr[k];
                sum_g += __g * w_g;
                norm_g += w_g;

                const int __r = cur_ptr[2];
                const double w_r = space_ptr[k];
                sum_r += __r * w_r;
                norm_r += w_r;
            }
            // 更新到结果图像上
            result_row_ptr[j] = cv::saturate_cast<uchar>(sum_b / norm_b);
            result_row_ptr[j + 1] = cv::saturate_cast<uchar>(sum_g / norm_g);
            result_row_ptr[j + 2] = cv::saturate_cast<uchar>(sum_r / norm_r);
        }
    }
    return result;
}








int main() {
        // 获取可执行文件目录
    get_dir();
    // 读取原始图像
    const std::string image_path("./images/woman_1.png");
    auto noise_image = cv::imread(image_path);
    if(noise_image.empty()) {
        std::cout << "读取图像  " << image_path << "  失败 !" << std::endl;
        return 0;
    }
    // Resize
    // cv::resize(noise_image, noise_image, {224, 224});
    std::cout << "要处理的图片大小为  (" << noise_image.rows << ", " << noise_image.cols << ", " << noise_image.channels() << ")" << std::endl;


    // 保存结果
    cv::Mat opencv_result, bilateral_cuda_result, bilateral_cpu_result, gaussi_result;

    // 准备双边滤波的参数
    const int window_size = 23;
    const double value_variance = 10;
    const double space_variance = 10;
    // 自己写的在 GPU 上的双边滤波
    run([&](){
        bilateral_cuda_result = bilateral_filter_cuda(noise_image, window_size, value_variance, space_variance);
    }, "cuda  :  ");

    // 自己写的在 CPU 上的双边滤波
    run([&](){
        bilateral_cpu_result = bilateral_filter_cpu(noise_image, window_size, value_variance, space_variance);
    }, "cpu  :  ");

    // opencv 的双边滤波
    run([&](){
        cv::bilateralFilter(noise_image, opencv_result, window_size, value_variance, space_variance);
    }, "opencv  :  ");

    // 我自己写的 cpu 上的高斯滤波
    run([&](){
        gaussi_result = gaussi_filter_cpu(noise_image, window_size, space_variance);
    }, "gaussi  :  ");

    // 后续保存结果
    cv::imwrite(string_replace(image_path, ".png", "_bilateral_filter_cuda.png"), bilateral_cuda_result);
    cv::imwrite(string_replace(image_path, ".png", "_bilateral_filter_cpu.png"), bilateral_cpu_result);
    cv::imwrite(string_replace(image_path, ".png", "_gaussi_filter_cpu.png"), gaussi_result);

    // 检查 CPU 与 GPU 的双边滤波结果相差大不大
    const auto psnr_value = cv::PSNR(bilateral_cpu_result, bilateral_cuda_result);
    std::cout << "cpu VS gpu ===> " << psnr_value << " db" << std::endl;

    // 展示对比结果
    cv::Mat comparison = bilateral_cuda_result;
    hconcat(noise_image, bilateral_cuda_result, comparison);
    hconcat(comparison, bilateral_cpu_result, comparison);
    hconcat(comparison, gaussi_result, comparison);
    // show(comparison, image_path);
    cv::imwrite(string_replace(image_path, ".png", "_comparison.png"), comparison);

    return 0;
}





















