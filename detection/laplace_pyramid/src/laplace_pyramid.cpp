// C++
#include <vector>
#include <chrono>
#include <fstream>
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
    cv::Mat toint8(const std::vector<T>& source, const int H, const int W, const int C, const int _type) {
        cv::Mat result(H, W, _type);
        const int length = H * W * C;
        for(int i = 0;i < length; ++i) result.data[i] = cv::saturate_cast<uchar>(std::abs(source[i]) * 2);
        return result;
    }
}



cv::Mat fast_gaussi_blur(
        const cv::Mat& source,
        const int radius=2,
        const float sigma=0.83,
        const float ratio=1.0,
        const bool mask=false) {
    // 根据半径和方差构造一个高斯模板
    const int filter_size = 2 * radius + 1;
    std::vector<float> filter(filter_size, 0);
    float cur_sum = 0;
    for(int i = -radius; i <= radius; ++i) {
        filter[i + radius] = 1.f / sigma * std::exp(-float(i * i) / (2 * sigma * sigma));
        cur_sum += filter[i + radius];
    }
    for(int i = 0;i < filter_size; ++i) filter[i] = filter[i] * ratio / cur_sum;
    // 先做 pad
    const auto source_pad = make_pad(source, radius, radius);
    // 获取图像信息
    const int H = source_pad.rows;
    const int W = source_pad.cols;
    const int C = source_pad.channels();
    // 先对 x 方向做高斯平滑
    cv::Mat temp = source_pad.clone();
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 0; i < H; ++i) {
            const uchar* const src_ptr = source_pad.data + i * W * C;
            uchar* const temp_ptr = temp.data + i * W * C;
            for(int j = radius; j < W - radius; ++j) {
                // if(mask and src_ptr[j * C + ch] != 0) continue;
                float intensity_sum = 0;
                for(int k = -radius; k <= radius; ++k)
                    intensity_sum += filter[radius + k] * src_ptr[(j + k) * C + ch];
                temp_ptr[j * C + ch] = cv::saturate_cast<uchar>(intensity_sum);
            }
        }
    }
    // 再对 y 方向做高斯平滑
    cv::Mat result = source.clone();
    for(int ch = 0; ch < C; ++ch) {
        for(int i = radius; i < H - radius; ++i) {
            const uchar* const temp_ptr = temp.data + i * W * C;
            uchar* const res_ptr = result.data + (i - radius) * source.cols * C;
            for(int j = radius; j < W - radius; ++j) {
                // if(mask and temp_ptr[j * C + ch] != 0) continue;
                float intensity_sum = 0;
                for(int k = -radius; k <= radius; ++k)
                    intensity_sum += filter[radius + k] * temp_ptr[k * W * C + j * C + ch];
                res_ptr[(j - radius) * C + ch] = cv::saturate_cast<uchar>(intensity_sum);
            }
        }
    }
    return result;
}


cv::Mat pyramid_downsample(const cv::Mat& source) {
    // 收集图像信息
    const int H = source.rows / 2, W = source.cols / 2;
    // 准备一个结果
    cv::Mat downsampled(H, W, source.type());
    const int C = source.channels();
    // 开始每隔一个点采一个样
    for(int i = 0;i < H; ++i) {
        uchar* const res_ptr = downsampled.data + i * W * C;
        for(int j = 0;j < W; ++j)
            std::memcpy(res_ptr + j * C, source.data + 2 * (i * source.cols + j) * C, sizeof(uchar) * C);
    }
    return downsampled;
}


std::vector<cv::Mat> build_gaussi_pyramid(const cv::Mat& source, const int layers_num) {
    // 首先需要把图像规整到 2 ^ layers_num 的整数倍
    const int new_H = (1 << layers_num) * int(source.rows / (1 << layers_num));
    const int new_W = (1 << layers_num) * int(source.cols / (1 << layers_num));
    auto source_croped = source(cv::Rect(0, 0, new_W, new_H)).clone();
    // 准备返回结果
    std::vector<cv::Mat> gaussi_pyramid;
    gaussi_pyramid.reserve(layers_num);
    gaussi_pyramid.emplace_back(source_croped);
    // 开始构造接下来的几层
    for(int i = 1;i < layers_num; ++i) {
        // 先对图像做高斯模糊
        source_croped = fast_gaussi_blur(source_croped, 2, 1.0, 1.0);
        // 做下采样
        source_croped = pyramid_downsample(source_croped);
        // 放到高斯金字塔中
        gaussi_pyramid.emplace_back(source_croped);
    }
    // 从低分辨率到高分辨率依次存储
    std::reverse(gaussi_pyramid.begin(), gaussi_pyramid.end());
    return gaussi_pyramid;
}


cv::Mat pyramid_upsample(const cv::Mat& source) {
    const int H = source.rows, W = source.cols;
    const int C = source.channels();
    // 准备一个结果
    cv::Mat upsampled = cv::Mat::zeros(2 * H, 2 * W, source.type());
    // 把值填充到上采样结果中
    for(int i = 0; i < H; ++i) {
        const uchar* const src_ptr = source.data + i * W * C;
        uchar* const res_ptr = upsampled.data + 2 * i * (2 * W) * C;
        for(int j = 0;j < W; ++j)
            std::memcpy(res_ptr + 2 * j * C, src_ptr + j * C, sizeof(uchar) * C);
    }
    return upsampled;
}

void pyramid_upsample_interpolate(const cv::Mat& source) {
    const int H = source.rows, W = source.cols;
    const int C = source.channels();
    // 先处理第一行
    for(int ch = 0; ch < C; ++ch) {
        for(int j = 1;j < W - 1; j += 2)
            source.data[j * C + ch] = (source.data[(j - 1) * C + ch] + source.data[(j + 1) * C + ch]) / 2;
        source.data[(W - 1) * C + ch] = source.data[(W - 2) * C + ch];
    }
    // 最后一行
    uchar* const row_ptr = source.data + (H - 1) * W * C;
    for(int ch = 0; ch < C; ++ch) {
        for(int j = 1;j < W - 1; j += 2)
            row_ptr[j * C + ch] = (row_ptr[(j - 1) * C + ch] + row_ptr[(j + 1) * C + ch]) / 2;
        row_ptr[(W - 1) * C + ch] = row_ptr[(W - 2) * C + ch];
    }
    // 第一列
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 1;i < H - 1; i += 2) {
            const int pos = i * W * C + ch;
            source.data[pos] = (source.data[pos - W * C] + source.data[pos + W * C]) / 2;
        }
    }
    // 剩下的行和列
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 1;i < H - 1; ++i) {
            uchar* const row_ptr = source.data + i * W * C;
            for(int j = 1;j < W - 1; ++j) {
                // 如果都是奇数, 说明是空的
                if((i & 1) and (j & 1)) {
                    row_ptr[j * C + ch] = (row_ptr[(j - 1 - W) * C + ch] + row_ptr[(j - 1 + W) * C + ch] + row_ptr[(j + 1 - W) * C + ch] + row_ptr[(j + 1 + W) * C + ch]) / 4;
                }
                // 如果奇数行, 偶数列
                else if(i & 1) {
                    row_ptr[j * C + ch] = (row_ptr[(j - W) * C + ch] + row_ptr[(j + W) * C + ch]) / 2;
                }
                // 如果偶数行, 奇数列
                else if(j & 1) {
                    row_ptr[j * C + ch] = (row_ptr[(j - 1) * C + ch] + row_ptr[(j + 1) * C + ch]) / 2;
                }
            }
        }
    }
    // 最后一列是全黑的 ! 因为之前 upsample 的时候偶数列全部都是 0, 直接把倒数第二列拷贝过去
    // 而倒数第二列在前面的大循环中才会赋值, 所以放到这里
    for(int ch = 0; ch < C; ++ch) {
        for(int i = 1;i < H - 1; ++i) {
            const int pos = (i * W + W - 1) * C + ch;
            source.data[pos] = source.data[pos - C];
        }
    }
}


using res_type = short;

std::vector< std::vector<res_type> > build_laplace_pyramid(const std::vector<cv::Mat>& gaussi_pyramid) {
    // 查看几层
    const int layers_num = gaussi_pyramid.size();
    // 准备一个结果
    std::vector< std::vector<res_type> > laplace_pyramid;
    laplace_pyramid.reserve(layers_num - 1);
    // 从低分辨率开始构建拉普拉斯金字塔
    for(int i = 0; i < layers_num - 1; ++i) {
        // 首先低分辨率先上采样到两倍大小
        cv::Mat upsampled = pyramid_upsample(gaussi_pyramid[i]);
        // 填补值
        pyramid_upsample_interpolate(upsampled);
        // 放到拉普拉斯金字塔
        const int length = upsampled.rows * upsampled.cols * upsampled.channels();
        std::vector<res_type> residual(length, 0);
        for(int k = 0;k < length; ++k)
            residual[k] = gaussi_pyramid[i + 1].data[k] - upsampled.data[k];
        laplace_pyramid.emplace_back(residual);
    }
    return laplace_pyramid;
}


using range_type = long;

cv::Mat rebuild_image_from_laplace_pyramid(
        const cv::Mat& low_res,
        const std::vector< std::vector<res_type> >& laplace_pyramid,
        const int layers_num,
        const std::vector<cv::Mat>& gaussi_pyramid={}) {
    // 从最低分辨率的开始
    cv::Mat reconstructed = low_res.clone();
    for(int i = 0;i <= layers_num - 2; ++i) {
        cv::Mat upsampled = pyramid_upsample(reconstructed);
        pyramid_upsample_interpolate(upsampled);
        // 将 laplace_pyramid 和 当前结果结合
        const int H = upsampled.rows, W = upsampled.cols, C = upsampled.channels();
        const int length = H * W * C;
        std::vector<uchar> temp(length, 0);
        for(int k = 0;k < length; ++k)
            temp[k] = cv::saturate_cast<uchar>(upsampled.data[k] + laplace_pyramid[i][k]);
        // 再把结果拷贝到 reconstructed
        reconstructed = cv::Mat(H, W, upsampled.type()); // reconstructed 的大小得变化一下
        std::memcpy(reconstructed.data, temp.data(), length);
        // 观察损失有多大
        if(gaussi_pyramid.size() == layers_num) {
            cv_info(reconstructed);
            cv_info(gaussi_pyramid[i]);
            std::cout << "PSNR ===>  " << cv::PSNR(reconstructed, gaussi_pyramid[i + 1]) << "db" << std::endl;
            cv_show(cv_concat({
                gaussi_pyramid[i + 1],
                upsampled,
                toint8(laplace_pyramid[i], H, W, C, upsampled.type()),
                reconstructed}));
        }
    }
    return reconstructed;
}


void laplace_decomposition_demo() {

    // 读取图像
    const std::string image_path("./images/input/a2376-IMG_2891.png");
    const std::string save_dir("./images/output/compression/1/");
    cv::Mat origin_image = cv::imread(image_path);
    assert(!origin_image.empty() and "图片读取失败");
    // 构建层数( 2 ^ layers_num 必须小于高和宽的最小值, 不然是错误的)
    int layers_num = 5;
    assert((1 << layers_num) < origin_image.rows and "金字塔层数太大了, 超出了图像的高");
    assert((1 << layers_num) < origin_image.cols and "金字塔层数太大了, 超出了图像的宽");
    // 根据图像构建高斯金字塔
    const auto gaussi_pyramid = build_gaussi_pyramid(origin_image, layers_num);
    // 构建拉普拉斯金字塔
    auto laplace_pyramid = build_laplace_pyramid(gaussi_pyramid);
    // 展示
    cv::Mat reconstructed = rebuild_image_from_laplace_pyramid(
        gaussi_pyramid[0],
        laplace_pyramid,
        layers_num
        , gaussi_pyramid
    );
    // 保存一波
    for(int i = 0;i < layers_num; ++i) cv_write(gaussi_pyramid[i], save_dir + "gaussi_pyramid_" + std::to_string(i) + ".png");
    cv_write(reconstructed, save_dir + "laplace_reconstructed.png");

    /*
        从上面可以看出, 如何压缩信息 ?
        最小分辨率的图像 + 拉普拉斯金字塔
        而拉普拉斯金字塔很多都是 0, 所以只需要保留那些大于某个阈值的点的坐标信息, 即可大概还原出图像来
        拉普拉斯金字塔压缩信息(base, detail 分离, 弱化一些不必要的 detail)
    */

    // 对拉普拉斯金字塔的结果过滤一些不必要的细节, 三通道之和小于 30 的都置为 0
    const int threshold = 15;
    std::vector< std::vector< std::pair<range_type, std::vector<res_type> > > > laplace_pyramid_compressd;
    laplace_pyramid_compressd.reserve(layers_num - 1);
    const int C = origin_image.channels();
    for(auto& layer : laplace_pyramid) {
        std::vector< std::pair<range_type, std::vector<res_type> > > cur_laplace; // 当前层压缩得到的拉普拉斯金字塔, 保留多少个点, 每个点 4 个信息(位置, 三个通道的值)
        const int length = layer.size() / 3;
        for(int i = 0;i < length; ++i) {
            const int pos = C * i;
            res_type res_sum = 0.0;
            for(int k = 0;k < C; ++k) // 叠加这个点, 在所有通道的细节
                res_sum += std::abs(layer[pos + k]);
            if(res_sum > threshold) { // 判断这个点的细节强弱程度, 大于则保留该细节
                std::vector<res_type> temp(C); // 收集 4 个信息, 当前点的坐标 i 和 BGR 三通道的值
                for(int k = 0;k < C; ++k) temp[k] = layer[pos + k];
                cur_laplace.emplace_back(i, temp);
            }
            else // 细节小于于阈值, 删除这个点
                for(int k = 0;k < C; ++k) layer[pos + k] = 0;
        }
        std::cout << cur_laplace.size() << "/" << length << std::endl;
        laplace_pyramid_compressd.emplace_back(cur_laplace);
    }
    // 重新构建一次
    cv::Mat compressed = rebuild_image_from_laplace_pyramid(
        gaussi_pyramid[0],
        laplace_pyramid,
        layers_num
        , gaussi_pyramid
    );
    cv_write(compressed, save_dir + "laplace_compressed.png");

    /*
        根据上面的方法, 对拉普拉斯金字塔压缩, 使用二进制存储到电脑上, 对比存储大小和信息损失
        对于原始拉普拉斯重建的图像 reconstructed, 写入文件只需要 高、宽、通道数目、数据类型, 以及整个 .data 图像内容
    */

    // 将 reconstructed 按照二进制存储
    std::ofstream writer_1(save_dir + "reconstructed", std::ios::binary);
    int H_1 = reconstructed.rows;
    int W_1 = reconstructed.cols;
    int C_1 = reconstructed.channels();
    const int length_1 = sizeof(uchar) * H_1 * W_1 * C_1;
    int info[4] = {reconstructed.rows, reconstructed.cols, reconstructed.channels(), reconstructed.type()};
    writer_1.write(reinterpret_cast<const char *>(&info), sizeof(info));
    writer_1.write(reinterpret_cast<const char *>(&reconstructed.data[0]), static_cast<std::streamsize>(length_1));
    writer_1.close();
    // 读取这个二进制文件, 重建
    run([&](){
        std::ifstream reader_1(save_dir + "reconstructed", std::ios::binary);
        int info_2[4];
        reader_1.read((char*)(&info_2), sizeof(info_2));
        cv::Mat reconstructed_from_file(info_2[0], info_2[1], info_2[3]); // 根据高宽和类型重建图像
        reader_1.read((char*)(&reconstructed_from_file.data[0]), sizeof(uchar) * info_2[0] * info_2[1] * info_2[2]); // 拷贝图像内容
        reader_1.close();
        cv_write(reconstructed_from_file, save_dir + "reconstructed_from_file.png");
        cv_show(reconstructed_from_file, std::to_string(cv::PSNR(reconstructed, reconstructed_from_file)).c_str());
    }, "从无信息损失的二进制文件中读取  "); // 算时间的话, 要把上面的 cv_show 注释掉


    /*
        如果是拉普拉斯金字塔压缩过后的图像, 只需要保存最小分辨率的图像和拉普拉斯金字塔
    */

    // 保存最小分辨率的图, 也是保存高、宽、通道数目、数据类型, 以及整个 .data 图像内容
    std::ofstream writer_2(save_dir + "compressed", std::ios::binary);
    const cv::Mat& start = gaussi_pyramid[0];
    int info_3[4] = {start.rows, start.cols, start.channels(), start.type()};
    int length_2 = sizeof(uchar) * info_3[0] * info_3[1] * info_3[2];
    writer_2.write(reinterpret_cast<const char *>(&info_3), sizeof(info_3));
    writer_2.write(reinterpret_cast<const char *>(&start.data[0]), static_cast<std::streamsize>(length_2));
    // 保存压缩之后的拉普拉斯金字塔
    const int pixel_info = origin_image.channels(); // 每个点有 4 个信息, int 存储
    const int laplace_num = laplace_pyramid_compressd.size();
    writer_2.write(reinterpret_cast<const char *>(&laplace_num), sizeof(int)); // 拉普拉斯金字塔有几层
    for(int i = 0;i < laplace_num; ++i) {
        const int pixel_num = laplace_pyramid_compressd[i].size(); // 这一层的拉普拉斯保留了多少个细节点
        writer_2.write(reinterpret_cast<const char *>(&pixel_num), sizeof(int));
        for(int k = 0;k < pixel_num; ++k) {
            // 把每一个保留点的 4 个信息写到文件
            range_type pos = laplace_pyramid_compressd[i][k].first;
            writer_2.write(reinterpret_cast<const char *>(&pos), sizeof(range_type));
            const auto& pixel_value = laplace_pyramid_compressd[i][k].second;
            writer_2.write(reinterpret_cast<const char *>(&pixel_value[0]), sizeof(res_type) * pixel_info);
        }
    }
    writer_2.close();
    // 从压缩过后的信息中恢复原图
    run([&](){
        // 先读取低分辨率的小图
        std::ifstream reader_2(save_dir + "compressed", std::ios::binary);
        int info_4[4];
        reader_2.read((char*)(&info_4), sizeof(info_4));
        cv::Mat reconstructed_from_compressed_file(info_4[0], info_4[1], info_4[3]); // 根据高宽生命图像
        reader_2.read((char*)(&reconstructed_from_compressed_file.data[0]), sizeof(uchar) * info_4[0] * info_4[1] * info_4[2]);
        cv_show(reconstructed_from_compressed_file);
        // 读取压缩过后的拉普拉斯金字塔
        int cur_H = info_4[0], cur_W = info_4[1], cur_C = info_4[2];  // 获取最开始的分辨率大小
        std::vector< std::vector<res_type> > laplace_pyramid_reconstructed;
        int laplace_num_2;
        reader_2.read((char*)(&laplace_num_2), sizeof(int)); // 这个拉普拉斯金字塔有几层
        laplace_pyramid_reconstructed.reserve(laplace_num_2);
        for(int i = 0;i < laplace_num_2; ++i) {
            cur_H *= 2, cur_W *= 2; // 这一层的分辨率是之前的两倍
            std::vector<res_type> cur_laplace(cur_H * cur_W * cur_C, 0); // 声明这一层拉普拉斯
            int pixel_num;
            reader_2.read((char*)(&pixel_num), sizeof(int)); // 这一层多少个保留的细节点
            for(int p = 0;p < pixel_num; ++p) {
                range_type pos;
                reader_2.read((char*)(&pos), sizeof(range_type)); // 找到这个细节点在 laplace 中的位置
                res_type* pos_c = cur_laplace.data() + pos * cur_C;
                for(int k = 0;k < cur_C; ++k)
                    reader_2.read((char*)(pos_c + k), sizeof(res_type)); // 这个点的三个通道的值赋值到这一层拉普拉斯
            }
            laplace_pyramid_reconstructed.emplace_back(cur_laplace);
        }
        reader_2.close();
        // 根据从文件中读到的低分辨率小图 和 压缩的拉普拉斯金字塔, 重建高分辨率图像
        cv::Mat reconstructed_from_compressed = rebuild_image_from_laplace_pyramid(
            gaussi_pyramid[0],
            laplace_pyramid_reconstructed,
            laplace_num_2 + 1
            , gaussi_pyramid
        );
        cv_write(reconstructed_from_compressed, save_dir + "reconstructed_from_compressed.png");
    }, "根据拉普拉斯金字塔压缩的方法, 恢复图像 ");
}





void laplace_image_blending_demo() {
    // 读取 lhs, rhs 图像, 以及 mask 图像

    std::string save_dir("./images/output/blending/3/");
    cv::Mat left_image, right_image, mask;
    if(false) {
        const std::string input_dir("./images/input/blending/1/");
        left_image = cv::imread(input_dir + "lhs_2.png");
        right_image = cv::imread(input_dir + "rhs_2.png");
        cv::resize(left_image, left_image, cv::Size(1024, 1024));
        cv::resize(right_image, right_image, cv::Size(1024, 1024));
        mask = cv::imread(input_dir + "mask_2.png", cv::IMREAD_GRAYSCALE);
    } else {
        const std::string input_dir("./images/input/blending/3/");
        right_image = cv::imread(input_dir + "background_2.png");
        cv::resize(right_image, right_image, cv::Size(874, 581));
        // 根据 mask 和 起始位置生成同分辨率的 left 和 mask
        cv::Mat foreground = cv::imread(input_dir + "foreground.bmp");
        cv::Mat mask_old = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
        left_image = cv::Mat::zeros(right_image.rows, right_image.cols, right_image.type());
        mask = cv::Mat::zeros(right_image.rows, right_image.cols, CV_8UC1);
        std::pair<int, int> pos({100, 20});
        for(int i = 0;i < mask_old.rows; ++i) {
            std::memcpy(mask.data + (pos.first + i) * mask.cols + pos.second, mask_old.data + i * mask_old.cols, sizeof(uchar) * mask_old.cols);
        }
        const int cur_C = foreground.channels();
        for(int i = 0;i < foreground.rows; ++i) {
            const uchar* mask_ptr = mask_old.data + i * mask_old.cols;
            const uchar* row_ptr = foreground.data + i * foreground.cols * cur_C;
            uchar* const res_ptr = left_image.data + (pos.first + i) * left_image.cols * cur_C;
            for(int j = 0;j < foreground.cols; ++j) {
                if(mask_ptr[j] < 128) continue;
                for(int k = 0;k < cur_C; ++k)
                    res_ptr[(j + pos.second) * cur_C + k] = row_ptr[cur_C * j + k];
            }
        }
    }

    // 去掉一些奇怪的输入
    assert(left_image.rows == right_image.rows and left_image.cols == right_image.cols and mask.rows == left_image.rows and mask.cols == left_image.cols and "左图右图的形状不对 !");

    // 是否要多频带融合 ?
    if(true) {
        // 设定金字塔的层数
        int layers_num = 7;
        assert(layers_num >= 1);
        assert((1 << layers_num) < left_image.rows and "金字塔层数太大了, 超出了图像的高");
        assert((1 << layers_num) < left_image.cols and "金字塔层数太大了, 超出了图像的宽");
        // 求左图, 右图的高斯金字塔
        const auto left_gaussi_pyramid = build_gaussi_pyramid(left_image, layers_num);
        const auto right_gaussi_pyramid = build_gaussi_pyramid(right_image, layers_num);
        // 求左图, 右图的拉普拉斯金字塔
        const auto left_laplace_pyramid = build_laplace_pyramid(left_gaussi_pyramid);
        const auto right_laplace_pyramid = build_laplace_pyramid(right_gaussi_pyramid);
        // 求 mask 的高斯金字塔
        const auto mask_pyramid = build_gaussi_pyramid(mask, layers_num);
        // 根据 mask 融合拉普拉斯金字塔
        std::vector< std::vector<res_type> > blend_laplace_pyramid;
        blend_laplace_pyramid.reserve(layers_num - 1);
        for(int i = 0; i < layers_num - 1; ++i) {
            const int cur_C = left_gaussi_pyramid[i + 1].channels();
            const int cur_length = left_gaussi_pyramid[i + 1].rows * left_gaussi_pyramid[i + 1].cols;
            std::vector<res_type> this_layer(cur_length * cur_C, 0);
            const auto& lhs = left_laplace_pyramid[i];
            const auto& rhs = right_laplace_pyramid[i];
            for(int ch = 0; ch < cur_C; ++ch) {
                for(int k = 0;k < cur_length; ++k) {
                    const float left_weight = 1.0f * mask_pyramid[i + 1].data[k] / 255;
                    const int pos = k * cur_C + ch;
                    this_layer[pos] = res_type(left_weight * lhs[pos] + (1 - left_weight) * rhs[pos]);
                }
            }
            // 这个例子, 高频细节很少很少, 不大看得出来, 到后面全黑了
            // cv_show(toint8(this_layer, left_gaussi_pyramid[i + 1].rows, left_gaussi_pyramid[i + 1].cols, cur_C, left_gaussi_pyramid[i + 1].type()));
            blend_laplace_pyramid.emplace_back(this_layer);
        }
        // 低分辨率的左图、右图根据 mask 融合得到起始图像
        const auto& start_mask = mask_pyramid[0];
        int cur_H = start_mask.rows;
        int cur_W = start_mask.cols;
        int cur_C = left_image.channels();
        const int cur_length = cur_H * cur_W;
        cv::Mat start(cur_H, cur_W, left_image.type());
        for(int ch = 0; ch < cur_C; ++ch) {
            for(int i = 0;i < cur_length; ++i) {
                const float left_weight = 1.0f * start_mask.data[i] / 255;
                const int pos = i * cur_C + ch;
                start.data[pos] = cv::saturate_cast<uchar>(left_weight * left_gaussi_pyramid[0].data[pos]
                          + (1 - left_weight) * right_gaussi_pyramid[0].data[pos]);
            }
        }
        // 根据 start 和融合之后的 laplace_pyramid, 重构
        cv::Mat blend_result = rebuild_image_from_laplace_pyramid(
            start,
            blend_laplace_pyramid,
            layers_num
        );
        cv_write(blend_result, save_dir + "blend_result_" + std::to_string(layers_num - 1) + ".png");
        cv_show(blend_result);
    }
    else {
        save_dir += "direct/";
        // 直接对中间那条缝做模糊
        for(int radius = 60; radius <= 300; radius += 60) {
            cv::Mat mask_blurred = fast_gaussi_blur(mask, radius, (radius * 2.f + 1) / 3);
            // cv_show(mask_blurred);
            cv_write(mask_blurred, save_dir + "mask_blurred_" + std::to_string(radius) + ".png");
            // 然后直接用这个叠加二者
            const int C = left_image.channels();
            const int H = left_image.rows;
            const int W = left_image.cols;
            // 准备一个结果
            cv::Mat result(H, W, left_image.type());
            for(int i = 0;i < H; ++i) {
                const uchar* const left_ptr = left_image.data + i * W * C;
                const uchar* const right_ptr = right_image.data + i * W * C;
                const uchar* const mask_ptr = mask_blurred.data + i * W;
                uchar* const res_ptr = result.data + i * W * C;
                for(int j = 0;j < W; ++j) {
                    const float weight = mask_ptr[j] * 1.f / 255;
                    const int pos = j * C;
                    for(int ch = 0;ch < C; ++ch)
                        res_ptr[pos + ch] = cv::saturate_cast<uchar>(weight * left_ptr[pos + ch] + (1 - weight) * right_ptr[pos + ch]);
                }
            }
            cv_write(result, save_dir + "radius_" + std::to_string(radius) + ".png");
        }
    }
}



int main() {

    // 拉普拉斯金字塔分解
    laplace_decomposition_demo();

    // 拉普拉斯图像融合
    laplace_image_blending_demo();


    return 0;
}
