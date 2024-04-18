// C++
#include <vector>
#include <chrono>
#include <iostream>
#include <functional>
// OpenCV
#include <opencv2/opencv.hpp>
// Eigen3
#include <Eigen/Sparse>



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

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REPLICATE);
        return padded_image;
    }
}


enum class EDIT_MODE {
    SEAMLESS_CLONE = 0,
    TEXTURE_FLATTEN = 1,
    CONTENT_FLIP = 2,
    ILLUMINATION_CHANGE = 3
};


std::vector<float> get_divergence(const cv::Mat& fore, const cv::Mat& back, const bool mix=true, const EDIT_MODE mode=EDIT_MODE::SEAMLESS_CLONE) {
    // 获取图像信息
    const int H = back.rows;
    const int W = back.cols;
    const int C = back.channels();
    const int length = H * W;
    // 计算
    std::vector<float> laplace_result(length * C, 0);
    const auto fore_padded = make_pad(fore, 1, 1);
    const auto back_padded = make_pad(back, 1, 1);
    const int W2 = back_padded.cols;
    std::vector<int> offset({-C, C, -W2 * C, W2 * C});
    for(int i = 0;i < H; ++i) {
        float* const res_ptr = laplace_result.data() + i * W * C;
        uchar* const fore_ptr = fore_padded.data + (1 + i) * W2 * C + C;
        uchar* const back_ptr = back_padded.data + (1 + i) * W2 * C + C;
        for(int j = 0;j < W * C; j += C) {
            // 三个方向上的, 还得分开计算
            for(int ch = 0;ch < C; ++ch) {
                // 四个方向上
                const int start = j + ch;
                if(false) {
                    // 如果不拆开四个方向分别比, 会得到错误的结果
                    std::vector<float> fore_grad(4, 0);
                    std::vector<float> back_grad(4, 0);
                    for(int k = 0;k < 4; ++k) {
                        fore_grad[k] = fore_ptr[start + offset[k]] - fore_ptr[start];
                        back_grad[k] = back_ptr[start + offset[k]] - back_ptr[start];
                    }
                    float fore_sum = 0, back_sum = 0;
                    for(int k = 0;k < 4; ++k) fore_sum += std::abs(fore_grad[k]), back_sum += std::abs(back_grad[k]);
                    if(fore_sum > back_sum)
                        res_ptr[start] = fore_grad[0] + fore_grad[1] + fore_grad[2] + fore_grad[3];
                    else
                        res_ptr[start] = back_grad[0] + back_grad[1] + back_grad[2] + back_grad[3];
                } else {
                    float grad_sum = 0;
                    for(int k = 0;k < 4; ++k) {
                        float lhs = fore_ptr[start + offset[k]] - fore_ptr[start];
                        // 纹理抹平, 在这里抹, mix 必须为 false, 代码还不够多变, 加入了这么多 if else 速度下降啊
                        if(mode == EDIT_MODE::TEXTURE_FLATTEN and !mix and std::abs(lhs) < 7)
                            lhs = 0;
                        // 亮度变换
                        else if(mode == EDIT_MODE::ILLUMINATION_CHANGE and !mix) {
                            lhs = std::pow(std::abs(lhs) / 255, 0.8) * lhs;
                        }
                        if(!mix) grad_sum += lhs; // 如果不考虑背景图像的梯度信息
                        else {
                            // 前景和背景的梯度信息融合
                            const float rhs = back_ptr[start + offset[k]] - back_ptr[start];
                            if(std::abs(lhs) > std::abs(rhs)) grad_sum += lhs;
                            else grad_sum += rhs;
                        }
                    }
                    res_ptr[start] = grad_sum;
                }
            }
        }
    }
    return laplace_result;
}


using cloned_type = std::vector< std::pair<int, std::vector<uchar> > >;
cloned_type build_and_solve_poisson_equations(
        const cv::Mat& back,
        const std::vector<float> divergence,
        const cv::Mat& mask) {
    // 获取图像信息
    const int H = back.rows;
    const int W = back.cols;
    const int C = back.channels();
    const int length = H * W;
    // 获取不规则区域内部(mask)的序号, 行优先; 非不规则区域为 -1, 不规则区域内部从 0 开始计数, 对应后面的 A, b 方程的行坐标
    std::vector<int> book(length, -1);
    int pixel_cnt = 0;
    for(int i = 0;i < length; ++i)
        if(mask.data[i] > 128)
            book[i] = pixel_cnt++;
    // 构建 Ax = b 线性方程, 需要填充 A 和 b 的值
    std::vector< Eigen::Triplet<float> > A_list; // 这个比 Eigen 稀疏矩阵 insert 要快很多
    A_list.reserve(pixel_cnt * 5);
    Eigen::MatrixXf b(pixel_cnt, C);
    b.setZero();
    const uchar* const back_data = back.ptr<uchar>();
    for (int i = 1; i < H - 1; ++i) {
        for (int j = 1; j < W - 1; ++j) {
            // 获取当前点, 判断在不在不规则区域范围内
            const int center = i * W + j;
            const int pid = book[center];
            if (pid == -1) continue;
            // A 的赋值
            A_list.emplace_back(pid, pid, -4.0);
            std::vector<float> missing(C);
            std::vector<int> offset({-1, 1, -W, W});
            for(int ori = 0;ori < 4; ++ori) { // 上下左右四个点的赋值
                const int pos = center + offset[ori];
                if(book[pos] == -1) // 如果旁边这个点在边界上, b 的这一项要等于 0, 右边系数 -1
                    for(int k = 0;k < C; ++k)
                        missing[k] += 1.f * back_data[C * pos + k];
                else A_list.emplace_back(pid, book[pos], 1.0);
            }
            // b 的赋值
            for(int k = 0; k < C; ++k)
                b(pid, k) = divergence[C * center + k] - missing[k];
        }
    }

    // Eigen3 解稀疏矩阵的非线性方程组 Ax = b
    Eigen::SparseMatrix<float> A(pixel_cnt, pixel_cnt);
    A.setFromTriplets(A_list.begin(), A_list.end());
    A.makeCompressed();
    Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
    solver.compute(A);
    // 道求解
    Eigen::MatrixXf X = solver.solve(b);
    // 把 Eigen3 的结果拷贝到背景图上
    cloned_type modified;
    modified.reserve(pixel_cnt);
    for(int i = 0;i < length; ++i) {
        if(book[i] != -1) {
            std::vector<uchar> temp(C);
            for(int ch = 0; ch < C; ++ch) temp[ch] = cv::saturate_cast<uchar>(X(book[i], ch));
            modified.emplace_back(i, temp);
        }
    }
    return modified;
}




cv::Mat possion_seamless_clone(
        const cv::Mat& foreground,
        const cv::Mat& background,
        const cv::Mat& mask,
        const std::pair<int, int> start,
        const bool mix=true,
        const EDIT_MODE mode=EDIT_MODE::SEAMLESS_CLONE) {

    // 异常处理
    assert(not foreground.empty() and "前景图 foreground 读取失败 !");
    assert(not background.empty() and "背景图 background 读取失败 !");
    assert(not mask.empty() and "mask 读取失败 !");
    assert(foreground.channels() == background.channels() and "前景图和背景图的通道数目不对等 !");
    assert(foreground.rows == mask.rows and foreground.cols == mask.cols and "前景图和 mask 的尺寸不对等 !");
    assert(start.first >= 0 and start.second >= 0 and "插入的起始位置不能为负 !");
    assert(start.first + foreground.rows <= background.rows and start.second + foreground.cols <= background.cols and "插入位置超出了背景图的界限 !");
    assert(foreground.isContinuous() and background.isContinuous() and mask.isContinuous());

    // 获取图像信息
    const int H = foreground.rows;
    const int W = foreground.cols;
    const int C = foreground.channels();

    // 背景区域对应位置切出来
    const auto background_crop = background(cv::Rect(start.second, start.first, W, H)).clone();

    // 求解要插入的内容的散度, 也可与背景图散度相融合
    const auto divergence = get_divergence(foreground, background_crop, mix, mode);

    // 根据泊松方程的条件, Ax = b, 构建 A, b 求解 x(不规则区域要填充的值)
    const auto modified = build_and_solve_poisson_equations(background_crop, divergence, mask);

    // 把结果到目标图像对应位置上修改像素值
    cv::Mat destination = background.clone();
    for(const auto & item : modified) {
        const int pos = (start.first + item.first / W) * destination.cols * C + (start.second + item.first % W) * C;
        for(int ch = 0; ch < C; ++ch) destination.data[pos + ch] = item.second[ch];
    }
    return destination;
}



void seamless_cloning_demo() {
    // 读取图像
    std::string input_dir("./images/edit/3/");
    std::string save_dir("./images/output/3/");
    cv::Mat background = cv::imread(input_dir + "background.jpg");
    cv::Mat foreground = cv::imread(input_dir + "src_image.png");
    cv::Mat mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);

    cv::Mat result;


    result = possion_seamless_clone(foreground, background,mask, {134, 140}, true);
    cv_show(result, "mixed and splited laplace");
    cv_write(result, save_dir + "mixed_splited_laplace.png");

    // 如果是单纯的 laplace, 不要 mix, 会是什么效果
    result = possion_seamless_clone(foreground, background,mask, {134, 140}, false);
    cv_show(result, "pure_laplace");
    cv_write(result, save_dir + "pure_laplace.png");


    // 其它经典例子
    input_dir = "./images/edit/1/";
    save_dir = "./images/output/1/";
    background = cv::imread(input_dir + "bg.jpg");
    foreground = cv::imread(input_dir + "fg.jpg");
    mask = cv::imread(input_dir + "mask.jpg", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background,mask, {30, 100}, true);
    cv_show(result, "demo_2");
    cv_write(result, save_dir + "mixed_splited_laplace_2.png");
    result = possion_seamless_clone(foreground, background,mask, {30, 100}, false);
    cv_show(result, "demo_2");
    cv_write(result, save_dir + "pure_laplace_2.png");


    input_dir = "./images/edit/4/";
    save_dir = "./images/output/4/";
    background = cv::imread(input_dir + "background.png");
    foreground = cv::imread(input_dir + "foreground.png");
    mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {200, 33}, false);
    cv_show(result, "demo_3");
    cv_write(result, save_dir + "pure_laplace_3.png");

    result = possion_seamless_clone(foreground, background, mask, {200, 33}, true);
    cv_show(result, "demo_3");
    cv_write(result, save_dir + "mixed_splited_laplace_3.png");

    /* 这个例子会报错, 应该是线性方程组无解
    input_dir = "./images/edit/2/";
    save_dir = "./images/output/2/";
    background = cv::imread(input_dir + "bg.jpg");
    foreground = cv::imread(input_dir + "fg.jpg");
    mask = cv::imread(input_dir + "mask.jpg", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {100, 100}, false);
    cv_show(result, "demo_4");
    cv_write(result, save_dir + "mixed_splited_laplace_4.png");
    */

    input_dir = "./images/edit/2/";
    save_dir = "./images/output/2/";
    background = cv::imread(input_dir + "background.jpg");
    foreground = cv::imread(input_dir + "foreground.png");
    mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {100, 100}, false);
    cv_show(result, "demo_4");
    cv_write(result, save_dir + "pure_laplace_4.png");

    result = possion_seamless_clone(foreground, background, mask, {100, 100}, true);
    cv_show(result, "demo_4");
    cv_write(result, save_dir + "mixed_splited_laplace_4.png");


    input_dir = "./images/edit/5/";
    save_dir = "./images/output/5/";
    background = cv::imread(input_dir + "background.png");
    foreground = cv::imread(input_dir + "foreground.png");
    mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {60, 0}, false);
    cv_show(result, "demo_5");
    cv_write(result, save_dir + "pure_laplace_5.png");
    result = possion_seamless_clone(foreground, background, mask, {60, 0}, true);
    cv_show(result, "demo_5");
    cv_write(result, save_dir + "mixed_splited_laplace_5.png");


    input_dir = "./images/edit/6/";
    save_dir = "./images/output/6/";
    background = cv::imread(input_dir + "background.png");
    foreground = cv::imread(input_dir + "foreground.png");
    mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {120, 100}, false);
    cv_show(result, "demo_6");
    cv_write(result, save_dir + "pure_laplace_6.png");
    result = possion_seamless_clone(foreground, background, mask, {120, 100}, true);
    cv_show(result, "demo_6");
    cv_write(result, save_dir + "mixed_splited_laplace_6.png");


    input_dir = "./images/edit/7/";
    save_dir = "./images/output/7/";
    background = cv::imread(input_dir + "background.png");
    foreground = cv::imread(input_dir + "foreground_1.png");
    mask = cv::imread(input_dir + "mask_1.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {10, 40}, true);
    foreground = cv::imread(input_dir + "foreground_2.png");
    mask = cv::imread(input_dir + "mask_2.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, result, mask, {200, 40}, false);
    cv_show(result, "demo_7");
    cv_write(result, save_dir + "pure_laplace_7.png");


    input_dir = "./images/edit/8/";
    save_dir = "./images/output/8/";
    background = cv::imread(input_dir + "background.png");
    cv::resize(background, background, cv::Size(background.cols + 100, background.rows));
    foreground = cv::imread(input_dir + "foreground_1.png");
    mask = cv::imread(input_dir + "mask_1.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {100, 40}, true);
    foreground = cv::imread(input_dir + "foreground_2.png");
    mask = cv::imread(input_dir + "mask_2.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, result, mask, {240, 60}, true);
    cv_show(result, "demo_8");
    cv_write(result, save_dir + "pure_laplace_8.png");


    input_dir = "./images/edit/9/";
    save_dir = "./images/output/9/";
    background = cv::imread(input_dir + "background.png");
    foreground = cv::imread(input_dir + "foreground_1.png");
    mask = cv::imread(input_dir + "mask_1.png", cv::IMREAD_GRAYSCALE);
    cv::resize(mask, mask, cv::Size(mask.cols + 30, mask.rows));
    cv::resize(foreground, foreground, cv::Size(foreground.cols + 30, foreground.rows));
    result = possion_seamless_clone(foreground, background, mask, {105, 340}, true);
    foreground = cv::imread(input_dir + "foreground_2.png");
    mask = cv::imread(input_dir + "mask_2.png", cv::IMREAD_GRAYSCALE);
    cv::resize(mask, mask, cv::Size(mask.cols - 20, mask.rows));
    cv::resize(foreground, foreground, cv::Size(foreground.cols - 20, foreground.rows));
    result = possion_seamless_clone(foreground, result, mask, {120, 30}, false);
    cv_show(result, "demo_9");
    cv_write(result, save_dir + "pure_laplace_9.png");


    input_dir = "./images/edit/14/";
    save_dir = "./images/output/14/";
    background = cv::imread(input_dir + "background.png");
    cv::resize(background, background, cv::Size(160, 160));
    foreground = cv::imread(input_dir + "foreground.png");
    mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
    result = possion_seamless_clone(foreground, background, mask, {35, 30}, true);
    cv_show(result, "demo_3");
    cv_write(result, save_dir + "mixed_splited_laplace_14.png");
    result = possion_seamless_clone(foreground, background, mask, {35, 30}, false);
    cv_show(result, "demo_3");
    cv_write(result, save_dir + "pure_laplace_14.png");

}



void possion_edit_demo_1() {
    // 读取图像
    std::string input_dir("./images/edit/10/");
    std::string save_dir("./images/output/10/");
    cv::Mat background = cv::imread(input_dir + "background.png");
    cv::Mat mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);

    // 选取目标图中的局部区域, 这里是一朵花
    cv::Mat foreground = background(cv::Rect(19, 85, mask.cols, mask.rows)).clone();
    const int H = foreground.rows;
    const int W = foreground.cols;
    const int C = foreground.channels();
    assert(C == 3);
    // 求解要插入的内容的散度, 也可与背景图散度相融合
    const auto divergence = get_divergence(foreground, foreground, false);

    // 更改三个通道的梯度
    int length = H * W;
    int length_2 = length * C;
    std::vector<float> temp(length * C);
    std::copy(divergence.begin(), divergence.end(), temp.begin());
    for(int i = 0;i < length_2; i += C) {
        temp[i] /= 2;
        temp[i + 1] /= 2;
        temp[i + 2] *= 1.5;
    }

    // 根据泊松方程的条件, Ax = b, 构建 A, b 求解 x(不规则区域要填充的值)
    auto modified = build_and_solve_poisson_equations(foreground, temp, mask);
    cv::Mat destination = background.clone();
    for(const auto & item : modified) {
        const int pos = (85 + item.first / W) * destination.cols * C + (19 + item.first % W) * C;
        for(int ch = 0; ch < C; ++ch) destination.data[pos + ch] = item.second[ch];
    }
    cv_show(destination);
    cv_write(destination, save_dir + "color_change_1.png");


    // 换另一种方式更改各通道的梯度信息
    std::copy(divergence.begin(), divergence.end(), temp.begin());
    for(int i = 0;i < length_2; i += C) {
        std::swap(temp[i], temp[i + 1]);
    }
    modified = build_and_solve_poisson_equations(foreground, temp, mask);
    for(const auto & item : modified) {
        const int pos = (85 + item.first / W) * destination.cols * C + (19 + item.first % W) * C;
        for(int ch = 0; ch < C; ++ch) destination.data[pos + ch] = item.second[ch];
    }
    cv_show(destination);
    cv_write(destination, save_dir + "color_change_2.png");
}





void possion_texture_flatten_demo() {
    // 读取图像
    std::string input_dir("./images/edit/11/");
    std::string save_dir("./images/output/11/");
    cv::Mat background = cv::imread(input_dir + "background.png");
    cv::Mat mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);

    // 选取目标图中的局部区域
    cv::Mat foreground = background(cv::Rect(54, 127, mask.cols, mask.rows)).clone();

    cv::Mat result = possion_seamless_clone(
            foreground, background, mask, {127, 54}, false, EDIT_MODE::TEXTURE_FLATTEN);

    cv_show(result);
    cv_write(result, save_dir + "texture_flatten_1.png");
}



void possion_content_flip_demo() {
        // 读取图像
    std::string input_dir("./images/edit/13/");
    std::string save_dir("./images/output/13/");
    cv::Mat background = cv::imread(input_dir + "background.png");
    cv::Mat mask = cv::imread(input_dir + "mask_2.png", cv::IMREAD_GRAYSCALE);

    // 保留叶子
    cv::Mat foreground = background(cv::Rect(50, 9, mask.cols, mask.rows)).clone();
    // cv_show(foreground);

    // 找个空白区域
    cv::Mat foreground_2 = background(cv::Rect(180, 9, mask.cols, mask.rows)).clone();

    cv::Mat result = possion_seamless_clone(
            foreground_2, background, mask, {9, 50}, false);

    // cv_show(result);
    cv_write(result, save_dir + "content_flip_1.png");


    const int H = foreground.rows;
    const int W = foreground.cols;
    const int C = foreground.channels();
    assert(C == 3);
    // 求解要插入的内容的散度, 也可与背景图散度相融合
    const auto divergence = get_divergence(foreground, foreground, true);

    // 在这里对图像进行翻转
    int length = H * W;
    int length_2 = length * C;
    std::vector<float> temp(length * C);
    std::copy(divergence.begin(), divergence.end(), temp.begin());
    for(int i = 0;i < H; ++i) {
        for(int j = 0;j < W; ++j) {
            const int pos = i * W * C + j * C;
            const int pos_2 = i * W * C + (W - 1 - j) * C;
            for(int k = 0;k < C; ++k)
                temp[pos + k] = divergence[pos_2 - k];
        }
    }
    // mask 也要 flip
    cv::flip(mask, mask, 0);


    // 根据泊松方程的条件, Ax = b, 构建 A, b 求解 x(不规则区域要填充的值)
    auto modified = build_and_solve_poisson_equations(foreground, temp, mask);
    cv::Mat destination = result.clone();
    for(const auto & item : modified) {
        const int pos = (9 + item.first / W) * destination.cols * C + (20 + mask.cols + item.first % W) * C;
        for(int ch = 0; ch < C; ++ch) destination.data[pos + ch] = item.second[ch];
    }
    cv_show(destination);
    cv_write(destination, save_dir + "content_flip_2.png");
}




void seam_clone_demo() {
    std::string input_dir = "./images/edit/14/";
    std::string save_dir = "./images/output/14/";
    cv::Mat background = cv::imread(input_dir + "background.png");
    cv::Mat foreground = cv::imread(input_dir + "foreground.png");
    cv::Mat mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);
    cv::resize(background, background, cv::Size(160, 160));

    std::pair<int, int> start({35, 30});

    // 直接粘贴
    const int length = foreground.rows * foreground.cols;
    for(int i = 0;i < length; ++i) {
        if(mask.data[i] > 128) {
            std::memcpy(background.data + (start.first + int(i / foreground.cols)) * background.cols * background.channels()
                          + (start.second + int(i % foreground.cols)) * background.channels(),
                        foreground.data + foreground.channels() * i,
                        sizeof(uchar) * foreground.channels());
        }
    }
    cv_show(background);
    cv_write(background, save_dir + "seam_clone.png");
}




void possion_texture_transform_demo() {
    std::string input_dir = "./images/edit/17/";
    std::string save_dir = "./images/output/17/";
    cv::Mat background = cv::imread(input_dir + "background.png");
    cv::Mat mask = cv::imread(input_dir + "mask_1.png", cv::IMREAD_GRAYSCALE);

    // 168.5000   61.5000  118.0000  265.0000
    cv::Mat background_crop = background(cv::Rect(168 - mask.cols + 20, 61, mask.cols, mask.rows)).clone();

    // 直接用背景纹理替代前景的人
    cv::Mat result = possion_seamless_clone(
            background_crop, background, mask, {61, 168}, false);
    cv_write(result, save_dir + "texture_transform_2.png");

    // 163.5000   99.5000   58.0000  124.00001
    mask = cv::imread(input_dir + "mask_2.png", cv::IMREAD_GRAYSCALE);
    background_crop = background(cv::Rect(163 - mask.cols, 99, mask.cols, mask.rows)).clone();

    result = possion_seamless_clone(
            background_crop, result, mask, {99, 163}, false);

    cv_write(result, save_dir + "texture_transform_3.png");
    cv_show(result);

    // 236.5000   67.5000   18.0000   46.0000
    mask = cv::imread(input_dir + "mask_3.png", cv::IMREAD_GRAYSCALE);
    background_crop = background(cv::Rect(236, 67, mask.cols, mask.rows)).clone();

    result = possion_seamless_clone(
            background_crop, result, mask, {67, 236 - mask.cols}, true);

    cv_write(result, save_dir + "texture_transform_4.png");
    cv_show(result);

}



void possion_illumination_change_demo() {
    std::string input_dir = "./images/edit/16/";
    std::string save_dir = "./images/output/16/";
    cv::Mat background = cv::imread(input_dir + "background.png");
    cv::Mat mask = cv::imread(input_dir + "mask.png", cv::IMREAD_GRAYSCALE);

    // 174.5000  168.5000
    cv::Mat background_crop = background(cv::Rect(174, 168, mask.cols, mask.rows)).clone();

    // 直接用背景纹理替代前景的人
    cv::Mat result = possion_seamless_clone(
            background_crop, background, mask, {168, 174}, false, EDIT_MODE::ILLUMINATION_CHANGE);
    cv_write(result, save_dir + "illumination_change_1.png");
    cv_show(result);
}


int main() {

    // 有缝合成
    seam_clone_demo();

    // 无缝隙合成图像
    seamless_cloning_demo();

    // 各种利用梯度的应用
    possion_edit_demo_1();

    // 纹理抹平
    possion_texture_flatten_demo();

    // 内容翻转
    possion_content_flip_demo();

    // 纹理去除
    possion_texture_transform_demo();

    // 局部亮度变换
    possion_illumination_change_demo();

    return 0;
}




