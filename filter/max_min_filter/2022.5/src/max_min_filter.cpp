#include "utils.h"
#include <random>


template<typename T>
class MonotonousQueue {
private:
    const int window_len;   // 窗口容量
    std::list<int> Q;     // 单调队列
    T* data;              // 数据指针
    std::function<T(const T, const T)> compare;
public:
    MonotonousQueue(T* _data, const int _win, const std::function<T(const T, const T)> _comp)
        : data(_data), window_len(_win), compare(_comp) {}

    // 插入窗口最右边的元素, 检查是否保持单调
    void emplace(const int i) {
        // 如果当前元素比前一个元素小, 则把单调队列中, 大于当前元素 data[i] 的都 pop 掉
        while(not Q.empty()) {
            if(this->compare(data[i], data[Q.back()]))  // 如果队列中都比当前元素小, 停止
                Q.pop_back();
            else break; // 否则, 把队列中大于当前元素的 pop_back 掉
        }
        // 当前元素的坐标放到这里
        Q.emplace_back(i);
        // 如果当前维护的区间长度超出了窗口
        if(i - Q.front() == window_len)
            Q.pop_front();
    }

    int front() const {
        return this->Q.front();
    }
};



template<typename T>
class MonotonousQueueContinous {
private:
    const int win_len;    // 窗口最大间隔
    T* data;              // 数据指针
    std::function<T(const T, const T)> compare;
private:
    std::vector<int> Q;   // 单调队列
    int _front = 0;       // 队列头
    int _back = 0;        // 队列尾
    const int capacity;   // 队列容量
public:
    MonotonousQueueContinous(T* _data, const int _win, const int _capacity, const std::function<T(const T, const T)> _comp)
        : data(_data), win_len(_win), capacity(_capacity + 1), compare(_comp), Q(_capacity, 0) {}

    MonotonousQueueContinous(const int _win, const int _capacity, const std::function<T(const T, const T)> _comp)
        : win_len(_win), capacity(_capacity + 1), compare(_comp), Q(_capacity, 0) {
        // std::cout << "当心这里没有要滤波的内容指针, 不能直接访问, 得 reset\n";
    }

    // 插入窗口最右边的元素, 检查是否保持单调
    void emplace(const int i) {
        // 如果当前元素比前一个元素小, 则把单调队列中, 大于当前元素 data[i] 的都 pop 掉
        while(_front != _back) {
            const int back_index = Q[_back];
            if(this->compare(data[i], data[back_index]))  // 如果队列中都比当前元素小, 停止
                _back = (_back - 1 + capacity) % capacity;
            else break; // 否则, 把队列中大于当前元素的 pop_back 掉
        }
        // 当前元素的坐标放到这里
        _back = (_back + 1) % capacity;
        Q[_back] = i;
        // 如果当前维护的区间长度超出了窗口
        const int front_next = (_front + 1) % capacity;
        if(i - Q[front_next] == win_len)
            _front = front_next;
    }

    int front() const {
        return this->Q[(_front + 1) % capacity];
    }

    void reset(T* new_data) {
        data = new_data;
        _front = _back = 0;
    }
};


/*
 * 拿 1d 数据测试一下单调队列的正确性
 */
void test_1d_extremum_filtering() {
    std::cout << "=============== 一维数据的最小值滤波 ===============\n";
    using data_type = int;
    auto display = [](data_type* const data, const int length) {
        for(int i = 0; i < length; ++i)
            std::cout << data[i] << "  ";
        std::cout << std::endl;
    };
    // 生成一个一维的数据
    std::vector<int> image_data({4, 1, 3, 0, 8, 9, -1, 2});
//    std::vector<int> image_data({1048576,	0,	18,	3,	4,	19,	8,	1048576	});
    display(image_data.data(), image_data.size());
    // 设定滤波核的长度
    const int kernel_size = 3;
    // 最大值滤波 or 最小值滤波
    constexpr bool MIN_FILTER = true;
    // 决定 padding 的值
    constexpr int EXTRENUM = MIN_FILTER ? 1 << 20 : -(1 << 20);
    auto comp = MIN_FILTER ? [](const int lhs, const int rhs) {return lhs < rhs;} : [](const int lhs, const int rhs) {return lhs > rhs;} ;
    // 对数据做 padding
    for(int i = 0; i < kernel_size; ++i)
        image_data.emplace_back(EXTRENUM);
    // 获取数据信息
    const int image_size = image_data.size();
    int* image_ptr = image_data.data();
    // 准备一个滤波结果
    const int result_size = image_size - kernel_size;
    std::vector<int> result(result_size);
    // 维护一个单调队列
    MonotonousQueue<int> Q(image_ptr, kernel_size, comp);
    // MonotonousQueueContinous<int> Q(image_ptr, kernel_size * 1, kernel_size + 1, comp);
    Q.emplace(0);
    // 开始最小值滤波
    for(int i = 1; i < image_size; ++i) {
        // 如果 i >= kernel, 说明当前窗口内已经记录了第 i - kernel 个窗口的最值
        if(i >= kernel_size)
            result[i - kernel_size] = image_data[Q.front()];
        // 尝试把第 i 个数据
        Q.emplace(i);
    }
    // 展示最值滤波结果
    display(result.data(), result_size);
}


/*
 * 拿 2d 数据测试一下单调队列的正确性, 水平做一次, 竖直方向做一次
 */
void test_2d_extremum_filtering() {
    std::cout << "\n\n\n=============== 二维数据的最小值滤波 ===============\n";
    using data_type = int;
    // 定义一个二维图像内容存储指针(存储是一维的)
    const int H = 4;
    const int W = 6;
    std::vector<data_type> image_data(H * W);
    // 定义随机种子, 生成二维数据
    std::default_random_engine seed(212);
    std::uniform_int_distribution engine(0, 20);
    // std::uniform_real_distribution<double> engine(0, 20);
    for(int i = 0, L = H * W; i < L; ++i)
        image_data[i] = engine(seed);
    // 打印
    auto display = [](data_type* const data, const int row, const int col){
        for(int i = 0; i < row; ++i) {
            for(int j = 0; j < col; ++j)
                std::cout << data[i * col + j] << "\t";
            std::cout << std::endl;
        }
    };
    display(image_data.data(), H, W);

    // 设定滤波的参数(默认是最小值滤波)
    const int kernel_size = 3;    // 滤波核长度
    assert(kernel_size > 0 and kernel_size & 1 and "kernel size must be odd");
    const int radius = (kernel_size - 1) >> 1;  // 滤波核半径
    const int EXTREMUM = (1 << 20);               // padding 填充值
    auto comp = [](const int l, const int r){return l <= r;};  // 决定是最小滤波还是最大滤波, 这个等于号很重要! 等于的数, 也要 pop 掉做更新
    // 对数据做 padding
    std::cout << "对数据做 padding\n";
    const int H2 = H + 2 * radius;   // padding 之后的图像高
    const int W2 = W + 2 * radius;   // padding 之后的图像宽
    std::vector<data_type> padded_data(H2 * W2, EXTREMUM);  // 存储 padding 之后的图像
    for(int i = 0; i < H; ++i) {
        data_type* const src_ptr = image_data.data() + i * W;  // 原图像第 i 行的指针
        data_type* const des_ptr = padded_data.data() + (i + radius) * W2 + radius;  // padding 后图像“有效内容”的第 i 行, 注意水平跟竖直方向上的 radius 偏移量
        std::memcpy(des_ptr, src_ptr, sizeof(data_type) * W);  // 拷贝这一行的内容
    }
    display(padded_data.data(), H2, W2);

    // 下一步, 准备做最小值滤波, 先做 H 行的最小值滤波
    std::cout << "做水平方向上的最小值滤波\n";
    std::vector<data_type> temp(H2 * W2, EXTREMUM);  // 找个临时变量, 存储水平滤波之后的结果
    for(int i = 0; i < H; ++i) {  // H 行, 做 H 次的水平滤波
        data_type* const row_ptr = padded_data.data() + (i + radius) * W2;  // 被滤波图像"有效内容"第 i 行的指针
        data_type* const res_ptr = temp.data() + (i + radius) * W2 + radius;// 水平滤波结果"有效内容"第 i 行的指针
        // MonotonousQueue<data_type> Q(row_ptr, kernel_size, comp);    // 直接 clear, empty 置换, 或者 list 改成数组
        MonotonousQueueContinous<data_type> Q(row_ptr, kernel_size, kernel_size, comp);
        Q.emplace(0);     // 当前方向要滤波的内容起点
        for(int j = 1; j < W2; ++j) {
            if(j >= kernel_size)
                res_ptr[j - kernel_size] = row_ptr[Q.front()];  // 窗口长达 kernel_size 之后, 每移动一次, 单调队列的 front() 代表当前窗口的最小值
            Q.emplace(j);    // 尝试把当前点放到单调队列中, 如果足够小, 则 pop 掉队列中比当前点大的值, 记录当前点坐标 j; 如果更大, 放进去, 检查窗口长度超出滤波核长度没有, 超出了就 pop 最老记录的数据
        }
        res_ptr[W2 - kernel_size] = row_ptr[Q.front()]; // 别忘了最后一个数据
    }
    display(temp.data(), H2, W2);
    std::vector<data_type>().swap(padded_data); // 清空, padded_data 用不着了, 释放对应内存

    // 做 W 列的最小值滤波
    std::cout << "做竖直方向上的最小值滤波\n";
    std::vector<data_type> result(H * W, EXTREMUM); // 存储最小值滤波的结果
    for(int i = 0; i < W; ++i) {
        data_type* const col_ptr = temp.data() + i + radius;  // 竖直方向上, 被滤波对象 temp “有效内容” 的第 i 列偏移地址, 在竖直方向上的下一个元素偏移 + W2
        data_type* const res_ptr = result.data() + i;         // 竖直方向上, 滤波结果放在第 i 列的起始偏移地址, 每次下一个元素偏移 + W
        // MonotonousQueue<data_type> Q(col_ptr, kernel_size * W2, comp); // 这一列内容的单调队列, 注意窗口间隔要乘以 W2(被滤波对象任意一行的元素个数)
        MonotonousQueueContinous<data_type> Q(col_ptr, kernel_size * W2, kernel_size, comp);
        Q.emplace(0);   // 这一列的滤波内容起点
        for(int j = 1; j < H2; ++j) {
            if(j >= kernel_size)
                res_ptr[(j - kernel_size) * W] = col_ptr[Q.front()]; // 窗口长达 kernel_size 之后, 每移动一次, 单调队列的 front() 代表当前窗口的最小值
            Q.emplace(j * W2); // 尝试把当前点放到单调队列中, 如果足够小, 则 pop 掉队列中比当前点大的值, 记录当前点坐标 j; 如果更大, 放进去, 检查窗口长度超出滤波核长度没有, 超出了就 pop 最老记录的数据
        }
        res_ptr[(H2 - kernel_size) * W] = col_ptr[Q.front()]; // 最后一个元素
    }
    display(result.data(), H, W);

}





/*
 * 这个是快速的最小值滤波, 算法来自 《Streaming Maximum-Minimum Filter Using No More than Three Comparisons per Element》
 * 其实也很简单, 就是用单调队列求移动窗口最大值, leetcode 里就有类似的经典题目
 * 如果要改成最大值滤波也很简单, 有下面两种手段
 *     1. 对滤波的目标全部取负, 然后最小值滤波, 滤波结果再取负(绝大多数情况都正确, 除了负数的最小值不能取负)
 *     2. 把下面代码中的 EXTREMUM 改成很小很小的值, 比如 -255, 甚至更小; 然后把 comp 里面的 l <= r 改成 l >= r 就可以做最大值滤波
 */
template<typename data_type>
void fast_min_filtering(cv::Mat& src, cv::Mat& des, const int kernel_size=3, const data_type EXTREMUM=255) {
    // 获取图像信息
    const int H = src.rows;
    const int W = src.cols;
    // 获取中间参数做准备
    const int radius = (kernel_size - 1) >> 1;  // 滤波核半径
    auto comp = [](const data_type l, const data_type r){ return l <= r; };  // 决定是最小滤波还是最大滤波, 这个等于号很重要! 等于的数, 也要 pop 掉做更新
    // 对数据做 padding
    const int H2 = H + 2 * radius;   // padding 之后的图像高
    const int W2 = W + 2 * radius;   // padding 之后的图像宽
    std::vector<data_type> padded_data(H2 * W2, EXTREMUM);  // 存储 padding 之后的图像
    for(int i = 0; i < H; ++i) {
        data_type* const src_ptr = src.ptr<data_type>() + i * W;  // 原图像第 i 行的指针
        data_type* const des_ptr = padded_data.data() + (i + radius) * W2 + radius;  // padding 后图像“有效内容”的第 i 行, 注意水平跟竖直方向上的 radius 偏移量
        std::memcpy(des_ptr, src_ptr, sizeof(data_type) * W);  // 拷贝这一行的内容
    }

    // 声明一个水平方向的单调队列
    MonotonousQueueContinous<data_type> Q1(kernel_size, kernel_size, comp);

    // 下一步, 准备做最小值滤波, 先做 H 行的最小值滤波
    std::vector<data_type> temp(H2 * W2, EXTREMUM);  // 找个临时变量, 存储水平滤波之后的结果
    for(int i = 0; i < H; ++i) {  // H 行, 做 H 次的水平滤波
        data_type* const row_ptr = padded_data.data() + (i + radius) * W2;  // 被滤波图像"有效内容"第 i 行的指针
        data_type* const res_ptr = temp.data() + (i + radius) * W2 + radius;// 水平滤波结果"有效内容"第 i 行的指针
        // MonotonousQueue<data_type> Q(row_ptr, kernel_size, comp);
        // MonotonousQueueContinous<data_type> Q(row_ptr, kernel_size, kernel_size, comp);
        Q1.reset(row_ptr);    // 如果不反复申请和析构单调队列, 需要在之前申明 Q1, 然后每次使用之前 reset 指定数据指针
        Q1.emplace(0);     // 当前方向要滤波的内容起点
        for(int j = 1; j < W2; ++j) {
            if(j >= kernel_size)
                res_ptr[j - kernel_size] = row_ptr[Q1.front()];  // 窗口长达 kernel_size 之后, 每移动一次, 单调队列的 front() 代表当前窗口的最小值
            Q1.emplace(j);    // 尝试把当前点放到单调队列中, 如果足够小, 则 pop 掉队列中比当前点大的值, 记录当前点坐标 j; 如果更大, 放进去, 检查窗口长度超出滤波核长度没有, 超出了就 pop 最老记录的数据
        }
        res_ptr[W2 - kernel_size] = row_ptr[Q1.front()]; // 别忘了最后一个数据
    }
    std::vector<data_type>().swap(padded_data); // 清空, padded_data 用不着了, 释放对应内存

    // 给结果创建为一个 H * W 的图像, 数据类型和被滤波图像 src 一致
    des.create(H, W, src.type());

    // 声明一个竖直方向上的单调队列, 最大窗口长度是 kernel_size 和 W2 的偏移量
    MonotonousQueueContinous<data_type> Q2(kernel_size * W2, kernel_size, comp);

    // 做 W 列的最小值滤波
    for(int i = 0; i < W; ++i) {
        data_type* const col_ptr = temp.data() + i + radius;  // 竖直方向上, 被滤波对象 temp “有效内容” 的第 i 列偏移地址, 在竖直方向上的下一个元素偏移 + W2
        data_type* const res_ptr = des.ptr<data_type>() + i;         // 竖直方向上, 滤波结果放在第 i 列的起始偏移地址, 每次下一个元素偏移 + W
        // MonotonousQueue<data_type> Q(col_ptr, kernel_size * W2, comp); // 这一列内容的单调队列, 注意窗口间隔要乘以 W2(被滤波对象任意一行的元素个数)
        // MonotonousQueueContinous<data_type> Q(col_ptr, kernel_size * W2, kernel_size, comp);
        Q2.reset(col_ptr);  // 重置指针到这一列
        Q2.emplace(0);   // 这一列的滤波内容起点
        for(int j = 1; j < H2; ++j) {
            if(j >= kernel_size)
                res_ptr[(j - kernel_size) * W] = col_ptr[Q2.front()]; // 窗口长达 kernel_size 之后, 每移动一次, 单调队列的 front() 代表当前窗口的最小值
            Q2.emplace(j * W2); // 尝试把当前点放到单调队列中, 如果足够小, 则 pop 掉队列中比当前点大的值, 记录当前点坐标 j; 如果更大, 放进去, 检查窗口长度超出滤波核长度没有, 超出了就 pop 最老记录的数据
        }
        res_ptr[(H2 - kernel_size) * W] = col_ptr[Q2.front()]; // 最后一个元素
    }
}


/*
 * 这个函数是暴力的最小值滤波器, 直接以当前点为中心, 边长 kernel_size 的滤波核暴力找
 * 时间复杂度 (H * W * k * k), k 是 kernel_size
 */
template<typename data_type>
void plain_min_filtering(cv::Mat& src, cv::Mat& des, const int kernel_size=3, const data_type EXTREMUM=255) {
    // 获取图像信息
    const int H = src.rows;
    const int W = src.cols;
    // 获取中间参数做准备
    const int radius = (kernel_size - 1) >> 1;  // 滤波核半径
    // 对数据做 padding
    const int H2 = H + 2 * radius;   // padding 之后的图像高
    const int W2 = W + 2 * radius;   // padding 之后的图像宽
    std::vector<data_type> padded_data(H2 * W2, EXTREMUM);  // 存储 padding 之后的图像
    for(int i = 0; i < H; ++i) {
        data_type* const src_ptr = src.ptr<data_type>() + i * W;  // 原图像第 i 行的指针
        data_type* const des_ptr = padded_data.data() + (i + radius) * W2 + radius;  // padding 后图像“有效内容”的第 i 行, 注意水平跟竖直方向上的 radius 偏移量
        std::memcpy(des_ptr, src_ptr, sizeof(data_type) * W);  // 拷贝这一行的内容
    }

    // 给结果分配空间
    des.create(H, W, src.type());

    // 准备一个偏移量模板
    int max_k = 0;
    std::vector<int> offset(kernel_size * kernel_size, 0);
    for(int i = -radius; i <= radius; ++i)
        for(int j = -radius; j <= radius; ++j) {
            if(i == 0 and j == 0)
                continue;
            offset[max_k++] = i * W2 + j;
        }

    // 直接暴力做最小值滤波
    for(int i = 0; i < H; ++i) {
        data_type* const res_ptr = des.ptr<data_type>() + i * W;
        data_type* const src_ptr = padded_data.data() + (radius + i) * W2 + radius;
        for(int j = 0; j < W; ++j) {
            // 开始找
            data_type min_value = src_ptr[j];
            for(int k = 0; k < max_k; ++k) {
                data_type neighbor = src_ptr[j + offset[k]];
                if(min_value > neighbor)
                    min_value = neighbor;
            }
            res_ptr[j] = min_value;
        }
    }
}





void test_dark_channel() {

    // 读取图像
    cv::Mat color_image = cv::imread("./images/input/demo.png");
    assert(not color_image.empty() and color_image.type() == CV_8UC3);
    const int H = color_image.rows;
    const int W = color_image.cols;

    // 求每个点在 rgb 上的最小值
    cv::Mat rgb_min(H, W, CV_8UC1);
    uchar* const color_ptr = color_image.ptr<uchar>();
    uchar* const min_ptr = rgb_min.ptr<uchar>();
    const int length = H * W;
    for(int i = 0; i < length; ++i) {
        const int p = 3 * i;
        min_ptr[i] = std::min(std::min(color_ptr[p], color_ptr[p + 1]), color_ptr[p + 2]);
    }

    const int kernel_size = 25;
    // 暴力的最小值滤波
    cv::Mat plain_dark_channel;
    run([&](){
        plain_min_filtering<uchar>(rgb_min, plain_dark_channel, kernel_size);
    }, "暴力最小值滤波");

    // 快速的最小值滤波
    cv::Mat fast_dark_channel;
    run([&](){
        fast_min_filtering<uchar>(rgb_min, fast_dark_channel, kernel_size);
    }, "快速最小值滤波");

    // 判断二者的内容是否一致
    int ii = 0;
    for(; ii < length; ++ii)
        if(plain_dark_channel.data[ii] != fast_dark_channel.data[ii])
            break;
    std::cout << "内容是否一致===>  " << std::boolalpha << (ii == length) << std::endl;

    // 展示
    cv_show(cv_concat({rgb_min, plain_dark_channel, fast_dark_channel}));
    cv_write(cv_concat({fast_dark_channel}), "./images/output/output.png");
}


int main() {
    std::setbuf(stdout, 0);

    // 测试一维的最小值滤波
    test_1d_extremum_filtering();

    // 测试一维的最小值滤波
    test_2d_extremum_filtering();

    // 测试二维图像的最小值滤波核
    test_dark_channel();

    return 0;
}



