// C & C++
#include <list>
#include <iostream>
#include <random>
#include <assert.h>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>


void run(const std::function<void()>& work=[]{}, const std::string message="") {
    auto start = std::chrono::high_resolution_clock::now();
    work();
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);
    std::cout << message << " " << duration.count() * 1.0 / 1e3 << " us" <<  std::endl;
}


template<typename T>
void display(const T* const matrix_ptr, const int height, const int width) {
	for (int i = 0; i < height; ++i) {
		const T* const row_ptr = matrix_ptr + i * width;
		for (int j = 0; j < width; ++j) 
			std::cout << row_ptr[j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


template<typename T>
bool verify_is_same(const T* const lhs, const T* const rhs, const int length) {
	for (int i = 0; i < length; ++i) 
		if (std::abs(lhs[i] - rhs[i]) > 1e-5)
			return false;
	return true;
}



template<typename T>
void plain_min_filter(T* const des_ptr, 
					  const T* const src_ptr, 
					  const int H, 
					  const int W, 
					  const int radius, 
					  const T EXTREMUM) {
	// 求解 H * W 个点
	for (int i = 0; i < H; ++i) {
		auto res_ptr = des_ptr + i * W;
		for (int j = 0; j < W; ++j) {
			// 初始值
			T temp{EXTREMUM};
			// 遍历半径为 radius 的局部区域
			for (int u = -radius; u <= radius; ++u) {
				const int i_u = i + u;
				if (i_u < 0 or i_u >= H)  // 越界判断
					continue;
				for (int v = -radius; v <= radius; ++v) {
					const int j_v = j + v;
					if (j_v < 0 or j_v >= W)
						continue;
					temp = std::min(temp, src_ptr[i_u * W + j_v]);
				}
			}
			res_ptr[j] = temp;
		}
	}
}




template<typename T>
void split_min_filter(T* const des_ptr, const T* const src_ptr, const int H, const int W, const int radius, const T EXTREMUM) {
	// 准备一个中间结果存放第一次水平滤波的结果
	std::vector<T> first_result(H * W);
	// “每一行” 先做水平方向上的最小值滤波
	for (int i = 0; i < H; ++i) {
		auto row_ptr = src_ptr + i * W;
		auto res_ptr = first_result.data() + i * W;
		for (int j = 0; j < W; ++j) {
			T temp{EXTREMUM};
			// 遍历水平方向
			for (int v = -radius; v <= radius; ++v) {
				const int j_v = j + v;
				if (j_v < 0 or j_v >= W)
					continue;
				temp = std::min(temp, row_ptr[j_v]);
			}
			res_ptr[j] = temp;
		}
	}
	// 在第一次滤波结果上, 对每一列做竖直方向上的最小值滤波
	for (int j = 0; j < W; ++j) {
		for (int i = 0; i < H; ++i) {
			T temp{EXTREMUM};
			// 遍历竖直方向
			for (int u = -radius; u <= radius; ++u) {
				const int i_u = i + u;
				if (i_u < 0 or i_u >= H)
					continue;
				temp = std::min(temp, first_result[i_u * W + j]);
			}
			des_ptr[i * W + j] = temp;
		}
	}
}





template<typename T>
void monotony_queue_min_filter(T* const des_ptr, const T* const src_ptr, const int H, const int W, const int radius, const T EXTREMUM) {
	
	struct LoopQueue final {
	private:
		int *data    = nullptr; 
		int __front  = 0;
		int __back   = 0;
		int win_len  = 0;
		int capacity = 0;
	public:
		LoopQueue(const int kernel_size)
			: data(new int[kernel_size]), win_len(kernel_size), capacity(kernel_size + 1) {}
		~LoopQueue() noexcept {
			if (data != nullptr) {
				delete data;
				data = nullptr;
			}
		}
		inline void clear() {
			__front = __back = 0;
		}
		inline bool empty() const {
			return __front == __back;
		}
		inline int front() const {
			return data[(__front + 1) % capacity];
		}
		inline int back() const {
			return data[__back];
		}
		inline void emplace_back(const int pos) {
			__back = (__back + 1) % capacity;
			data[__back] = pos;
		}
		inline void pop_back() {
			__back = (__back - 1 + capacity) % capacity;
		}
		inline void pop_front() {
			__front = (__front + 1) % capacity;
		}
	};

	// 准备一块缓冲区, 存储当前滤波的一行或者一列
	const int H_2radius = H + 2 * radius;
	const int W_2radius = W + 2 * radius;
	const int pad_size  = std::max(H_2radius, W_2radius);
	T buffer[pad_size];
	std::fill(buffer, buffer + pad_size, EXTREMUM);

	// 准备一个单调队列
	const int kernel_size = 2 * radius + 1;
	LoopQueue Q(kernel_size);
	// std::list<int> Q;

	// 先做水平方向的最小值滤波, 做 H 次
	for (int i = 0; i < H; ++i) {
		// 把当前行数据放到 buffer(栈区), 
		std::memcpy(buffer + radius, src_ptr + i * W, sizeof(T) * W);
		// 放置结果
		T* res_ptr = des_ptr + i * W;
		// 队列置空
		Q.clear();

		for (int j = 0; j < W_2radius; ++j) {
			// j >= kernel 说明第一个滤波结果已经算出来了, 记录结果
			if (j >= kernel_size)
				res_ptr[j - kernel_size] = buffer[Q.front()];
			// 尝试把 buffer[j] 放到单调队列, 比当前值 buffer[j] 更大的都弹出去
			while (not Q.empty()) {
				const auto tail = buffer[Q.back()];
				if (buffer[j] < tail)
					Q.pop_back();
				else break;
			}
			// push 当前值
			Q.emplace_back(j);
			// 如果维护的区间超过了滤波窗口长度, pop_front()
			if (j - Q.front() == kernel_size)
				Q.pop_front();
		}
		res_ptr[W_2radius - kernel_size] = buffer[Q.front()];
	}

	// 做竖直方向上的最小值滤波
	std::fill(buffer, buffer + pad_size, EXTREMUM);

	for (int j = 0; j < W; ++j) {
		// 把图像数据从 image(堆) 拷贝到 buffer(栈)
		for (int k = 0; k < H; ++k)
			buffer[radius + k] = des_ptr[k * W + j];
		// 放置结果
		auto res_ptr = des_ptr + j;
		// 队列置空
		Q.clear();

		for (int i = 0; i < H_2radius; ++i) {
			if (i >= kernel_size)
				res_ptr[(i - kernel_size) * W] = buffer[Q.front()];
			// 尝试把 buffer[i] 放到单调队列
			while (not Q.empty()) {
				const auto tail = buffer[Q.back()];
				if (buffer[i] < tail)
					Q.pop_back();
				else break;
			}
			// 放入 buffer[i]
			Q.emplace_back(i);
			// 如果超出了窗口
			if (i - Q.front() == kernel_size)
				Q.pop_front();
		}
		res_ptr[(H_2radius - kernel_size) * W] = buffer[Q.front()];
	}
}








// 把队列放在栈上
template<typename T>
void faster_monotony_queue_min_filter(T* const des_ptr, const T* const src_ptr, const int H, const int W, const int radius, const T EXTREMUM) {
	
	// 准备一块缓冲区, 存储当前滤波的一行或者一列
	const int H_2radius = H + 2 * radius;
	const int W_2radius = W + 2 * radius;
	const int pad_size  = std::max(H_2radius, W_2radius);
	T buffer[pad_size];
	std::fill(buffer, buffer + pad_size, EXTREMUM);

	// 准备一个单调队列
	const int kernel_size = 2 * radius + 1;
	int Q[kernel_size];
	int front    = 0;
	int back     = 0;
	int capacity = kernel_size + 1;

	// 先做水平方向的最小值滤波, 做 H 次
	for (int i = 0; i < H; ++i) {
		// 把当前行数据放到 buffer(栈区), 
		std::memcpy(buffer + radius, src_ptr + i * W, sizeof(T) * W);
		// 放置结果
		T* res_ptr = des_ptr + i * W;
		// 队列置空
		front = back = 0;

		for (int j = 0; j < W_2radius; ++j) {
			// j >= kernel 说明第一个滤波结果已经算出来了, 记录结果
			if (j >= kernel_size)
				res_ptr[j - kernel_size] = buffer[Q[(front + 1) % capacity]];
			// 尝试把 buffer[j] 放到单调队列, 比当前值 buffer[j] 更大的都弹出去
			while (front != back) {
				const auto tail = buffer[Q[back]];
				if (buffer[j] < tail)
					back = (back - 1 + capacity) % capacity;
				else break;
			}
			// push 当前值
			back = (back + 1) % capacity;
            Q[back] = j;
			// 如果维护的区间超过了滤波窗口长度, pop_front()
			int front_next = (front + 1) % capacity;
            if(j - Q[front_next] == kernel_size)
                front = front_next;
		}
		res_ptr[W_2radius - kernel_size] = buffer[Q[(front + 1) % capacity]];
	}

	// 做竖直方向上的最小值滤波
	std::fill(buffer, buffer + pad_size, EXTREMUM);

	for (int j = 0; j < W; ++j) {
		// 把图像数据从 image(堆) 拷贝到 buffer(栈)
		for (int k = 0; k < H; ++k)
			buffer[radius + k] = des_ptr[k * W + j];
		// 放置结果
		auto res_ptr = des_ptr + j;
		// 队列置空
		front = back = 0;

		for (int i = 0; i < H_2radius; ++i) {
			if (i >= kernel_size)
				res_ptr[(i - kernel_size) * W] = buffer[Q[(front + 1) % capacity]];
			// 尝试把 buffer[i] 放到单调队列
			while (front != back) {
				const auto tail = buffer[Q[back]];
				if (buffer[i] < tail)
					back = (back - 1 + capacity) % capacity;
				else break;
			}
			// 放入 buffer[i]
			back = (back + 1) % capacity;
            Q[back] = i;
			// 如果超出了窗口
			int front_next = (front + 1) % capacity;
            if(i - Q[front_next] == kernel_size)
                front = front_next;
		}
		res_ptr[(H_2radius - kernel_size) * W] = buffer[Q[(front + 1) % capacity]];
	}
}

















template<typename T>
void dynamic_programming_min_filter(T* const des_ptr, const T* const src_ptr, const int H, const int W, const int radius, const T EXTREMUM) {
	// 计算水平和竖直 buffer 缓冲区的大小
	const int H_2radius = H + 2 * radius;
	const int W_2radius = W + 2 * radius;
	const int pad_size  = std::max(H_2radius, W_2radius);

	// 算至少需要分成几段, 以此计算出最大的预留空间
	const int kernel_size   = 2 * radius + 1;
	const int segment_count = (pad_size + kernel_size - 1) / kernel_size;
	const int max_length    = segment_count * kernel_size;

	// 申请 buffer
	T buffer[max_length];
	std::fill(buffer, buffer + max_length, EXTREMUM);

	// 申请前向数组跟反向数组
	T forward[max_length];
	T backward[max_length];

	// 先对水平方向做最小值滤波, 做 H 次
	for (int i = 0; i < H; ++i) {
		// 把这一行拷贝到 buffer
		std::memcpy(buffer + radius, src_ptr + i * W, sizeof(T) * W);
		// 对 segment_count 段, 分别计算前向数组和反向数组
		for (int s = 0; s < segment_count; ++s) {
			// 找到对应数据的起始位置
			const int offset = s * kernel_size;
			auto for_ptr  = forward + offset;
			auto back_ptr = backward + offset;
			auto buf_ptr  = buffer + offset;
			// 计算前向数组
			for_ptr[0] = buf_ptr[0];
			for (int k = 1; k < kernel_size; ++k)
				for_ptr[k] = std::min(for_ptr[k - 1], buf_ptr[k]);
			// 计算反向数组
			back_ptr[kernel_size - 1] = buf_ptr[kernel_size - 1];
			for (int k = kernel_size - 2; k >= 0; --k)
				back_ptr[k] = std::min(back_ptr[k + 1], buf_ptr[k]);
		}
		// 这一行的上升数组和下降数组都计算完成, 开始写回
		auto res_ptr = des_ptr + i * W;
		for (int j = radius, j_end = W + radius; j < j_end; ++j)
			res_ptr[j - radius] = std::min(forward[j + radius], backward[j - radius]);
	}

	// 再做竖直方向的最小值滤波, 做 W 次
	for (int j = 0; j < W; ++j) {
		// 把这一列拷贝到 buffer
		for (int k = 0; k < H; ++k) 
			buffer[radius + k] = des_ptr[k * W + j];
		// 对 segment_count 段, 分别计算前向数组和反向数组
		for (int s = 0; s < segment_count; ++s) {
			// 找到对应数据的起始位置
			const int offset = s * kernel_size;
			auto for_ptr  = forward + offset;
			auto back_ptr = backward + offset;
			auto buf_ptr  = buffer + offset;
			// 计算前向数组
			for_ptr[0] = buf_ptr[0];
			for (int k = 1; k < kernel_size; ++k)
				for_ptr[k] = std::min(for_ptr[k - 1], buf_ptr[k]);
			// 计算反向数组
			back_ptr[kernel_size - 1] = buf_ptr[kernel_size - 1];
			for (int k = kernel_size - 2; k >= 0; --k)
				back_ptr[k] = std::min(back_ptr[k + 1], buf_ptr[k]);
		}
		// 这一行的上升数组和下降数组都计算完成, 开始写回
		for (int i = radius, i_end = H + radius; i < i_end; ++i) 
			des_ptr[(i - radius) * W + j] = std::min(forward[i + radius], backward[i - radius]);
	}
}	


// 是否有其它的考虑分块提高 cache 命中率的? 第二次竖直方向写回在不断跳跃






int main() {

	std::vector<int> image({
		25, 43, 79, 24,  0, 48, 99, 76, 20,
        75,  7, 41, 31, 57, 41, 86, 24, 91,
        28, 43,  6, 20, 94,  0, 42, 19, 94,
        32, 13, 99, 59, 10, 64, 46, 66, 38,
        92,  8, 90, 88, 82, 69,  0, 27, 70
	});

	const int height = 5;
	const int width  = 9;
	assert(height * width == static_cast<int>(image.size()));
	
	// 先来个暴力的
	{
		std::vector<int> result(height * width, -1);
		plain_min_filter(result.data(), image.data(), height, width, 1, (1 << 30));
		display(result.data(), height, width);
	}

	// 分水平跟竖直方向上的处理
	{
		std::vector<int> result(height * width, -1);
		split_min_filter(result.data(), image.data(), height, width, 1, (1 << 30));
		display(result.data(), height, width);
	}

	// 来个单调队列
	{
		std::vector<int> result(height * width, -1);
		monotony_queue_min_filter(result.data(), image.data(), height, width, 1, (1 << 30));
		display(result.data(), height, width);
	}


	// 单调队列放栈上
	{
		std::vector<int> result(height * width, -1);
		faster_monotony_queue_min_filter(result.data(), image.data(), height, width, 1, (1 << 30));
		display(result.data(), height, width);
	}


	// 动态规划
	{
		std::vector<int> result(height * width, -1);
		dynamic_programming_min_filter(result.data(), image.data(), height, width, 1, (1 << 30));
		display(result.data(), height, width);
	}


	return 0;
}






extern "C" {
	void plain_min_filter_uint8(
			unsigned char* const des_ptr, 
			const unsigned char* const src_ptr, 
			const int H, 
			const int W, 
			const int radius, 
			const unsigned char EXTREMUM) {
		plain_min_filter<unsigned char>(des_ptr, src_ptr, H, W, radius, EXTREMUM);
	}	

	void split_min_filter_uint8(
			unsigned char* const des_ptr, 
			const unsigned char* const src_ptr, 
			const int H, 
			const int W, 
			const int radius, 
			const unsigned char EXTREMUM) {
		split_min_filter<unsigned char>(des_ptr, src_ptr, H, W, radius, EXTREMUM);
	}	

	void monotony_queue_min_filter_uint8(
			unsigned char* const des_ptr, 
			const unsigned char* const src_ptr, 
			const int H, 
			const int W, 
			const int radius, 
			const unsigned char EXTREMUM) {
		monotony_queue_min_filter<unsigned char>(des_ptr, src_ptr, H, W, radius, EXTREMUM);
	}	

	void faster_monotony_queue_min_filter_uint8(
			unsigned char* const des_ptr, 
			const unsigned char* const src_ptr, 
			const int H, 
			const int W, 
			const int radius, 
			const unsigned char EXTREMUM) {
		faster_monotony_queue_min_filter<unsigned char>(des_ptr, src_ptr, H, W, radius, EXTREMUM);
	}	

	void dynamic_programming_min_filter_uint8(
			unsigned char* const des_ptr, 
			const unsigned char* const src_ptr, 
			const int H, 
			const int W, 
			const int radius, 
			const unsigned char EXTREMUM) {
		dynamic_programming_min_filter<unsigned char>(des_ptr, src_ptr, H, W, radius, EXTREMUM);
	}	
}

