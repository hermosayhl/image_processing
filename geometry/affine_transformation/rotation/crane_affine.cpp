// C++
#include <cmath>
#include <vector>
#include <cstring>
#include <iostream>




template<typename src_type=float, typename res_type=int>
res_type clip(const src_type x, const res_type low, const res_type high) {
	if (x < low)       return low;
	else if (x > high) return high;
	else               return x;
}



template<typename src_type=unsigned char>
void affine_transform_inplementation(
		src_type* result,
		src_type* source,
		float* H,
		const int height,
		const int width,
		const int channel,
		const char* mode) {
	// 如果是最近邻插值
	if (std::strcmp(mode, "nearest") == 0) {
		// 不考虑孔洞
		if (false) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					// H 和坐标 (i, j, 1) 操作, 得到变换的新坐标
					float x = H[0] * i + H[1] * j + H[2];
					float y = H[3] * i + H[4] * j + H[5];
					// 最近邻, 找到最相近的坐标
					int __x = int(x + 0.5f);
					int __y = int(y + 0.5f);
					// 判断是否超出边界
					if (__x < 0 or __x >= height or __y < 0 or __y >= width)
						continue;
					// 将位置 (i, j) 三个通道的值赋给位置 (x, y)
					src_type* src_ptr = source + (i   * width + j)   * channel;
					src_type* res_ptr = result + (__x * width + __y) * channel;
					for (int c = 0; c < channel; ++c) {
						res_ptr[c] = src_ptr[c];
					} 
				}
			}
		}
		// 记录每一个孔洞, 做填充
		else {
			std::vector<int> visited(height * width, 0);
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					// H 和坐标 (i, j, 1) 操作, 得到变换的新坐标
					float x = H[0] * i + H[1] * j + H[2];
					float y = H[3] * i + H[4] * j + H[5];
					// 最近邻, 找到最相近的坐标
					int __x = int(x + 0.5f);
					int __y = int(y + 0.5f);
					// 判断是否超出边界
					if (__x < 0 or __x >= height or __y < 0 or __y >= width)
						continue;
					// 标记新位置是被赋值了的
					const int target_pos = __x * width + __y;
					visited[target_pos]  = 1;
					// 将位置 (i, j) 三个通道的值赋给位置 (x, y)
					src_type* src_ptr = source + (i * width + j) * channel;
					src_type* res_ptr = result + target_pos      * channel;
					for (int c = 0; c < channel; ++c) {
						res_ptr[c] = src_ptr[c];
					} 
				}
			}
			// 对剩下的空洞做填充, 比如简单的上下左右四个取均值
			std::vector<int> offset({-width, width, 1, -1});
			for (int i = 1, i_end = height - 1; i < i_end; ++i) {
				for (int j = 1, j_end = width - 1; j < j_end; ++j) {
					// 判断当前位置是否是空的
					const int this_pos = i * width + j;
					if (visited[this_pos] != 0)
						continue;
					// 初始化累积量
					std::vector<float> temp(channel, 0.f);
					// 遍历周围四个点
					int spatial_count = 0;
					for (int k = 0; k < 4; ++k) {
						const int neighbor_pos = this_pos + offset[k];
						if (visited[neighbor_pos] == 0)
							continue;
						// 记录实际遍历的点的个数
						++spatial_count;
						// 找到对应位置的值加权
						src_type* src_ptr = result + neighbor_pos * channel;
						for (int c = 0; c < channel; ++c) {
							temp[c] += src_ptr[c];
						}
					}
					// 如果四周都是空的, 不做插值
					if (spatial_count == 0) 
						continue;
					// 对 this_pos 的多通道进行赋值
					src_type* res_ptr = result + this_pos * channel;
					for (int c = 0; c < channel; ++c) {
						res_ptr[c] = temp[c] / spatial_count;
					}
				}
			}
		}
	}
	// 如果是双线性插值
	else if (strcmp(mode, "biliear") == 0) {
		// 生成每一个点
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				// 注意使用的是逆矩阵 (x, y) = H^-1 * [x, y, 1]
				float x = H[0] * i + H[1] * j + H[2];
				float y = H[3] * i + H[4] * j + H[5];
				// 如果超出了图像范围, 找不到
				constexpr float eps{0.00001f};
				if ((x + eps > height - 1) or (x - eps < 0) or (y + eps > width - 1) or (y - eps < 0))
					continue;
				// 找到 (x, y) 的下界
				int x_low  = std::floor(x);
				int y_low  = std::floor(y);
				// 计算 x, y 方向上的加权值
				float x_high_weight = x - x_low;
				float y_high_weight = y - y_low;
				// 找到对应在 source 中的位置, 做双线性加权
				src_type* Q1 = source + (x_low * width + y_low)  * channel;
				src_type* Q2 = Q1 + channel;
				src_type* Q3 = Q1 + width * channel;
				src_type* Q4 = Q3 + channel;
				// 找到被赋值的位置
				src_type* res_ptr = result + (i * width + j) * channel;
				// 多通道分别算
				for (int c = 0; c < channel; ++c) {
					float up   = (1.f - y_high_weight) * Q1[c] + y_high_weight * Q2[c];
					float down = (1.f - y_high_weight) * Q3[c] + y_high_weight * Q4[c];
					res_ptr[c] = (1.f - x_high_weight) * up + x_high_weight * down;
				}
			}
		}
	}
}



extern "C" {
	void affine_transform(
		unsigned char* result,
		unsigned char* source,
		float* H,
		const int height,
		const int width,
		const int channel,
		const char* mode) {
		affine_transform_inplementation(result, source, H, height, width, channel, mode);
	}
}