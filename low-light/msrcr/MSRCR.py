import os
import cv2
import math
import numpy
import matplotlib.pyplot as plt

def cv_show(image, message='crane'):
	cv2.imshow(message, image)
	cv2.waitKey(0);
	cv2.destroyAllWindows()


def numpy_log(x): 
	return numpy.log(x + 1)


def mean_std_normalize(result, dynamic=2.0):
	mean = numpy.mean(result, axis=(0, 1))
	stdvar = numpy.sqrt(numpy.var(result, axis=(0, 1)))
	min_value = mean - dynamic * stdvar
	max_value = mean + dynamic * stdvar
	result = (result - min_value) / (max_value - min_value)
	result = 255 * numpy.clip(result, 0, 1)
	return result.astype("uint8")


def max_min_normalize(result):
	min_value = result.min()
	dynamic = result.max() - min_value
	result = 255 * (result - min_value) / dynamic
	return result.astype("uint8")


# 画直方图分布
def plot_hist(picture):
	(n, bins) = numpy.histogram(picture, bins=100, range=(picture.min(), picture.max()), density=True)
	plt.plot(.5*(bins[1:]+bins[:-1]), n)
	plt.show()


# 单尺度的 SSR 增强, Single scale retinex
def SSR(low_light, sigma=15, dynamic=2):
	low_light = low_light.astype("float32")
	log_I = numpy_log(low_light)
	log_L = cv2.GaussianBlur(log_I, (0, 0), sigma)
	log_R = log_I - log_L
	# return max_min_normalize(log_R) # 直接最大最小值标准化
	# plot_hist(log_R)
	log_R = numpy.exp(log_R)  # 做了 exp 可能会放大范围
	# plot_hist(log_R)
	return mean_std_normalize(log_R, dynamic)


def MSR(low_light, sigmas=[10, 50, 100], weights=[0., 0, 0], dynamic=2):
	weights = numpy.array(weights) / numpy.sum(weights)
	low_light = low_light.astype("float32")
	log_I = numpy_log(low_light)
	log_Ls = [cv2.GaussianBlur(log_I, (0, 0), sig) for sig in sigmas]
	log_R = weights[0] * (log_I - log_Ls[0])
	for i in range(1, len(weights)):
		log_R += weights[i] * (log_I - log_Ls[i])
	temp = numpy.exp(log_R)
	return mean_std_normalize(temp, dynamic)


def MSRCR(low_light, sigmas=[15, 80, 200], weights=[0.33, 0.33, 0.34], alpha=128, dynamic=2.0):
	assert len(sigmas) == len(weights), "scales are not consistent !"
	weights = numpy.array(weights) / numpy.sum(weights)
	# 图像转成 float 处理
	low_light = low_light.astype("float32")
	# 转到 log 域
	log_I = numpy_log(low_light)
	# 每个尺度下做高斯模糊, 提取不同的平滑层, 作为光照图的估计
	log_Ls = [cv2.GaussianBlur(log_I, (0, 0), sig) for sig in sigmas]
	# 多个尺度的 MSR 叠加
	log_R = numpy.stack([weights[i] * (log_I - log_Ls[i]) for i in range(len(sigmas))])
	log_R = numpy.sum(log_R, axis=0)
	# 颜色恢复
	norm_sum = numpy_log(numpy.sum(low_light, axis=2))
	result = log_R * (numpy_log(alpha * low_light) - numpy.atleast_3d(norm_sum))
	# result = numpy.exp(result)
	# 标准化
	return mean_std_normalize(result, dynamic)



# 读取图像
image_path = './input/Balloons.png'
low_light = cv2.imread(image_path)

# 多尺度分解加权
result = MSRCR(
	low_light, 
	sigmas=[5, 25, 50, 75, 100], 
	weights=[0.2, 0.2, 0.2, 0.3, 0.1], 
	alpha=128, 
	dynamic=2.0)
cv2.imwrite("./output/MSRCR.png", result)
cv_show(numpy.concatenate([low_light, result], axis=1))

# 单尺度增强
result = SSR(low_light, 10)
cv_show(numpy.concatenate([low_light, result], axis=1))

result = SSR(low_light, 50)
cv_show(numpy.concatenate([low_light, result], axis=1))

result = SSR(low_light, 100)
cv2.imwrite("./output/SSR.png", result)
cv_show(numpy.concatenate([low_light, result], axis=1))

# 多尺度增强
result = MSR(low_light, [5, 25, 50, 75, 100], [0.2, 0.2, 0.2, 0.3, 0.1])
cv2.imwrite("./output/MSR.png", result)
cv_show(numpy.concatenate([low_light, result], axis=1))