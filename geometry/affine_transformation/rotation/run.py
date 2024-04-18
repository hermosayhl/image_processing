# Python
import os
import sys
import time
import math
import ctypes
# 3rd party
import cv2
import numpy


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# 读取图像
image_path   = "./a0372-WP_CRW_6207.png"
origin_image = cv2.imread(image_path)
height, width, channel = origin_image.shape
# cv_show(origin_image)


# 根据可视角度生成一个变换矩阵
def get_rotation_matrix(theta, center=(0, 0), scale=1.0):
	sin_theta = math.sin(math.radians(theta))
	cos_theta = math.cos(math.radians(theta))
	cx, cy = center
	# 这个是逆时针旋转! OpenCV 跟下面不一样, 它默认是顺时针旋转
	# https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
	return numpy.array([
		scale * cos_theta, scale * -sin_theta, cx - (cx * cos_theta - cy * sin_theta),
		scale * sin_theta, scale * cos_theta,  cy - (cx * sin_theta + cy * cos_theta),
		0,                 0,                  1
	]).reshape((3, 3)).astype("float32")

# 获取旋转角度 30° 的变换矩阵
rotate_matrix = get_rotation_matrix(theta=30)
print(rotate_matrix)


# 编译 C++ 代码生成
affine_lib_code = "./crane_affine.cpp"
affine_lib_path = "./crane_affine.so"
os.system("g++ -fPIC -shared -O2 {} -o {}".format(affine_lib_code, affine_lib_path))

# 加载动态库
affine_lib = ctypes.cdll.LoadLibrary(affine_lib_path)


# 准备一个函数做接口
def make_affine(x, H, mode="nearest"):
	if (len(x) < 2): 
		return
	elif (len(x) < 3): 
		x = numpy.expand_dims(x, axis=-1)
	if (H.dtype != numpy.float32): 
		H = H.astype(numpy.float32)
	h, w, c = x.shape
	# 准备一个结果存储旋转结果
	result = numpy.zeros(x.shape, x.dtype)
	affine_lib.affine_transform(
		result.ctypes.data_as(ctypes.c_char_p),
		x.ctypes.data_as(ctypes.c_char_p),
		H.ctypes.data_as(ctypes.c_char_p),
		h, w, c,
		mode.encode()
	)
	return result


# 做一次最简单的以 (0, 0) 为中心的旋转
rotate_result = make_affine(origin_image, rotate_matrix)
cv_write("./rotate_result.png", rotate_result)
cv_show(rotate_result)


# 获取角度 30°, 以图像中心为中心的旋转矩阵
rotate_matrix_2 = get_rotation_matrix(theta=30, center=(int(height / 2), int(width / 2)))
print(rotate_matrix_2)
# 再做一次旋转
rotate_result_2 = make_affine(origin_image, rotate_matrix_2)
cv_write("./rotate_result_2.png", rotate_result_2)
cv_show(rotate_result_2)


# 试一试放缩
rotate_matrix_3 = get_rotation_matrix(
	theta=30, 
	center=(int(height / 2), int(width / 2)),
	scale=0.5)
print(rotate_matrix_3)
# 再做一次旋转
rotate_result_3 = make_affine(origin_image, rotate_matrix_3)
cv_write("./rotate_result_3.png", rotate_result_3)
cv_show(rotate_result_3)

# 简述下两个平移项的数学意义
# 可以看成两步, 先按照 (0, 0) 为中心的旋转, 可以看到图像的中心点也被旋转了
# 如果是按照图像中心点为中心的旋转, 则图像中心的坐标是不变的, 所以要加上新旧坐标的插值
# 即 (x, y) 减去 (xcosθ - ysinθ, xsinθ + ycosθ)
