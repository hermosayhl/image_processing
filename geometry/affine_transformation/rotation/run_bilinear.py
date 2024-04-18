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
rotate_matrix = get_rotation_matrix(theta=30, center=(int(height / 2), int(width / 2)))
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


# 求得旋转矩阵的逆矩阵
rotate_matrix_inv = numpy.linalg.inv(rotate_matrix)

# 可以使用双线性(其实, 如果用逆这种方式可以解出来, 最近邻这些都可以用)
rotate_result = make_affine(origin_image, rotate_matrix_inv, mode="biliear")
cv_write("./rotate_result_bilinear.png", rotate_result)
cv_show(rotate_result)