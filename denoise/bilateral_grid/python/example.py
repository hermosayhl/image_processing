import ctypes
import numpy
import cv2
 
# 加载动态库
lib = ctypes.cdll.LoadLibrary("./example.so")
# 准备一个输入
image = cv2.imread("greekdome.ppm")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype("float32") / 255;
input_ptr = image.ctypes.data_as(ctypes.c_char_p)

# 准备一个结果
rows, cols = image.shape
result = numpy.zeros((rows, cols)).astype("float32")
res_ptr = result.ctypes.data_as(ctypes.c_char_p)

# 做插值
lib.fast_bilateral_approximation(
	res_ptr, 
	input_ptr, 
	input_ptr, 
	rows, 
	cols, 
	ctypes.c_float(4.0), 
	ctypes.c_float(0.05), 
	ctypes.c_int(2))

# 转换
result = (result * 255).astype("uint8")
cv2.imwrite("output.png", result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imshow('crane', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result.shape)