import ctypes
import numpy
import cv2
 
# 加载动态库
lib = ctypes.cdll.LoadLibrary("./example.so")
# 准备一个输入
image = cv2.imread("greekdome.ppm")
image = image.astype("float32") / 255;
image = numpy.ascontiguousarray(image.transpose(2, 0, 1))
channel, rows, cols = image.shape

result = []
for c in range(image.shape[0]):
	input_ptr = image[c].ctypes.data_as(ctypes.c_char_p)
	result_current = numpy.zeros((rows, cols)).astype("float32")
	res_ptr = result_current.ctypes.data_as(ctypes.c_char_p)
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
	result.append(result_current)
result = numpy.stack(result, axis=-1).__mul__(255).astype("uint8")

# 转换
cv2.imwrite("output.png", result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imshow('crane', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

