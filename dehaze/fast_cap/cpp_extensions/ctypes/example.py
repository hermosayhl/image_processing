import ctypes
import numpy
 
# 加载动态库
lib = ctypes.cdll.LoadLibrary("./example.so")
# 准备一个输入
tmp = numpy.arange(0, 100, 1).reshape((10, 10)).astype("float32")
pad_ptr = tmp.ctypes.data_as(ctypes.c_char_p)
# 准备一个结果
radius = 1
rows, cols = tmp.shape
result = numpy.zeros((rows - 2 * radius, cols - 2 * radius)).astype("float32")
res_ptr = result.ctypes.data_as(ctypes.c_char_p)

# lib.display(pad_ptr, rows, cols)
# print(tmp)

lib.compute_dark(pad_ptr, res_ptr, rows, cols, radius)

print(result)