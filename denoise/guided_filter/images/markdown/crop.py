import os
import cv2
import numpy

def show(image):
	cv2.imshow('crane', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

origin = cv2.imread("../output/comparison_detail_enhancement.png");

x = 1270
y = 400
width = 60
height = 60
lhs = origin[y: y + height, x: x + width]
rhs = origin[y: y + height, x + 800: x + 800 + width]
composed = numpy.concatenate([lhs, rhs], axis=1)
cv2.imwrite("./comparison_detail_enhancement_croped_reversed_gradients.png", 
	composed, [cv2.IMWRITE_PNG_COMPRESSION, 0])

x2 = 1010
y2 = 90
lhs = origin[y2: y2 + height, x2: x2 + width]
rhs = origin[y2: y2 + height, x2 + 800: x2 + 800 + width]
composed = numpy.concatenate([lhs, rhs], axis=1)
cv2.imwrite("./comparison_detail_enhancement_croped_edge_artifacts.png", 
	composed, [cv2.IMWRITE_PNG_COMPRESSION, 0])


cv2.rectangle(origin, (x, y), (x + 60, y + 60), (255, 0, 0), 10)
cv2.rectangle(origin, (x + 800, y), (x + 800 + 60, y + 60), (255, 0, 0), 10)

cv2.rectangle(origin, (x2, y2), (x2 + 60, y2 + 60), (0, 255, 255), 10)
cv2.rectangle(origin, (x2 + 800, y2), (x2 + 800 + 60, y2 + 60), (0, 255, 255), 10)
# show(origin)
cv2.imwrite("./comparison_detail_enhancement_marked.png", origin, [cv2.IMWRITE_PNG_COMPRESSION, 0])