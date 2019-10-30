# -*- coding: utf-8 -*-
# author: dengfan
# datetime:2018-11-05 16:51
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('E:\\program\\zhuyou\\data\\1.png')
image_color = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image_color,cv2.COLOR_RGB2GRAY)
print(np.shape(image))
sl1 = cv2.Sobel(image,cv2.CV_16S,1,0)
sl2 = cv2.Sobel(image,cv2.CV_16S,0,1)

absX = cv2.convertScaleAbs(sl1)
absY = cv2.convertScaleAbs(sl2)
abs_q = absX + absY

wide = cv2.Canny(image, 20, 60)
wide = wide>100

line_image = np.copy(image)

ax = plt.figure()
ax.add_subplot(3,1,1)
plt.imshow(image_color)
ax.add_subplot(3,1,2)
plt.imshow(image*wide,cmap='gray')
ax.add_subplot(3,1,3)
plt.imshow(abs_q,cmap='gray')
plt.show()