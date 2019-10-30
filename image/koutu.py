# -*- coding: utf-8 -*-
# author: dengfan
# datetime:2018-11-08 14:39
import cv2
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import image
data_dir = r'E:\program\zhuyou\data\5.jpg'
image = cv2.imread(data_dir)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image_copy = np.copy(image)
red_up = np.array([255,20,20])
red_low = np.array([240,0,0])

mask = cv2.inRange(image_copy, red_low, red_up)
mask_image = np.copy(image)
mask_image[mask!=0] = [0,0,0]

# ax = plt.figure()
# ax.add_subplot(3,1,1)
# plt.imshow(image[:,:,0],cmap='gray')
# ax.add_subplot(3,1,2)
# plt.imshow(image[:,:,1],cmap='gray')
# ax.add_subplot(3,1,3)
plt.imshow(image[:,:,2],cmap='gray')
# plt.imshow(mask,cmap='gray')
plt.show()