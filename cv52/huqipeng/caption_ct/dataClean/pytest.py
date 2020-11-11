import json
import os
import sys
import pydicom
import numpy as np
import matplotlib.image as mpimg # mpimg 用于读取图片
import glob
import xlrd
import matplotlib.pyplot as plt

from PIL import Image

root_path = r'E:\data\data1024\data_01'
min = 999
max = 0
for root, dirs, files in os.walk(root_path):
    if len(files) > 0 & len(dirs) == 0:
        if len(files) < 10 or len(files) >= 100:
            continue
        if len(files) < min:
            min = len(files)
        if len(files) > max:
            max = len(files)
print(min,max)
            # img = img.convert('RGB')
            # print(img.mode)
            # img = np.array(img)
            # print(type(img))  # 显示类型
            # print(img.shape)  # 显示尺寸
            # print(img.shape[0])  # 图片宽度
            # print(img.shape[1])  # 图片高度
            # print(img.shape[2])  # 图片通道数
            # print(img.size)  # 显示总像素个数
            # print(img.max())  # 最大像素值
            # print(img.min())  # 最小像素值
            # print(img.mean())  # 像素平均值

