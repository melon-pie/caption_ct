import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import os
import random
import glob
import joblib
import json
import xlrd
import win32com.client
import numpy as np
import PIL
import pydicom
import tqdm
import logging as l
windows = ['Default window', 'Brain window', 'Subdural window', 'Bone window', 'BSB window', 'SigmoidBSB window']


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


'''
返回不同window的CT图
1. Brain Matter window : W:80 L:40
2. Blood/subdural window: W:130-300 L:50-100
3. Soft tissue window: W:350–400 L:20–60
4. Bone window: W:2800 L:600
5. Grey-white differentiation window: W:8 L:32 or W:40 L:40
'''


def window_image(img_dicom, window_center, window_width, rescale=True):
    _, _, intercept, slope = get_windowing(img_dicom)

    img = (img_dicom.pixel_array * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)

    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)



# 扫描一个患者的目录，加载所有的切片，按切换的z方向排序切片，并获取切片厚度
# 只使用了排序和获取厚度功能
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices = [pydicom.read_file(path) for path in paths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


'''
某些扫描仪具有圆柱扫描边界，但输出图像为方形。 落在这些边界之外的像素获得固定值-2000。 
第一步是将这些值设置为0，当前对应于air。 
接下来，回到HU单位，乘以重新缩放斜率并添加截距（方便地存储在扫描的元数据中！）。

注：输入必须是同一次扫描的切片 不同的series不可以
'''


def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])
    # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
    image = image.astype(np.int16)

    # 设置边界外的元素为0
    image[image == -2000] = 0

    # 转换为HU单位
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)
    print(np.array(image, dtype=np.int16).shape)
    return np.array(image, dtype=np.int16)


def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)



def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)


def sigmoid_window(img_dicom, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    _, _, intercept, slope = get_windowing(img_dicom)
    img = img_dicom.pixel_array * slope + intercept
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def bsb_window(img_dicom):
    brain_img = window_image(img_dicom, 40, 80)
    subdural_img = window_image(img_dicom, 80, 200)
    bone_img = window_image(img_dicom, 600, 2800)

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


def sigmoid_bsb_window(img_dicom):
    brain_img = sigmoid_window(img_dicom, 40, 80)
    subdural_img = sigmoid_window(img_dicom, 80, 200)
    bone_img = sigmoid_window(img_dicom, 600, 2800)

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img


def view_images(img_dicoms):
    height = len(img_dicoms)
    width = len(windows)
    fig, axs = plt.subplots(height, width, figsize=(15, 8),gridspec_kw={'hspace': 0.3, 'wspace': 0})

    if height == 1:
        for i in range(width):
            if i == 0:
                axs[i].imshow(img_dicoms[0].pixel_array, cmap=plt.cm.bone)
                axs[i].axis('off')
                axs[i].set_title('Default window')
            elif i == 1:
                img1 = window_image(img_dicoms[0], 40, 80)
                axs[i].imshow(img1, cmap=plt.cm.bone)
                axs[i].axis('off')
                axs[i].set_title('Brain window')
            elif i == 2:
                img2 = window_image(img_dicoms[0], 80, 200)
                axs[i].imshow(img2, cmap=plt.cm.bone)
                axs[i].axis('off')
                axs[i].set_title( 'Subdural window')
            elif i == 3:
                img3 = window_image(img_dicoms[0], 600, 2800)
                axs[i].imshow(img3, cmap=plt.cm.bone)
                axs[i].axis('off')
                axs[i].set_title('Bone window')
            elif i == 4:
                img4 = bsb_window(img_dicoms[0])
                axs[i].imshow(img4, cmap=plt.cm.bone)
                axs[i].axis('off')
                axs[i].set_title('BSB window')
            else:
                img4 = sigmoid_bsb_window(img_dicoms[0])
                axs[i].imshow(img4, cmap=plt.cm.bone)
                axs[i].axis('off')
                axs[i].set_title('SigmoidBSB window')
    else:
        for i in range(height):
            for j in range(width):
                if j == 0:
                    axs[i, j].imshow(img_dicoms[i].pixel_array, cmap=plt.cm.bone)
                    axs[i, j].axis('off')
                    axs[i, j].set_title('Default window')
                elif j == 1:
                    img1 = window_image(img_dicoms[i], 40, 80)
                    axs[i, j].imshow(img1, cmap=plt.cm.bone)
                    axs[i, j].axis('off')
                    axs[i, j].set_title('Brain window')
                elif j == 2:
                    img2 = window_image(img_dicoms[i], 80, 200)
                    axs[i, j].imshow(img2, cmap=plt.cm.bone)
                    axs[i, j].axis('off')
                    axs[i, j].set_title( 'Subdural window')
                elif j == 3:
                    img3 = window_image(img_dicoms[i], 600, 2800)
                    axs[i, j].imshow(img3, cmap=plt.cm.bone)
                    axs[i, j].axis('off')
                    axs[i, j].set_title('Bone window')
                elif j == 4:
                    img4 = bsb_window(img_dicoms[i])
                    axs[i, j].imshow(img4, cmap=plt.cm.bone)
                    axs[i, j].axis('off')
                    axs[i, j].set_title('BSB window')
                else:
                    img4 = sigmoid_bsb_window(img_dicoms[i])
                    axs[i, j].imshow(img4, cmap=plt.cm.bone)
                    axs[i, j].axis('off')
                    axs[i, j].set_title('SigmoidBSB window')
    plt.show()

def test():
    # 运行实例
    TRAIN_IMG_PATH = r'E:\data\NewCT\正常\01200825110104'
    case = os.path.join(TRAIN_IMG_PATH, r'1.2.840.113619.2.428.3.403321453.349.1596699726.22')
    data = pydicom.read_file(case)

    print(data.pixel_array)
    # print(data.ImagePositionPatient[2])
    img = bsb_window(data)
    view_images([img])
    # first_patient_pixels = get_pixels_hu(dicom_imags)
    # plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()
    #
    # samp_Distribution = random.sample(dicom_imags, 1)
    # samp_slices = random.sample(dicom_imags, 4)
    # plt.title('Distribution of DICOM Pixel Values')
    # print(img.pixel_array for img in dicom_imags)
    # ax = plt.hist(np.array(samp_Distribution[0].pixel_array).flatten(), bins=50, color='c')
    # plt.xlabel("Pixel Values")
    # plt.ylabel("Frequency")
    # plt.show()
    #
    # view_images(samp_slices, len(samp_slices))
    print(data)


test()

