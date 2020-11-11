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
import shutil
import win32com.client
import numpy as np
import PIL
import pydicom
import tqdm
import logging as l


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

windows = ['Default window', 'Brain window', 'Subdural window', 'Bone window', 'BSB window', 'SigmoidBSB window']

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
def load_scan(paths):
    # slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = [pydicom.read_file(path) for path in paths]
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
注：处理一个切片
'''


def set_bound_0(img_dicom):
    temp_path = r'E:\data\temp.dcm'
    img = img_dicom.pixel_array
    img[img == -2000] = 0
    img_dicom.PixelData = img.tobytes()
    img_dicom.save_as(temp_path)
    return temp_path


def resize_img(img, new_w, new_h):
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


def pre_pro():
    patients = {}
    id_list = ["pad", "pad"]
    root_path = r'E:\data\北工大合作课题'
    save_root = r'E:\data\data1023\old_train'
    annotation_path = r'E:\data\data1023\report.xlsx'
    json_patients = r'E:\data\data1023\patients.json'
    temp_path = r'E:\data\temp.dcm'
    many_ID = {
      "UZIpjM5b3Lv2GyoP5JAUcg==",
      "9bdJyUiSp3R0CCEWRkHAdw==",
      "vx+xLn0Xc7fAG3brTDHRFQ==",
      "76IE/keDoNFEBey4FxQ9XA==",
      "9upfEckiXydn6wAmdVtyow==",
      "d2sNh5MZ1BZjuE1xnAzrzw=="
    }
    many = [
    "1.2.528.1.1001.200.10.1853.6657.2217472578.20200303014527206",
    "1.2.528.1.1001.200.10.1853.6657.2217472578.20200302083410921",
    "1.2.528.1.1001.200.10.1853.6657.2217472578.20200302052647618",
    "1.2.528.1.1001.200.10.1853.6657.2217472578.20200303013358955",
    "1.2.528.1.1001.200.10.1853.6657.2217472578.20200302080231143",
    "1.2.528.1.1001.200.10.1853.6657.2217472578.20200303015234142"
    ]
    workbook = xlrd.open_workbook(annotation_path)
    sheet = workbook.sheet_by_index(0)
    num = sheet.nrows - 1

    for i in range(num):
        id = sheet.cell(i+1, 0).value
        id_list.append(id)
        if patients.get(id) is None:
            patients[id] = {'numkey': [i+2], 'cap_count': 1, 'rest_count': 1, 'series_path': [], 'root_path': [], 'SeriesInstanceUID':{},'StudyInstanceID':{}}
        else:
            patients[id]['numkey'].append(i+2)
            patients[id]['cap_count'] +=1
            patients[id]['rest_count'] += 1


    for root, dirs, files in os.walk(root_path):
        if (len(files) > 20) & (len(dirs) == 0):
            dcm = pydicom.read_file(root + '\\' + files[0])
            id = dcm.PatientID
            patients[id]['series_path'].append(root)
            patients[id]['root_path'].append(root.split('\\')[-3])
            for file in files:
                dcm1 = pydicom.read_file(root + '\\' + file)
                SeriesID = dcm1.SeriesInstanceUID
                StudyID = dcm1.StudyID

                if patients[id]['SeriesInstanceUID'].get(SeriesID) is None:
                    patients[id]['SeriesInstanceUID'][SeriesID] = 1
                else:
                    patients[id]['SeriesInstanceUID'][SeriesID] += 1

                if patients[id]['StudyInstanceID'].get(StudyID) is None:
                    patients[id]['StudyInstanceID'][StudyID] = [SeriesID]
                else:
                    if SeriesID not in patients[id]['StudyInstanceID'][StudyID]:
                        patients[id]['StudyInstanceID'][StudyID].append(SeriesID)
    with open(json_patients, 'w') as f:
        json.dump(patients, f)

# plt.hist(img_clean_ct.flatten(), bins=80, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show(

# TRAIN_IMG_PATH = r'E:\data\NewCT\出血\01130304105003'
# temp_path = r'E:\data\temp.dcm'
# png_path = r'E:\data\temp.png'
# case = os.path.join(TRAIN_IMG_PATH, r'1.2.840.113619.2.55.3.386549861.896.1362355549.192.6')
# dicom = pydicom.read_file(case)
#
# # 边界像素值设置为0 并储存到temp.dcm中
# arr = dicom.pixel_array
# arr[arr == -2000] = 0
# dicom.PixelData = arr.tobytes()
# dicom.save_as(temp_path)
#
# dicom1 = pydicom.read_file(temp_path)
# img = bsb_window(dicom1)
# # plt.imsave(png_path, img, cmap=plt.cm.bone)

#
# view_images([dicom, dicom1])

pre_pro()
