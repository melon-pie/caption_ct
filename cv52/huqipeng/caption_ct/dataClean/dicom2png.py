import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    total_img = 17257
    saved_imgs = 0
    patients = json.load(open(json_patients, 'r'))
    workbook = xlrd.open_workbook(annotation_path)
    sheet = workbook.sheet_by_index(0)
    num = sheet.nrows - 1

    mkdir = 0
    # 为只出现一次的病例进行转换
    for i in range(num):
        id = sheet.cell(i+1, 0).value
        if id in many_ID:
            continue
        else:
            save_num = str(patients[id]['numkey'][0])
            save_path = save_root + '\\' + save_num
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                mkdir += 1

            series_path = patients[id]['series_path'][-1]
            for root, dirs, files in os.walk(series_path):
                for file in files:
                    dcm = pydicom.read_file(root + '\\' + file)
                    temp_path = set_bound_0(dcm)
                    clean_dicom = pydicom.read_file(temp_path)
                    # view_images([dcm, clean_dicom])
                    img = bsb_window(clean_dicom)
                    img_save_path = save_path + '\\' + file[:-3] + 'png'
                    plt.imsave(img_save_path, img, cmap=plt.cm.bone)
                saved_imgs += len(files)
                print("已完成%.2f%%" % (saved_imgs*100.0/total_img))

    # 处理多个study的病例
    for id in many_ID:
        for k, v in patients[id]['StudyInstanceID'].items():
            # 每个study选最后一个series
            series = patients[id]['StudyInstanceID'][k][-1]
            for path in patients[id]['series_path']:
                # 扫描didom文件
                for root, dirs, files in os.walk(path):
                    dcm = pydicom.read_file(root + '\\' + files[0])
                    SeriesID = dcm.SeriesInstanceUID
                    if SeriesID == series:
                        patients[id]['rest_count'] -= 1
                        # tem表示第几个numkey
                        tem = patients[id]['cap_count'] - patients[id]['rest_count']
                        save_num = str(patients[id]['numkey'][tem-1])
                        save_path = save_root + '\\' + save_num
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                            mkdir += 1
                        # 循环转png
                        for file in files:
                            dcm = pydicom.read_file(root + '\\' + file)
                            temp_path = set_bound_0(dcm)
                            clean_dicom = pydicom.read_file(temp_path)
                            # view_images([dcm, clean_dicom])
                            img = bsb_window(clean_dicom)
                            img_save_path = save_path + '\\' + file[:-3] + 'png'
                            plt.imsave(img_save_path, img, cmap=plt.cm.bone)
                        saved_imgs += len(files)
                        print("已完成%.2f%%" % (saved_imgs*100.0/total_img))

    print('mkdir:%d' % mkdir)


pre_pro()
