import csv
import json
import os
import sys
import pydicom
import numpy as np

# import win32com.client
import xlrd
import matplotlib.pyplot as plt
# import scipy.misc
# import SimpleITK as sitk
from PIL import Image
# import cv2py
import csv
with open(r'E:\data\NewCT\脑出血.csv', 'r') as f:
    reader = csv.reader(f)
    print(type(reader))
    for row in reader:
        print(row)

def Max_MinNormalization(image, dcm):
    try:
        MIN_BOUND = (2 * np.min(dcm.WindowCenter) -
                     np.min(dcm.WindowWidth)) / 2
        MAX_BOUND = (2 * np.min(dcm.WindowCenter) +
                     np.min(dcm.WindowWidth)) / 2
    except TypeError:
        MIN_BOUND = (2 * dcm.WindowCenter - dcm.WindowWidth) / 2
        MAX_BOUND = (2 * dcm.WindowCenter + dcm.WindowWidth) / 2

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    return image


# path = 'IMG00020.DCM'
# dcm=pydicom.read_file(path)

# print(dcm.RescaleIntercept)
# # print(dcm.pixel_array.max())
# # im=Image.fromarray(dcm.pixel_array)
# im=dcm.pixel_array

# im = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
# lv = dcm.]WindowCenter[1 - dcm.WindowWidth[1]/2
# uv = dcm.WindowCenter[1] + dcm.WindowWidth[1]/2
# im = (im-lv)/(lv-uv)*255

# im[im < 0] = 0
# im[im > 255] = 255

# im=np.uint8(im)
# print(im.dtype)
# print(dcm.RescaleSlope , dcm.RescaleIntercept)
# print(len(dcm.WindowCenter),len(dcm.WindowWidth))
# print(dcm.WindowCenter,dcm.WindowWidth)
# center=dcm.WindowCenter/dcm.RescaleSlope - dcm.RescaleIntercept
# width=dcm.WindowWidth/dcm.RescaleSlope - dcm.RescaleIntercept
# im=Max_MinNormalization(im,dcm)
# im = np.uint8(im)
# im = Image.fromarray(im)
# im.show()

# ret,img = cv2.threshold(im, 90,3071, cv2.THRESH_BINARY)
# img = np.uint8(img)

# contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# mask = np.zeros(img.shape, np.uint8)
# for contour in contours:
#  cv2.fillPoly(mask, [contour], 255)
# img[(mask > 0)] = 255

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# im[(img == 0)] = -2000

# plt.imsave('test2.png',im,cmap='gray')
# plt.imshow(im,'gray')
# plt.show()

# root_path = '北工大合作课题'
root_path = r'E:\CQ500-CT-276\CQ500CT276 CQ500CT276\Unknown Study\CT PRE CONTRAST THIN'
annotation_path = 'annotation-jiami.xlsx'
ID2num_path = 'ID2num.json'
save_root = 'test'
unfit = []

workbook = xlrd.open_workbook(annotation_path)
sheet = workbook.sheet_by_index(0)
print (sheet.name,sheet.nrows,sheet.ncols)

xlApp = win32com.client.Dispatch("Excel.Application")
xlApp.Visible = False
filename,password = 'D:\\data\\annotation-jiami.xlsx', '305'
excel = xlApp.Workbooks.Open(filename, False, True, None, Password='305')
sheet = excel.Worksheets('Sheet1')
num = sheet.UsedRange.rows.Count-1
IDnum = []
for i in range(num):
    IDnum.append(sheet.Cells(i+2,1).Value)

json_name = 'ID2num.json'
with open(json_name, 'w') as f:
    json.dump(IDnum, f)

ID2num = json.load(open(ID2num_path))
max,min=0,0
count=np.zeros(567)
c=0
search = 'qXUjLTgyyk2EJ5eTbJmXWA=='
for root, dirs, files in os.walk(root_path):

    if ((len(files) > 20) & (len(dirs) == 0)):
        print(root, files, dirs)
        dcm = pydicom.read_file(root + '\\' + files[0])
        # ID = dcm.PatientID
        # ID = ID.replace('/', '-')

        # try:
        #     num = ID2num.index(ID)
        # except ValueError:
        #     unfit.append(ID)
        #     continue
        # if (ID == search):
        #     print(root)
        # try:
        #     print(np.min(dcm.WindowCenter),np.min(dcm.WindowWidth))
        # except TypeError:
        #     print(dcm.WindowCenter,dcm.WindowWidth)
        
        # count[num]+=1
        # if count[num]>=3:
        #     c+=1
        #     print(num)
        # folder = save_root + '\\' + str(num)
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        for i in range(len(files)):
            if (files[i][-3:] == 'dcm'):
                #print(files[i])    
                img_path = root + '\\' + files[i]
                # print(img_path)
                dcm = pydicom.read_file(img_path)
                print(dcm)
                # save_path = save_root + '\\' + files[i][:-4] + '.png'
                save_path =save_root + '\\' + dcm.PatientID + '_CT'+ str(dcm.InstanceNumber).zfill(3) + '.png'
                # dcm1 = sitk.ReadImage(img_path)
                # im = sitk.GetArrayFromImage(dcm1).squeeze(0)
                im = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
                im = Max_MinNormalization(im, dcm)
                plt.imsave(save_path, im, cmap='gray')
# json_name = 'unfit.json'
# with open(json_name, 'w') as f:
#     json.dump(unfit, f)
