#coding = gbk
import jieba
from PIL import Image
import matplotlib
import pandas as pd
import  os
import datetime
import json
from glob import glob
import xlrd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 将image的信息写入字典
def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info

# 将注释的信息写入字典
def create_annotation_info(annotation_id, image_id, sentence):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "caption":sentence,
    }

    return annotation_info

#写注释的字典
def createannotationfile(url,image_id, sentence):
    annotation = {
        "url": url,
        "image_id": image_id,
        "caption":sentence,

    }
    return annotation


def append2ann(ann=[], image_root= None, report_root=None ):
    file_info = pd.read_excel(report_root, sheet_name='Sheet1')
    findings = file_info['诊断结论']
    dirs = os.listdir(image_root)

    for dir in dirs:
        if image_root == "/devdata/data1024/pngnew/png":
            row = int(dir)+2
        else:
            row = int(dir.split("_")[0])

        if "_" in dir:
            data_id = int(dir.split("_")[1])
        else:
            data_id = 0
        dir_path = image_root + "/"+ dir
        files = os.listdir(dir_path)
        files.sort(key=lambda x: int(x[-8:-4]))
        img_path = []
        for file in files:
            # 这里不是单张图片，而是一个人的图片集合
            img_path.append(dir_path + "/" + file)

        if data_id == 2:
            annotions = "脑部CT检查未见出血,请结合临床"
        else:annotions = findings[row-2]
        # 小于10张影像的不使用
        if len(img_path) < 10:
            continue
        ann.append(createannotationfile("http://news.sogou.com", img_path, annotions))
    return ann


data_root = [
    ("/devdata/data1024/data", "/devdata/data1024/report-refine.xlsx")
    ]
# [   ("/devdata/data1024/pngnew/png", "/devdata/data1024/report.xlsx")
#     ("/devdata/data1024/data", "/devdata/data1024/report-refine.xlsx"),
#     ("/devdata/data1024/data_01", "/devdata/data1024/report_01.xlsx"),
#     ("/devdata/data1024/data_02", "/devdata/data1024/report_02.xlsx")
#     ]

ann = []
for data in data_root:
    image_root = data[0]
    report_root = data[1]
    ann = append2ann(ann, image_root, report_root)

print(len(ann))
json_name = '/home/ai/data/huqipeng/caption_ct/annotations/raw_annotationfile_11_10.json'
with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(ann, f, ensure_ascii=False)
print(u'生成annotation文件完成...')