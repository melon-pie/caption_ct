"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
import math
from skimage.color import  rgba2rgb
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
from tqdm import tqdm
import joblib
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
import random
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms as trn

# preprocess = trn.Compose([
#     # trn.ToTensor(),
#     trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

data_transform = trn.Compose(
    [
     trn.CenterCrop(512),
     trn.ToTensor(),
     trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

import sys
sys.path.append("..")
from misc.my_resnet import resnet101


def RandomSampling(dataMat, number):
    try:
        slice = random.sample(dataMat, number)
        return slice
    except:
        print('sample larger than population')


def SystematicSampling(dataMat, number):
    length = len(dataMat)
    k = length / number
    # print(k)
    sample = []
    i = 0
    if k > 0:
        while len(sample) != number:
            sample.append(random.sample(list(dataMat[math.floor(k * i):math.ceil((i + 1) * k)]), 1)[0])
            i += 1
        return sample
    else:
        return RandomSampling(dataMat, number)


def extract_img_features(img,params):
    my_resnet = resnet101()
    my_resnet.eval()

    # cat imgs for every person
    for j,sub_img in enumerate(img['file_path']):
        if j == 0:
            pic = Image.open(sub_img).convert('RGB')
            pic = data_transform(pic)
            pic = torch.unsqueeze(pic, 0)
            tem_cated_images = pic
        else:
            pic = Image.open(sub_img).convert('RGB')
            pic = data_transform(pic)
            pic = torch.unsqueeze(pic, 0)
            tem_cated_images = torch.cat((tem_cated_images, pic), 0)

    # print(tem_cated_images.shape)
    # 一个病例均匀采样images_per_person = 24张图像，如果小于24张则补零，大于等于24张则等距随机采样
    if len(img['file_path']) < params['images_per_person']:
        tem_cated_images_ = torch.zeros(params['images_per_person'], 3, 512, 512)
        tem_cated_images_[0:len(img['file_path']), :, :, :] = tem_cated_images
    else:
        tem_cated_images = SystematicSampling(dataMat=tem_cated_images, number=params['images_per_person'])
        for j, sub_img in enumerate(tem_cated_images):
            if j == 0:
                tem_cated_images_ = torch.unsqueeze(tem_cated_images[0], 0)
            else:
                tci = torch.unsqueeze(tem_cated_images[j], 0)
                tem_cated_images_ = torch.cat((tem_cated_images_, tci), 0)

    tmp_fc, tmp_att = my_resnet(tem_cated_images_, 14)
    return tmp_fc, tmp_att


def img_features_to_h5(i, img, dset_fc, dset_att, params):
    tmp_fc,  tmp_att = extract_img_features(img, params)
    dset_fc[i] = tmp_fc.data.cpu().float().numpy()
    # dset_att[i] = tmp_att.data.cpu().float().numpy()


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    N = len(imgs)
    seed(123)

    # create fc,att h5 file
    f_fc = h5py.File(params['output_h5'] + '_fc.h5', "w")
    f_att = h5py.File(params['output_h5'] + '_att.h5', "w")
    dset_fc = f_fc.create_dataset("fc", (N, params['images_per_person'], 2048), dtype='float32')
    dset_att = f_att.create_dataset("att", (N, params['images_per_person'], 14, 14, 2048), dtype='float32')

    # write imges_feats to h5 file
    joblib.Parallel(n_jobs=1)(joblib.delayed(img_features_to_h5)
                              (imgs.index(img), img, dset_fc, dset_att, params) for img in tqdm(imgs))

    f_fc.close()
    # f_att.close()
    print('wrote ', params['output_h5'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='/home/ai/data/huqipeng/caption_ct/annotations/middle_annotationfile_11_09.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--output_h5', default='/home/ai/data/huqipeng/caption_ct/data/coco_chinese_talk_11_09', help='output h5 file')

    # options
    parser.add_argument('--images_per_person', type=int, default='24', help='')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
