#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: github.com/duinodu

from __future__ import print_function
import os
import argparse
import json
from PIL import Image
import jieba

def ai_challenger_preprocess(caption_json=None, pre_label_json=None):
    import os
    import json
    val = json.load(open(caption_json, 'r'))

    print(val.keys())
    print(val['info'])
    print(len(val['images']))
    print(len(val['annotations']))
    print(val['images'][0])
    print(val['annotations'][0])

    import json

    imgs = val['images']
    annots = val['annotations']
    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    # create the json blob
    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']
        jimg = {}
        jimg['file_path'] =  img['file_name']
        jimg['id'] = imgid

        sents = []
        annotsi = itoa[imgid]
        for a in annotsi:
            sents = a['caption']
        jimg['captions'] = sents
        out.append(jimg)

    json.dump(out, open(pre_label_json, 'w'))

'''
Generate json file for preprocessing!!!
'''
def convert2coco(caption_json):
    dataset = json.load(open(caption_json, 'r'))

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()

    for ind, sample in enumerate(dataset):
        # img = Image.open(os.path.join(imgdir, sample['image_id']))
        # width, height = 512

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = sample['image_id']
        coco_img[u'width'] = 512
        coco_img[u'height'] = 512
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = ind

        coco_anno = {}
        coco_anno[u'image_id'] = ind
        coco_anno[u'id'] = ind
        coco_anno[u'caption'] = sample['caption']

        coco[u'images'].append(coco_img)
        coco[u'annotations'].append(coco_anno)

        print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join(os.path.dirname(caption_json),"coco_"+ os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))
    return output_file
'''
Generate json file for evaluation
'''
def convert2coco_eval(caption_json):
    dataset = json.load(open(caption_json, 'r'))

    coco = dict()
    coco[u'info'] = { u'desciption':u'AI challenger image caption in mscoco format'}
    coco[u'licenses'] = ['Unknown', 'Unknown']
    coco[u'images'] = list()
    coco[u'annotations'] = list()
    coco[u'type'] = u'captions'
    for ind, sample in enumerate(dataset):

        coco_img = {}
        coco_img[u'license'] = 0
        coco_img[u'file_name'] = sample['image_id']
        coco_img[u'width'] = 512
        coco_img[u'height'] = 512
        coco_img[u'date_captured'] = 0
        coco_img[u'coco_url'] = sample['url']
        coco_img[u'flickr_url'] = sample['url']
        coco_img['id'] = ind

        coco_anno = {}
        coco_anno[u'image_id'] = ind
        coco_anno[u'id'] = ind
        coco_anno[u'caption'] = sample['caption']

        coco[u'images'].append(coco_img)
        coco_anno_ = coco_anno['caption']
        coco_anno_s = {}
        coco_anno_s[u'image_id'] = coco_anno[u'image_id']
        coco_anno_s[u'id'] = coco_anno[u'id']
        # w = jieba.cut(coco_anno_.strip(), cut_all=False)
        # p = ' '.join(w)
        # coco_anno_ = p
        coco_anno_s[u'caption'] = coco_anno_
        coco[u'annotations'].append(coco_anno_s)

        print('{}/{}'.format(ind, len(dataset)))

    output_file = os.path.join(os.path.dirname(caption_json), 'coco_val_'+os.path.basename(caption_json))
    with open(output_file, 'w') as fid:
        json.dump(coco, fid)
    print('Saved to {}'.format(output_file))


if __name__ == "__main__":
    
    raw_json = '/home/ai/data/huqipeng/caption_ct/annotations/raw_annotationfile_11_10.json'
    middle_json = '/home/ai/data/huqipeng/caption_ct/annotations/middle_annotationfile_11_10.json'
    
    coco_json = convert2coco(raw_json)
    ai_challenger_preprocess(coco_json, middle_json)
    convert2coco_eval(raw_json)
