
# -*- coding:utf-8 -*-

from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
# import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
global true,false
true=True
false=False
'''
keypoints是一个长度为3*k的数组，其中k是category中keypoints的总数量。
每一个keypoint是一个长度为3的数组，第一和第二个元素分别是x和y坐标值，第三个元素是个标志位v，
v为0时表示这个关键点没有标注（这种情况下x=y=v=0），v为1时表示这个关键点标注了但是不可见（被遮挡了），v为2时表示这个关键点标注了同时也可见。
num_keypoints表示这个目标上被标注的关键点的数量（v>0），比较小的目标上可能就无法标注关键点。
"keypoints": [623, 171, 2, 
0, 0, 0, 
602, 152, 2, 
0, 0, 0, 
571, 126, 1, 
0, 0, 0, 
499, 222, 2, 
0, 0, 0, 
438, 358, 2, 
592, 407, 2,
 399, 465, 2], 
'''
def convert_coco2custom():
    fn=open('./custom_coco_train_17.json', 'w')
    json_file = 'coco/person_keypoints_train2017.json'  # # Object Instance 类型的标注
    # person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
    # captions_val2017.json  # Image Caption的标注格式
    count=0
    data = json.load(open(json_file, 'r'))
    annotation_dict = {}
    for ann in data['annotations']:
        # print(ann)
        annotation_dict.update({ann['id']:ann})

    for image in data['images']:
        # print(image)
        img_name=image['file_name']
        img_height=image['height']
        img_width=image['width']
        id=image['id']
        instances=[]
        annotations = data['annotations']
        GT_NUM=len(annotations)

        # print(GT_NUM)
        for i in range(GT_NUM):
            ann=annotations[i]
            if ann['image_id']==id:
                ann_bbox=ann['bbox']
                bbox=[ann_bbox[0],ann_bbox[1],ann_bbox[0]+ann_bbox[2],ann_bbox[1]+ann_bbox[3]]
                keypoints=ann['keypoints']
                need_keypoints=[]
                for i in range(17):
                    need_keypoints.append(keypoints[0+i*3:3+i*3])
                instances.append({
                    'is_ignored': False,
                    'bbox': bbox,
                    'keypoints': need_keypoints,
                    'label': 1})
        if len(instances)<1:
            continue
        data_write = {
            'filename': 'data/train2017/'+img_name,
            'image_height': img_height,
            'image_width': img_width,
            'instances': instances
        }
        fn.write(json.dumps(data_write, ensure_ascii=False) + '\n')
        count+=1
        print(count)


    # data_need={}
    # data_need['images'] = [data['images'][27]]  # 只提取第一张图片
    # print(data['images'][27])
    # annotation = []
    # # 通过imgID 找到其所有对象
    # # for
    # imgID = data_need['images'][0]['id']
    # print(len(data['images']))
    # print(imgID)
    # for ann in data['annotations']:
    #     print(ann)
    #     annotation.append(ann)
    #     # if ann['image_id'] == imgID:
    #     #     annotation.append(ann)
    #     fn.write(json.dumps(ann, ensure_ascii=False) + '\n')
    # data_need['annotations'] = annotation

    # 保存到新的JSON文件，便于查看数据特点


    # json.dump(data_need, open('./new_instances_val2017.json', 'w'), indent=1)  # indent=4 更加美观显示
convert_coco2custom()


def read_one_image():
    json_file = 'coco/person_keypoints_val2017.json'  # # Object Instance 类型的标注
    # person_keypoints_val2017.json  # Object Keypoint 类型的标注格式
    # captions_val2017.json  # Image Caption的标注格式

    data = json.load(open(json_file, 'r'))
    data_2 = {}
    data_2['info'] = data['info']
    data_2['licenses'] = data['licenses']
    data_2['images'] = [data['images'][27]]  # 只提取第一张图片
    data_2['categories'] = data['categories']
    annotation = []

    # 通过imgID 找到其所有对象
    imgID = data_2['images'][0]['id']
    for ann in data['annotations']:
        if ann['image_id'] == imgID:
            annotation.append(ann)

    data_2['annotations'] = annotation

    # 保存到新的JSON文件，便于查看数据特点
    json.dump(data_2, open('./one.json', 'w'), indent=4)  # indent=4 更加美观显示
# read_one_image()
