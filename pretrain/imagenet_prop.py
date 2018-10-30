import os
import pickle
from os import listdir
import numpy as np
import xml.etree.ElementTree as EL
from PIL import Image
import matplotlib.pyplot as plt

seq_home = '/media/hdd1/datasets/ImageNet/Image'
anno_home = '/media/hdd1/datasets/ImageNet/Annotation'
output_path = 'data/imagenet.pkl'

seq_list = [f for f in listdir(anno_home)]

data = {}
for i, seq in enumerate(seq_list):
    img_list = []
    img_all_list = sorted([p for p in os.listdir(seq_home + '/' + seq) if os.path.splitext(p)[1] == '.JPEG'])
    anno_list = sorted([p for p in os.listdir(anno_home + '/' + seq) if os.path.splitext(p)[1] == '.xml'])
    gt = []
    for j, anno in enumerate(anno_list):
        if anno[:-4] + '.JPEG' not in img_all_list:
            continue
        try:
            image = Image.open(seq_home + '/' + seq + '/' + anno[:-4] + '.JPEG').convert('RGB')
        except:
            continue
        e = EL.parse(anno_home + '/' + seq + '/' + anno).getroot()
        x_min = int(e[5][4][0].text)
        y_min = int(e[5][4][1].text)
        x_max = int(e[5][4][2].text)
        y_max = int(e[5][4][3].text)
        gt.append([x_min, y_min, x_max - x_min, y_max - y_min])
        img_list.append(anno[:-4] + '.JPEG')

    assert len(img_list) == len(gt), "Lengths do not match!!"
    data[seq] = {'images': img_list, 'gt': np.array(gt)}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)