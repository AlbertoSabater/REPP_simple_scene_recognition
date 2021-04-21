#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:37:10 2021

@author: asabater
"""

import json
import pandas as pd
import numpy as np

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--preds_filename', type=str, required=True, help='predictions (COCO format) filename')
parser.add_argument('--min_score', type=float, default=0.1, help='threshold to filter out low-scoring detections')
parser.add_argument('--video_filename', type=str, required=True, help='video filename to be rendered')
args = parser.parse_args()
    


    

cat_agg_mode = 'size'
# preds_filename = './predictions/preds_repp_casa_repp_coco.json'
# video_filename = '/mnt/hdd/datasets_hdd/filovi_videos/casa.mp4'


classes_coco_ids = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
                44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
                49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 
                59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 
                64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
                69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


classes_coco = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 
                'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 
                'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 
                'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 
                'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 
                'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 
                'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 
                'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 
                'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 
                'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 
                'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 
                'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 
                'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 
                'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 
                'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 
                'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}




rooms_class_name_dict = {
        'kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'microwave', 'oven', 
                'toaster', 'sink', 'refrigerator'],
        'dining_room': ['chair', 'couch', 'potted plant', 'dining table', 'tv', 'remote', 'book'],
        'bedroom': ['bed', 'book', 'scissors', 'chair'],
        'bathroom': ['toilet', 'hair drier', 'toothbrush', 'sink'],
    }
room_names = rooms_class_name_dict.keys()



classes_useful_name = sum( list(rooms_class_name_dict.values()), [])
classes_useful_id = list(set([ classes_coco[k] for k in classes_useful_name ]))
classes_useful_dict = { k:i for i,k in enumerate(classes_useful_id) }
classes_useful_len = len(classes_useful_name)

room_descriptors = { k:np.zeros(classes_useful_len) for k in room_names }
for k in room_names: room_descriptors[k][[ classes_useful_dict[classes_coco[class_id]] for class_id in rooms_class_name_dict[k] ]] = 1



# Load COCO predictions
preds = json.load(open(args.preds_filename))

df = pd.DataFrame(preds)
df = df[df.category_id.isin(classes_useful_id)]
df = df[df.score >= args.min_score]
df['category_name'] = df.category_id.apply(lambda x: classes_coco_ids[x])


total_room_scores = {}

for image_id, g in df.groupby('image_id'):

    cat_vector = np.zeros(classes_useful_len)
    
    # Summarize frame-objects by category
    # if cat_agg_mode == 'mean':
    #     for category_id, res in g.groupby('category_id').mean().iterrows(): cat_vector[classes_useful_dict[category_id]] += res.score
    # elif cat_agg_mode == 'median':
    #     for category_id, res in g.groupby('category_id').median().iterrows(): cat_vector[classes_useful_dict[category_id]] += res.score
    # elif cat_agg_mode == 'size':
    #     for category_id, res in g.groupby('category_id').size().iteritems(): cat_vector[classes_useful_dict[category_id]] += res
    # else: raise ValueError('cat_agg_mode {} not valid'.format(cat_agg_mode))
    for category_id, res in g.groupby('category_id').size().iteritems(): cat_vector[classes_useful_dict[category_id]] += res

    room_scores = np.array([ np.dot(cat_vector, room_descriptors[k]) for k in room_names ])
    room_scores = room_scores/room_scores.sum(axis=0,keepdims=1)
    
    total_room_scores[image_id] = room_scores
    



import matplotlib.pyplot as plt

plt.figure(figsize=(12,8), dpi=150)

for i, k in enumerate(room_descriptors.keys()):
    plt.plot([ s[i] for s in total_room_scores.values() ], label=k)

plt.legend()



from skvideo.io import FFmpegWriter
import cv2
from PIL import Image, ImageFont, ImageDraw
import colorsys


output_path = args.video_filename.replace('.mp4', '_scene_recognition_{:.2f}.mp4'.format(args.min_score))
video_fps = cv2.VideoCapture(args.video_filename).get(cv2.CAP_PROP_FPS)
out = FFmpegWriter(output_path, inputdict={'-r': str(video_fps)}, outputdict={'-r': str(video_fps)})
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), [(x / classes_useful_len, 1., 1.) for x in range(classes_useful_len)]))
colors = [ (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors ]

def video_iterator(video_file):
    vid = cv2.VideoCapture(video_file)
    num_frame = 0
    while True:
        ret, frame = vid.read()
        print('Frame:', num_frame)
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        num_frame += 1
        yield frame, '{:06d}'.format(num_frame), False
        

def print_box(image, box, label, color, label_size=0.5):
    
    font = ImageFont.truetype(font='./FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + label_size).astype('int32'))
    
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    x_min, y_min, width, height = box
    x_min = max(0, np.floor(x_min + 0.5).astype('int32'))
    y_min = max(0, np.floor(y_min + 0.5).astype('int32'))
    x_max = min(image.size[0], np.floor(x_min + width + 0.5).astype('int32'))
    y_max = min(image.size[1], np.floor(y_min + height + 0.5).astype('int32'))
    top, left, bottom, right = y_min, x_min, y_max, x_max

    if top - label_size[1] >= 0: text_origin = np.array([left, top - label_size[1]])
    else: text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range((image.size[0] + image.size[1]) // 300):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=color)
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=color)
    draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
    del draw
    
    return image



for frame, image_id, _ in video_iterator(args.video_filename):
        
    # Plot predictions
    frame_preds = df[df.image_id == image_id]
    for _, row in frame_preds.iterrows():
        print_box(frame, row.bbox, row.category_name, colors[classes_useful_dict[row.category_id]], label_size=3)
    frame = np.asarray(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        
    # Plot classification
    if image_id in total_room_scores: frame_scores = total_room_scores[image_id]
    else: frame_scores = np.zeros(classes_useful_len)
    for i, k in enumerate(room_names): 
        if frame_scores[i] == 0 or frame_scores[i] != frame_scores.max(): color = (255, 0, 0)
        else: color = (0, 255, 0)
        cv2.putText(frame, text='{}: {:.1f}%'.format(k, frame_scores[i]*100), 
                    org=(3, 40*(i+1)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3, color=color, thickness=3)
    
    
    out.writeFrame(frame)
    

out.close()
print('Rendered video stored in:', output_path)

