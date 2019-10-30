import time
import mmcv
import cv2
import json
import pycocotools.mask as maskutil
from itertools import groupby
from skimage import measure,draw
import json
import copy
from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result, show_result_pyplot
import os
import numpy as np
from mmcv.image import imread, imwrite


config_file = './faster_rcnn_mdconv_c3-c5_r50_fpn_1x.py'
checkpoint_file = './work_dirs/faster_rcnn_mdconv_c3-c5_r50_fpn_1x/epoch_20.pth'
imgs_dir = '/data/sdv1/datasets/xs_bbox/bbox_train/test_pic'
save_path = '/data/sdv1/datasets/xs_bbox/bbox_train/save_pic'
json_path = '/data/sdv1/datasets/xs_bbox/train.json'

data = open(json_path, encoding='utf-8')
other = json.load(data)
categories = other['categories']
class_nums = []
for i in categories:
	if i['name'] not in class_nums:
		class_nums.append(i['name'])

names = []
files = os.listdir(imgs_dir)
imgs = [os.path.join(imgs_dir,i) for i in files]

model = init_detector(config_file, checkpoint_file, device='cuda:0')

'''
for index in validResult:
    left_top = (box_final[index][0], box_final[index][1])
    right_bottom = (box_final[index][2], box_final[index][3])
    cv2.rectangle(img_data, left_top, right_bottom, (0, 0, 255), thickness=2)
    strText = str(class_names[clsIndex_final[index]]) + ': ' + str(format(box_final[index][4], '.2f'))
    cv2.putText(img_data, strText, (box_final[index][0], box_final[index][1]),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
	  cv2.imwrite(img_data, out_dir)
'''
def selectClsScoreBoxFromResult(result):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    selectedClsIndex = []
    selectedScore = []
    selectedBox = []
    assert(len(labels) == len(bboxes))
    for i in range(0, len(labels)):
        selectedClsIndex.append(labels[i])
        selectedScore.append(bboxes[i][-1])
        tempBox = int(bboxes[i][0]), int(bboxes[i][1]), int(bboxes[i][2]), int(bboxes[i][3]), bboxes[i][4]
        selectedBox.append(tempBox)
    return selectedClsIndex, selectedScore, selectedBox


for i in range(len(imgs)):
  print(imgs[i])
  img = Image.open(imgs[i])
  img = np.asarray(img)
  result = inference_detector(model, img)
  img_data = copy.copy(img)
  print(result)
  indx, score, bbox = selectClsScoreBoxFromResult(result)
  
  score_thr=0.4
  for n in range(len(indx)):
    if score[n] < score_thr:
      continue
    left_top = (bbox[n][0], bbox[n][1])
    right_bottom = (bbox[n][2], bbox[n][3])
    cv2.rectangle(img_data, left_top, right_bottom, (0, 0, 255), thickness=2)
    strText = str(class_nums[indx[n]]) + ': ' + str(format(score[n], '.4f'))
    cv2.putText(img_data, strText, (bbox[n][0], bbox[n][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
  imwrite(img_data, os.path.join(save_path, files[i]))
    
  '''
  out_img = show_result(img, result, class_nums, show=False, score_thr=0.5)
  print(np.shape(out_img))
  image = Image.fromarray(out_img)
  image.save(os.path.join(save_path, files[i]))
  print('..............................................................')
  '''

