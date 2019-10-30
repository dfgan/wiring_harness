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

def NMS_alphaRatioConf(bboxes, score, thresh, nms_alpha):
    """ Pure Python NMS baseline."""
    # bounding box and score
    boxes = np.array(bboxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = np.array(score)
    # the area of candidate
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # score in descending order
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Calculate the intersection between current box and other boxes
        # using numpy->broadcast, obtain vector
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # intersection area, return zero if no intersection
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # IoU: intersection area / (area1+area2-intersection area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # find out the boxes which's "overlap ratio smaller than threshold
        inds1 = np.where(ovr <= thresh)[0]
        # find out the boxes which's "overlap ratio bigger than threshold
        inds2 = np.where(ovr > thresh)[0]
        # find out the boxes which's "overlap ratio bigger than threshold" but "score >= (currentBoxScore * nms_alpha)"
        for find_ in inds2:
            thd_temp = scores[i] * nms_alpha
            if thd_temp <= score[find_]:
                keep.append(find_)
        # '+1' as the compared boxes do not include "order[0]"
        inds = inds1 + 1     # all elements in "inds1[0]" "+1"

        # update order
        order = order[inds]
    return keep

def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分�?
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方�?
    :param thresh: 最后的分数门限
    :param method: 使用的方�?
    :return:       留下�?boxes �?index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep


def nms(boxes, scores, iou_threshold, max_output_size, soft_nms=False):
    keep = []
    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]  # 按得分从大到小排�?
    num = len(boxes)
    suppressed = np.zeros((num), dtype=np.int)  # 抑制
    for _i in range(num):
        if len(keep) >= max_output_size:
            break
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ####boxes左上和右下角坐标####
        xi1 = boxes[i, 0]
        yi1 = boxes[i, 1]
        xi2 = boxes[i, 2]
        yi2 = boxes[i, 3]
        areas1 = (xi2 - xi1 + 1) * (yi2 - yi1 + 1)  # box1面积
        for _j in range(_i + 1, num):  # start，stop
            j = order[_j]
            if suppressed[i] == 1:
                continue
            xj1 = boxes[j, 0]
            yj1 = boxes[j, 1]
            xj2 = boxes[j, 2]
            yj2 = boxes[j, 3]
            areas2 = (xj2 - xj1 + 1) * (yj2 - yj1 + 1)  # box2面积

            xx1 = np.maximum(xi1, xj1)
            yy1 = np.maximum(yi1, yj1)
            xx2 = np.minimum(xi2, xj2)
            yy2 = np.minimum(yi2, yj2)

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            int_area = w * h  # 重叠区域面积

            inter = 0.0

            if int_area > 0:
                inter = int_area * 1.0 / (areas1 + areas2 - int_area)  # IOU
            ###softnms
            if soft_nms:
                sigma = 0.6
                if inter >= iou_threshold:
                    scores[j] = np.exp(-(inter * inter) / sigma) * scores[j]
            ###nms
            else:
                if inter >= iou_threshold:
                    suppressed[j] = 1
    return keep  # 返回保留下来的下�?

for i in range(len(imgs)):
    print(imgs[i])
    img = Image.open(imgs[i])
    img = np.asarray(img)
    result = inference_detector(model, img)
    img_data = copy.copy(img)
    print(result)
    indx, score, bbox = selectClsScoreBoxFromResult(result)
    # keep = NMS_alphaRatioConf(bbox, score, 0.7, 0.7)
    keep = nms(bbox, score, 0.3, 10, soft_nms=True)
    for n in keep:
        # if score[n] < 0.6*max(score):
        #     continue
        left_top = (bbox[n][0], bbox[n][1])
        right_bottom = (bbox[n][2], bbox[n][3])
        cv2.rectangle(img_data, left_top, right_bottom, (0, 0, 255), thickness=2)
        strText = str(class_nums[indx[n]]) + ': ' + str(format(score[n], '.4f'))
        cv2.putText(img_data, strText, (bbox[n][0], bbox[n][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1)
    imwrite(img_data, os.path.join(save_path, files[i]))

