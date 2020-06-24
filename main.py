# -*- coding: utf-8 -*-

import time
from PIL import Image
import numpy as np

from util.utils_text import detect_lines
from util.utils_ocr import process_alphabet, predict
from config import scale, maxScale, TEXT_LINE_SCORE
from config import ocrPath, textPath



def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def rotate_cut_img(im,box,leftAdjust=0.0,rightAdjust=0.0):
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    degree,w,h,x_center,y_center = solve(box)
    xmin_ = min(box[0::2])
    xmax_ = max(box[0::2])
    ymin_ = min(box[1::2])
    ymax_ = max(box[1::2])
    x_center = x_center-xmin_
    y_center =y_center-ymin_
    im = im.crop([xmin_,ymin_,xmax_,ymax_])
    degree_ = degree*180.0/np.pi
    xmin = max(1,x_center-w/2-leftAdjust*(w/2))
    ymin =y_center-h/2
    xmax = min(x_center+w/2+rightAdjust*(w/2),im.size[0]-1)
    ymax = y_center+h/2
    newW = xmax-xmin
    newH = ymax-ymin
    tmpImg = im.rotate(degree_,center=(x_center,y_center)).crop([xmin,ymin,xmax,ymax])
    box = {'cx':x_center+xmin_,'cy':y_center+ymin_,'w':newW,'h':newH,'degree':degree_}
    return tmpImg


def text_ocr(img, textModel, ocrModel):

    # get boxes (textModel)
    start = time.perf_counter()

    boxes, scores = detect_lines(textModel, img, scale=scale, maxScale=maxScale)

    end = time.perf_counter()
    print('Detection model processing time is {}'.format(end - start))

    # process boxes
    start = time.perf_counter()
    tt = 0

    im = Image.fromarray(img)
    result = []
    for i, box in enumerate(boxes):
        if scores[i] > TEXT_LINE_SCORE:
            # extract proposal
            tmpImg = rotate_cut_img(im, box, leftAdjust=0.01, rightAdjust=0.01)

            start_t = time.perf_counter()

            # load alphabet
            alphabet = process_alphabet('./weight/ocr/ocr.json')

            end_t = time.perf_counter()
            tt = tt+end_t-start_t

            # recognize words (ocrModel)
            text_dict = predict(ocrModel, tmpImg, alphabet)

            if text_dict['text'] != '':
                text_dict['box'] = [int(x) for x in box]
                text_dict['textprob'] = round(float(scores[i]), 2)
                result.append(text_dict)

    result = sorted(result, key=lambda x: sum(x['box'][1::2]))

    print('Recognition Model inference time is {}'.format(tt))
    end = time.perf_counter()
    print('Recognition Model processing time is {}'.format(end - start))

    return result


if __name__ == '__main__':
    import cv2
    import torch
    from model.vgg import VGG
    from model.cnn import CNN

    start_p = time.perf_counter()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1"

    # load image for text
    image_path = './test/text/1032838434.jpg'
    img = cv2.imread(image_path) # ndarray

    # load model
    textModel = VGG(3).cuda()
    checkpoint = torch.load(textPath)
    textModel.load_state_dict(checkpoint)
    print(textModel)

    ocrModel = CNN(1).cuda()
    checkpoint = torch.load(ocrPath)
    ocrModel.load_state_dict(checkpoint)
    print(ocrModel)

    end = time.perf_counter()
    print('Load Model time is {}'.format(end - start_p))

    text = text_ocr(img, textModel, ocrModel)

    end_p = time.perf_counter()
    print('OCR processing time is {}'.format(end_p - start_p))

    print(text)




