# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
import time

import torch

from config import anchors, GPU



class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def __init__(self, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6):
        """
        @@param:MAX_HORIZONTAL_GAP:文本行间隔最大值
        @@param:MIN_V_OVERLAPS
        @@param:MIN_SIZE_SIM
        MIN_V_OVERLAPS=0.6
        MIN_SIZE_SIM=0.6
        """
        self.MAX_HORIZONTAL_GAP = MAX_HORIZONTAL_GAP
        self.MIN_V_OVERLAPS = MIN_V_OVERLAPS
        self.MIN_SIZE_SIM = MIN_SIZE_SIM

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + self.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - self.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= self.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= self.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1
        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)

        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.

                graph[index, succession_index] = True

        return Graph(graph)


class TextProposalConnector:
    """
        Connect text proposals into text lines
    """

    def __init__(self, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6):
        self.graph_builder = TextProposalGraphBuilder(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes

        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  ##find the text line

        text_lines = np.zeros((len(tp_groups), 8), np.float32)
        newscores = np.zeros((len(tp_groups),), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]
            # num = np.size(text_line_boxes)##find
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)
            # p1 = np.poly1d(z1)

            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score
            text_lines[index, 5] = z1[0]
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))
            text_lines[index, 7] = height + 2.5
            newscores[index] = score
        return text_lines, newscores


def normalize(data):
    if data.shape[0]==0:
        return data
    max_=data.max()
    min_=data.min()
    return (data-min_)/(max_-min_) if max_-min_!=0 else data-min_


def nms(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    def box_to_center(box):
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        return [xmin, ymin, w, h]

    newBoxes = [box_to_center(box) for box in boxes]
    newscores = [float(x) for x in scores]
    index = cv2.dnn.NMSBoxes(newBoxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(index) > 0:
        index = index.reshape((-1,))
        return boxes[index], scores[index]
    else:
        return [], []


def get_boxes(bboxes):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    boxes = []
    for box in text_recs:
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[6], box[7])
        x4, y4 = (box[4], box[5])
        boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    boxes = np.array(boxes)
    return boxes


class TextDetector:
    """
        Detect text from an image
    """

    def __init__(self, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6):
        """
        pass
        """

        self.text_proposal_connector = TextProposalConnector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)

    def detect(self, text_proposals, scores, size,
               TEXT_PROPOSALS_MIN_SCORE=0.7,
               TEXT_PROPOSALS_NMS_THRESH=0.3,
               TEXT_LINE_NMS_THRESH=0.3,
               TEXT_LINE_SCORE=0.7
               ):
        text_proposals, scores = nms(text_proposals, scores, TEXT_PROPOSALS_MIN_SCORE, TEXT_PROPOSALS_NMS_THRESH)
        if len(text_proposals) > 0:
            scores = normalize(scores)
            text_lines, scores = self.text_proposal_connector.get_text_lines(text_proposals, scores,
                                                                             size)  ##cluster lines
            text_lines = get_boxes(text_lines)
            # text_lines, scores     = rotate_nms(text_lines,scores,TEXT_LINE_SCORE,TEXT_LINE_NMS_THRESH)##?cv2.dnn.rotate_nms error
            return text_lines, scores
        else:
            return [], []


def resize_img(image, scale, maxScale=None):
    """
    image :BGR array
    """
    image = np.copy(image)
    vggMeans = [122.7717, 102.9801, 115.9465]  # used to normalize
    imageList = cv2.split(image.astype(np.float32))  # split the channel
    imageList[0] = imageList[0] - vggMeans[0]  # B
    imageList[1] = imageList[1] - vggMeans[1]  # G
    imageList[2] = imageList[2] - vggMeans[2]  # R
    image = cv2.merge(imageList)  # merge to bgr, (h,w,c)

    h, w = image.shape[:2]  # image height and width
    rate = scale / min(h, w)  # 600/min(h,w) is the ratio of future w/h to real w/h
    if maxScale is not None:  # is 900
        if rate * max(h, w) > maxScale:  # max(future w/h) v.s. the upper boundary
            rate = maxScale / max(h, w)  # do not higher than boundary

    image = cv2.resize(image, None, None, fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)  # resize the image by ratio

    return image, rate


def resize_img2(img, scale, maxScale=None):
    """
    image :BGR array
    """

    img = np.copy(img)
    img = img.astype(np.float32)
    vggMeans = [122.7717, 102.9801, 115.9465]  # used to normalize
    img[:, :, 0] = img[:, :, 0] - vggMeans[0]
    img[:, :, 1] = img[:, :, 1] - vggMeans[1]
    img[:, :, 2] = img[:, :, 2] - vggMeans[2]

    h, w = img.shape[:2]  # image height and width
    rate = scale / min(h, w)  # 600/min(h,w) is the ratio of future w/h to real w/h
    if maxScale is not None:  # is 900
        if rate * max(h, w) > maxScale:  # max(future w/h) v.s. the upper boundary
            rate = maxScale / max(h, w)  # do not higher than boundary

    img = cv2.resize(img, None, None, fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)  # resize the image by ratio

    return img, rate


def reshape(x):
    b = x.shape
    x = x.transpose(0, 2, 3, 1) # to h,w,c
    b = x.shape
    x = np.reshape(x,[b[0],b[1]*b[2]*10,2]) # change c from 20 to 2
    return x


def get_origin_box(size, anchors, boxes, scale=16):
    """
    size:(w,h) --h,w =img.shape[:2]//16 --- vggnet 8 maxpool
    boxes.shape = iw*ih*len(anchors)
    """

    w, h = size
    iw = int(np.ceil(w / scale)) * scale  # what SAO process
    ih = int(np.ceil(h / scale)) * scale
    anchors = np.array(anchors.split(',')).astype(int)  # load anchors for config file
    anchors = np.repeat(anchors, 2, axis=0).reshape((-1, 4))  # ????
    anchors[:, [1, 2]] = anchors[:, [2, 1]]
    anchors = anchors / 2.0
    cscale = (scale - 1) / 2.0
    anchors[:, [0, 1]] = cscale - anchors[:, [0, 1]]
    anchors[:, [2, 3]] = cscale + anchors[:, [2, 3]]
    gridbox = [[[i, j, i, j] + anchors for i in range(0, iw, scale)] for j in range(0, ih, scale)]
    gridbox = np.array(gridbox)
    gridbox = gridbox.reshape((-1, 4))
    gridcy = (gridbox[:, 1] + gridbox[:, 3]) / 2.0
    gridh = (gridbox[:, 3] - gridbox[:, 1] + 1)
    cy = boxes[:, 0] * gridh + gridcy
    ch = np.exp(boxes[:, 1]) * gridh
    ymin = cy - ch / 2
    ymax = cy + ch / 2
    gridbox[:, 1] = ymin
    gridbox[:, 3] = ymax
    return gridbox


def soft_max(x):
    """numpy softmax"""
    expz = np.exp(x)
    sumz = np.sum(expz,axis=1)
    return expz[:,1]/sumz


def detect_box(model, image, scale=600, maxScale=900):
    image, rate = resize_img2(image, scale, maxScale=maxScale)  # resize the opencv image and convert to
    image = torch.from_numpy(image).cuda() if GPU else torch.from_numpy(image)  #######
    image = image.permute(2, 0, 1).unsqueeze(0)  ######
    h, w = image.size()[2:]  # input image height and width

    start = time.perf_counter()

    res = model(image)  # conduct textModel

    end = time.perf_counter()
    print('Detection Model inference Time is {}'.format(end-start))

    # print('________________')
    # print(out)
    # print(out.shape)
    # print('________________')

    res = res.data.cpu().numpy()
    clsOut = reshape(res[:, :20, ...])  # process cls
    boxOut = reshape(res[:, 20:, ...])  # process box

    # clsOut = res[:, :20, :, :].permute(0, 2, 3, 1).reshape(1, -1, 2).data.cpu().numpy()
    # boxOut = res[:, 20:, :, :].permute(0, 2, 3, 1).reshape(1, -1, 2).data.cpu().numpy()

    boxes = get_origin_box((w, h), anchors, boxOut[0])
    scores = soft_max(clsOut[0])
    boxes[:, 0:4][boxes[:, 0:4] < 0] = 0
    boxes[:, 0][boxes[:, 0] >= w] = w - 1
    boxes[:, 1][boxes[:, 1] >= h] = h - 1
    boxes[:, 2][boxes[:, 2] >= w] = w - 1
    boxes[:, 3][boxes[:, 3] >= h] = h - 1

    return scores, boxes, rate, w, h


def detect_lines(model, image,scale=600,
                 maxScale=900,
                 MAX_HORIZONTAL_GAP=30,
                 MIN_V_OVERLAPS=0.6,
                 MIN_SIZE_SIM=0.6,
                 TEXT_PROPOSALS_MIN_SCORE=0.7,
                 TEXT_PROPOSALS_NMS_THRESH=0.3,
                 TEXT_LINE_NMS_THRESH = 0.9,
                 TEXT_LINE_SCORE=0.9
                ):
    MAX_HORIZONTAL_GAP = max(16,MAX_HORIZONTAL_GAP)
    detectors = TextDetector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)
    scores,boxes,rate,w,h = detect_box(model, image,scale,maxScale) # textModel, image is opencv image
    size = (h,w)
    text_lines, scores =detectors.detect( boxes,scores,size,\
           TEXT_PROPOSALS_MIN_SCORE,TEXT_PROPOSALS_NMS_THRESH,TEXT_LINE_NMS_THRESH,TEXT_LINE_SCORE)
    if len(text_lines)>0:
        text_lines = text_lines/rate
    return text_lines, scores



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1"

    sys.path.append('../')
    from model.vgg import VGG

    img = cv2.imread('../test/text/1032838434.jpg')
    print(img.shape)

    scale = 900
    maxScale = 1800

    model = VGG(3).cuda()
    checkpoint = torch.load('../weight/text/text.pth.tar')
    model.load_state_dict(checkpoint)

    boxes, scores = detect_lines(model, img, scale=scale, maxScale=maxScale)
    print(scores)
    print(boxes)























