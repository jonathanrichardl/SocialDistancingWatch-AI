import cv2
from models.common import DetectMultiBackend
import sys
import torch
from utils.general import (non_max_suppression, scale_coords)
from utils.augmentations import letterbox
from utils.plots import Annotator, colors, save_one_box
import numpy as np
from utils.torch_utils import select_device
import os

import cv2
import numpy as np
from sys import path
ROOT = sys.path[0]
EXCLUDE = 0.5
THRESHOLD = 0.3
NMSTHRESHOLD=0.5
class ImageProcessor():
    def __init__(self):
        self.device = select_device('')
        self.model = DetectMultiBackend(ROOT + '/weight/capstone.pt', device=self.device, dnn=False, data=ROOT + '/data/data.yaml', fp16=False)
        self.names = self.model.names
        self.model.warmup(imgsz=(1,3,640,640))

    def centroid(self, boxes):
        center=np.zeros((len(boxes),2),dtype=int)
        i=0
        for box in boxes:
            x= box [0]+ 0.5*box[2]
            y= box[1]+0.5*box[3]
            center[i]=(int(x),int(y))
            i+=1
        return center


    def distance(self, center, indices):
        length=len(indices)
        dist = np.zeros((length,length))
        for i in range(length):
            dist[i]=(((center[i]-center)**2).sum(axis=1))**0.5
        return dist

    def violations(self, centre, img, boxes, distance, alpha, color1, color2):
        i0=0
        detected = 0
        avgwidth = 0
        flag = np.zeros(len(boxes))
        length = len(boxes)
        # iterate through detections
        for box in boxes:
            (x, y) = box[0], box[1]
            (w, h) = box[2], box[3]
            # for each detection, check if its distance with other points is lower
            # than treshold value
            for iter in range(i0+1,length):
                w2 = boxes[iter][2]
                avgwidth = (w+w2)/2 
                # violation detected (lower than a certain value)
                if distance[i0][iter] < avgwidth * alpha:
                    detected+=1
                    flag[i0]=1
                    flag[iter]=1
            if flag[i0]:
                cv2.rectangle(img, (x, y), (x + w, y + h), color1, 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), color2, 2)
        
            cv2.circle(img,tuple(centre[i0]),3,(0,255,0),2 )
            i0+=1
        return detected
    
    def preprocess(self, img : np.ndarray):
        resized = letterbox(img)[0]
        resized = resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        resized = np.ascontiguousarray(resized)
        return resized
    
    def detect(self, original_img):
        blob = self.preprocess(original_img)
        im = torch.from_numpy(blob).to(self.device)
        im = im.float()  
        im /= 255  
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        predictions = self.model(im, augment=False, visualize=False )
        predictions = non_max_suppression(predictions, 0.25, 0.45, None, False, max_det=1000)
        boxes = []
        for detection in predictions: 
            detection[:, :4] = scale_coords(im.shape[2:], detection[:, :4], original_img.shape).round()
            for *coordinates, _, _ in reversed(detection):
                coordinates = list(map(int, coordinates))
                boxes.append((coordinates[0], coordinates[1] , (coordinates[2] - coordinates[0]), (coordinates[3] - coordinates[1])))
        if boxes:
            centre = self.centroid(boxes)
            dist = self.distance(centre, boxes)
            num_of_viols=self.violations(centre, original_img, boxes, dist, 1, (0,0,255), (0,255,0))
            return num_of_viols
        return 0


if __name__ == '__main__':
    from time import time
    image_processor = ImageProcessor()
    folder_name= ROOT + "/test"
    start= time()
    n = len(os.listdir(f'{sys.path[0]}/test'))
    for file in os.listdir(folder_name):
        if not file.endswith('.jpg'):
            continue
        frame = cv2.imread(folder_name + "/" + file)
        image_processor.detect(frame)
        cv2.imwrite(ROOT + "/test/yolov5/" + file, frame)
    end = time()
    print(f"Average Processing time for each image = {(end-start)/n} seconds")