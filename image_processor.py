import cv2
import numpy as np
EXCLUDE = 0.5
THRESHOLD = 0.3
NMSTHRESHOLD=0.5
class ImageProcessor():
    def __init__(self):
        self.tensorflow_net = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco_2018_03_29.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    def centroid(self,boxes, indices):
        """Detect centroid of detection boxes
        Parameters
        ----------
        boxes : 2D array
            array of detection boxes [[left, top, width, height]...]
        indices : 2D array with 1 dimension == 1
            indices of detection boxes of interest"""
        center=np.zeros((len(indices),2),dtype=int)
        i=0
        for index in indices.flatten():
            x= boxes[index][0]+0.5*boxes[index][2]
            y= boxes[index][1]+0.5*boxes[index][3]
            center[i]=[int(x),int(y)]
            i+=1
        return center
    # fungsi distance generate distance matrix tapi dalam float
    # eg dist[1][2] isinya jarak antara titik 1 dan 2, dist[1][3] jarak antar 1 sama 3 dst
    def distance(self,center, indices):
        """Perform distance measurement, returns a nxn eucliadean distance matrix.
        return type: 2d numpy ndarray
        Parameters
        ----------
        center : 2d array
            the center point of detection boxes
        indices : 1D/2D with one dimension=1
            indices of the detections
        indices -- the detection indexes returned from nmsboxes"""
        length=len(indices)
        dist = np.zeros((length,length))
        for i in range(length):
            dist[i]=(((center[i]-center)**2).sum(axis=1))**0.5
        return dist

    def violations(self, centre, img, boxes, indices, confidences, distance, alpha, color1, color2):
        """Visualize detection boxes and detect violations, returns the number of violations
        detected
        Parameters
        ----------
        img : image
            original image
        boxes : 2D array
            detection boxes [[left, top, width, height]...]
        indices : 1D/2D array with 1 dimension ==1
            indices of detections of interest
        confidences : 1D array
            detection confidences
        distance : 2D array
            distance matrix
        alpha : int/float
            violation distance scaling
        color1 : tuple
            colour of violation boxes
        color2 : tuple
            colour of non-violation boxes"""
        i0=0
        length=len(indices)
        detected = 0
        avgwidth = 0
        index = indices.flatten()
        flag = np.zeros(max(index)+1)
        # iterate through detections
        for i in index:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # for each detection, check if its distance with other points is lower
            # than treshold value
            if i != index[-1]:
                for iter in range(i0+1,length):
                    w2 = boxes[index[iter]][2]
                    avgwidth = (w+w2)/2 
                    # violation detected (lower than a certain value)
                    if distance[i0][iter]<avgwidth*alpha:
                        detected+=1
                        flag[i]=1
                        flag[index[iter]]=1
            if flag[i]==1:
                cv2.rectangle(img, (x, y), (x + w, y + h), color1, 2)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), color2, 2)
            cv2.circle(img,tuple(centre[i0]),3,(0,255,0),2 )
            text = "Conf:"+str(round(confidences[i],3))
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139,0,0), 1)
            i0+=1
        return detected
    
    def detect(self, frame):
        rows, cols, _ = frame.shape
        self.tensorflow_net.setInput(cv2.dnn.blobFromImage(frame, size=(320, 320), swapRB=True, crop=False))
        network_output = self.tensorflow_net.forward()
        boxes=[]
        confidences=[]
        for detection in network_output[0,0]:
            score = float(detection[2])
            if score > THRESHOLD and int(detection[1])==1:
            
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                box = [int(left),int(top),int(right-left),int(bottom-top)]
                if(detection[5]-detection[3]<EXCLUDE):
                    boxes.append(box)
                    confidences.append(float(round(score,3)))
        indices = cv2.dnn.NMSBoxes(boxes,confidences,THRESHOLD,NMSTHRESHOLD)
        if len(indices)!=0:
            centre = self.centroid(boxes,indices)
            dist = self.distance(centre,indices)
            num_of_viols=self.violations(centre, frame, boxes, indices, confidences, dist, 1.5, (0,0,255), (0,255,0))
        return num_of_viols



if __name__ == '__main__':
    image_processor = ImageProcessor()
    filename="../trials/kelasft1.jpg"
    frame=cv2.imread(filename)
    image_processor.detect(frame)

