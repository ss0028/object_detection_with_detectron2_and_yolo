# object_detection_with_detectron2_and_yolo

Introduction

Object detection is a computer vision technique for locating instances of objects in images or videos. In this,  we have to classify the objects in the image and also locate where these objects are present in the image. So, broadly we have three tasks for object detection problems:
1. To identify if there is an object present in the image,
2.  Where is this object located,
3. What is this object?
Also, the object detection problem can be divided into multiple categories:
    • Single object detection- detect only 1 object in the provided image
    • Multiple object detection- detect more than one object in the provided image

In this task, our aim is to detect 2 objects in the provided image, which are:
→ person
→ car

![prediction_output11](https://user-images.githubusercontent.com/31015605/169228260-d6a64290-2ef1-4bec-aea0-a028ca85448e.png)


About the model

The given approach to detect person and car in a provided image is implemented in 2 ways:

    1. Using Yolo v3
    2. Using Detectron2 

YOLO(You Only Look Once):

The YOLO framework takes the entire image in a single instance and predicts the bounding box coordinates and class probabilities for these boxes. 
The biggest advantage of using YOLO is its superb speed- it’s incredibly fast and can process 45frames/sec.

Detectron2:

Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms implemented in Pytorch. It supports a number of computer vision tasks such as object  detection, image segmentation, etc.

Detectron 2 includes high-quality implementations of state-of-the-art  object detection algorithms including DensePose, panoptic feature pyramid networks and numerous variants of the pioneering Mask-RCNN model family also developed by FAIR. Its extensible design makes it easy to implement cutting-edge research projects.


Primary Analysis

This object detection task is performed on a dataset containing a total of 2239 images. These images consist of both the classes which need to be detected: person and car.

The annotations for the objects in the images are provided in COCO annotation format. These are converted into annotation format required by the YOLO (a .txt file corresponding to each image)  and Detectron 2(BoxMode.XYXY_ABS) framework respectively.

Also, the bounding box coordinates in COCO annotation represents (x, y, width, height) whereas the boxmode in Detectron2 requires the coordinates as (xmin, ymin, xmax, ymax). This has been taken care of while carrying out the conversion from one annotation format to another. 

False Positives

In object detection, there are 2 mistakes that the algorithm can make. The first error is detecting an object when it's not there. This is the false positive. The second error is not detecting an object when it's there. This is the false negative.
To compute the false positive rate we need to compute how often it detected an object when the object was not there. This is the quotient of the false positives and all potential false positives: i.e., :
False Positive = (detect box when no box is present) / (all no box)

To compute the false negative rate, we compute how often an object is not detected divided by how often the object could have been detected, i.e., 
 False Negative = (detect no box when box) / (all box)

The key distinction between the false positive and false negative rate is which situation / hypothesis is true. False positives exist only when there are no boxes. False negatives exist only when there are boxes.

Evaluation Metric

Evaluation metric for object detection consist of:
    • Intersection over Union(IoU)
    • Mean Average Precision(mAP)

Intersection over Union(IoU) is a term used to describe the extent of overlap of two boxes. The greater the region of overlap, the greater the IOU.

![image](https://user-images.githubusercontent.com/31015605/169039394-b7fedf2f-7886-4af6-90eb-f3925360e880.png)

IoU = (Area of intersection of 2 boxes) / (Area of union of 2 boxes)

![image](https://user-images.githubusercontent.com/31015605/169039429-f59a4095-833b-4453-b06e-e5b2b86bb990.png)

Mean Average Precision(mAP) is a metric used to evaluate object detection models such as Fast R-CNN, YOLO, Mask R-CNN, etc. The mean of average precision(AP) values are calculated over recall values from 0 to 1.

mAP formula is based on the following sub metrics:
    • Confusion matrix
    • Intersection over Union(IoU)
    • Precision
    • Recall
    
  
Training and Validation Loss

A learning curve is a plot of model learning performance over experience or time.

Learning curves are a widely used diagnostic tool in machine learning for algorithms that learn from a training dataset incrementally. The model can be evaluated on the training dataset and on a hold out validation dataset after each update during training and plots of the measured performance can be created to show learning curves.
Below are the learning curves for the trained model:


![image](https://user-images.githubusercontent.com/31015605/169039546-7a74f723-c3ea-4b1d-93b7-a06811528de9.png)

Output

![prediction_output15](https://user-images.githubusercontent.com/31015605/169228664-af7f62e5-1dc9-4df9-8bd3-c6d74d24090f.png)


![image](https://user-images.githubusercontent.com/31015605/169039589-75ba8c10-bf6e-45d5-84bc-fce4cc5ed7cc.png)

![image](https://user-images.githubusercontent.com/31015605/169039600-f80ea08d-f3b5-45a6-a178-889c083b63fa.png)

![image](https://user-images.githubusercontent.com/31015605/169039622-9150d4e3-f831-45d6-94ca-64f67df0f2a8.png)

![prediction_output6](https://user-images.githubusercontent.com/31015605/169039671-1520b562-835f-4fb4-a6b4-48f401374ece.png)

![prediction_output7](https://user-images.githubusercontent.com/31015605/169039709-1f39fd00-e071-49b5-bc07-ea9fd3c02feb.png)





