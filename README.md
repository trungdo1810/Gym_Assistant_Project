# Augmented Statups YOLOv7 Pose-Estimation Tutorial

## 1.	Clone the repo to your working directory using Terminal
``` git clone  https://github.com/augmentedstartups/pose-estimation-yolov7.git ```

Or 

Download YOLOv7 Pose Files from Course (Includes YOLOv7 Pose Weights)

## 2.	Create a new Conda Environment

``` conda env create -f environment.yml ```

Activate new Conda Environment  
``` conda activate yolov7pose ```


## 3.	Navigate to the cloned pose-estimation folder 
``` cd pose-estimation ```

## 4.	Download the Pose Weights and put it in pose-estimation folder
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt 

Otherwise use on Ubuntu:

``` wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt ```


## 5. Run Pose Estimation:

``` python run_pose.py  –-source 0 ```

To run inference on video:

``` python run_pose.py  –-source [path to video]```

To run on GPU:

``` python run_pose.py  –-source 0  –-device 0 ```

 
## References
YOLOv7 - https://github.com/WongKinYiu/yolov7 
