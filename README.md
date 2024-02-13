# yolov4_real_time_object_detecion

# Task Approaches
 - we will do this task in VSCODE
 - first we will create envoroment
 - we will create a requirements.txt for required library which is we will use for this project.
 - models are - we can use yolov1, yolov2, yolov3, yolov4, yolov5.
 - nowadays these all models are use for object detection
 - we will use yoloV4 model.
 - by using this model we can do own task like - person detection, vehicle detection etc.


## What specific steps would you take to detect each object in the image?
- There is some steps 
 - first we will install all library which is located in requirements.txt
 - *pdf_process* -  In this file i write code for extracting all links from pdf file and by using request library i download all images from links.
 - *main.py* - i write code in this file for real-time object detection. I have use some library in this
 - By using this we can extract Text, images, Extracting Links from Page
 - fitz library is specifically designed for working with PDF documents
 - aslo we can use *request* library for extracting  images form web.
 - The fitz library is a Python binding for the MuPDF library, and it is commonly used for working with PDF documents


#### import request
 - The requests module in Python is a popular HTTP library that allows you to send HTTP requests and handle the responses easily


## Import necessary library
*import cv2* - Open Source Computer Vision Library (OpenCV for image processing)
*import numpy as np* - NumPy for numerical operation.
*import matplotlib.pyplot as plt* - visulization tools in python library
*numpy* 
*matplotlib*
*PyMuPDF*
*requests*
*Pillow*
*mplcursors*
*io*


## we will use pretrained model YOLOv4 for object detection
 -  for this we will clone the github reprosetry where all the requirements are there like *weight initilize* and yolo configure.
 -  not everyone have high and pc so we can't train owned model so we use pretrainde model for any task.
 -  YOLOv4 is trained on large data set.
 -  pretrained means - we are using previous knowledge of model for doing current task.


## some potential challanges are there
 - This is realtime object detection task. but i have not done all task. I'll update when it's complete.


## What technologies would you use and why?
 - *computer visionn (OpenCV (cv2))* - OpenCV is a popular computer vision library in Python. It provides functions for image and video processing and analysis, and computer vision tasks.

 - *NumPy* - It is a numerical computing library in python. it supports  multi-dimensional arrays and matrices. Used this code for array manipulation.
 - *Matplotlib* - Matplotlib is a 2D plotting library for Python. Used in this code for displaying images using plt.imshow()
 - *Darknet (YOLOv4):* - YOLOv4 is a  real-time object detection system.
