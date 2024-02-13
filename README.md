# yolov4_real_time_object_detecion

# Task Approaches
 - we will do this task in VSCODE
 - first we will create envoroment
 - we will create a requirements.txt for required library which is we will use for this project.
 - models are - we can use yolov1, yolov2, yolov3, yolov4, yolov5.
 - nowadays these all models are use for object detection
 - by using this model we will do own task like - person detection, vehicle detection etc.


## What specific steps would you take to detect each object in the image?
- There is some steps 
 - pip install pymupdf
 - import fitz
 - By using this we can extract Text, images, Extracting Links from Page
 - fitz library is specifically designed for working with PDF documents
 - aslo we can use *request* library for extracting  images form web.
 - The fitz library is a Python binding for the MuPDF library, and it is commonly used for working with PDF documents


#### import subprocess
 - This code is using the subprocess module to run the wget command in a loop for each link in the page_links list. The purpose is to download images from the specified links using the wget utility.


## Import necessary library
*import cv2* - Open Source Computer Vision Library (OpenCV for image processing)
*import numpy as np* - NumPy for numerical operations
*import time* - Time for measuring execution time
*import os* - OS is used for interacting with the operating system
*import matplotlib.pyplot as plt* - visulization tools in python library
*from google.colab.patches import cv2_imshow* - For displaying images in Google Colab


## we will use pretrained model YOLOv4 for object detection
 -  for this we will clone the github reprosetry where all the requirements are there like *weight initilize* and yolo configure.
 -  not everyone have high and pc so we can't train owned model so we use pretrainde model for any task.
 -  YOLOv4 is trained on large data set.
 -  pretrained means - we are using previous knowledge of model for doing current task.


## some potential challanges are there
 - First i was trying to do these task on jupyter notebook and VSCODE but my pc is not compatible so i use google colab.
 - It is possible that it may not be able to detect well on some images.


## What technologies would you use and why?
 - *computer visionn (OpenCV (cv2))* - OpenCV is a popular computer vision library in Python. It provides functions for image and video processing and analysis, and computer vision tasks.

 - *NumPy* - It is a numerical computing library in python. it supports  multi-dimensional arrays and matrices. Used this code for array manipulation.
 - *Matplotlib* - Matplotlib is a 2D plotting library for Python. Used in this code for displaying images using plt.imshow()
 - *Google Colab* - Google Colab is a cloud-based Jupyter notebook environment provided by Google.
 - *Darknet (YOLOv4):* - YOLOv4 is a  real-time object detection system.
 - *wget* - wget is a utility for downloading files from the web. Used in the code to download the pre-trained YOLOv4 weights.
 - *Exception Handling* - For handling errors.
