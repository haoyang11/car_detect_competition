# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:46:12 2018

@author: conna
"""


#coding=utf-8
import cv2
import numpy as np  

img = cv2.imread("E:\\1aaaMCDC\\out\\54.jpg", 0)
img = cv2.resize(img,(400,400))
cv2.imshow('1', img)
img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img, 50, 150)

cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()