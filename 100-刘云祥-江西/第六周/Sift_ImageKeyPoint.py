import cv2
import numpy as np
from skimage.io import sift

SrcImg = cv2.imread('HomeworkImg.jpg')
GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
# Sift = cv2.xfeatures2d.SIFT_create()
Keypoints, descriptor = sift.detectAndCompute(GrayImg, None)
SrcImg = cv2.drawKeypoints(image=SrcImg, outImage=SrcImg, keypoints=Keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(50, 150, 200))
cv2.namedWindow('keypointImg', 2)
cv2.imshow('keypointImg', SrcImg)
cv2.waitKey(0)

