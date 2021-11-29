import cv2
import imutils
#resizedimg=imutils.resize(img,width=20)
img=cv2.imread("download.jpg")
#gsblur=cv2.GaussianBlur(img,(21,21),0)
#gryimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#threshholdImg=cv2.threshold(gryimg,220,255,cv2.THRESH_BINARY)(saving error)
cv2.imwrite("threshimg.jpg",threshholdImg)