import cv2
import os
alg="haarcascade_frontalface_default.xml"
haar=cv2.CascadeClassifier(alg)

cam=cv2.VideoCapture(0)

dataset="dataset"
name="champ"

path=os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height)=(130,100)
count=1

while (count<30):
    print(count)
    _,img=cam.read()
    gryimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=haar.detectMultiScale(gryimg,1.3,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        onlyFace=gryimg[y:y+h,x:x+h]
        resixeImg=cv2.resize(gryimg,(width,height))
        cv2.imwrite("%s%s.jpg"%(path,count),resixeImg)
        count+=1
    cv2.imshow("facedetection",img)
    key=cv2.waitKey(10)
    if key==27:
        break
print("Faces captured sucessfully")
cam.release()
cv2.destroyAllWindows()