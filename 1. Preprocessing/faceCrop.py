import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, os.path
import sys

# Notes: 'face.xml' is 'opencv/data/haarcascades/haarcascade_frontalface_default.xml'

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('face.xml')

# Notes : 'JoYuris' element is file directory
# raw image file is saved 'raw' dir and cropped file is saved 'cropped' dir
# ex) raw image : './Yuri/raw/1.jpg', cropped image : './Yuri/cropped/1.jpg' 
JoYuris =['Yuri', 'Caewon', 'Yaena']
for member in JoYuris:
    path = './' + member
    imageList = os.listdir(path+'/raw/')
    cnt = 0 
    for imageName in imageList:
        image = cv2.imread(path+'/raw/'+imageName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # If it cannot find face in image, then waste that image
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        crop  = image[y:y+h, x:x+w]
        resized= cv2.resize(crop, (128, 128),interpolation = cv2.INTER_CUBIC)
        plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        plt.show()
        cnt = cnt + 1
        cv2.imwrite(path+'/croped/'+str(cnt)+'.jpg', resized)
