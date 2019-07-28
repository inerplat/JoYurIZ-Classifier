import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, os.path
import sys

# Notes: 'face.xml' is 'opencv/data/haarcascades/haarcascade_frontalface_default.xml'

cascPath = sys.argv[1]
frontFaceCascade = cv2.CascadeClassifier('frontFace_default.xml')
profileFaceCascade = cv2.CascadeClassifier('profileFace.xml')

# Notes : 'JoYuris' element is file directory
# raw image file is saved 'raw' dir and cropped file is saved 'cropped' dir
# ex) raw image : './Yuri/raw/1.jpg', cropped image : './Yuri/cropped/1.jpg' 

JoYuris =['Yuri', 'Caewon', 'Yaena']
for member in JoYuris:
    path = './' + member
    imageList = os.listdir(path+'/raw/')
    print(format(imageList))
    cnt = 0
    for imageName in imageList:
        print(path+'/raw/'+imageName)
        image = cv2.imread(path+'/raw/'+imageName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frontFaces = frontFaceCascade.detectMultiScale(gray, 1.3, 5)
        profileFaces = profileFaceCascade.detectMultiScale(gray, 1.3, 5)
        if len(frontFaces) > 0:
            faces = frontFaces
        elif len(profileFaces) > 0:
            faces = profileFaces
        else:
            continue
        x, y, w, h = faces[0]
        print(x, y, w, h)
        
        cropType = [[y, y+h, x, x+w], [int(y-h/5), int(y+h-h/5), int(x-w/5), int(x+w-w/5)], [int(y+h/5),int( y+h+h/5), int(x-w/5), int(x+w-w/5)], 
                    [int(y-h/5), int(y+h-h/5), int(x+w/5),int( x+w+w/5)], [int(y+h/5), int(y+h+h/5), int(x+w/5),int( x+w+w/5)]] 
        crop  = image[y:y+h, x:x+w]
        cnt = cnt + 1
        tcnt = 0
        for cropType in cropType:
            tcnt = tcnt + 1
            crop = image[cropType[0]:cropType[1], cropType[2]:cropType[3]]
            resized= cv2.resize(crop, (128, 128),interpolation = cv2.INTER_CUBIC)
            plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            plt.show()
            cv2.imwrite(path+'/croped/'+str(cnt)+'-'+str(tcnt)+'.jpg', resized)
        
