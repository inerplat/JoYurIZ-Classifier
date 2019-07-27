import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, os.path
import sys

# notes: 'face.xml' is 'opencv/data/haarcascades/haarcascade_frontalface_default.xml'
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('face.xml')
image = cv2.imread('test2.jpg')

'''DEBUG
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()
'''

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.3, 5)
tmp = 0
print(faces)
for (x, y, w, h) in faces:
    '''DEBUG
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
'''
crop  = image[y:y+h, x:x+w]
resized= cv2.resize(crop, (128, 128),interpolation = cv2.INTER_CUBIC)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
cv2.imwrite("resized.jpg", resized)
