import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, os.path
import sys
from PIL import Image
from PIL import GifImagePlugin
import face_recognition as FR
import magic
import shutil
JoYuris =['Yuri', 'Chaewon', 'Yena']
for member in JoYuris:
    newImageName = ''
    path = '../Dataset/Raw/' + member + '/'
    savePath = '../Dataset/Train/' + member + '/'
    failPath = '../Dataset/Fail/' + member + '/'
    imageList = os.listdir(path)
    cnt = 0
    for imageName in imageList:
        print("path+imageName : ",path+imageName)
        if os.path.isfile(savePath+imageName) == True:
            continue
        extention = magic.from_file(path+imageName).split()[0].upper()
        if extention == 'GIF':
            imageObject = Image.open(path+imageName)
            imageObject.seek(0) 
            imageObject = imageObject.convert('RGB')
            image = np.array(imageObject)
        elif extention != 'JPEG' and extention != 'PNG':
            continue
        else:
            image = FR.load_image_file(path+imageName)
        faces = FR.face_locations(image, number_of_times_to_upsample=0, model="hog")
        if len(faces) != 1 :
            shutil.move(path+imageName, failPath+imageName)
            continue
        for T, R, B, L in faces:
            cnt = cnt + 1
            tcnt = 0
            crop  = image[T:B, L:R]
            resized = Image.fromarray(crop).resize((256,256), Image.BICUBIC)
            print(type(resized))
            newImageName = savePath + imageName.split('.')[0] + '.jpg'
            print(newImageName)
            resized.save(newImageName)