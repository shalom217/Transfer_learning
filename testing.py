# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:52:23 2020

@author: shalo
"""

from keras.models import load_model
import numpy as np
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.preprocessing import image
path=r"C:\Users\shalo\Desktop\ML stuffs\DL\pneumonia_detection\using_transfer_learning\Pneumonia_TFL_MdlCHCKPNT(91%).h5"
classifier = load_model(path)

#Testing our classifer on some test images

import os
import cv2
import numpy as np
import re
from os import listdir
from os.path import isfile, join

#LOADING THE VALIDATION FOLDER PATH
mypath=r"C:\Users\shalo\Desktop\ML stuffs\DL\pneumonia_detection\dataset\lung cancer\chest_xray\val\NORMAL"

file_names=[f for f in listdir(mypath) if isfile(join(mypath,f))]   

PNEUMONIA_dict = {"[0]": "NORMAL= ", 
                      "[1]": "PNEUMONIA="}

def draw_test(name, pred, im,Pred_val):
    check = PNEUMONIA_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 0, 0, 500, 550 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, check+str(Pred_val)+str("%"), (152, 60) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)
  
for file in file_names: 
    
    image_path=mypath+"\\"+file
    
    image_path = image_path.replace('\\', '//')#making PATHS READABLE
    input_im=image_path
    input_im=cv2.imread(input_im)
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    res2=classifier.predict(input_im, 1, verbose = 0)
    value=res2.max()
    value=value*100
    value=np.floor(value)
    # Show image with predicted class
    draw_test("Prediction", res, input_original,value) 
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()   
"""Use q to change the images"""   