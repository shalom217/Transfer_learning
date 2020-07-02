# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:59:09 2020

@author: shalo
"""


# Loading the vgg16 Model
"""
Using vgg16 for our Pneumonia Detection

"""

#Freeze all layers except the top 4, as we'll only be training the top 4

import keras
from keras.applications import VGG16

#VGG16=keras.applications.vgg16.VGG16()#to check ALL the layers
#VGG16.summary()
#VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

#FC=fully connected
# Re-loads the VGG16 model without the top or FC layers
# Here we freeze the last 4 layers 
VGG16 = VGG16(weights = 'imagenet', 
                 include_top = False,#not to train the top layer(or last layer because last layer is classifying 1000 classes but we want only two classes to be classified)
                 input_shape = (img_rows, img_cols, 3))
VGG16.summary()

# Layers which are set to trainable as True by default we make them false(not to train because it require high computational power and they are already trained)
for layer in VGG16.layers:
    layer.trainable = False
    
    
# Let's print our layers 
for (i,layer) in enumerate(VGG16.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


#Let's make a function that returns our FC Head

def addTopModelVGG16(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed on top of the last layers of VGG16"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model

#Let's add our FC Head back onto VGG16


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D#dropout=leave some perceptron of last layer(reduce overfitting )
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 2 (Normal and Pneumonia)
num_classes = 2

FC_Head = addTopModelVGG16(VGG16, num_classes)

model = Model(inputs = VGG16.input, outputs = FC_Head)#integrating both cnn vgg16 model and user model

print(model.summary())

#Loading our Pneumonia Dataset

from keras.preprocessing.image import ImageDataGenerator#for data augmentation
import time
start=time.time()

train_data_dir = 'C:/Users/shalo/Desktop/ML stuffs/DL/pneumonia_detection/dataset/lung cancer/chest_xray/train'
validation_data_dir = 'C:/Users/shalo/Desktop/ML stuffs/DL/pneumonia_detection/dataset/lung cancer/chest_xray/test'

# Let's use some data augmentation 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)# we do not require data augmentation on test data
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32#will use 32 images at one time
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


#Training out Model


from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("Pneumonia_TFL_MdlCHCKPNT.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = adam(lr = 0.0001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 5216# images in training folder of both classes
nb_validation_samples = 624# images in test folder of both classes

# We only train 5 EPOCHS 
epochs = 5
batch_size = 32

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)#// means it wont consider float value only int values will be considered


model.save('Pneumonia_TFL_Erlystop.h5')


stop=time.time()

print("time taken ", (stop-start)," s")

import matplotlib.pyplot as plt

# from IPython.display import Inline
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print(train_generator.class_indices)

#Testing our classifer on some test images

from keras.models import load_model
import numpy as np
path=r"C:\Users\shalo\Desktop\ML stuffs\DL\pneumonia_detection\using_transfer_learning\Pneumonia_TFL_Erlystop(92%).h5"
classifier = load_model(path)


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

PNEUMONIA_dict = {"[0]": "NORMAL ", 
                      "[1]": "PNEUMONIA"}

def draw_test(name, pred, im):
    check = PNEUMONIA_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 0, 0, 500, 550 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, check, (152, 60) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)
    
    
image_path=r"C:\Users\shalo\Desktop\ML stuffs\DL\pneumonia_detection\dataset\lung cancer\chest_xray\val\PNEUMONIA\person1952_bacteria_4883.jpeg"
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
# Show image with predicted class
draw_test("Prediction", res, input_original) 
cv2.waitKey(0)

cv2.destroyAllWindows()    