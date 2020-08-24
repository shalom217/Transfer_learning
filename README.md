# Transfer_learning
According to https://www.healthline.com/health/pneumonia#is-it-contagious? Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. This infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe. The germs that cause pneumonia are contagious. This means they can spread from person to person. Pneumonia can be Bacterial, Viral, Fungal. Anyone can get pneumonia from 2 years old to people ages 65 years and older people who smoke, use certain types of drugs, or drink excessive amounts of alcohol and many others. So basically this disease is related to the lungs and the people who have mild systems must gone through the check-up. AI provide us that way to detect the people having pneumonia. The model is trained on lots of lung X-ray images of several people, some of them have pneumonia and some are normal(Download dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). So when a person walks for the check up, the X-ray image of his lungs must be passed through the classifier and it will detect weather the person is having pneumonia or not?
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/images%20(1).jpg)
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/images.jpg)

# Transfer Learning techinque
This classifier is built using Transfer Learning techinque on one of the famous CNN architecture that is VGG16(using trained weights of imagenet dataset)which was itself built on imagenet dataset of 1000 differnt classes.![alt text](https://github.com/shalom217/Transfer_learning/blob/master/transfer_l.jpeg)

# comparision between old/default VGG16model

Here a detailed comparision between old/default VGG16model(which was built to classify 1000 categories) and our custom model using trained weights of VGG16model and classifying only 2 classes.Have a look----- ![alt text](https://github.com/shalom217/Transfer_learning/blob/master/DEFAULTvsOURS.png)
Here we implementing(optimizer = 'adam') with Callbacks method having (EarlyStopping=92% accuracy and ModelCheckpoint=91% accuracy).Here Accuracy log is shown-----![alt text](https://github.com/shalom217/Transfer_learning/blob/master/accuracy_log.png)

# accuracyVSepoch and lossVSepoch curves
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/accuracyVSepoch.png)
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/lossVSepoch.png)

We have also accuracyVSepoch and lossVSepoch curves of both test and train datasets.
Go check it out----train the model, save it, and try predicting by your own.

# REQUIREMENTS----
keras 2.3.1,  
python 3.7,
tensorflow 2.0.0, 
cuda installed,
openCV 4.1.1.26, 
imagenet weights.

# Results--------------
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/NORMAL_RESULT.png)
![alt text](https://github.com/shalom217/Transfer_learning/blob/master/pneumonia_result.png)
