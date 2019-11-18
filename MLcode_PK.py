# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:21:55 2019

@author: Panagiotis Korkos
"""

#ML Competition
#Panagiotis Korkos


import os, sys

#Create an index of class names
#path = "C:\Users\Παναγιώτης Κορκός\Documents\PhD_courses\Pattern recognition and Machine Learning\Competition\train\train"
class_names = sorted(os.listdir(r"C:\Users\Παναγιώτης Κορκός\Documents\PhD_courses\Pattern recognition and Machine Learning\Competition\train\train"))

#Prepare a pretrained CNN for feature extraction
import tensorflow as tf
#from tf.keras.applications.mobilenet import MobileNet
base_model = tf.keras.applications.mobilenet.MobileNet(
             input_shape = (224,224,3),
             include_top = False)

in_tensor = base_model.inputs[0] # Grab the input of base model
out_tensor = base_model.outputs[0] # Grab the output of base model

# Add an average pooling layer (averaging each of the 1024 channels):
out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

# Define the full model by the endpoints.
model = tf.keras.models.Model(inputs = [in_tensor],
                              outputs = [out_tensor])

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')

# Find all image files in the data directory.
import numpy as np
import matplotlib.pyplot as plt
import cv2

X = [] # Feature vectors will go here.
y = [] # Class ids will go here.
for root, dirs, files in os.walk(r"C:\Users\Παναγιώτης Κορκός\Documents\PhD_courses\Pattern recognition and Machine Learning\Competition\train\train"):
    for name in files:
        if name.endswith(".jpg"):
            # Load the image:
            img = plt.imread(root + os.sep + name)
            
            # Resize it to the net input size:
            img = cv2.resize(img, (224,224))
            
            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128
            
            # Push the data through the model:
            x = model.predict(img[np.newaxis, ...])[0]
            
            # And append the feature vector to our list.
            X.append(x)
            
            # Extract class name from the directory name:
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))
            
# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

#Split the training dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#LDA method
# Training code:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
model_LDA = LinearDiscriminantAnalysis()
# X contains all samples, and y their class
# labels: y = [0,1,1,0,...]
model_LDA.fit(x_train, y_train)

# Testing code:
y_hat_LDA = model_LDA.predict(x_test)
p_LDA = model_LDA.predict_proba(x_test)

accuracy_LDA = cross_val_score(model_LDA, X, y, cv = 3)

print('The accuracy for LDA method is', accuracy_LDA)


#Linear SVC method
# Training code:
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model_linSVC = SVC(kernel='linear', probability=True)
# X contains all samples, and y their class
# labels: y = [0,1,1,0,...]
model_linSVC.fit(x_train, y_train)

# Testing code:
y_hat_linSVC = model_linSVC.predict(x_test)
p_linSVC = model_linSVC.predict_proba(x_test)

accuracy_linSVC = cross_val_score(model_linSVC, X, y, cv = 3)

print('The accuracy for linear SVC method is', accuracy_linSVC)

#RBF SVC method
# Training code:
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model_rbfSVC = SVC(kernel='rbf', probability=True)
# X contains all samples, and y their class
# labels: y = [0,1,1,0,...]
model_rbfSVC.fit(x_train, y_train)

# Testing code:
y_hat_rbfSVC = model_rbfSVC.predict(x_test)
p_rbfSVC = model_rbfSVC.predict_proba(x_test)

accuracy_rbfSVC = cross_val_score(model_rbfSVC, X, y, cv = 3)

print('The accuracy for rbf SVC method is', accuracy_rbfSVC)

#Logistic Regression method
# Training code:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model_logreg = LogisticRegression()
# X contains all samples, and y their class
# labels: y = [0,1,1,0,...]
model_logreg.fit(x_train, y_train)

# Testing code:
y_hat_logreg = model_logreg.predict(x_test)
p_logreg = model_logreg.predict_proba(x_test)

accuracy_logreg = cross_val_score(model_logreg, X, y, cv = 3)

#100-Tree Random Forests classification
# Training code:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model_randf = RandomForestClassifier(n_estimators=100)
model_randf.fit(x_train, y_train)

# Testing code:
y_hat_randf = model_randf.predict(x_test)
p_randf = model_randf.predict_proba(x_test)

accuracy_randf = cross_val_score(model_randf, X, y, cv = 3)

print('The accuracy for 100-Tree Random Forests classification method is', accuracy_randf)