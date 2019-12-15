#!/usr/bin/env python
# coding: utf-8

# Tame the pretrained model and add the following layers.
# Take the global average pooling, add a few dence layers
# Last activations softmax, not relu
# Augment, with imgaug
# keras model fit generator, a good place for augmentation

testrun = True

import os
# Load the data
# "C:\Work\sgndataset\train\""

# Extract class_names from directories inside the main one
# TODO
# "C:\\Work\\sgndataset\\train"
train_location = "C:\\Users\\enehl\\Desktop\\train"
class_names = sorted(os.listdir(train_location))

from tensorflow.keras.applications.mobilenet import MobileNet 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
# Base model, without the top layer, just the convolutional layers

model_mobilenet = MobileNet(input_shape = (224,224,3), include_top = False)
model_mobilenetV2 = MobileNetV2(input_shape = (224,224,3), include_top = False)
model_InceptionV3 = InceptionV3(input_shape = (224,224,3), include_top = False)

models = [model_mobilenet, model_mobilenetV2, model_InceptionV3]

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

i = 0
for model in models:
    in_tensor = model.inputs[0] 
    out_tensor = model.outputs[0]
    
    # Add layers
    out_tensor = tensorflow.keras.layers.GlobalAveragePooling2D()(out_tensor)
    out_tensor = tensorflow.keras.layers.Dense(100, activation='relu')(out_tensor)
    out_tensor = tensorflow.keras.layers.Dense(17, activation='softmax')(out_tensor)

    # Full model defined by endpoints
    models[i] = tensorflow.keras.models.Model(inputs=[in_tensor], outputs=[out_tensor])
    i += 1
print("Models ready")

for model in models:
    
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print("Compiling ready")

import matplotlib.pyplot as plt
import cv2
import numpy as np


print("Starting to prepare X and y...")
X = []
y = []
# "C:\\Work\\sgndataset\\train"
location = train_location

if testrun:
    test_num = 0
    max_test = 3

for root, dirs, files in os.walk(location):
    if testrun:
        test_num = 0
    for name in files:
        if name.endswith(".jpg"):

            # Load the image
            img = plt.imread(root + os.sep + name)

            # Resize (consider zeropadding)
            img = cv2.resize(img, (224, 224))

            # Convert data to float and extract mean (this is how the network was trained)
            img = img.astype(np.float32)
            img -= 128

            # Add the feature vector to our list
            X.append(img)

            # Extract class name from the directory name
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))
            print(label)

            if testrun:
                test_num += 1
                if test_num > max_test:
                    break
            
# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

print("X and y ready")

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

y_categorical = to_categorical(y,17)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=0)
      
print("Test-train split done!")

all_results = []
from sklearn.metrics import accuracy_score
with open(".\\test2.txt", "w") as fp:

    i = 1
    for model in models:
        model.fit(X_train, y_train)
        result = model.evaluate(X_test, y_test)
        # y_pred = model.predict(X_test)
        # result.append(sklearn.metrics.accuracy_score(y_test, y_pred))
        fp.write("%d,%d\n" %(result[0],result[1]))
        all_results.append(result[1])
        print(i, "/3 trained with a result of ", result)
        i += 1
fp.close()
print("Done!")
print(all_results)

all_results = np.array(all_results)

# Just an idea?
best = np.argmax(all_results)
winner_model = models[best]

winner_model.fit(X, y_categorical)

# Number to be printed to see how the prediction advances
num = 1

# TODO
# "C:\\Work\\sgndataset\\testset"
test_location = "C:\\Users\\enehl\\Desktop\\testset"
# Create file called submissions.csv
with open(".\\sub_test2.csv", "w") as subfile:
    subfile.write("Id,Category\n")

    # The for loop will do the following.
    # Loop through the images in testset. For each image, modify it to the right format using
    # the pretrainde network called nn_model. The result will be a vector, similar to the ones in
    # training data X. Predic the class using the model we just trained. Finally, get the
    # label of the prediction and save image-id and predicted label to the submissions file.

    if testrun:
        test_num = 0

    for root, dirs, files in os.walk(test_location):
        if testrun:
            test_num = 0
        for name in files:
            if name.endswith(".jpg"):
                # Load the image
                img = plt.imread(root + os.sep + name)

                # Resize (consider zero padding)
                img = cv2.resize(img, (224, 224))

                # Convert data to float and extract mean (this is how the network was trained)
                img = img.astype(np.float32)
                img -= 128

                pred = winner_model.predict(img[np.newaxis, ...])

                ind_of_pred = np.argmax(pred)

                # Class id to name
                label = class_names[ind_of_pred]
                ind = name.split('.')[0]

                subfile.write("%d,%s\n" % (int(ind), label))
                num += 1
                print(num, "out of 7958")

                if testrun:
                    test_num += 1
                    if test_num > max_test:
                        break
subfile.close()
print("Files ready!")
