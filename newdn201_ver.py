# -*- coding: utf-8 -*-
"""NewDN201_Ver.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eQCRGZtdONlnaFKImHi9of5vQckmpTbx
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalMaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tqdm import tqdm


test = False
model_name = 'DenseNet201'

def build_model():
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalMaxPool2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def predict_one(model):
    image_batch = X_test[:5]
    classes_batch = y_test[:5]
    predicted_batch = model.predict(image_batch)
    for k in range(0, image_batch.shape[0]):
        image = image_batch[k]
        pred = predicted_batch[k]
        the_pred = np.argmax(pred)
        predicted = class_names[the_pred]
        val_pred = max(pred)
        the_class = np.argmax(classes_batch[k])
        value = class_names[np.argmax(classes_batch[k])]
        plt.figure(k)
        isTrue = (the_pred == the_class)
        plt.title(str(isTrue) + ' - class: ' + value + ' - ' + 'predicted: ' + predicted + '[' + str(val_pred) + ']')
        plt.imshow(image)
        plt.show()


if test:
    epochs = 5
    X_size = 600
else:
    epochs = 32
    X_size = 1

gpu_options = tensorflow.compat.v1.GPUOptions(allow_growth=True)
sess = tensorflow.compat.v1.Session(
    config=tensorflow.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

#Create an index of class names
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

class_names = sorted(os.listdir(r'/content/drive/My Drive/Colab Notebooks/train'))

# Load np.array X from h5 file:
import h5py
with h5py.File('/content/drive/My Drive/Colab Notebooks/mydata2.h5', 'r') as h5:
     X = np.array(h5["X"])
     y = np.array(h5["y"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print(class_names)
batch_size = 32
K.set_learning_phase(1)
model = build_model()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

from sklearn.utils import class_weight

# Balance the number of images for each class  
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

class_weight_dict = dict(enumerate(class_weights))

# with tensorflow.device('/GPU:0'):
model_history = model.fit(x=X_train,
                          y=y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks_list,
                          validation_data=(X_test, y_test),
                          class_weight=class_weight_dict)

model.evaluate(X_test, y_test, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
pred = model.predict(X_test, batch_size=batch_size, max_queue_size=10, workers=1,
                     use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)

with open("/content/drive/My Drive/Colab Notebooks/submissions_Dense201.csv", "w") as fp:
    fp.write("Id,Category\n")
    print("Testing results\n")
    for root, dirs, files in os.walk("./test/testset"):
        for name in tqdm(files[0::X_size]):
            if name.endswith(".jpg"):
                # Load the image
                img = plt.imread(root + os.sep + name)
                
                #Image modification by adding zero arrays for images less than 224x224
                if img.shape[1] >=224 and img.shape[0] >=224:
                  img = cv2.resize(img, (224, 224))
                elif img.shape[1] < 224 and img.shape[0] >=224:
                  advec_dim = (224-img.shape[1])/2
                  if (advec_dim % 2) == 0:
                    advec = np.zeros(advec_dim,img.shape[0])
                    img = np.hstack(advec,img,advec)
                  else:
                    advec_dim1 = advec_dim + 0.5
                    advec1 = np.zeros(advec_dim1,img.shape[0])
                    advec_dim2 = advec_dim - 0.5
                    advec2 = np.zeros(advec_dim2,img.shape[0])
                    img = np.hstack(advec1,img,advec2)
                  
                  img = cv2.resize(img, (224, 224))

                elif img.shape[1] >= 224 and img.shape[0] < 224:
                  advec_dim = (224-img.shape[0])/2
                  if (advec_dim % 2) == 0:
                    advec = np.zeros(img.shape[1],advec_dim)
                    img = np.vstack(advec,img,advec)
                  else:
                    advec_dim1 = advec_dim + 0.5
                    advec1 = np.zeros(img.shape[1],advec_dim1)
                    advec_dim2 = advec_dim - 0.5
                    advec2 = np.zeros(img.shape[1],advec_dim2)
                    img = np.vstack(advec1,img,advec2)
                  
                  img = cv2.resize(img, (224, 224))

                else:
                  advec_dimh = (224-img.shape[1])/2
                  if (advec_dimh % 2) == 0:
                    advech = np.zeros(advec_dimh,img.shape[0])
                    img = np.hstack(advech,img,advech)
                  else:
                    advec_dimh1 = advec_dimh + 0.5
                    advech1 = np.zeros(advec_dimh1,img.shape[0])
                    advec_dimh2 = advec_dimh - 0.5
                    advech2 = np.zeros(advec_dimh2,img.shape[0])
                    img = np.hstack(advech1,img,advech2)

                  advec_dimv = (224-img.shape[0])/2
                  if (advec_dimv % 2) == 0:
                    advecv = np.zeros(img.shape[1],advec_dimv)
                    img = np.vstack(advecv,img,advecv)
                  else:
                    advec_dimv1 = advec_dimv + 0.5
                    advecv1 = np.zeros(img.shape[1],advec_dimv1)
                    advec_dimv2 = advec_dimv - 0.5
                    advecv2 = np.zeros(img.shape[1],advec_dimv2)
                    img = np.vstack(advecv1,img,advecv2)
                  
                  img = cv2.resize(img, (224, 224))


                img = img.astype(np.float32)
                # Resize (consider zeropadding)
                img = preprocess_input(img)
                # Push data through the model to get prediction index i
                res = model.predict(img[np.newaxis, ...])
                i = np.argmax(res)
                label = class_names[i]
                ind = name.split('.')[0]
                fp.write("%s,%s\n" % (ind, label))