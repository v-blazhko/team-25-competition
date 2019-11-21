import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications import densenet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model


def build_model():
    base_model = densenet.DenseNet121(input_shape=(224, 224, 3),
                                      weights='./full-keras-pretrained-no-top'
                                              '/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                      include_top=False,
                                      pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    created_model = Model(inputs=base_model.input, outputs=predictions)

    return created_model


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


class_names = sorted(os.listdir("./train/train"))
# Get all images
X = []
y = []
location = "./train/train"
with tensorflow.device('/GPU:0'):
    for root, dirs, files in os.walk(location):
        for name in files[0::200]:
            if name.endswith(".jpg"):
                # Load the image
                img = plt.imread(root + os.sep + name)
                # Resize (consider zeropadding)
                img = cv2.resize(img, (224, 224))
                # Convert data to float and extacrt mean (this is how the network was trained)
                img = img.astype(np.float32)
                img -= 128
                X.append(img)
                # Extract class name from the directory name
                label = root.split(os.sep)[-1]
                y.append(class_names.index(label))

# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# image = Image.open("./train/train/Ambulance/000040_09.jpg")
# imgplot = plt.imshow(image)
# plt.show()
print(class_names)
epochs = 10
batch_size = 32
K.set_learning_phase(1)
model = build_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]
model_history = model.fit(x=X_train,
                          y=y_train,
                          epochs=epochs,
                          validation_data=(X_test, y_test),
                          validation_steps=len(y_test) // batch_size,
                          callbacks=callbacks_list)

plt.figure(0)
plt.plot(model_history.history['acc'], 'r')
plt.plot(model_history.history['val_acc'], 'g')
plt.xticks(np.arange(0, 20, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train', 'validation'])

plt.figure(1)
plt.plot(model_history.history['loss'], 'r')
plt.plot(model_history.history['val_loss'], 'g')
plt.xticks(np.arange(0, 20, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.figure(2)
plt.plot(model_history.history['mean_squared_error'], 'r')
plt.plot(model_history.history['val_mean_squared_error'], 'g')
plt.xticks(np.arange(0, 20, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("MSE")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train', 'validation'])

plt.show()

model.evaluate(X_test, y_test, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
pred = model.predict(X_test, y_test, steps=None, max_queue_size=10, workers=1,
                     use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)

predict_one(model)
