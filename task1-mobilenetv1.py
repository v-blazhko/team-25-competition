import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tqdm import tqdm


def build_model():
    base_model = MobileNet(input_shape=(224, 224, 3),
                           weights='imagenet',
                           include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(
        x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(len(class_names), activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
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

gpu_options = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

class_names = sorted(os.listdir("./train/train"))
# Get all images
X = []
y = []
location = "./train/train"
for root, dirs, files in os.walk(location):
    print(root.split(os.sep)[-1])
    for name in tqdm(files):
        if name.endswith(".jpg"):
            # Load the image
            img = plt.imread(root + os.sep + name)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32)
            # Resize (consider zeropadding)
            img = preprocess_input(img)
            X.append(img)
            # Extract class name from the directory name
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))

# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# image = Image.open("./train/train/Ambulance/000040_09.jpg")
# imgplot = plt.imshow(image)
# plt.show()
print(class_names)
epochs = 25
batch_size = 32
K.set_learning_phase(1)
model = build_model()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]
#with tensorflow.device('/GPU:0'):
model_history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks_list)

model.evaluate(X_test, y_test, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
pred = model.predict(X_test, batch_size=batch_size, max_queue_size=10, workers=1,
                     use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)

with open("submissions.csv", "w") as fp:
    fp.write("Id,Category\n")
    print("Testing results\n")
    for root, dirs, files in os.walk("./test/testset"):
        for name in tqdm(files):
            if name.endswith(".jpg"):
                # Load the image
                img = plt.imread(root + os.sep + name)
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
