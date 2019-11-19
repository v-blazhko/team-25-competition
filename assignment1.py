# %%

import os

# Load the data
# "C:\Work\sgndataset\train\""

# Extract class_names from directories inside the main one
# r"C:\Work\sgndataset\train\"
class_names = sorted(os.listdir("./train"))

# %%

from tensorflow.keras.applications.mobilenet import MobileNet

# Base model, without the top layer, just the convolutional layers
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)

# base_model.summary()

# %%

import tensorflow

in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0]

# Add one player
out_tensor = tensorflow.keras.layers.GlobalAveragePooling2D()(out_tensor)

# Full model defined by enpoints
model = tensorflow.keras.models.Model(inputs=[in_tensor], outputs=[out_tensor])

# Compile the model for execution. Losses and optimizers
# can be anything here, since we don’t train the model.
model.compile(loss="categorical_crossentropy", optimizer='sgd')
# model.summary()

# %%

import matplotlib.pyplot as plt
import cv2
import numpy as np

# Get all images
X = []
y = []
location = "./train"
for root, dirs, files in os.walk(location):
    for name in files:
        if name.endswith(".jpg"):
            # Load the image
            img = plt.imread(root + os.sep + name)

            # Resize (consider zeropadding)
            img = cv2.resize(img, (224, 224))

            # Convert data to float and extacrt mean (this is how the network was trained)
            img = img.astype(np.float32)
            img -= 128

            # Push data through the model
            x = model.predict(img[np.newaxis, ...])[0]

            # Add the feature vector to our list
            X.append(x)

            # Extract class name from the directory name
            label = root.split(os.sep)[-1]
            y.append(class_names.index(label))

            print(label)

# Cast the python lists to a numpy array.
X = np.array(X)
y = np.array(y)

print(X, y)

# %%

# Classify using different models
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

result = []
# Add others?
models = [LinearDiscriminantAnalysis(),
          SVC(kernel='linear'),
          SVC(kernel='rbf'),
          LogisticRegression(),
          RandomForestClassifier(n_estimators=100)]
with tensorflow.device('/GPU:0'):
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result.append(sklearn.metrics.accuracy_score(y_test, y_pred))

best_loc = np.argmax(result)

print("Best classifier: ", type(models[best_loc]).__name__)
print(results)

# %%

# Create submission file
best_model = models[best_loc]
best_model.fit(X, y)
with open("submissions.csv", "w") as fp:
    fp.write("Id,Category\n")

    for root, dirs, files in os.walk("C:\\Work\\sgndataset\\testset"):
        for name in files:
            if name.endswith(".jpg"):
                # Load the image
                img = plt.imread(root + os.sep + name)

                # Resize (consider zero padding)
                img = cv2.resize(img, (224, 224))

                # Convert data to float and extacrt mean (this is how the network was trained)
                img = img.astype(np.float32)
                img -= 128

                # Push data through the model to get prediction index i
                i = best_model.predict(img[np.newaxis, ...])[0]

                # Class id to name
                label = class_names[i]
                ind = name.split('.')[0]
                fp.write("%d,%s\n" % (ind, label))

