import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tqdm import tqdm

test = True
model_name = 'InceptionResNetV2'


def build_model():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def rescale_image(image):
    new_size = 224
    # rescale image, the smallest side should always be equal to 224 (both for upscaling and downscaling)
    width = int(image.shape[1])
    height = int(image.shape[0])

    ratio = width / height
    if width < height:
        width = new_size
        height = round(width / ratio)
    else:
        height = new_size
        width = round(height * ratio)
    dim = (width, height)
    image = cv2.resize(image, dim)
    return image


def crop_image_center(image):
    width = image.shape[1]
    height = image.shape[0]
    new_size = 224

    left = int(np.ceil((width - new_size) / 2))
    right = width - int(np.floor((width - new_size) / 2))

    top = int(np.ceil((height - new_size) / 2))
    bottom = height - int(np.floor((height - new_size) / 2))

    if len(image.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


def crop_divide_image(image):
    # crop into 2 pieces depending on the align (horizontal or vertical)
    width = image.shape[1]
    height = image.shape[0]
    if width > height:
        r = width/height
    else:
        r = height/width
    if r <= 1.2:
        img_0 = crop_image_center(image)
        return [img_0]
    new_size = 224
    images = []
    if width == new_size:
        ratio = np.ceil(height / width)
        # moving from up to down
        if ratio == 2:
            left = 0
            right = new_size
            top = 0
            bottom = new_size
            img_1 = image[top:bottom, left:right, ...]
            left = 0
            right = new_size
            top = height - new_size
            bottom = height
            img_2 = image[top:bottom, left:right, ...]
            images = [img_1, img_2]
        elif ratio == 3:
            left = 0
            right = new_size
            top = 0
            bottom = new_size
            img_1 = image[top:bottom, left:right, ...]
            left = 0
            right = new_size
            top = height - new_size
            bottom = height
            img_2 = image[top:bottom, left:right, ...]
            left = 0
            right = new_size
            top = int(np.ceil((height - new_size) / 2))
            bottom = height - int(np.floor((height - new_size) / 2))
            img_3 = image[top:bottom, left:right, ...]
            images = [img_1, img_2, img_3]
        elif ratio >= 4:
            left = 0
            right = new_size
            top = 0
            bottom = new_size
            img_1 = image[top:bottom, left:right, ...]
            left = 0
            right = new_size
            top = height - new_size
            bottom = height
            img_2 = image[top:bottom, left:right, ...]

            left = 0
            right = new_size
            top = int(round((height / 4 - new_size / 2)))
            bottom = int(round((height / 4 + new_size / 2)))
            img_3 = image[top:bottom, left:right, ...]
            left = 0
            right = new_size
            top = int(round((3 * height / 4 - new_size / 2)))
            bottom = int(round((3 * height / 4 + new_size / 2)))
            img_4 = image[top:bottom, left:right, ...]
            images = [img_1, img_2, img_3, img_4]
    else:
        ratio = np.ceil(width / height)
        # moving from left to the right
        if ratio == 2:
            top = 0
            bottom = new_size
            left = 0
            right = new_size
            img_1 = image[top:bottom, left:right, ...]
            top = 0
            bottom = new_size
            left = width - new_size
            right = width
            img_2 = image[top:bottom, left:right, ...]
            images = [img_1, img_2]
        elif ratio == 3:
            top = 0
            bottom = new_size
            left = 0
            right = new_size
            img_1 = image[top:bottom, left:right, ...]
            top = 0
            bottom = new_size
            left = width - new_size
            right = width
            img_2 = image[top:bottom, left:right, ...]
            top = 0
            bottom = new_size
            left = int(np.ceil((width - new_size) / 2))
            right = width - int(np.floor((width - new_size) / 2))
            img_3 = image[top:bottom, left:right, ...]
            images = [img_1, img_2, img_3]
        elif ratio >= 4:
            top = 0
            bottom = new_size
            left = 0
            right = new_size
            img_1 = image[top:bottom, left:right, ...]
            top = 0
            bottom = new_size
            left = width - new_size
            right = width
            img_2 = image[top:bottom, left:right, ...]

            top = 0
            bottom = new_size
            left = int(round((width - 2 * new_size) / 4))
            right = int(round((width + 2 * new_size) / 4))
            img_3 = image[top:bottom, left:right, ...]
            top = 0
            bottom = new_size
            left = int(round((3 * width - 2 * new_size) / 4))
            right = int(round((3 * width + 2 * new_size) / 4))
            img_4 = image[top:bottom, left:right, ...]
            images = [img_1, img_2, img_3, img_4]

    return images


def flip_image(image):
    image = np.fliplr(image)
    return image


def add_noise(image, noise_typ):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


def diversify_image(image):
    # creates a horizontally flipped, noised and cropped copies of the image
    orig_images = crop_divide_image(image)
    changed_images = crop_divide_image(add_noise(flip_image(image), 'gauss'))
    orig_images.extend(changed_images)
    return orig_images


def sort_func(image):
    width = image.shape[1]
    height = image.shape[0]
    if height > width:
        return height / width
    else:
        return width / height


if test:
    epochs = 5
    X_size = 600
else:
    epochs = 32
    X_size = 1

gpu_options = tensorflow.compat.v1.GPUOptions(allow_growth=True)
sess = tensorflow.compat.v1.Session(
    config=tensorflow.compat.v1.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

class_names = sorted(os.listdir("./train/train"))
# Get all images
X = []
y = []
location = "./train/train"
upsampling_large_dirs = ['Ambulance', 'Cart', 'Caterpillar', 'Limousine', 'Segway', 'Snowmobile', 'Tank']

upsampling_dirs = ['Helicopter', 'Taxi', 'Van']

keeping_all_dirs = ['Bus', 'Bicycle', 'Motorcycle', 'Truck']

downsampling_dirs = ['Boat', 'Car']

for up_dir in upsampling_dirs:
    loc = "./train/train/" + up_dir
    for root, dirs, files in os.walk(loc):
        print(root.split(os.sep)[-1])
        for name in tqdm(files[0::X_size]):
            if name.endswith(".jpg"):
                img = plt.imread(root + os.sep + name)
                img = rescale_image(img)
                img = img.astype(np.float32)
                img = preprocess_input(img)
                images = crop_divide_image(img)
                for image in images:
                    wi = image.shape[1]
                    he = image.shape[0]
                    if (wi != 224) or (he != 224):
                        print('Wrong shape' + up_dir)

                    X.append(image)
                    y.append(class_names.index(up_dir))

for up_large_dir in upsampling_large_dirs:
    loc = "./train/train/" + up_large_dir
    for root, dirs, files in os.walk(loc):
        print(root.split(os.sep)[-1])
        for name in tqdm(files[0::X_size]):
            if name.endswith(".jpg"):
                img = plt.imread(root + os.sep + name)
                img = rescale_image(img)
                img = img.astype(np.float32)
                img = preprocess_input(img)
                images = diversify_image(img)
                for image in images:
                    wi = image.shape[1]
                    he = image.shape[0]
                    if (wi != 224) or (he != 224):
                        print('Wrong shape' + up_large_dir)
                    X.append(image)
                    y.append(class_names.index(up_large_dir))

for keep_dir in keeping_all_dirs:
    loc = "./train/train/" + keep_dir
    for root, dirs, files in os.walk(loc):
        print(root.split(os.sep)[-1])
        for name in tqdm(files[0::X_size]):
            if name.endswith(".jpg"):
                img = plt.imread(root + os.sep + name)
                img = rescale_image(img)
                img = img.astype(np.float32)
                img = preprocess_input(img)
                image = crop_image_center(img)
                wi = image.shape[1]
                he = image.shape[0]
                if (wi != 224) or (he != 224):
                    print('Wrong shape' + keep_dir)
                X.append(image)
                y.append(class_names.index(keep_dir))

for down_dir in downsampling_dirs:
    loc = "./train/train/" + down_dir
    for root, dirs, files in os.walk(loc):
        print(root.split(os.sep)[-1])
        dicts = {}
        images = []
        for name in tqdm(files[0::X_size]):
            if name.endswith(".jpg"):
                img = plt.imread(root + os.sep + name)
                images.append(img)
        sorted_images = sorted(images, key=sort_func)[:1800]
        for img in tqdm(images):
            img = rescale_image(img)
            img = img.astype(np.float32)
            img = preprocess_input(img)
            image = crop_image_center(img)
            wi = image.shape[1]
            he = image.shape[0]
            if (wi != 224) or (he != 224):
                print('Wrong shape' + down_dir)
            X.append(image)
            y.append(class_names.index(down_dir))

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print(class_names)
batch_size = 32
K.set_learning_phase(1)
model = build_model()
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]
# with tensorflow.device('/GPU:0'):
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

with open("submissions" + model_name + ".csv", "w") as fp:
    fp.write("Id,Category\n")
    print("Testing results\n")
    for root, dirs, files in os.walk("./test/testset"):
        for name in tqdm(files[0::X_size]):
            if name.endswith(".jpg"):
                # Load the image
                img = plt.imread(root + os.sep + name)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32)
                # Resize (consider zeropadding)
                img = preprocess_input(img)
                images = crop_divide_image(img)
                results = []
                for image in images:
                    res = model.predict(img[np.newaxis, ...])
                    results.append(res)
                best_score = 0
                best_indx = 0
                for res in results:
                    best = np.amax(res)
                    best_i = np.argmax(res)
                    if best > best_score:
                        best_score = best
                        best_indx = best_i
                label = class_names[best_indx]
                ind = name.split('.')[0]
                fp.write("%s,%s\n" % (ind, label))
