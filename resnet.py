import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# Initial learning rate, can be extended
intial_learning_rate = 1e-4

# number of epochs to train
epochs = 45  # @param {type:"slider", min:20, max:100, step:5}

# The size of the batch to train on
batch_size = 4  # @param {type:"slider", min:4, max:100, step:4}

# Directory
dir: str = os.path.abspath(os.getcwd())

categories = ["with_mask", "without_mask"]

# Data Variables
data = []
labels = []

for category in categories:
    path = os.path.join(dir, category)

    # Store as list

    for img in os.listdir(path):
        path_img = os.path.join(path, img)
        print("Loading %s" % path_img)
        # see from tensorflow.keras.preprocessing.image import load_img
        image = load_img(path_img, target_size=(224, 224))
        # from tensorflow.keras.preprocessing.image import img_to_array
        image = img_to_array(image)
        # from tensorflow.keras.applications.* import preprocess_input
        image = preprocess_input(image)
        # Append the image at index (x)
        data.append(image)
        # Append the label for each image... at an index (x)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels)

# Setting the data (images) to pixels of float32
data = np.array(data, dtype="float32")

# Converting labels to array
labels = np.array(labels)

# Dataset split w.r.t ratio.
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)
# Data Augmentation
# Extra sampling of images.
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


dense_Model = DenseNet201(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# construct the head of the model that will be placed on top of the
# the base model
head_dense_Model = dense_Model.output
# 224 / 32 = 7
# Window create -> Shifting
head_dense_Model = MaxPooling2D(pool_size=(7, 7))(head_dense_Model)

# Feature Map flatted to 1D Formation
head_dense_Model = Flatten(name="flatten")(head_dense_Model)

# Action Function = RELU.. (Possibility)... Image Thresholding (Function making)
# Rectified Linear Unit
head_dense_Model = Dense(128, activation="relu")(head_dense_Model)

# Dropout -> 1/2 is expunged [It is a bias]
head_dense_Model = Dropout(0.5)(head_dense_Model)

# Max=[0, MAX] (Binary Classification)
head_dense_Model = Dense(2, activation="softmax")(head_dense_Model)


model = Model(inputs=dense_Model.input, outputs=head_dense_Model)

# Layers
for layer in dense_Model.layers:
    # Freezing so that a pre-set model at transfer learning ....
    layer.trainable = False

# TODO: Explaination needed?
opt = Adam(lr=intial_learning_rate, decay=(intial_learning_rate / epochs))

model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# TODO: Explaination needed?
Head = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs)

# TODO: Explaination needed?
predIdxs = model.predict(testX, batch_size=batch_size)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# TODO: Explaination needed?
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
# TODO: Explaination needed?
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# Save the model
# evaluate(), save()
models_path = os.path.join(os.path.abspath("models"), "mask_detector_resnet.model")
model.save(models_path, save_format="h5")

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Head.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), Head.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), Head.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), Head.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
