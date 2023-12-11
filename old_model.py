import os, sys, argparse
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    required=False,
    help="Number of epochs to train",
)
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    help="Path to the training dataset",
)
parser.add_argument(
    "-nt",
    "--not_train",
    action="store_true",
    help="Use this parameter to not train the model",
)
parser.add_argument(
    "--evaluate",
    action="store_true",
    help="Use this parameter to not train the model",
)

default_data_dir = "food_data/train_set"
default_epochs = 10
args = parser.parse_args()
training_data_dir = args.dir or default_data_dir
training_epochs = args.epochs or default_epochs
not_train = args.not_train
evaluate = args.evaluate

existed_model_file_dir = "models/food_recognition.h5"
img_file_type = ["jpg"]
log_dir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
optimizer="adam" 
metrics=["accuracy"]


def process_and_show_chart(fit_result, content_type=""):
    """
    Process the trained result from the training process and show the chart
    about accuracy and loss
    """
    if not content_type:
        return

    fig = plt.figure()
    plt.plot(fit_result.history[content_type], color="teal", label=content_type)
    plt.plot(
        fit_result.history[f"val_{content_type}"],
        color="orange",
        label=f"val_{content_type}",
    )
    fig.suptitle(content_type.capitalize(), fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


def evaluate_result(model, test_data):
    """
    Using this function to evaluate the result from training the model.
    Should run every time the model is trained
    Especially useful in case user wants to make some charts about loss and accuracy
    when training the model
    """
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test_data.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(pre.result(), re.result(), acc.result())
    return pre, re, acc


def predict(model, img_dir):
    img = cv2.imread(img_dir)
    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))


# create and config model
try:
    model = load_model(existed_model_file_dir)
except Exception:
    # handle case first time run and there's no existed model
    model = Sequential(
        [
            Conv2D(16, (3, 3), 1, activation="relu", input_shape=(256, 256, 3)),
            MaxPooling2D(),
            Conv2D(32, (3, 3), 1, activation="relu"),
            MaxPooling2D(),
            Conv2D(16, (3, 3), 1, activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(256, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=metrics)

# load data
data = tf.keras.utils.image_dataset_from_directory(training_data_dir)
# process and scale data
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
# scale data
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

# split data
train_size = int(len(data) * 0.7) # 70% of the images is used for training
val_size = int(len(data) * 0.2) # 20% of 30% remain images is used for validating
test_size = int(len(data) * 0.1) # 10% remain images is used for testing

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


if not not_train:
    fit_result = model.fit(
        train, epochs=training_epochs, validation_data=val, callbacks=[tensorboard_callback]
    )

    # save or update the model
    model.save(os.path.join('models','food_recognition.h5'))

    if evaluate:
        evaluate_result(model, test)
else:
    fit_result = None
