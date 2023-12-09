import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import load_model


default_data_dir = '/Users/manh/Downloads/data/food-101-tiny/train'
image_exts = ['jpeg','jpg', 'bmp', 'png']
log_dir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def create_model():
    model = Sequential(
        [
            Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)),
            MaxPooling2D(),
            Conv2D(32, (3,3), 1, activation='relu'),
            MaxPooling2D(),
            Conv2D(16, (3,3), 1, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    )
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


def train_model(model, data=None):
    if not data:
        data = get_data(default_data_dir)
    
    train = data["train"]
    val = data["val"]
    fit_result = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
    
    return model, fit_result


def load_existed_model():
    # TODO: should implement this feature later
    pass


def get_data(dir):
    # load data from dataset directory
    data = tf.keras.utils.image_dataset_from_directory(dir)

    # process and scale data
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    # scale data
    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()


    # split data
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return {
        "train": train,
        "val": val,
        "test": test
    }


def process_and_show_chart(fit_result, content_type=""):
    if not content_type:
        return
        
    fig = plt.figure()
    plt.plot(fit_result.history[content_type], color='teal', label='loss')
    plt.plot(fit_result.history[f'val_{content_type}'], color='orange', label='val_loss')
    fig.suptitle(content_type.capitalize(), fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


def evaluate(model, test_data):
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
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
