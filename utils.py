import collections
import json
import os

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from bots import get_answer_from_chat_gpt


def setup_generator(train_path, test_path, batch_size, dimentions):
    train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.125,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.125,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=1.0 / 255,
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=dimentions,
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        test_path,  # this is the target directory
        target_size=dimentions,
        batch_size=batch_size,
    )

    return train_generator, validation_generator


class Rescaling(tf.keras.layers.Layer):
    """Multiply inputs by `scale` and adds `offset`.
    For instance:
    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.
    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]`
    range,
    you would pass `scale=1./127.5, offset=-1`.
    The rescaling is applied both during training and inference.
    Input shape:
    Arbitrary.
    Output shape:
    Same as input.
    Arguments:
    scale: Float, the scale to apply to the inputs.
    offset: Float, the offset to apply to the inputs.
    name: A string, the name of the layer.
    """

    def __init__(self, scale, offset=0.0, name=None, **kwargs):
        self.scale = scale
        self.offset = offset
        super(Rescaling, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        dtype = self._compute_dtype
        scale = tf.cast(self.scale, dtype)
        offset = tf.cast(self.offset, dtype)
        return tf.cast(inputs, dtype) * scale + offset

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "scale": self.scale,
            "offset": self.offset,
        }
        base_config = super(Rescaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_existed_model(version):
    try:
        saved_folder_dir = "saved_models"
        statistic_file_name = "training_statistics.json"
        all_acc_statistics = []
        subfolders = [
            f.name for f in os.scandir(saved_folder_dir) if f.is_dir()
        ]
        if not subfolders:
            return None
        if not version:
            for folder in subfolders:
                version = f"{saved_folder_dir}/{folder}"
                with open(f"{version}/{statistic_file_name}", "r") as f:
                    statistics = f.read()
                    json_statistics = json.loads(statistics)
                    acc_list = json_statistics["acc"]
                    acc_list.sort()
                    all_acc_statistics.append(
                        {"version": version, "acc": acc_list[-1]}
                    )

            sorted_all_acc_statistics = sorted(
                all_acc_statistics, key=lambda d: d["acc"]
            )
            version = sorted_all_acc_statistics[-1]["version"]
        print("================================================")
        print(f"Loaded model {version}")

        model = load_model(version, compile=False)
        model.compile()
        return model
    except Exception as ex:
        print(ex)
        return None


def img_to_array(img_path, img_height, img_width):
    img = tf.keras.utils.load_img(
        img_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array


def search_recipe_instruction(dish_name):
    dish_name = format_dish_name(dish_name)
    question = make_question(dish_name)
    return get_answer_from_chat_gpt(question)


def make_question(dish_name):
    return f"Can you show me an instruction on making {dish_name}?"


def format_dish_name(dish_name):
    return dish_name.replace("_", " ")


def load_class_names():
    with open("class_names.json", "r") as f:
        data = json.loads(f.read())
        return data["names"]


def img_prediction(model, pic_path):
    pic = img.imread(pic_path)
    with open("class_names.json", "r") as f:
        class_names = json.loads(f.read())
    preds = predict_10_crop(model, np.array(pic), 0)[0]
    best_pred = collections.Counter(preds).most_common(1)[0][0]
    return {
        "preds": preds,
        "best_pred": best_pred,
        "label": class_names[str(best_pred)],
    }


def images_with_class_names(class_dir="class_names.txt"):
    class_to_ix = {}
    ix_to_class = {}
    with open(class_dir, "r") as txt:
        classes = [l.strip() for l in txt.readlines()]
        class_to_ix = dict(zip(classes, range(len(classes))))
        ix_to_class = dict(zip(range(len(classes)), classes))
        class_to_ix = {v: k for k, v in ix_to_class.items()}
    sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))
    return {
        "ix_to_class": ix_to_class,
        "class_to_ix": class_to_ix,
        "sorted_class_to_ix": sorted_class_to_ix,
    }


def predict_10_crop(
    model, img, ix, top_n=5, plot=False, preprocess=True, debug=False
):
    flipped_X = np.fliplr(img)
    crops = [
        img[:299, :299, :],  # Upper Left
        img[:299, img.shape[1] - 299 :, :],  # Upper Right
        img[img.shape[0] - 299 :, :299, :],  # Lower Left
        img[img.shape[0] - 299 :, img.shape[1] - 299 :, :],  # Lower Right
        center_crop(img, (299, 299)),
        flipped_X[:299, :299, :],
        flipped_X[:299, flipped_X.shape[1] - 299 :, :],
        flipped_X[flipped_X.shape[0] - 299 :, :299, :],
        flipped_X[flipped_X.shape[0] - 299 :, flipped_X.shape[1] - 299 :, :],
        center_crop(flipped_X, (299, 299)),
    ]
    if preprocess:
        crops = [preprocess_input(x.astype("float32")) for x in crops]

    if plot:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])

    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds = np.argpartition(y_pred, -top_n)[:, -top_n:]
    if debug:
        print("Top-1 Predicted:", preds)
        print("Top-5 Predicted:", top_n_preds)
    return preds, top_n_preds


def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[
        centerw - halfw : centerw + halfw + 1,
        centerh - halfh : centerh + halfh + 1,
        :,
    ]
