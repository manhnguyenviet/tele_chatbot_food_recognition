import argparse, os
import json
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from openai import OpenAI

from bots import get_answer_from_chat_gpt


def setup_generator(train_path, test_path, batch_size, dimentions):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
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


# def load_image(img_path, dimentions, rescale=1. / 255):
#     img = load_img(img_path, target_size=dimentions)
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x *= rescale # rescale the same as when trained

#     return x


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


def load_existed_model():
    try:
        saved_folder_dir = "saved_models"
        statistic_file_name = "training_statistics.json"
        all_acc_statistics = []
        subfolders = [
            f.name for f in os.scandir(saved_folder_dir) if f.is_dir()
        ]
        if not subfolders:
            return None

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

def predict(model, img_path=""):
    img_height = model.input_shape[1]
    img_width = model.input_shape[2]
    # get class names
    class_names = load_class_names()

    # process the image
    img_array = img_to_array(img_path, img_height, img_width) 

    # making predition
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return {
        "dish_name": class_names[np.argmax(score)],
        "score": score,
        "rounded_score": round(100 * np.max(score), 4),
    }


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