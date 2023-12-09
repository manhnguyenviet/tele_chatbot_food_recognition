import tensorflow as tf
import tensorflow_datasets as tfds

# load food class names
with open("food_class.txt", "r") as file:
    food_class_names = file.read()


# load food dataset
# ref: https://www.tensorflow.org/datasets/catalog/food101?hl=vi
food_101 = tfds.image_classification.Food101


# load dataset
# mnist = tf.keras.datasets.mnist
# tf.keras.datasets.imdb
ds = tfds.load('food101', split='train', shuffle_files=True)
(x_train, y_train), (x_test, y_test) = ds.load_data()
breakpoint()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])