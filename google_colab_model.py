import json
import logging
import os

import tensorflow as tf
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from keras.models import Sequential
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


train_dir = "/content/drive/MyDrive/CS331 Chatbot/food-101-tiny/train"
val_dir = "/content/drive/MyDrive/CS331 Chatbot/food-101-tiny/train"

# img information and format for training
batch_size = 32
img_height = 256
img_width = 256


# load data from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    validation_split=0.9,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# get all class names (names of all subfolders from traing directory)
class_names = train_ds.class_names
num_classes = len(class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = Rescaling(1.0 / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))


# declare model
data_augmentation = Sequential(
    [
        RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ]
)
model = Sequential(
    [
        data_augmentation,
        Rescaling(1.0 / 255),
        Conv2D(16, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, name="outputs"),
    ]
)


# compile model with configs
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# show the model summary statistics
model.summary()


# train
epochs = 30
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# save the model
overwrite = False
saved_folder_dir = "/content/drive/MyDrive/CS331 Chatbot/saved_models"
save_model_prefix = "food_recog_v"
statistic_file_name = "training_statistics.json"

saved_subfolders = [f.name for f in os.scandir(saved_folder_dir) if f.is_dir()]
saved_subfolders.sort()
if not saved_subfolders:
    version = 1
else:
    version = int(saved_subfolders[-1].strip(save_model_prefix)) + 1
save_dir = f"{saved_folder_dir}/{save_model_prefix}{version}"

model.save(save_dir, overwrite=overwrite)
print("Model saved successfully")


# Visualize training results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

# save the loss and accuracy
training_statistics = {
    "acc": acc,
    "val_acc": val_acc,
    "loss": loss,
    "val_loss": val_loss,
}
with open(f"{save_dir}/training_statistics.json", "w") as file:
    json.dump(training_statistics, file)

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig(f"{save_dir}/train_result.png")  # save the training result image
plt.show()  # show the training and validation results as a chart figure, uncomment if needed
