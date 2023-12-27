import os, json

from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Activation,
    Flatten,
    AveragePooling2D,
)
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K

from utils import *


def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0002
    elif epoch < 15:
        return 0.00002
    else:
        return 0.0000005


shape = (224, 224, 3)
X_train, X_test = setup_generator(
    "food_data/train", "food_data/train", 32, shape[:2]
)

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3)),
)
x = base_model.output
x = AveragePooling2D()(x)

x = Dropout(0.5)(x)
x = Flatten()(x)
predictions = Dense(
    X_train.num_classes,
    # init="glorot_uniform",
    # W_regularizer=l2(0.0005),
    activation="softmax",
)(x)

model = Model(inputs=base_model.input, outputs=predictions)

opt = SGD(lr=0.1, momentum=0.9)
model.compile(
    optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
)

checkpointer = ModelCheckpoint(
    filepath="model.{epoch:02d}-{val_loss:.2f}.hdf5",
    verbose=1,
    save_best_only=True,
)
csv_logger = CSVLogger("model.log")


lr_scheduler = LearningRateScheduler(schedule)
model.summary()

history = model.fit_generator(
    X_train,
    validation_data=X_test,
    epochs=10,
    steps_per_epoch=X_train.samples // 32,
    validation_steps=X_test.samples // 32,
    callbacks=[lr_scheduler, csv_logger, checkpointer],
)

saved_folder_dir = "saved_models"
save_model_prefix = "food_recog_v"
statistic_file_name = "training_statistics.json"
saved_subfolders = [
    f.name for f in os.scandir(saved_folder_dir) if f.is_dir()
]
saved_subfolders.sort()
if not saved_subfolders:
    version = 1
else:
    version = int(saved_subfolders[-1].strip(save_model_prefix)) + 1
save_dir = f"{saved_folder_dir}/{save_model_prefix}{version}"

model.save(save_dir)
print(f"Model saved successfully to {version}")


# Visualize training results
# should also save the loss and accuracy to json file,
# and use that to select the model to use in main.py
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


# Predict on new data
img_height = 224
img_width = 224
img_path = "img/test_imgs/Apple_pie.jpg"
with open("class_names.json") as f:
    class_names = json.loads(f.read())["names"]
img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f"Class names: {class_names}")
print(f"Score: {score}")
print(
    f"This image most likely belongs to {class_names[np.argmax(score)]}"
    f" with a {round(100 * np.max(score), 4)} percent confidence."
)
