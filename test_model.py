import json
import os

from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# from utils import *


def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0002
    elif epoch < 15:
        return 0.00002
    else:
        return 0.0000005


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


shape = (224, 224, 3)
X_train, X_test = setup_generator(
    "food_data/train_set", "food_data/train_set", 32, shape[:2]
)

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(299, 299, 3)),
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# # x = Flatten()(x)
x = Dense(4096)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(X_train.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

checkpointer = ModelCheckpoint(
    filepath="first.3.{epoch:02d}-{val_loss:.2f}.hdf5",
    verbose=1,
    save_best_only=True,
)
csv_logger = CSVLogger("first.3.log")
model.fit_generator(
    X_train,
    validation_data=X_test,
    validation_steps=X_train.samples // 32,
    steps_per_epoch=X_train.samples // 32,
    epochs=10,
    verbose=1,
    callbacks=[csv_logger, checkpointer],
)

for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

opt = SGD(lr=0.0001, momentum=0.9)
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
saved_subfolders = [f.name for f in os.scandir(saved_folder_dir) if f.is_dir()]
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
