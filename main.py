import logging, os, json
import numpy as np
import cv2
import tensorflow as tf

from io import BytesIO
from telegram import Update
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Updater,
)
from keras.models import load_model

from settings import TELEGRAM_BOT_API_KEY, TELEGRAM_BOT_LINK
from food_recognition import recognize_food


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


use_trained_model = False


def load_existed_model():
    saved_folder_dir = "saved_models"
    statistic_file_name = "training_statistics.json"
    all_acc_statistics = []
    subfolders = [f.name for f in os.scandir(saved_folder_dir) if f.is_dir()]
    if not subfolders:
        return None

    for folder in subfolders:
        version = f"{saved_folder_dir}/{folder}"
        with open(f"{version}/{statistic_file_name}", "r") as f:
            statistics = f.read()
            json_statistics = json.loads(statistics)
            acc_list = json_statistics["acc"]
            acc_list.sort()
            all_acc_statistics.append({"version": version, "acc": acc_list[-1]})

    sorted_all_acc_statistics = sorted(all_acc_statistics, key=lambda d: d["acc"])
    version = sorted_all_acc_statistics[-1]["version"]

    model = load_model(version)
    return model


def load_class_names():
    with open("class_names.json", "r") as f:
        data = json.loads(f.read())
        return data["names"]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
    /start: Start to chat with the bot
    /echo: Nothing, just for fun
    /help: show help commands
    /train: Train the bot
    
    """
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="This is Echo"
    )


async def google_cloud_vision_handle_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    biyte_io_file = BytesIO(await file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(biyte_io_file.read()), dtype=np.uint8)

    # decode to get the image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # because cv2 using BGR color space,
    # we need to convert the image from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # recognize the food inside the image
    food_name, percent_match = recognize_food(img)
    reply_text = f"The food is {food_name} with {percent_match}% confidence"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        """
        should save the image in path, then use it to process the prediction
        """
        img_path = "tele_img.jpg"
        await file.download_to_drive(img_path)

        model = load_existed_model()
        img_height = 180
        img_width = 180
        img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = load_class_names()

        text = (
            f"This image is most likely belongs to {class_names[np.argmax(score)]} with"
            f" {round(100 * np.max(score), 4)}% confidence"
        )
        os.remove(img_path)
    except Exception as ex:
        text = f"Error loading and processing image: {ex}"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_API_KEY).build()

    start_handler = CommandHandler("start", start)
    echo_handler = CommandHandler("echo", echo)
    help_handler = CommandHandler("help", help)
    # help_handler = CommandHandler("help", help)
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)

    # add the handlers to the bot
    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(help_handler)
    application.add_handler(photo_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
