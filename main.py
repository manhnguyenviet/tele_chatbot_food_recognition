import logging
import numpy as np
import cv2

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

from settings import TELEGRAM_BOT_API_KEY, TELEGRAM_BOT_LINK
from food_recognition import recognize_food


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


use_trained_model = False


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


async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Training...")
    # put logic to train the bot here
    use_trained_model = True
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm successfully trained!"
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="This is Echo"
    )


async def google_cloud_vision_handle_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    # TODO: should change logic of this function to using trained model,
    # currently using GG Cloud Vision for recognizing photo

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


def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_API_KEY).build()

    start_handler = CommandHandler("start", start)
    echo_handler = CommandHandler("echo", echo)
    help_handler = CommandHandler("help", help)
    train_handler = CommandHandler("train", train)
    # help_handler = CommandHandler("help", help)
    if not use_trained_model:
        photo_handler = MessageHandler(filters.PHOTO, google_cloud_vision_handle_photo)
    else:
        pass

    # add the handlers to the bot
    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(help_handler)
    application.add_handler(train_handler)
    application.add_handler(photo_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
