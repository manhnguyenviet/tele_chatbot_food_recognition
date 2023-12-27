import logging, os
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
)

from settings import TELEGRAM_BOT_API_KEY
from food_recognition import recognize_food
from utils import (
    load_existed_model,
    predict,
    search_recipe_instruction,
    format_dish_name,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
    /start: Start to chat with the bot
    /echo: Nothing, just for fun
    /search_instruction: Search for making instruction of given dish name
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
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=reply_text
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # download the image
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        """
        should save the image in path, then use it to process the prediction
        """
        img_path = "tele_img.jpg"
        await file.download_to_drive(img_path)

        model = load_existed_model()
        result = predict(model, img_path)

        dish_name = result["dish_name"]
        rounded_score = result["rounded_score"]

        text = (
            f"This image is most likely belongs to {dish_name} with"
            f" {rounded_score}% confidence"
        )

        # remove temp img
        os.remove(img_path)

        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=text
        )

        # search for dish instruction
        temp_text = (
            "Searching for the instruction of making"
            f" {format_dish_name(dish_name)}..."
        )
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=temp_text
        )

        instruction = search_recipe_instruction(dish_name)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=instruction,
            parse_mode="markdown",
        )
    except Exception as ex:
        text = f"Error loading and processing image: {ex}"
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=text
        )


async def search_instruction(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    dish_name = update.message.text.replace("/search_instruction", "").strip()
    if not dish_name:
        text = "No dish detected. Please provide a dish to search."
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=text
        )    
        return

    search_noti = (
        "Searching for the instruction of making"
        f" {format_dish_name(dish_name)}..."
    )
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=search_noti
    )

    instruction = search_recipe_instruction(dish_name)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=instruction,
        parse_mode="markdown",
    )


def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_API_KEY).build()

    start_handler = CommandHandler("start", start)
    echo_handler = CommandHandler("echo", echo)
    search_instruction_handler = CommandHandler(
        "search_instruction", search_instruction
    )
    help_handler = CommandHandler("help", help)
    # help_handler = CommandHandler("help", help)
    photo_handler = MessageHandler(filters.PHOTO, handle_photo)

    # add the handlers to the bot
    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(search_instruction_handler)
    application.add_handler(help_handler)
    application.add_handler(photo_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
