import os

import configparser

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
SETTINGS_DIR = os.path.dirname(os.path.realpath(__file__))

config_parser = configparser.ConfigParser()
config_parser.read(f"{SETTINGS_DIR}/.env")

TELEGRAM_BOT_API_KEY = config_parser.get("telegram", "API_KEY")
TELEGRAM_BOT_LINK = config_parser.get("telegram", "BOT_LINK")
