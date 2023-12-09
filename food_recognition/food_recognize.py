"""
Recognize food: fruit, vegetable
"""

import io
import os

import cv2
from google.cloud import vision_v1p3beta1 as vision

# Setup google authen client key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "food_recognition/client_key.json"

# Source path content all images
SOURCE_PATH = "food_data/train_set"

FOOD_TYPE = "Fruit"  # 'Vegetable'
FOOD_NAMES = [
    line.rstrip("\n").lower()
    for line in open("food_recognition/dict/" + FOOD_TYPE + ".dict")
]


def recognize_food(img):
    # Get image size
    height, width = img.shape[:2]

    # Scale image
    img = cv2.resize(img, (800, int((height * 800) / width)))

    # Save the image to temp file
    cv2.imwrite(SOURCE_PATH + "output.jpg", img)

    # Create new img path for google vision
    img_path = SOURCE_PATH + "output.jpg"

    # Create google vision client
    client = vision.ImageAnnotatorClient()

    # Read image file
    with io.open(img_path, "rb") as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    # Recognize text
    response = client.label_detection(image=image)
    labels = response.label_annotations

    food_name = labels[0].description.lower()
    percent_match = round(labels[0].score, 2)
    return (food_name, percent_match)
