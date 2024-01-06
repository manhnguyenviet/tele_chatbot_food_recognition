# Telegram Chatbot Food Recogintion

A chatbot for recognition food using Tensorflow, Keras and python-telegram-bot

## Installation

Clone the repository:

```python
git clone https://github.com/manhnguyenviet/tele_chatbot_food_recognition.git chatbot_food_recognition
cd chatbot_food_recognition
```

or download the code and cd to the extracted folder

Setup virtual envronment:

```python
pyenv install 3.10.8
pyenv virtualenv 3.10.8 chatbot_food_recognition
pyenv local chatbot_food_recognition
```

Install dependencies

```python
pip install -r requirements.txt
```

Copy and fill out environment variables:

```python
cp settings/.env.tpl settings/.env
```

NOTE: `API_KEY` and `BOT_LINK` MUST be filled, else will cause error

Create a folder named “food_data” in the root project folder.

Download the Food101 from https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/, extract to `food_data` folder you’ve just created.

### Train the model

Train the model with default training set:

```python
python model.py
```

Train the model with your specific training dataset:

```python
python model.py --train_dir "your training dataset directory"
```

Train the model with your specific number of epochs (For example if you want to train for 10 epochs):

```python
python model.py -e 10
```

### Run the bot

You can simply run the telegram bot using:

```python
python main.py
```

Navigate to the telegram window, enter the chat box with your bot, and send them a pic of a dish, bot will response you the name of the dish
