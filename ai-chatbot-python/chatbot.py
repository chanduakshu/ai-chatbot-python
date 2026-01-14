# AI Chatbot using Python
# Approach: NLP + Cosine Similarity
# Author: Your Name

import json
import random
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

with open("intents.json") as file:
    data = json.load(file)

def chatbot_response(user_input):
    user_words = preprocess(user_input)

    best_match = None
    max_overlap = 0

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            pattern_words = preprocess(pattern)
            overlap = len(set(user_words) & set(pattern_words))

            if overlap > max_overlap:
                max_overlap = overlap
                best_match = intent

    if best_match:
        return random.choice(best_match["responses"])
    return "Sorry, I didn't understand that."

print("🤖 Chatbot is running! (type 'quit' to exit)")
while True:
    user = input("You: ")
    if user.lower() == "quit":
        print("Bot: Goodbye 👋")
        break
    print("Bot:", chatbot_response(user))

