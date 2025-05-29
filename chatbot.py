import io
import random
import string
import warnings
import numpy as np
import time
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Set NLTK data path to a local directory
import os
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data
try:
    nltk.download('all', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Warning: Could not download all NLTK data. Error: {str(e)}")
    for package in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.download(package, download_dir=nltk_data_dir)
        except Exception as e:
            print(f"Warning: Could not download {package}. Error: {str(e)}")

# Initialize Ollama client
client = ollama.Client(host='http://localhost:11434')

# Define base context for the chatbot
raw = """I am a helpful chatbot assistant. I can help answer questions and engage in conversation.
I aim to be friendly and informative in my responses.
I can provide information on various topics and help with different tasks.
""".lower()

# Tokenization
sent_tokens = raw.split('.')  # Simple sentence tokenization
word_tokens = raw.split()     # Simple word tokenization

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching for Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Global variable for typing delay
typing_delay = 0.01  # Default delay in seconds per character

# Function to simulate typing effect
def simulate_typing(text):
    """Prints the text character by character with a delay to simulate typing."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(typing_delay)
    print()  # New line after the response

# Function to set typing speed
def set_typing_speed():
    global typing_delay
    try:
        new_delay = float(input("Enter new typing delay in seconds (e.g., 0.05): "))
        if new_delay > 0:
            typing_delay = new_delay
            simulate_typing(f"Typing speed set to {new_delay} seconds per character.")
        else:
            simulate_typing("Delay must be a positive number. Keeping current speed.")
    except ValueError:
        simulate_typing("Invalid input. Please enter a number. Keeping current speed.")

# Generating response with conversation history
def response(user_response, conversation_history):
    robo_response = ''
    source = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_tokens.remove(user_response)  # Remove user input from sent_tokens after TF-IDF

    if req_tfidf < 0.1:  # If no good match in base context, use Ollama with history
        history_str = format_history_for_prompt(conversation_history)
        try:
            response = client.generate(
                model="mistral:7b",
                prompt=history_str,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            robo_response = response['response']
            source = '[Using mistral:7b]'
        except Exception as e:
            robo_response = "I am sorry! I encountered an error while processing your request."
            source = '[Error]'
    else:
        robo_response = sent_tokens[idx]
        source = '[Using base context]'
    return robo_response, source

# Function to format conversation history for Ollama prompt
def format_history_for_prompt(history):
    if not history:
        return ""
    formatted = ""
    for message in history[:-1]:  # All messages except the latest user input
        if message["role"] == "user":
            formatted += "User: " + message["content"] + "\n"
        else:
            formatted += "Assistant: " + message["content"] + "\n"
    formatted += "User: " + history[-1]["content"] + "\nAssistant:"
    return formatted

# Main chatbot loop with conversation history
flag = True
conversation_history = []  # Initialize empty history
simulate_typing("Chatbot: My name is Chatbot. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while flag:
    user_response = input("YOU: ").lower()
    if user_response == 'bye':
        flag = False
        simulate_typing("Chatbot: Bye! Take care..")
    elif user_response in ['thanks', 'thank you']:
        flag = False
        simulate_typing("Chatbot: You are welcome..")
    elif user_response == 'set speed':
        set_typing_speed()
    else:
        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_response})
        # Limit history to last 10 messages
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        if greeting(user_response):
            robo_response = greeting(user_response)
            source = '[Greeting]'
        else:
            robo_response, source = response(user_response, conversation_history)
        
        simulate_typing(f"Chatbot: {robo_response} {source}")
        # Add chatbot response to conversation history
        conversation_history.append({"role": "assistant", "content": robo_response})
