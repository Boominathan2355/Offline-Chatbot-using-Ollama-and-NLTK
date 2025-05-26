import io
import random
import string # to process standard python strings
import warnings
import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# First, set the NLTK data path to a local directory
import os
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download all required NLTK data
try:
    nltk.download('all', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Warning: Could not download all NLTK data. Error: {str(e)}")
    # Try downloading essential packages individually
    for package in ['punkt', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.download(package, download_dir=nltk_data_dir)
        except Exception as e:
            print(f"Warning: Could not download {package}. Error: {str(e)}")
    
# Initialize Ollama client
client = ollama.Client(host='http://localhost:11434')

# Define a base context for the chatbot
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


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf < 0.1):  # If no good match found, use Ollama
        try:
            # Generate response using Ollama
            response = client.generate(
                model="mistral:7b",  # or any other model you have installed
                prompt=user_response,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            robo_response = response['response']
        except Exception as e:
            robo_response = "I am sorry! I encountered an error while processing your request."
    else:
        robo_response = sent_tokens[idx]
    
    return robo_response


flag=True
print("Chatbot: My name is Chatbot. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input("YOU:")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("chatbot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("chatbot: "+greeting(user_response))
            else:
                print("chatbot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("chatbot: Bye! take care..")

