import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def load_data():
    train_data = pd.read_json('data/train.json')
    test_data = pd.read_json('data/test.json')
    train_data['text'] = train_data['text'].apply(preprocess_text)
    train_data['title'] = train_data['title'].apply(preprocess_text)
    train_data['full_text'] = train_data['title'] + " " + train_data['text']
    test_data['text'] = test_data['text'].apply(preprocess_text)
    test_data['title'] = test_data['title'].apply(preprocess_text)
    test_data['full_text'] = test_data['title'] + " " + test_data['text']
    return train_data, test_data

def encode_labels(train_data):
    le = LabelEncoder()
    train_data['rating'] = le.fit_transform(train_data['rating'])
    return train_data, le

def tokenize_pad_sequences(X_train, X_val, X_test, max_words=20000, max_len=300):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    return X_train_pad, X_val_pad, X_test_pad, tokenizer
