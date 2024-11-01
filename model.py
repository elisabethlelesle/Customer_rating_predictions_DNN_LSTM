from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.initializers import Constant
import numpy as np

def load_embedding_matrix(tokenizer, embedding_dim=100, max_words=20000):
    embedding_index = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_model(embedding_matrix, max_words=20000, embedding_dim=100, input_length=300):
    model = Sequential([
        Embedding(input_dim=max_words, 
                  output_dim=embedding_dim, 
                  embeddings_initializer=Constant(embedding_matrix), 
                  input_length=input_length, 
                  trainable=False),
        Bidirectional(LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)),
        LayerNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
