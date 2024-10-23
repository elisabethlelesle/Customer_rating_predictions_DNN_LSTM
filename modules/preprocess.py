import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preprocessing Function
def preprocess_text(text):
    """
    Cleans text by converting to lowercase and removing special characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Function to tokenize and pad sequences
def tokenize_and_pad(text_data, vocab_size=10000, max_len=100):
    """
    Tokenizes and pads text data to ensure consistent input length.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(text_data)
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, tokenizer
