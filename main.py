import pandas as pd
from modules.preprocess import preprocess_text, tokenize_and_pad
from modules.model import create_dnn_model
from modules.train import train_model, plot_training_history

# Load the data
train_data = pd.read_json('data/train.json')
test_data = pd.read_json('data/test.json')

# Preprocess the data
train_data['clean_text'] = train_data['text'].apply(preprocess_text)
train_data['clean_title'] = train_data['title'].apply(preprocess_text)
train_data['combined_text'] = train_data['clean_title'] + ' ' + train_data['clean_text']

test_data['clean_text'] = test_data['text'].apply(preprocess_text)
test_data['clean_title'] = test_data['title'].apply(preprocess_text)
test_data['combined_text'] = test_data['clean_title'] + ' ' + test_data['clean_text']

# Tokenize and pad sequences
X_train_padded, tokenizer = tokenize_and_pad(train_data['combined_text'])
X_test_padded, _ = tokenize_and_pad(test_data['combined_text'])

# Prepare the labels
y_train = train_data['rating'] - 1  # Convert to 0-4 range

# Create the DNN model
vocab_size = 10000
max_len = 100
dnn_model = create_dnn_model(vocab_size, max_len)

# Train the model
history = train_model(dnn_model, X_train_padded, y_train, epochs=10, batch_size=32)

# Plot the training history
plot_training_history(history)

# Predict the test data
y_pred = dnn_model.predict(X_test_padded)
y_pred_classes = y_pred.argmax(axis=1) + 1  # Convert to 1-5 range

# Create a submission file
submission = pd.DataFrame({'index': test_data.index, 'rating': y_pred_classes})
submission.to_csv('submission.csv', index=False)
