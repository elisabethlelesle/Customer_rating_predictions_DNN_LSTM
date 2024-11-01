from preprocessing import load_data, preprocess_text, encode_labels, tokenize_pad_sequences
from model import load_embedding_matrix, build_model
from training import train_model, evaluate_model, prepare_submission

# Load and preprocess data
train_data, test_data = load_data()
train_data, label_encoder = encode_labels(train_data)

# Split data
X = train_data['full_text']
y = train_data['rating']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad sequences
X_train_pad, X_val_pad, X_test_pad, tokenizer = tokenize_pad_sequences(X_train, X_val, test_data['full_text'])

# Load embedding matrix
embedding_matrix = load_embedding_matrix(tokenizer)

# Build and train model
model = build_model(embedding_matrix)
train_model(model, X_train_pad, y_train, X_val_pad, y_val)

# Evaluate model
evaluate_model(model, X_val_pad, y_val)

# Prepare submission
prepare_submission(model, X_test_pad, test_data, label_encoder)
