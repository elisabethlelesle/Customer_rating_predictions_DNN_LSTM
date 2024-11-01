# Customer_rating_predictions_DNN_LSTM

## Business Overview

Text classification is a powerful application in Natural Language Processing (NLP) with uses across industries to understand customer sentiments, reviews, and feedback. This project applies an LSTM model with an attention mechanism to classify product reviews based on their ratings. The model leverages pre-trained word vectors from the GloVe dataset to enhance text representation, ensuring a robust understanding of contextual language in the reviews.

---

## Aim

The main objective is to predict product review ratings based on textual data using an LSTM model enhanced with an attention layer. The attention mechanism aims to capture significant patterns in text sequences, making the model suitable for complex NLP tasks.

---

## Data Description

The dataset consists of product reviews, each paired with a corresponding rating category. Text preprocessing techniques and tokenization steps are employed, followed by GloVe embeddings to represent the textual data.

---

## Tech Stack

- **Language**: `Python`
- **Libraries**: `pandas`, `tensorflow`, `nltk`, `numpy`, `scikit-learn`

---

## Approach

### 1. Installation and Imports

Install necessary packages using the `pip` command. Import the libraries and modules required for the project.

### 2. Configuration

Define paths and settings to manage data and model-related parameters.

### 3. Process GloVe Embeddings

- Load GloVe embeddings from the text file.
- Convert embeddings to a numerical array.
- Incorporate embeddings for padding and unknown words.
- Save embeddings to enhance model performance.

### 4. Text Preprocessing

- Load the dataset and handle null values.
- Apply text preprocessing techniques: lowercasing, punctuation removal, lemmatization, and stopword removal.
- Tokenize the preprocessed text.

### 5. Data Preparation

- Encode rating labels using `LabelEncoder`.
- Split the data into training and validation sets.
- Pad tokenized sequences to a fixed length for model input.

### 6. Model Building

- Define an LSTM architecture with an attention mechanism.
- Initialize pre-trained embeddings and set up dropout layers for regularization.

### 7. Model Training

Train the LSTM model using the preprocessed dataset with specified epochs and batch size.

### 8. Evaluation on Validation Data

Calculate the F1 score on the validation data to evaluate model performance.

### 9. Prediction on Test Data

Use the trained model to make predictions on unseen test data and prepare a submission file.

---

## Modular Code Overview

1. **Input**: Contains the data files, including:
   - `train.json` – training data
   - `test.json` – test data
   - `glove.6B.100d.txt` – GloVe embeddings (download from [here](https://nlp.stanford.edu/projects/glove/))

2. **Source**: Contains modularized code for each project step:
   - `preprocessing.py` – handles text preprocessing and data preparation
   - `model.py` – defines the LSTM model architecture
   - `training.py` – trains the model and evaluates it

3. **Output**: Generated files required for model deployment and evaluation:
   - `lstm_attention_submission.csv` – model predictions on test data

4. **main.py**: Integrates all modules, allowing end-to-end model training and evaluation.
