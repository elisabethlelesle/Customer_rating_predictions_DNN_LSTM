from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score

def train_model(model, X_train_pad, y_train, X_val_pad, y_val, epochs=16, batch_size=32):
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    history = model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return history

def evaluate_model(model, X_val_pad, y_val):
    predictions = model.predict(X_val_pad)
    predictions_class = np.argmax(predictions, axis=1)
    f1 = f1_score(y_val, predictions_class, average='weighted')
    print(f"LSTM with Attention F1 Score: {f1}")
    return f1

def prepare_submission(model, X_test_pad, test_data, label_encoder):
    predictions = model.predict(X_test_pad)
    predictions_class = np.argmax(predictions, axis=1)
    submission = pd.DataFrame({
        'index': ['index_' + str(i) for i in test_data.index],
        'rating': label_encoder.inverse_transform(predictions_class)
    })
    submission.to_csv('lstm_attention_submission.csv', index=False)
    print("Submission file saved as lstm_attention_submission.csv")
