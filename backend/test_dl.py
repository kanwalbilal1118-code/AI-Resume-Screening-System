from tensorflow.keras.models import load_model

cnn = load_model("models/cnn_model.h5")
print("CNN Loaded")

lstm = load_model("models/lstm_model.h5")
print("LSTM Loaded")