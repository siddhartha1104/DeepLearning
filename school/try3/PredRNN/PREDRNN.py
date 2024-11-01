import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM
import os

# Change to your dataset directory
os.chdir('D:\dataScience\deepLearning\school\try3\PredRNN')

asl_dataset_path = 'D:\dataScience\deepLearning\school\try3\PredRNN\data'

categories = os.listdir(asl_dataset_path)

# Define the sequence length and frame dimensions
seq_length = 10
frame_width, frame_height = 224, 224

# Load and preprocess the images
X = []
y = []
for category in categories:
    category_path = os.path.join(asl_dataset_path, category)
    for file in os.listdir(category_path):
        img_path = os.path.join(category_path, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        img = cv2.resize(img, (frame_width, frame_height))
        img = img / 255.0
        X.append(img)
        y.append(categories.index(category))

# Convert the data into sequences of frames
X_seq, y_seq = [], []
for i in range(0, len(X) - (len(X) % seq_length), seq_length):
    X_seq.append(X[i:i + seq_length])
    y_seq.append(y[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential()
model.add(Reshape((seq_length, -1), input_shape=(seq_length, frame_width, frame_height, 3)))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and save loss history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Test accuracy: {accuracy:.2f}')

# Plotting training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Use the model to make predictions on new sequences of frames
def predict_sign(img_seq):
    img_seq = np.array(img_seq) / 255.0
    img_seq = np.expand_dims(img_seq, axis=0)
    prediction = model.predict(img_seq)
    predicted_class = np.argmax(prediction, axis=1)
    return categories[predicted_class[0]]

# Test the prediction function
img_seq = [cv2.imread(f'image_{i}.jpeg') for i in range(seq_length)]  # Adjust image names accordingly
result = predict_sign(img_seq)
if result:
    print(result.encode('utf-8', errors='ignore').decode('utf-8'))