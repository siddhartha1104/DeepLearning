import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LSTM, Dense

os.chdir('E:/Win_sem 2024-25/BCSE332L-Deep Learning/LAB/Assessment -1DL/PredRNN')

asl_dataset_path = 'E:/Win_sem 2024-25/BCSE332L-Deep Learning/LAB/Assessment -1DL/PredRNN/data'  # Update the path if necessary

categories = os.listdir(asl_dataset_path)

# Define the sequence length (number of frames) and frame dimensions
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

        # Check if the image was loaded correctly
        if img is None:
            print(f"load image: {img_path}")
            continue  # Skip to the next image

        img = cv2.resize(img, (frame_width, frame_height))
        img = img / 255.0
        X.append(img)
        y.append(categories.index(category))

# Convert the data into sequences of frames
X_seq = []
y_seq = []
for i in range(0, len(X) - (len(X) % seq_length), seq_length): # Stop iteration before remainder images
    X_seq.append(X[i:i+seq_length])
    y_seq.append(y[i])
    
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Define the epsilon decay transformation function
def epsilon_decay(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# Initialize epsilon and decay rate
epsilon = 1.0
decay_rate = 0.95
min_epsilon = 0.1

# Define the RNN model
model = Sequential()
model.add(Reshape((seq_length, -1), input_shape=(seq_length, frame_width, frame_height, 3)))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with epsilon decay
for epoch in range(3):
    epsilon = epsilon_decay(epsilon, decay_rate, min_epsilon)
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    print(f'Epoch {epoch+1}, Epsilon: {epsilon:.2f}')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Test accuracy: {accuracy:.2f}')

# Use the model to make predictions on new sequences of frames
def predict_sign(img_seq):
    img_seq = np.array(img_seq)
    img_seq = img_seq / 255.0
    img_seq = np.expand_dims(img_seq, axis=0)
    prediction = model.predict(img_seq)
    predicted_class = np.argmax(prediction, axis=1)
    return categories[predicted_class[0]]

# Test the prediction function
img_seq = [cv2.imread('image.jpeg'.format(i)) for i in range(seq_length)]
result = predict_sign(img_seq)
if result:
    print(result.encode('utf-8'), errors='ignore')