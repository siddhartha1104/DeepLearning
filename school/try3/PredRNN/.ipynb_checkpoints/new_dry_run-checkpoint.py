import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the ASL dataset
asl_dataset_path = 'data/asl_dataset'
categories = os.listdir(asl_dataset_path)

# Define the image dimensions
img_width, img_height = 224, 224

# Load and preprocess the images
X = []
y = []
for category in categories:
    category_path = os.path.join(asl_dataset_path, category)
    for file in os.listdir(category_path):
        #img_path = os.path.join(category_path, file)
        img_path = 'image.jpeg'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0
        X.append(img)
        y.append(categories.index(category))

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Test accuracy: {accuracy:.2f}')

# Use the model to make predictions on new images
def predict_sign(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return categories[predicted_class[0]]

# Test the prediction function
img_path = 'image.jpeg'
print(predict_sign(img_path))