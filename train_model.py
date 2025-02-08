import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dummy dataset (using random numbers)
X_train = np.random.rand(1000, 150, 150, 3)  # 1000 images, 150x150 resolution, 3 color channels
y_train = np.random.randint(0, 26, size=(1000,))  # 26 classes

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # Output layer (26 classes)
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model (using dummy data)
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Save Model
model.save("my_model.h5")
print("Model saved successfully as 'my_model.h5'")
