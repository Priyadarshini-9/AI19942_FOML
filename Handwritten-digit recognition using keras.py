import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess data
x_train = x_train / 255.0  # Normalize to [0,1]
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  # One-hot encode labels
y_test = to_categorical(y_test, 10)

# Step 3: Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),        # Flatten 2D image to 1D vector
    Dense(128, activation='relu'),        # Hidden layer
    Dense(10, activation='softmax')       # Output layer for 10 classes
])

# Step 4: Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Step 6: Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Step 7: Predict and visualize
predictions = model.predict(x_test)

# Plot the first test image and predicted label
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted Label: {tf.argmax(predictions[0]).numpy()}")
plt.axis('off')
plt.show()
