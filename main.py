# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# import tensorflow as tf
# from tensorflow.keras import layers, models

# # Step 1: Build the CNN Model
# model = models.Sequential ([
#     layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),  # Adjust input shape if needed
#     layers.MaxPooling2D((2, 2)),

#     layers.Conv2D(64, (3, 3), activation="relu"),
#     layers.MaxPooling2D((2, 2)),ds

#     layers.Conv2D(128, (3, 3), activation="relu"),
#     layers.MaxPooling2D((2, 2)),

#     layers.Flatten(),
#     layers.Dense(128, activation="relu"),
#     layers.Dropout(0.5),  # Prevent overfitting
#     layers.Dense(1, activation="sigmoid")
#     # layers.Dense(10, activation="softmax")  # Replace 10 with the number of your classes
# ])

# # Step 2: Compile the Model
# model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",  # Binary cross-entropy loss for binary classification
#     # loss="categorical_crossentropy",  # Use "sparse_categorical_crossentropy" if y_train is not one-hot encoded
#     metrics=["accuracy"]
# )

# # Step 3: Train the Model
# history = model.fit(
#     X_train,  # Training images as a numpy array of shape (num_samples, 128, 128, 3)
#     y_train,  # Training labels
#     epochs=10,  # Adjust based on your needs
#     batch_size=32,  # Number of samples per batch
#     validation_data=(X_val, y_val)  # Validation data
# )

# # Step 4: Save the Model
# model.save("cnn_model.h5")
# print("Training complete! Model saved as 'cnn_model.h5'")
