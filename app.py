import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load and preprocess the data
data_dir = r"C:\Users\hemas\Downloads\archive (6)\images\train"
img_size = 48
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Step 2: Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # 7 emotions
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Step 3: Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)

# Step 4: Save the trained model
model.save("emotion_detection_model.keras")
print("Model saved as 'emotion_detection_model.keras'")