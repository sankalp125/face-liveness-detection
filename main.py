# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
#
# # Count dataset balance
# real_images = len(os.listdir("training_dataset/real"))
# fake_images = len(os.listdir("training_dataset/fake"))
# print(f"real_images: {real_images}")
# print(f"fake_images: {fake_images}")
#
# # Image size and batch config
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
#
# # Train Data Generator (with augmentation)
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
# )
#
# # Validation Data Generator (no augmentation)
# val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
#
# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     "training_dataset",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary"
# )
#
# # Load validation data
# val_generator = val_datagen.flow_from_directory(
#     "validation_dataset",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary"
# )
#
# print("Class labels:", train_generator.class_indices)
#
# # Load pre-trained MobileNetV2 base
# base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False  # freeze base for transfer learning
#
# # Custom top classifier
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dropout(0.3),
#     Dense(128, activation="relu"),
#     Dropout(0.3),
#     Dense(1, activation="sigmoid")
# ])
#
# # Compile the model
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# print("Model created and compiled successfully!")
#
# # Train the model
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=val_generator
# )
#
# # Save the trained model
# model.save("liveness_model.h5")
# print("Model saved successfully!")
#
# # Convert to TensorFlow Lite (with optimization)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()
#
# # Save the TFLite model
# with open("liveness_model.tflite", "wb") as f:
#     f.write(tflite_model)
#
# print("TFLite model saved as liveness_model.tflite")

##  New Model ##################################

import os
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Extract frames (no change here)
attack_type_to_label = {
    "real": "real_z",
    "mask": "fake",
    "mask3d": "fake",
    "monitor": "fake",
    "outline": "fake",
    "outline3d": "fake"
}

output_dir = "frame_dataset"
os.makedirs(f"{output_dir}/real_z", exist_ok=True)
os.makedirs(f"{output_dir}/fake", exist_ok=True)

video_files = glob("dataset/rose_p3/**/**/*/*.mp4", recursive=True)

for video_path in video_files:
    print(f"video_path: {video_path}")
    attack_type = video_path.split(os.sep)[1]
    label = attack_type_to_label.get(attack_type)
    print(f"attack Type: {attack_type}, label: {label}")
    if not label:
        continue

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    while success and frame_count < 10:
        success, frame = cap.read()
        if success:
            frame_name = os.path.splitext(os.path.basename(video_path))[0] + f"_f{frame_count}.jpg"
            save_path = os.path.join(output_dir, label, frame_name)
            cv2.imwrite(save_path, frame)
            frame_count += 1
    cap.release()

print("✅ Frame extraction complete.")

# Count samples
real_count = len(os.listdir("frame_dataset/real_z"))
fake_count = len(os.listdir("frame_dataset/fake"))
print(f"Total Real: {real_count}, Total Fake: {fake_count}")

# Training parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    "frame_dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "frame_dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Double-check label mapping
print("✅ Class indices:", train_generator.class_indices)
# Should print: {'fake': 0, 'real_z': 1}

# Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Save model
model.save("liveness_model.h5")
print("✅ Model saved as liveness_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("liveness_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved.")




