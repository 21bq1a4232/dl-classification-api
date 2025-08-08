# train.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")  # .keras format
EPOCHS = int(os.environ.get("EPOCHS", 20))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
IMG_SIZE = (32, 32, 3)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def build_model(input_shape=IMG_SIZE, num_classes=10):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.3))

    # Block 2
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.4))

    # Block 3
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.5))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def main():
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_data()

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    model = build_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    print("Training...")
    model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(x_test, y_test),
              epochs=EPOCHS)

    print(f"Saving model to {MODEL_PATH} ...")
    model.save(MODEL_PATH)  # Native Keras format
    print("Saved.")

if __name__ == "__main__":
    main()
