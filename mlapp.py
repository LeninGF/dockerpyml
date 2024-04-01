"""
Dockerizar aplicacion de Machine Learning
To dockerize a machine learning python script
Autor/Author: LeninGF
Fecha/Date: 2024-03-25

We use Mnist train to practice to create a docker image where we can make predictions on new
data with the trained model and save predicted results to disk

Se practica como crear una imagen docker donde se pueda hacer predicciones sobre nuevos
datos con el modelo entrenado y salver las predicciones a disco
"""
import argparse
import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.keras.datasets import mnist
# from tensorflow.python.keras.models import Sequential, load_model
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.python.keras.utils import to_categorical
#  probably because of the cpu?
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    return image

def predict_single_image(model, image_path):
    image = load_image(image_path)
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model (LeNet expects 4D images_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class

def predict_images_in_folder(model, folder_path):
    predictions = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict_single_image(model, image_path)
            predictions.append((image_path, predicted_class))
    return predictions

def evaluate_model(model_file_name, x_test, y_test):
    model = load_model_from_disk(model_file_name)
    loss, accuracy = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
    print(f"Model accuracy: {accuracy*100:.2f}%\bModel Loss: {loss:.2f}")
    

def load_model_from_disk(model_name):
    this_model = load_model(model_name)
    return this_model


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a LeNet model for handwritten digit classification.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', action='store_true', help='Predict images in a folder')
    parser.add_argument('--images_path', type=str, help='Location of the image folder for prediction')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model to the specified location')
    parser.add_argument('--model_h5', default='lenet.h5',  type=str, help='Load a pre-trained model from the specified location')
    args = parser.parse_args()

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the LeNet model
    model = Sequential([
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    if args.train:
        model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5, verbose=1)

        # Save the trained model
        if args.save_model:
            output_file = 'lenet.h5'
            model.save(output_file)
            print(f"Model saved to {output_file}")

    # Load a pre-trained model
    print(args.model_h5)
    loaded_model = load_model(args.model_h5)
    print(f"Model loaded from {args.model_h5}")

    # Evaluate the model
    if args.evaluate:
        evaluate_model(model_file_name=args.model_h5, x_test=x_test, y_test=y_test)

    # Predict images in a folder
    if args.predict and args.images_path:
        predictions = predict_images_in_folder(loaded_model, args.images_path)
        df = pd.DataFrame(predictions, columns=['Image', 'Predicted'])
        df.to_csv('predictions.csv', index=False)
        print(f"Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()
