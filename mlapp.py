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
import tensorflow as tf
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
from src.utils import conectar_sql
import numpy as np
import pandas as pd
from PIL import Image

def check_gpu_avaible():
    if tf.test.is_gpu_available():
        print(f"GPU name: {tf.config.list_physical_devices('GPU')}")
    else:
        print(f"Tensorflow is using CPU")


def load_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    return image


def predict_single_image(model, image_path):
    image = load_image(image_path)
    image = image.reshape(-1, 28, 28, 1)  # Reshape for the model (LeNet expects 4D images_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return predicted_class


def predict_images_in_folder(model, folder_path):
    predictions = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # print(filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
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
    parser.add_argument('--model_h5', default='/falconiel/app/models/lenet.h5',  type=str, help='Load a pre-trained model from the specified location')
    parser.add_argument('--read_sql', action='store_true', help='To execute a sql query to read from a mysql database')
    parser.add_argument('--save_sql', action='store_true', help='To write to sql table a pandas dataframe')
    args = parser.parse_args()
    # Read env variables
    
    DB_USER = os.environ.get('DB_USER')
    DB_BBDD_PASSWORD = os.environ.get('DB_PASSWORD')
    DB_ANALITICA = os.environ.get('DB_ANALITICA')
    DB_ANALITICA_PASSWORD = os.environ.get('DB_ANALITICA_PASSWORD')
    DB_BBDD_HOST = os.environ.get('DB_HOST')

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
        check_gpu_avaible()
        model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5, verbose=1)

        # Save the trained model
        if args.save_model:
            output_file = os.path.join(os.getcwd(), 'models', 'lenet.h5')
            model.save(output_file)
            print(f"Model saved to {output_file}")

    # Evaluate the model
    if args.evaluate:
        evaluate_model(model_file_name=args.model_h5, x_test=x_test, y_test=y_test)

    # Predict images in a folder
    if args.predict:
        # Load a pre-trained model
        print(args.model_h5)
        loaded_model = load_model(args.model_h5)
        print(f"Model loaded from {args.model_h5}")
        predictions = predict_images_in_folder(loaded_model, args.images_path)
        # print(predictions)
        df = pd.DataFrame(predictions, columns=['Image', 'Predicted'])
        predictions_path = os.path.join(os.getcwd(),'outputs', 'predictions.csv')
        df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to predictions.csv")

    if args.read_sql:
        # Executing a sql query to retrieve data from database
        print(DB_USER, DB_BBDD_PASSWORD, DB_BBDD_HOST)
        conx = conectar_sql(db_user=DB_USER,
                            analitica_user_password=DB_BBDD_PASSWORD,
                            analitica_host=DB_BBDD_HOST)
        query = """SELECT robos.delitos_validados_unified_siaf, count(robos.NDD) 
                    FROM DaaS.robosML_copy robos
                    group by robos.delitos_validados_unified_siaf
                    order by count(robos.NDD) desc;"""
        df_sql = pd.read_sql(query, conx)
        print(f"La consulta SQL genera: {df_sql.shape} resultados")
        output_sql = os.path.join(os.getcwd(),'outputs', 'reading_sql.csv')
        df_sql.to_csv(output_sql)

if __name__ == "__main__":
    main()
