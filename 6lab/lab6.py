import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Функция для предсказания класса изображения
def predict_image(model, image_path):
    # Загрузка и подготовка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255

    # Предсказание
    prediction = model.predict(img)
    return np.argmax(prediction)


# Основная программа
def main():
    # Загрузка данных MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Подготовка данных
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_test = to_categorical(y_test, 10)

    # Загрузка модели
    model = load_model('mnist_cnn_model.h5')

    # Оценка модели
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Предсказание на новом изображении
    image_path = './img/digit_8.png'  # Укажите путь к вашему изображению
    predicted_class = predict_image(model, image_path)
    if predicted_class is not None:
        print(f'Predicted class: {predicted_class}')


if __name__ == "__main__":
    main()
