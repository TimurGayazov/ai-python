import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Подготовка данных
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# 1. Сверточные слои (Convolutional Layers): Применяют фильтры к входным изображениям для извлечения признаков.
# Каждый фильтр скользит по изображению и вычисляет свертку, выделяя такие признаки, как края, углы и текстуры.

# 2. Слои подвыборки (Pooling Layers): Уменьшают размерность данных, обобщая информацию и снижая вероятность переобучения.
# Наиболее распространенный тип - это max-pooling, который выбирает максимальное значение из каждого участка изображения.

# 3. Полносвязные слои (Fully Connected Layers): Соединяют все нейроны предыдущего слоя с каждым нейроном текущего слоя,
# что позволяет модели интегрировать признаки и выполнять классификацию.
model = Sequential([
    # relu нелинейность
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),# Полносвязные слои
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

model.save('mnist_cnn_model.h5')
