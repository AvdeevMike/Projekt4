import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Преобразование и нормализация данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Разделение на тренировочный и валидационный наборы
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Шаг 2: Создание сверточной нейронной сети
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Шаг 3: Обучение сверточной нейронной сети
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Шаг 4: Оценка производительности модели
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)