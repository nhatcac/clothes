import os
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, RandomRotation, RandomZoom, RandomFlip
from keras.callbacks import EarlyStopping

# 1. Chuẩn bị dữ liệu
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Kiến trúc AI với Data Augmentation tích hợp (Cách mới nhất)
model = Sequential([
    # Các lớp Augmentation (tự động xoay, lật, zoom ảnh khi training)
    RandomRotation(0.1, input_shape=(28, 28, 1)),
    RandomZoom(0.1),
    RandomFlip("horizontal"),
    
    # Kiến trúc CNN
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3), # Chống học vẹt
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Công tắc an toàn EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Huấn luyện
print("--- Đang huấn luyện AI bản tối ưu hóa (Keras 3+) ---")
if not os.path.exists('model_data'): os.makedirs('model_data')

model.fit(
    x_train, y_train,
    epochs=30,
    validation_split=0.2, # Kiểm tra chéo 20%
    callbacks=[early_stop],
    batch_size=64
)

model.save('model_data/model_quanao.h5')
print("--- ĐÃ LƯU XONG: model_data/model_quanao.h5 ---")