import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 및 전처리
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# 훈련/검증 데이터 분할 (sklearn 없이)
val_size = int(len(x_train_full) * 0.2)
x_val = x_train_full[:val_size]
y_val = y_train_full[:val_size]
x_train = x_train_full[val_size:]
y_train = y_train_full[val_size:]

# SGD 모델
model_sgd = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_sgd.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("SGD 모델 학습 중...")
history_sgd = model_sgd.fit(x_train, y_train, epochs=20,
                            validation_data=(x_val, y_val),
                            verbose=1)

# Adam 모델
model_adam = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_adam.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

print("\nAdam 모델 학습 중...")
history_adam = model_adam.fit(x_train, y_train, epochs=20,
                              validation_data=(x_val, y_val),
                              verbose=1)

# 테스트 성능 평가
sgd_loss, sgd_acc = model_sgd.evaluate(x_test, y_test, verbose=0)
adam_loss, adam_acc = model_adam.evaluate(x_test, y_test, verbose=0)

print(f"\n=== 테스트 성능 비교 ===")
print(f"SGD  - Loss: {sgd_loss:.4f}, Accuracy: {sgd_acc:.4f}")
print(f"Adam - Loss: {adam_loss:.4f}, Accuracy: {adam_acc:.4f}")

# 시각화
plt.figure(figsize=(14, 5))

# 훈련 손실 비교
plt.subplot(1, 3, 1)
plt.plot(history_sgd.history['loss'], label='SGD Train', linestyle='--', marker='o')
plt.plot(history_adam.history['loss'], label='Adam Train', linestyle='--', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)

# 검증 손실 비교
plt.subplot(1, 3, 2)
plt.plot(history_sgd.history['val_loss'], label='SGD Validation', linestyle='--', marker='o')
plt.plot(history_adam.history['val_loss'], label='Adam Validation', linestyle='--', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)

# 검증 정확도 비교
plt.subplot(1, 3, 3)
plt.plot(history_sgd.history['val_accuracy'], label='SGD Validation', linestyle='--', marker='o')
plt.plot(history_adam.history['val_accuracy'], label='Adam Validation', linestyle='--', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('sgd_vs_adam_comparison.png', dpi=150)
print("\n그래프가 'sgd_vs_adam_comparison.png'로 저장되었습니다.")
plt.show()

# 상세 비교 표
plt.figure(figsize=(8, 5))
epochs = range(1, 21)
plt.subplot(2, 1, 1)
plt.plot(epochs, history_sgd.history['accuracy'], 'b-o', label='SGD Train Acc')
plt.plot(epochs, history_adam.history['accuracy'], 'r-s', label='Adam Train Acc')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, history_sgd.history['val_accuracy'], 'b-o', label='SGD Val Acc')
plt.plot(epochs, history_adam.history['val_accuracy'], 'r-s', label='Adam Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('sgd_vs_adam_accuracy.png', dpi=150)
print("정확도 비교 그래프가 'sgd_vs_adam_accuracy.png'로 저장되었습니다.")
plt.show()