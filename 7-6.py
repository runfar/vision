import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

n_class = 10
img_siz = (32, 32, 3)

patch_size = 4
p2 = (img_siz[0] // patch_size) ** 2

d_mdoel = 64 # 임베딩 벡터 차원
h = 9        # 헤드 갯수
N = 6        # 인코더 블럭의 개수

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.p_siz = patch_size

    def call(self, img):
        batch_size = tf.shape(img)[0]
        patches = tf.image.extract_patches(images=img, sizes = [1, self.p_siz, self.p_siz, 1],
                                           strides=[1, self.p_siz, self.p_siz, 1], rates=[1, 1, 1, 1], padding='VALID')
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) 
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, p2, d_model):
        super(PatchEncoder, self).__init__()
        self.p2 = p2
        self.projection = layers.Dense(units = d_model)
        self.position_embedding = layers.Embedding(input_dim = p2, output_dim = d_model)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.p2, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifer():
    input = layers.Input(shape = (img_siz))
    nor = layers.Normalization()(input)

    patches = Patches(patch_size)(nor)     # 패치 생성
    x = PatchEncoder(p2, d_mdoel)(patches) #패치 인코딩

    for _ in range(N): # 다중 인코더 블록
        x1 = layers.LayerNormalization(epsilon=1e-6)(x) # 첫번째 층 정규화
        x2 = layers.MultiHeadAttention(num_heads=h, key_dim=d_mdoel//h, dropout=0.1)(x1, x1) # 멀티 헤드 어텐션
        x3 = layers.Add()([x2, x]) # 지름길 연결
        x4 = layers.LayerNormalization(epsilon=1e-6)(x3) # 두번째 층 정규화
        x5 = layers.Dense(d_mdoel * 2, activation = tf.nn.gelu)(x4) # 완전 연결층
        x6 = layers.Dropout(0.1)(x5)
        x7 = layers.Dense(d_mdoel, activation = tf.nn.gelu)(x6) # 완전 연결층
        x8 = layers.Dropout(0.1)(x7)
        x = layers.Add()([x8, x3]) # 지름길 연결

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation = tf.nn.gelu)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation = tf.nn.gelu)(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(n_class, activation = 'softmax')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

model = create_vit_classifer()
model.layers[1].adapt(x_train)

model.compile(optimizer=Adam(learning_rate=0.001), loss = SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), verbose=1)

res = model.evaluate(x_test, y_test, verbose=0)
print("정확률:", res[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


        
