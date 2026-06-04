import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar dataset
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
train_ds = dataset['train']

# 2. Pré-processamento
IMG_SIZE = 160
BATCH_SIZE = 32

def format_example(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(format_example)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Divisão treino/validação
train_size = int(0.8 * info.splits['train'].num_examples)
val_size = int(0.2 * info.splits['train'].num_examples)

train_dataset = train_ds.take(train_size)
val_dataset = train_ds.skip(train_size).take(val_size)

# 3. Modelo pré-treinado
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Visualizar algumas imagens antes do treino
class_names = info.features['label'].names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
plt.suptitle("Amostras antes do treino")
plt.show()

# 5. Treinar
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=2)

# 6. Avaliar
loss, acc = model.evaluate(val_dataset)
print(f"Acurácia no conjunto de validação: {acc:.2f}")

# 7. Visualizar previsões depois do treino
plt.figure(figsize=(10, 10))
for images, labels in val_dataset.take(1):
    predictions = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        pred_label = "dog" if predictions[i] > 0.5 else "cat"
        true_label = class_names[labels[i].numpy()]
        plt.title(f"Pred: {pred_label} | True: {true_label}")
        plt.axis("off")
plt.suptitle("Amostras depois do treino")
plt.show()
