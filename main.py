# working this model in jupyter or colab would be better.

# Importing required Libraries

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Constants

IMAGE_SIZE = 150
BATCH_SIZE = 32

# Extracting fle path from Google drive

data_sets = keras.preprocessing.image_dataset_from_directory(
    "images's file path",
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = 32
)
data_sets

# Defining lables

classes = data_sets.class_names
classes
# len(data_sets)

'''p
rint(f"No. of images in the 1st batch: {len(batch_image)}")
print(f"No. of labels in the 1st batch: {len(batch_class)}")'''

print("FIRST IMAGE:")
print()
print(batch_image[0].numpy())

print("FIRST ROW OF THE IMAGE:")
print()
print(batch_image[0][0].numpy())

plt.figure(figsize = (8, 6))
for batch_image, lable_batch in data_sets.take(1):
  for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(batch_image[i].numpy().astype('int32'), cmap = 'Oranges')
    plt.title(classes[batch_class[i]])
    plt.axis('off')

print("Approx. images in the sample:", len(data_sets)*32)

def data_set_split(ds, train_percent=0.8, test_percent=0.1, val_percent=0.1, shuffle=True, shuffle_size=1000):
  ds_size = len(ds)  # 50

  if shuffle:
    ds = ds.shuffle(shuffle_size, seed=12)
  train_size = int(ds_size*train_percent)  # 40
  valid_size = int(ds_size*val_percent)  #  5

  train_data = ds.take(train_size)
  valid_data = ds.skip(train_size).take(valid_size)
  test_data = ds.skip(train_size).skip(valid_size)

  return train_data, valid_data, test_data

train_ds, valid_ds, test_ds = data_set_split(data_sets)

# print(len(train_ds))
# print(len(valid_ds))
# print(len(test_ds))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

resize_image_and_rescale_image = keras.models.Sequential([
    keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    keras.layers.experimental.preprocessing.Rescaling(1./255)
])

data_argumentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),
    keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
cnn = keras.Sequential()


cnn.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
cnn.add(keras.layers.MaxPooling2D(pool_size=3, strides=1))

cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D((3,3), 1))

cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(3, 1))

cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(3, 1))

'''cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(3, 1))'''

cnn.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPooling2D(3, 1))

cnn.add(keras.layers.Flatten())

cnn.add(keras.layers.Dense(64, activation='relu'))
cnn.add(keras.layers.Dropout(0.5))
cnn.add(keras.layers.Dense(3, activation='softmax'))

# cnn.summary()

cnn.compile(
    optimizer='adam',
    loss = 'SparseCategoricalCrossentropy',
    metrics=['accuracy']
)


from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights=True)
history = cnn.fit(
    train_ds,
    validation_data = valid_ds,
    epochs=25,
    verbose=1,
    callbacks=[early_stop]
)

scores = cnn.evaluate(test_ds)

for i in range(len(scores)):
  if i==0:
    print("Loss:", round(scores[i], 2))
  else:
    print("Accuracy:", round(scores[i]*100, 2), '%')

# print(history.params)
# print(history.history.keys())

for i in range(len(history.history['accuracy'])):
  print(round(history.history['accuracy'][i]*100, 2), "%")

# Assigning accuracy and losses in variables

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# Plotting Training and validation accuracy

plt.figure(figsize=(5, 5), dpi=100, facecolor='lightgrey', edgecolor='black', tight_layout=True)
plt.plot(range(len(acc)), acc, label='Training accuracy', color='blue', linestyle='-', marker='o', linewidth=2)
plt.plot(range(len(acc)), val_acc, label='Validation accuracy', color='red', linestyle='--', marker='s', linewidth=2)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("TRAINING AND VALIDATION ACCURACY")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

predicted = cnn.predict(image_batch)
# print(predicted)

import numpy as np

plt.figure(figsize=(4, 4))
plt.imshow(image_batch[0].numpy().astype('int32'))
print((classes[lable_batch[0].numpy()]))
plt.axis('off')

def to_predict(img):
  img_array = keras.utils.img_to_array(images[i].numpy())
  img_array = np.expand_dims(img_array, axis=0)

  predictions = cnn.predict(img_array)

  predicted_class = classes[np.argmax(predictions[0])]
  confidence = round(100*np.max(predictions[0]), 2)
  return predicted_class, confidence

plt.figure(figsize=(8, 12), dpi=100, facecolor='lightgrey', edgecolor='black', tight_layout=True)
for images, labels in test_ds.take(1):
  for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i].numpy().astype('int32'))
    plt.axis("off")
    predicted_class, confidence = to_predict(images[i].numpy())
    actual_class = classes[labels[i]]
    plt.title(f"Actual : {actual_class}\n Predicted : {predicted_class}\nConfidence : {confidence}")
plt.show()
