1.XOR 
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])


input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1
learning_rate = 0.1
epochs = 10000


np.random.seed(42)
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))


for epoch in range(epochs):
    
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)

    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(wo.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    wo += hidden_output.T.dot(d_predicted_output) * learning_rate
    bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Final output after training:")
print(np.round(predicted_output, 3))

2.CNN Image Classification 

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.3f}')

predictions = model.predict(x_test)

def plot_images(images, true_labels, pred_probs, start_idx=0, num=10):
    plt.figure(figsize=(12, 6))
    for i in range(num):
        idx = start_idx + i
        plt.subplot(2, 5, i+1)
        plt.imshow(images[idx].reshape(28,28), cmap='gray')
        true_label = true_labels[idx]
        pred_label = np.argmax(pred_probs[idx])
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.axis('off')
    plt.show()

plot_images(x_test, y_test, predictions, start_idx=0, num=10)


3.Sentiment Analysis 

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

dataset = tfds.load('imdb_reviews', as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
batch_size = 32
train_dataset = train_dataset.shuffle(10000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
vectorize_layer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=100)
vectorize_layer.adapt(train_dataset.map(lambda x, y: x))
model = tf.keras.Sequential([
 vectorize_layer,
 tf.keras.layers.Embedding(len(vectorize_layer.get_vocabulary()), 64, mask_zero=True),
 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
 tf.keras.layers.Dense(64, activation='relu'),
 tf.keras.layers.Dense(1)
])
model.build(input_shape=(None,))
model.compile(
 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
 optimizer=tf.keras.optimizers.Adam(),
 metrics=['accuracy']
)
model.fit(train_dataset, epochs=3, validation_data=test_dataset)


4.Data Augmentation 

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data_augmentation = tf.keras.Sequential([
 layers.RandomFlip("horizontal_and_vertical"),
 layers.RandomRotation(0.2),
 layers.RandomZoom(height_factor=0.2, width_factor=0.2),
 layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
 layers.RandomContrast(0.2),
 layers.RandomBrightness(0.2),
 layers.GaussianNoise(0.05),
 layers.RandomCrop(height=200, width=200),
 layers.Rescaling(1./255)
])
img_raw = tf.io.read_file('/content/drive/MyDrive/Kamesh_HT.jpeg')
img = tf.image.decode_jpeg(img_raw, channels=3)
img = tf.image.resize(img, [224, 224])
img = tf.expand_dims(img, 0)
plt.figure(figsize=(15,5))
for i in range(5):
    augmented_img = data_augmentation(img)
    plt.subplot(1,5,i+1)
    plt.imshow(tf.cast(augmented_img[0]*255, tf.uint8))
    plt.axis('off')
    plt.title(f"Augmented {i+1}")
plt.show()



5.iris dataset

import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = keras.Sequential([
 keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
 keras.layers.Dense(10, activation='relu'),
 keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


6. Implementation of Confusion Matrix for Multi-Class Classification


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.1)
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels



7.Neural Network architecture 


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = Sequential([
 Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
 MaxPooling2D((2,2)),
 Conv2D(64, (3,3), activation='relu'),
 MaxPooling2D((2,2)),
 Flatten(),
 Dense(128, activation='relu'),
 Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
