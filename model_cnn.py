import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#import data from mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#How many images do we have?
print(x_train.shape)
print(x_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model
model.fit(x_train, y_train, epochs=20)

#evaluate model
model.evaluate(x_test,  y_test, verbose=2)

model.save('Degit_Recognizer_cnn.h5')

# Print the model summary
model.summary()
