import tensorflow as tf
from tensorflow.keras import Sequential


model = Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

#import data from mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0]/255)

"""
#normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model
model.fit(x_train, y_train, epochs=20)

#evaluate model
model.evaluate(x_test,  y_test, verbose=2)

#save model
model.save('Degit_Recognizer.h5')
"""