import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

x_train = pickle.load(open("x_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))

x_test = pickle.load(open("x_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

x_train = x_train / 255.0


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=x_train.shape[1:], activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=40, validation_split=0.1, batch_size=100)

metrics = model.evaluate(x_test, y_test)

print('Loss of {} and Accuracy is {} %'.format(metrics[0], metrics[1] * 100))
model.save('v2_cancer_model.h5')
