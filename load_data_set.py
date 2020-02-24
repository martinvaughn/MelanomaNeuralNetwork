import numpy as np
import os
import cv2
import random
import pickle

training_data = []
test_data = []
IMG_SIZE = 50


def create_training_data():
    DIR = "/Users/martinvaughn/Project_Pictures"
    CATS = ["Malignant", "Benign"]  # 0 is MALIGNANT, 1 is BENIGN
    for cat in CATS:
        path = os.path.join(DIR, cat)  # path to malignant or benign
        class_num = CATS.index(cat)
        for img in os.listdir(path):
            if img != ".DS_Store":
                try:
                    img_array = cv2.imread(os.path.join(path, img))
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    print('Error saving train_data to array')


def create_test_data():
    DIR = "/Users/martinvaughn/Project_Pictures"
    CATS = ["Test_Malignant", "Test_Benign"]  # 0 is MALIGNANT, 1 is BENIGN
    for cat in CATS:
        path = os.path.join(DIR, cat)  # path to malignant or benign
        class_num = CATS.index(cat)
        for img in os.listdir(path):
            if img != ".DS_Store":
                try:
                    img_array = cv2.imread(os.path.join(path, img))
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    test_data.append([new_array, class_num])
                except Exception as e:
                    print('Error saving test_data to array')


create_training_data()
print(len(training_data))
random.shuffle(training_data)

create_test_data()
print(len(test_data))
random.shuffle(test_data)

x_train = []
y_train = []

for features, label in training_data:
    x_train.append(features)
    y_train.append(label)

x_test = []
y_test = []

for features, label in test_data:
    x_test.append(features)
    y_test.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = np.array(y_train)
y_test = np.array(y_test)


pickle_out = open("x_train.pickle", "wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("x_test.pickle", "wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

#  TO OPEN -->
#  pickle_in = open("x_train.pickle", "rb") // or x_test.pickle
#  X = pickle.load(pickle_in)
