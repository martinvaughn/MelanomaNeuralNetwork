import tensorflow as tf
import cv2


CATS = ["Malignant", "Benign"]  # 0 is MALIGNANT, 1 is BENIGN
IMG_SIZE = 50
model = tf.keras.models.load_model('v2_cancer_model.h5')


def prepare(filepath):
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) #perhaps?  return np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


prediction = model.predict([prepare('/Users/martinvaughn/Project_Pictures/Validate_Benign/IMG_5650.jpg')])

print(CATS[int(prediction[0][0])])
