# load model
model = tf.keras.models.load_model('v4_cancer_model.tflite')

# convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_buffer = converter.convert()

# open file and write to it
open('v4_cancer_model.tflite', 'wb').write(tflite_buffer)
print("Tf model created")
