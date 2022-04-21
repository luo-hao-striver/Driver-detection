import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('D:\code_py\driver\models\Resnet18_epoch15_batch8.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("./Resnet18.tflite", "wb").write(tflite_model)
