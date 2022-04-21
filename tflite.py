import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('D:\code_py\driver\models\Resnet18_epoch15_batch8.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("./Resnet18.tflite", "wb") as f:
    f.write(tflite_model)

# 显示tflite模型的名字
# tflite_model = tf.lite.Interpreter(model_path="./MobileNet_V1.tflite")  # .contrib
# input_details = tflite_model.get_input_details()
# output_details = tflite_model.get_output_details()
# print(input_details)
# print(output_details)
