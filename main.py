import sys
import os
sys.path.append("D:\\soft\\anaconda\\envs\\tensorgpu\\Lib\\site-packages\\")

import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')
'''
===============================常用工具包======================================
'''
import time
import PIL
import pathlib
import cv2
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

'''
=========================keras或者tensorflow工具包==============================
'''
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image_dataset import image_dataset_from_directory
from keras.models import load_model,model_from_json
from glob import glob



'''
========================自定义网络结构=========================================
'''
from my_models.my_VGG11 import VGG_11
from my_models.my_VGG13 import VGG_13
from my_models.my_VGG16 import VGG_16
from my_models.my_VGG19 import VGG_19
from my_models.my_Resnet18 import Resnet_18
from my_models.Moblienet import MobileNet




'''
==================自定义全局变量====================================
'''
BASE_DIR = './state-farm-distracted-driver-detection/imgs'
class_labels = ['safe driving', 'texting - right', 'talking on the phone - right', 'texting - left',
                'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
                'hair and makeup', 'talking to passenger']
# data_dir = "./kaggle/working/train_dataset/"
updated_train_dir="./kaggle/working/train_dataset/"
updated_val_dir="./kaggle/working/val_dataset/"
updated_test_dir="./kaggle/working/test_dataset/"

'''
==================自定义函数====================================
'''
def distribution(param1, param2, param3):

    a=os.listdir(param3)

    test = a[:200]
    val = a[201:401]
    train = a[402:]

    for images in test:
#         print(f"../input/state-farm-distracted-driver-detection/imgs/train/{param1}/"+images, f"/kaggle/working/test_dataset/{param2}/")
        shutil.copy(f"./state-farm-distracted-driver-detection/imgs/train/{param1}/"+images, f"./kaggle/working/test_dataset/{param2}/")
    for images in val:
        shutil.copy(f"./state-farm-distracted-driver-detection/imgs/train/{param1}/"+images, f"./kaggle/working/val_dataset/{param2}/")

    for images in train:
        shutil.copy(f"./state-farm-distracted-driver-detection/imgs/train/{param1}/"+images, f"./kaggle/working/train_dataset/{param2}/")




    #
    # print(f"The count of images for test_dataset > {param2} ",len(os.listdir(f"./kaggle/working/test_dataset/{param2}")))
    # print(f"The count of images for val_dataset > {param2} ",len(os.listdir(f"./kaggle/working/val_dataset/{param2}")))
    # print(f"The count of images for train_dataset > {param2} ",len(os.listdir(f"./kaggle/working/train_dataset/{param2}")))

'''
===========================进入正题==========================================
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driver action detection')

    subparsers = parser.add_subparsers(dest='mode')
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--model_type', type=str, required=True,
                              choices=['VGG11', 'VGG13','Resnet18','MobileNet'])
    parser_train.add_argument('--batch_size', type=int, required=True,
                              help='set batch_size')
    parser_train.add_argument('--epoch', type=int, required=True,
                              help='set epoch')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--model_type', type=str, required=True,
                              choices=['VGG11', 'VGG13', 'Resnet18','MobileNet'])
    parser_test.add_argument('--batch_size', type=int, required=True,
                              help='set batch_size')
    parser_test.add_argument('--epoch', type=int, required=True,
                              help='set epoch')

    args = parser.parse_args()

    '''
    1.数据的加载与处理
    '''

    batch = args.batch_size
    epoch = args.epoch

    # train_dir = os.path.join(BASE_DIR, 'train/')
    # test_dir = os.path.join(BASE_DIR, 'test/')
    #
    # print('Number of images in training set = ', str(len(glob(train_dir + '*/*'))))
    # print('Number of images in testing set = ', str(len(glob(test_dir + '*'))))
    #
    #
    #
    # # training directories
    # for label in class_labels:
    #     tf.io.gfile.makedirs('./kaggle/working/train_dataset/' + label + '/')
    #
    # # validation directories
    # for label in class_labels:
    #     tf.io.gfile.makedirs('./kaggle/working/val_dataset/' + label + '/')
    #
    # # test directories
    # for label in class_labels:
    #     tf.io.gfile.makedirs('./kaggle/working/test_dataset/' + label + '/')


     SAFE_DRIVING = os.path.join(train_dir, 'c0/')
     TEXTING_RIGHT = os.path.join(train_dir, 'c1/')
     TALKING_ON_PHONE_RIGHT = os.path.join(train_dir, 'c2/')
     TEXTING_LEFT = os.path.join(train_dir, 'c3/')
     TALKING_ON_PHONE_LEFT = os.path.join(train_dir, 'c4/')
     OPERATING_THE_RADIO = os.path.join(train_dir, 'c5/')
     DRINKING = os.path.join(train_dir, 'c6/')
     REACHING_BEHIND = os.path.join(train_dir, 'c7/')
     HAIR_MAKEUP = os.path.join(train_dir, 'c8/')
     TALKING_TO_PASSENGER = os.path.join(train_dir, 'c9/')

    # print("Safe driving = ", len(os.listdir(SAFE_DRIVING)))
    # print("Texting right = ", len(os.listdir(TEXTING_RIGHT)))
    # print("Talking on phone right = ", len(os.listdir(TALKING_ON_PHONE_RIGHT)))
    # print("Texting left = ", len(os.listdir(TEXTING_LEFT)))
    # print("Talking on phone left = ", len(os.listdir(TALKING_ON_PHONE_LEFT)))
    # print("Operating the radio = ", len(os.listdir(OPERATING_THE_RADIO)))
    # print("Drinking = ", len(os.listdir(DRINKING)))
    # print("Reaching behind = ", len(os.listdir(REACHING_BEHIND)))
    # print("Hair makeup = ", len(os.listdir(HAIR_MAKEUP)))
    # print("Talking to passenger = ", len(os.listdir(TALKING_TO_PASSENGER)))

     dir_list = [SAFE_DRIVING, TEXTING_RIGHT, TALKING_ON_PHONE_RIGHT, TEXTING_LEFT, TALKING_ON_PHONE_LEFT,
                OPERATING_THE_RADIO, DRINKING, REACHING_BEHIND, HAIR_MAKEUP, TALKING_TO_PASSENGER]
     i = 0
     for class_label in class_labels:
        # print(f"c{i}")
         distribution(f"c{i}", class_label, dir_list[i])
         i += 1

    # data_dir = pathlib.Path(data_dir)

    # print("The count of total images for training set ", len(list(data_dir.glob('*/*.jpg'))))


    # for class_label, img_count in all_training_images.items():
    #     print(class_label)
    #     print(len(img_count))

    train_datagen3 = ImageDataGenerator(rescale=1.0 / 255,
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.2,
                                        )

    val_datagen3 = ImageDataGenerator(rescale=1.0 / 255)

    test_datagen3 = ImageDataGenerator(rescale=1.0 / 255)
    width = 224
    height = 224

    train_generator3 = train_datagen3.flow_from_directory(updated_train_dir, target_size=(width, height),
                                                          batch_size=batch, class_mode='categorical')

    val_generator3 = val_datagen3.flow_from_directory(updated_val_dir, target_size=(width, height),
                                                      batch_size=batch,class_mode='categorical')

    test_generator3 = test_datagen3.flow_from_directory(updated_test_dir, target_size=(width, height),
                                                        batch_size=batch,class_mode='categorical')


    # print(train_generator3.class_indices)
    # print(val_generator3.class_indices)
    # print(test_generator3.class_indices)
    '''
    2.模型的选择以及训练
    '''
    if args.mode == 'train':
        if args.model_type == 'VGG11':
            model = VGG_11()
        elif args.model_type == 'VGG13':
            model = VGG_13()
        elif args.model_type == 'Resnet18':
            model = Resnet_18()
        elif args.model_type == 'MobileNet':
            model = MobileNet()


        # model.summary()
        time_begin = time.time()

        H = model.fit_generator(train_generator3,
                                 steps_per_epoch = 18404/batch,
                                 epochs = epoch,
                                 validation_data = val_generator3,
                                 validation_steps = 2000/batch)
        time_end = time.time()

        time = time_end - time_begin

        print('time:', time)

        model.save(f'./models/{args.model_type}_epoch{args.epoch}_batch{args.batch_size}.h5')
        fd = open(f'./recoder/data_{args.model_type}_epoch{args.epoch}_batch{args.batch_size}.txt', 'a+')
        print(H.history, file=fd)
        fd.close()

        eval_result3 = model.evaluate_generator(test_generator3)
        print('loss rate at evaluation data :', eval_result3[0])
        print('accuracy rate at evaluation data :', eval_result3[1])


    elif args.mode == 'test':
        model = load_model(f'./models/{args.model_type}_epoch{args.epoch}_batch{args.batch_size}.h5')
        eval_result3 = model.evaluate_generator(test_generator3)
        print('loss rate at evaluation data :', eval_result3[0])
        print('accuracy rate at evaluation data :', eval_result3[1])




