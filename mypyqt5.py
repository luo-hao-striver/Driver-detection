import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import cv2
import keras
from keras.models import load_model
import numpy as np
import re
import os
from keras.preprocessing import image
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()

        self.resize(600, 400)
        self.setWindowTitle("驾驶员行为分类")
        self.btn = QPushButton()
        self.btn.setText("打开图片")
        self.btn.clicked.connect(self.openimage)
        self.label = QLabel()
        self.label.setText('图片路径')


        self.labelimage = QLabel()
        self.labelimage.setText("显示图片")
        #self.labelimage.setFixedSize(500, 400)#设置尺寸
        self.labelimage.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        #预测按钮
        self.btnclass=QPushButton()
        self.btnclass.setText('点击预测分类')
        self.btnclass.clicked.connect(self.fenlei)
        self.labelclass=QLabel()
        self.labelclass.setText('预测类别')
        self.labelclass.setStyleSheet("font:16pt '楷体';border-width:2px;border-style: inset;border-color:gray")


        layout1=QVBoxLayout()
        layout1.addWidget(self.btn)
        layout1.addWidget(self.label)
        layout1.addWidget(self.labelimage)

        layout2 = QVBoxLayout()
        layout2.addWidget(self.btnclass)
        layout2.addWidget(self.labelclass)

        layout=QVBoxLayout()
        layout.addLayout(layout1)
        layout.addLayout(layout2)

        self.setLayout(layout)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        #jpg = QtGui.QPixmap(imgName).scaled(self.labelimage.width(), self.label.height())#适应labelimage尺寸，前提是label设置了尺寸
        jpg = QtGui.QPixmap(imgName)
        self.labelimage.setPixmap(jpg)
        self.label.setText(str(imgName))
    def fenlei(self):
        biaoqian = {'0': 'dringking', '1': 'hair and makeup', '2': 'operating the radio', '3': 'reaching behind', '4': 'safe driving'
                    , '5': 'talking on the phone - left', '6': 'talking on the phone - right',
                    '7': 'talking to passenger','8': 'texting - left', '9': 'texting - right'}
        path=self.label.text()
        # newName = re.sub('(D:/code_py/driver/kaggle/working/test1)','', path)
        print(path)
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)  # 三维（224，224，3）
        x = np.expand_dims(x, axis=0)  # 四维（1，224，224，3）#因为keras要求的维度是这样的，所以要增加一个维度
        x = x / 255

        model = load_model('./models/Resnet18_epoch15_batch8.h5')
        predict_y = model.predict(x)
        pred_y = np.argmax(predict_y)
        pro = np.max(predict_y)*100
        pro = f"{pro:.4f}"
        print(pred_y)
        self.labelclass.setText("Resnet18：\n"+
                                "预测结果："+biaoqian[str(pred_y)]+'\n'+
                                "预测概率："+str(pro)+"%")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
