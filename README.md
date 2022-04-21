# Driver-detection
The project bases on keras to detect the driver's actions which are correct or dangerous, thank you for your reading
The project will use the VGG , ResNet and MobileNet_v1 to detect the driver datasets.
The project running the pycharm, so for the .sh file you should use the Git (recommend).

# 1.my_models directory
      In the directory, you can get the VGG models ,ResNet models, and MobileNet_v1 based on the keras or tensorflow.
      
# 2.picture_data directory
      This is a result. when you run the code ,you can get the result in here.
      
# 3.state-farm-distracted-driver-detection directory
      There are your datasets 
      dowmloading from https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data.
    
# 4.  0_train.sh 
      This is a .sh file ,you should run it for training the models
      
# 5.  1_test.sh
      This is a .sh file too, you can run it for testing the models.
      
# 6. main.py
      The file is the important, you can change something you want.
      In the here , you should pay attention to your datasets、models... 's file path!!!!
      
# 7.Draw_picture_train1.py(may be unnecessary for reader,the file just to me!!!)
      The file is about show many result after training the model and saving it.

# 8.data_argument.py(may be unnecessary for reader,the file just to me!!!)
      This is about the process from data.
      
# 9.tflite.py
      If you want to create a app or use MCU、CPU which is removable,you need run the code to transform your models.
     
         
# Application（if you want to use the model running into a app, you should look the topic!!!）
      https://github.com/yeyupiaoling/ClassificationForAndroid/tree/master/TFLiteClassification
      and 
      https://resource.doiduoyi.com/#c1uo2s4
      
      The above is important for the app created, and we can run the two projects to load ourselves' models in the mobile phone.
      我制作的app核心代码全部来自于这两个网址，处于诚实且保护原作者成果的原则，只贴附上原作者代码。
      以下两点是其中修改符合自己模型的处理：
      1.  第一个网址的代码中：注意自己tflite的输入层和输出层的名字是否一致数据部分。记得修改均值为0，方差为255f!（这里根据读者预处理决定）
      2.  第二个网址的代码中：因为是10个类别，需要修改数组大小，本人网络训练预处理预处理用的是除以255f，因此这里也需要这么处理!

# Summary
      This is my first create project in the github, If you can see here ,thank you!!.
      The project is my undergraduate graduation thesis's code, and I want to remenber the beautiful time.


