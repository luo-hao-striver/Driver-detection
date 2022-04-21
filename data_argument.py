
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
l = []
img = load_img("./kaggle/working/train_dataset/drinking/img_72.jpg")  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
l.append(x)
l = np.array(l)


k = datagen.flow(l, batch_size=1,save_to_dir=r'./', save_format='jpg')

img.show()
with open("./123.txt","a+") as fd:
    print(k.next(),file=fd)


# i = 0
# for batch in datagen.flow(l, batch_size=1,save_to_dir=r'./', save_format='jpg'):
#     i += 1
#     if i > 2:
#         break  # otherwise the generator would loop indefinitely