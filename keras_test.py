from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add
from keras.preprocessing import image
from scipy.misc import imsave, imread, imresize, toimage
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import nibabel as nib
from PIL import Image
import scipy.io as sio
from prepare_data import(
read_npy,
read_bmp,
down_up_samples,
wavelet_transform,
array_to_patch
)

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

data_label = read_bmp()
data_input = down_up_samples(data_label)

data_label = np.array(data_label, dtype=np.float)
data_label /= 255
data_label = np.reshape(data_label, newshape=[1, 256, 256, 1])
data_input = np.array(data_input, dtype=np.float)
data_input /= 255
data_input = np.reshape(data_input, newshape=[1, 256, 256, 1])
img_shape = (256, 256, 1)

input_img = Input(shape=(img_shape))

model = Conv2D(64, (3, 3), padding='same', name='conv1')(input_img)
model = Activation('relu', name='act1')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv2')(model)
model = Activation('relu', name='act2')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv3')(model)
model = Activation('relu', name='act3')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv4')(model)
model = Activation('relu', name='act4')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv5')(model)
model = Activation('relu', name='act5')(model)

model = Conv2D(64, (3, 3), padding='same', name='conv6')(model)
model = Activation('relu', name='act6')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv7')(model)
model = Activation('relu', name='act7')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv8')(model)
model = Activation('relu', name='act8')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv9')(model)
model = Activation('relu', name='act9')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv10')(model)
model = Activation('relu', name='act10')(model)

model = Conv2D(64, (3, 3), padding='same', name='conv11')(model)
model = Activation('relu', name='act11')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv12')(model)
model = Activation('relu', name='act12')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv13')(model)
model = Activation('relu', name='act13')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv14')(model)
model = Activation('relu', name='act14')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv15')(model)
model = Activation('relu', name='act15')(model)

model = Conv2D(64, (3, 3), padding='same', name='conv16')(model)
model = Activation('relu', name='act16')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv17')(model)
model = Activation('relu', name='act17')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv18')(model)
model = Activation('relu', name='act18')(model)
model = Conv2D(64, (3, 3), padding='same', name='conv19')(model)
model = Activation('relu', name='act19')(model)
model = Conv2D(1, (3, 3), padding='same', name='conv20')(model)
# model = Activation('relu', name='act20')(model)
res_img = model

output_img = add([res_img, input_img])

model = Model(input_img, output_img)

model.load_weights('checkpoints2/vdsr-200-32.21.hdf5')

pred = model.predict(data_input, batch_size=1)
sess = tf.InteractiveSession()
print(sess.run(PSNR(data_label, pred)))
print(data_label.shape)
print(data_input.shape)
print(tf.shape(pred))
y = np.reshape(data_label, [256, 256])
t = np.reshape(data_input, [256, 256])
c = np.reshape(pred, [256, 256])

# sio.savemat("yuantu.mat", {'yuan': y})
# sio.savemat("chongjian.mat", {'jian': c})

ax1 = plt.subplot(1, 3, 1)
plt.imshow(y, cmap='gray')
ax2 = plt.subplot(1, 3, 2)
plt.imshow(t, cmap='gray')
ax3 = plt.subplot(1, 3, 3)
plt.imshow(c, cmap='gray')
plt.show()



# path = "./kirby_NPY/KKI2009-33-MPRAGE.npy"
# file = np.array(np.load(path), dtype=np.float32)
# print("the shape of file is {}".format(file.shape))
# test_input = down_up_samples(file)
# test_input = np.array(test_input)
# test_label = file
# result = []
# for i in range(len(data_input)):
#     # tmp = test_input[i] / np.max(test_input[i])
#     tmp = data_input[i]
#     tmp_test = np.reshape(tmp, newshape=[1, len(data_input[i]), len(data_input[i][0]), 1])
#     pred = model.predict(tmp_test, batch_size=1)
#     result.append(pred)

# result = np.reshape(result, newshape=[170, 256, 256])
# sess = tf.InteractiveSession()
# score_psnr = PSNR(test_label, result)
# sums = 0
# psnr = []
# for i in range(len(data_label)):
#     # test_label[i] /= np.max(test_label[i])
#     print("psnr is {}".format(sess.run(PSNR(data_label[i], result[i]))))
#     sums += sess.run(PSNR(data_label[i], result[i]))
#     print()
#     psnr.append(sess.run(PSNR(data_label[i], result[i])))
# psnr = np.array(psnr)
# max_psnr = np.argmax(psnr)
# min_psnr = np.argmin(psnr)
# ax1 = plt.subplot(2, 3, 1)
# plt.imshow(test_label[max_psnr], cmap='gray')
# ax2 = plt.subplot(2, 3, 2)
# plt.imshow(result[max_psnr], cmap='gray')
# ax3 = plt.subplot(2, 3, 3)
# plt.imshow(test_label[min_psnr], cmap='gray')
# ax4 = plt.subplot(2, 2, 4)
# plt.imshow(result[min_psnr], cmap='gray')
# print("average is {}".format(sums/len(test_label)))
# plt.show()

# img = image.load_img('./patch.png', grayscale=True, target_size=(256, 256, 1))

# x = x.astype('float32') / 255
# x = np.expand_dims(x, axis=0)
#
# pred = model.predict(x)
#
# test_img = np.reshape(pred, (41, 41))
#
# imsave('test_img.png', test_img)