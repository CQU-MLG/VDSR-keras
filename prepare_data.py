import os
import glob
import numpy as np
import nibabel
from PIL import Image
from scipy import ndimage
import pywt



# 读取bmp文件
def read_bmp():
    test_path = './ScSR/Data/Testing/*.bmp'
    path = './ScSR/Data/Training/*.bmp'
    data = []
    for file_name in glob.glob(test_path):
        img = Image.open(file_name).convert('L')
        img = np.array(img)
        data.append(img)
    print(np.array(data).shape)
    return data
# 读取文件并转化为npy格式

def nii_npy():
    path = './HCP_NPY'
    if not os.path.exists(path):
        os.mkdir(path)
    for file_name in glob.glob('./HCP/*.nii'):
        file = nibabel.load(file_name).get_data()
        print(file.shape)
        npyfilename_ = file_name.split('/')[-1].split('.')[0]
        npyfilename = npyfilename_ + '.npy'
        full_path = path + '/' + npyfilename
        np.save(full_path, file)
        print('File ' + npyfilename_ + ' is saved in ' + full_path + ' .')



# 读取npy格式文件
def read_npy():
    npy_path = './kirby_NPY/*.npy'
    # npy_path = './HCP_NPY/*.npy'
    x = []
    for file_name in glob.glob(npy_path):
        file = np.array(np.load(file_name), dtype=np.float32)
        print("the shape of data is {}".format(file.shape))
        for i in range(20, len(file)-20):
            x.append(file[i][20:236, 20:236])
    x = np.array(x)
    print("The number of data is {}".format(x.shape[0]))
    return x


# # 增大数据量(没写完)（调用读取函数并增强，也可以将增强后的图像保存）
# def data_augment(data):
#     over_lap_rate = 2
#     x_data = []
#     x_data = np.array(x_data)
#     return x_data

# 先高斯模糊而后上下采样（调用数据增强和高斯函数）
def down_up_samples(data, scale=2):
    data1 = []
    count = 0
    for tmp in data:
        count += 1
        print("%d-th" % count)
        guss = ndimage.filters.gaussian_filter(tmp, sigma=scale)
        gussdownsample = ndimage.zoom(guss, 1/scale, order=3)
        gussupsample = ndimage.zoom(gussdownsample, scale, order=3)
        data1.append(gussupsample)
    return data1


# 小波变换（）
def wavelet_transform(data):
    coeffs = pywt.dwt2(data, "haar")
    cA, (cH, cV, cD) = coeffs

    # 拼接4副图像
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    after_wavelet = np.concatenate([AH, VD], axis=0)

    return after_wavelet

# 归一化处理
def Nomalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    for i in range(len(data)):
        for j in range(len(data)):
            data[i][j] = (float(data[i][j])-m) / (mx - mn)
    return data
# 将图片切片(先上下采样然后再切片)
def array_to_patch(strid, inputs, label, image_size):
    input_patches = []
    label_patches = []
    # 二维图像
    for k in range(len(inputs)):
        h = len(inputs[k])
        w = len(inputs[k][0])
        for i in range(0, h-image_size, strid):
            for j in range(0, w-image_size, strid):
                tmp_input = inputs[k][i:i+image_size, j:j+image_size]
                tmp_label = label[k][i:i+image_size, j:j+image_size]
                input_patches.append(tmp_input)
                label_patches.append(tmp_label)

    input_patches = np.array(input_patches, dtype=np.float)
    label_patches = np.array(label_patches, dtype=np.float)
    # max_input = np.max(input_patches)
    # max_label = np.max(label_patches)
    # print(max_input)
    # print(max_label)
    for i in range(len(input_patches)):
        max_input = np.max(input_patches[i])
        if max_input == 0:
            max_input += 1
        input_patches[i] /= 255
    print("input data is ready")
    for i in range(len(label_patches)):
        max_label = np.max(label_patches[i])
        if max_label == 0:
            max_label += 1
        label_patches[i] /= 255
    print("label data is ready")
    input_patches = np.reshape(input_patches, newshape=[len(input_patches), image_size, image_size, 1])
    label_patches = np.reshape(label_patches, newshape=[len(label_patches), image_size, image_size, 1])
    print("the shape of train data is {}".format(input_patches.shape))
    return input_patches, label_patches





















# 插值法
#
# from scipy import ndimage as scn
# import numpy as np
#
# x = np.array((range(16))).reshape(4, 4)
# print(x)
# print('\n')
# y = scn.zoom(x, 0.5, order=3)
# print(y)
