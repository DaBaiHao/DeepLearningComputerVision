#
from PIL import Image
import matplotlib.pyplot as plt
# import matplotlib.image as image
from PIL import Image
import scipy.ndimage as sp
import pdb
import os
import numpy as np
import tensorflow as tf
'''
#%%
img_dir = 'image/butterfly_GT.bmp'


image = Image.open(img_dir).convert('LA')
plt.imshow(image)
plt.show()
width, height = image.size
# pdb.set_trace() # check point


image_H = height*3
image_W = width*3


def resize_image(image):
    # image = tf.image.decode_image(image, channels=1)

    image = tf.image.resize_images(image,
                                   (image_H, image_W),
                                   method=tf.image.ResizeMethod.BICUBIC)
    #image = np.asarray(image.eval(), dtype='uint8')  # convert to uint8 to run
    return image
with tf.Session() as sess:
    # image = Image.open(img_dir)
    image = resize_image(image)
    image = sess.run(image)
    image = image[:, :, 0]
    plt.imshow(image)
    plt.show()

#%%
'''
# To load the pre-trained model named model.npy
pre_train_dir = 'model/model.npy'
model = np.load(pre_train_dir, encoding='latin1').item()

# pdb.set_trace()
# weight
conv1_w = model['w1']
conv2_w = model['w2']
conv3_w = model['w3']

# bias
conv1_b = model['b1']
conv2_b = model['b2']
conv3_b = model['b3']

# conv1 layer with biases: 64 filters with size 9 x 9
num_filters_conv1 = len(conv1_b)
size_filters_conv1 = len(conv1_w)

# conv2 layer with biases and relu: 32 filters with size 1 x 1
num_filters_conv2 = len(conv2_b)
size_filters_conv2 = len(conv2_w)

# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
num_filters_conv3 = len(conv3_b)
size_filters_conv3 = len(conv3_w)

# To show the value	of the 1st filter
print('The 1st filter in first convolutional layer is %.2f')
print(conv1_w[0])
# To show the bias of the 10th filter
print('The 10th bias in first convolutional layer is ')
print(conv1_b[9])

# To show the value	of the 5st filter
print('The 1st filter in first convolutional layer is ')
print(conv2_w[0])
# To show the bias of the 6th filter
print('The 10th bias in first convolutional layer is ')
print(conv2_b[0])

# To show the value	of the 1st filter
print('The 1st filter in first convolutional layer is ')
print(conv2_w[0])
# To show the bias of the 1th filter
print('The 10th bias in first convolutional layer is ')
print(conv2_b[0])
pdb.set_trace()
# check point

weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }