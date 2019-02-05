#%%
import time
import os
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
import scipy
import pdb

#%%
def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers 
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
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

"""Define the model layers with three convolutional layers
"""
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + biases['b1'])

##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b3']


"""Load the pre-trained model file
"""
model_path='./model/model.npy'
model = np.load(model_path, encoding='latin1').item()

##------ Add your code here: show the weights of model and try to visualisa
# variabiles (w1, w2, w3)
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


"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groudtruth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
ouput_ = sess.run(conv3, feed_dict={inputs: input_})
ouput_ = ouput_.squeeze()
import cv2
ouput_ = cv2.resize(ouput_, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)


##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'
scipy.misc.imsave('blurred_image.jpg', blurred_image)
scipy.misc.imsave('SR_image.jpg', ouput_)

import skimage.measure as sk
psnr_ = sk.compare_psnr(groudtruth_image, ouput_)
print(psnr_)



