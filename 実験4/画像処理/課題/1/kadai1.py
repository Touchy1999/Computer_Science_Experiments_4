import numpy as np
import mnist
from numpy.core.fromnumeric import shape
from scipy.special import expit
from sklearn import preprocessing
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz") 

# sigmoid function(シグモイド関数だけど、使ってない)
def sigmoid(x):
  return 1/(1+np.exp(-x))

number_of_image = 10000
C = 10
size_of_image = 28
M = 100
d = size_of_image*size_of_image

# prepare for input number(入力を受け取る)
def input_number():
  global number_of_image
  while(1):
    i = input('Input number(0~9999)!!!>>> ')
    if i.isdecimal():
     i = int(i)
     if 0 <= i and i < number_of_image:
      np.random.seed(seed=i)
      return i
     else:
      pass

# input layer(index番目の画像ファイルを正規化して1次元配列にして返す)
def input_layer(index):
  target_image = X[index]
  flattened_image = target_image.flatten()
  x = preprocessing.minmax_scale(flattened_image)
  return x

# fully_connected_layer_1(1次元配列を受け取って、重み付けしてからシグモイド関数を適用する)
def fully_connected_layer_1(input_vector):
  W_1 = np.random.normal(0, 1.0/len(input_vector), (M, len(input_vector)))
  b_1 = np.random.normal(0, 1.0/len(input_vector), (M, ))
  y_1 = expit(np.dot(W_1, input_vector) + b_1)
  return y_1

# softmax(ソフトマックス関数)
def softmax(input_data):
  alpha = np.amax(input_data)
  return np.exp(input_data-alpha)/np.sum(np.exp(input_data-alpha))

# fully_connected_layer_2(1次元配列を受け取って、重み付けしてからソフトマックス関数を適用する)
def fully_connected_layer_2(input_vector):
  W_2 = np.random.normal(0, 1.0/len(input_vector), (C, len(input_vector)))
  b_2 = np.random.normal(0, 1.0/len(input_vector), (C, ))
  y_2 = softmax(np.add(np.dot(W_2, input_vector), b_2))
  return y_2

# output(配列の最大値のインデックスを返す)
def output_answer(input_data):
  print(np.argmax(input_data))

output_answer(fully_connected_layer_2(fully_connected_layer_1(input_layer(input_number()))))