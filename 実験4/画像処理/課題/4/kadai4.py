import numpy as np
import mnist
import math
import random
import time
from numpy.core.fromnumeric import shape
from scipy.special import expit
from sklearn import preprocessing
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz") 

number_of_image = 10000
C = 10
size_of_image = 28
M = 50
d = size_of_image*size_of_image
B = 100
iters_number = 600
learning_rate = 0.01
W_1 = np.load('W_1.npy')
W_2 = np.load('W_2.npy')
b_1 = np.load('b_1.npy')
b_2 = np.load('b_2.npy')

def input_number():
  global number_of_image
  while(1):
    i = input('Input number(0~9999)!!!>>> ')
    if i.isdecimal():
     i = int(i)
     if 0 <= i and i < number_of_image:
      np.random.seed(seed=i)
      print("True number is ", Y[i])
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
  global W_1, b_1
  y_1 = expit(np.dot(W_1, input_vector) + b_1)
  return y_1

# softmax(ソフトマックス関数)
def softmax(input_data):
  alpha = np.amax(input_data)
  return np.exp(input_data-alpha)/np.sum(np.exp(input_data-alpha))

# fully_connected_layer_2(1次元配列を受け取って、重み付けしてからソフトマックス関数を適用する)
def fully_connected_layer_2(input_vector):
  global W_2, y_2, W_1, b_2
  y_2 = softmax(np.add(np.dot(W_2, input_vector), b_2))
  return y_2

# output(配列の最大値のインデックスを返す)
def output_answer(input_data):
  return np.argmax(input_data)

estimated_ans = output_answer(fully_connected_layer_2(fully_connected_layer_1(input_layer(input_number()))))
print("Estimated answer is ", estimated_ans)


def show_accuracy():
  true_cnt = 0
  false_cnt = 0
  for i in range(60000):
    print("\r"+str(i+1),end="/60000...")
    time.sleep(0.00)
    estimated_ans = output_answer(fully_connected_layer_2(fully_connected_layer_1(input_layer(i))))
    real_ans = Y[i]
    if estimated_ans == real_ans:
      true_cnt += 1
    else:
      false_cnt += 1
  print("")
  print("Accuracy is ", true_cnt/(true_cnt+false_cnt))

def question():
  i = input('Do you want to calculate accuracy?(Y/N).It will take some time.>>> ')
  if i == 'Y':
    show_accuracy()
  else:
    print("OK. See you.")

question()