import numpy as np
import mnist
import math
import random
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt
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
epoch_size = 30
learning_rate = 0.002
W_1 = np.random.normal(0, 1.0/d, (M, d))
b_1 = np.random.normal(0, 1.0/d, (M, ))
W_2 = np.random.normal(0, 1.0/M, (C, M))
b_2 = np.random.normal(0, 1.0/M, (C, ))

# input layer(index番目の画像ファイルを正規化して1次元配列にして返す)
def input_layer(index):
  target_image = X[index]
  flattened_image = target_image.flatten()
  x = preprocessing.minmax_scale(flattened_image)
  return x

def reLU(x):
  if x >= 0:
    f = x
  else:
    f = 0
  return f

def d_reLU(x):
  if x >= 0:
    f = 1
  else:
    f = 0
  return f

with np.errstate(invalid='ignore'):
  univ_reLU = np.vectorize(reLU, otypes=[float])

univ_d_reLU = np.vectorize(d_reLU, otypes=[float])

# fully_connected_layer_1(1次元配列を受け取って、重み付けしてからシグモイド関数を適用する)
def fully_connected_layer_1(input_vector):
  global W_1, b_1
  y_1 = univ_reLU(np.dot(W_1, input_vector) + b_1)
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
  print(np.argmax(input_data))

def log_fun(x):
  return math.log(x)

# mini_batch
def mini_batch():
  global W_2, W_1, b_1, b_2
  i = input('Do you reuse parameter files?(Y/N) ')
  if i == 'Y':
    print('Yes')
    W_1 = np.load('W_1.npy')
    W_2 = np.load('W_2.npy')
    b_1 = np.load('b_1.npy')
    b_2 = np.load('b_2.npy')
  else:
    print('No')
  
  for e in range(epoch_size):
    epoch_error = 0
    for j in range(iters_number):
      random_list = np.array(random.sample(range(60000),B))
      error = 0
      dE_da = np.empty((C, 0), dtype=np.float32)
      X_2 = np.empty((M, 0), dtype=np.float32)
      X_1 = np.empty((d, 0), dtype=np.float32)
      for i in range(B):
        index = random_list[i]
        output_l = np.array(fully_connected_layer_2(fully_connected_layer_1(input_layer(index))))
        after_sigmoid = np.array(fully_connected_layer_1(input_layer(index))).reshape(M,1)
        before_sigmoid = np.array(input_layer(index)).reshape(d,1)
        y = Y[index]
        one_hot = np.array([0] * C)
        one_hot[y] = 1
        new_func = np.frompyfunc(log_fun, 1, 1)
        z = new_func(output_l)
        error -= np.dot(z, one_hot)
        new_l = np.array(output_l - one_hot).reshape(C,1)
        dE_da =  np.append(dE_da, new_l, axis=1)
        X_2 = np.append(X_2, after_sigmoid, axis=1)
        X_1 = np.append(X_1, before_sigmoid, axis=1)
      E = error/B
      epoch_error += E
      dE_dX = np.dot(W_2.T, dE_da)
      dE_dW_2 = np.dot(dE_da, X_2.T)
      dE_db_2 = np.sum(dE_da, axis=1)
      
      dE_da2 = dE_dX * univ_d_reLU(X_2)
      #print(X_2.shape)
      dE_dX = np.dot(W_1.T, dE_da2)
      dE_dW_1 = np.dot(dE_da2, X_1.T)
      dE_db_1 = np.sum(dE_da2, axis=1)

      W_1 = W_1 - learning_rate * dE_dW_1
      W_2 = W_2 - learning_rate * dE_dW_2
      b_1 = b_1 - learning_rate * dE_db_1
      b_2 = b_2 - learning_rate * dE_db_2

    print(e+1)
    print(epoch_error/iters_number)
    plt.plot(e,epoch_error/iters_number,"b",marker='.')
    plt.title('Cross-Entropy Error',fontsize=15)
    plt.xlabel('epoch',fontsize=10)
    plt.ylabel('cross-entropy error',fontsize=10)
  plt.show()


def final_input():
  i2 = input('Do you save parameter files?(Y/N) ')
  if i2 == 'Y':
    np.save('W_1',W_1)
    np.save('W_2',W_2)
    np.save('b_1',b_1)
    np.save('b_2',b_2)
    print("Parameter files was saved.")
  else:
    print("Parameter files was not saved.")  

mini_batch()
final_input()
