import numpy as np
import mnist
import math
import random
import time
from numpy.core.fromnumeric import shape
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import triu_indices_from
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
learning_rate = 0.001
W_1 = np.random.normal(0, 1.0/d, (M, d))
b_1 = np.random.normal(0, 1.0/d, (M, ))
W_2 = np.random.normal(0, 1.0/M, (C, M))
b_2 = np.random.normal(0, 1.0/M, (C, ))
rho = 0.1

# input layer(index番目の画像ファイルを正規化して1次元配列にして返す)
def input_layer(index):
  target_image = X[index]
  flattened_image = target_image.flatten()
  x = preprocessing.minmax_scale(flattened_image)
  return x

# fully_connected_layer_1(1次元配列を受け取って、重み付けしてからdropoutを適用する)
def fully_connected_layer_1(input_vector):
  global W_1, b_1, true_or_false
  ratio = np.random.rand(M)
  true_or_false = ratio > rho
  y_1 = true_or_false *( (np.dot(W_1, input_vector) + b_1))
  return y_1

def fully_connected_layer_1_TEST(input_vector):
  global W_1, b_1
  W_1 = np.load('W_1.npy')
  b_1 = np.load('b_1.npy')
  y_1 = (1-rho) * (np.dot(W_1, input_vector) + b_1)
  return y_1

# softmax(ソフトマックス関数)
def softmax(input_data):
  alpha = np.amax(input_data)
  return np.exp(input_data-alpha)/np.sum(np.exp(input_data-alpha))

# fully_connected_layer_2(1次元配列を受け取って、重み付けしてからソフトマックス関数を適用する)
def fully_connected_layer_2(input_vector):
  global W_2, y_2, b_2
  y_2 = softmax(np.add(np.dot(W_2, input_vector), b_2))
  return y_2

def fully_connected_layer_2_TEST(input_vector):
  global W_2, y_2, b_2
  W_2 = np.load('W_2.npy')
  b_2 = np.load('b_2.npy')
  y_2 = softmax(np.add(np.dot(W_2, input_vector), b_2))
  return y_2

# output(配列の最大値のインデックスを返す)
def output_answer(input_data):
  return np.argmax(input_data)

def log_fun(x):
  return math.log(x)

# mini_batch
def mini_batch():
  global W_2, W_1, b_1, b_2, true_or_false
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
      d_X_2 = np.empty((M, 0), dtype=np.float32)
      X_1 = np.empty((d, 0), dtype=np.float32)
      for i in range(B):
        index = random_list[i]
        output_l = np.array(fully_connected_layer_2(fully_connected_layer_1(input_layer(index))))
        after_sigmoid = np.array(fully_connected_layer_1(input_layer(index))).reshape(M,1)

        z = fully_connected_layer_1(input_layer(index))*true_or_false
        zz = np.where(z == 0, 0, 1)
        dropout = np.array(zz.reshape(M,1))
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
        d_X_2 = np.append(d_X_2, dropout, axis=1)
        X_1 = np.append(X_1, before_sigmoid, axis=1)
      E = error/B
      epoch_error += E
      dE_dX = np.dot(W_2.T, dE_da)
      dE_dW_2 = np.dot(dE_da, X_2.T)
      dE_db_2 = np.sum(dE_da, axis=1)
      
      dE_da2 = dE_dX * d_X_2
      
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

def show_accuracy():
  true_cnt = 0
  false_cnt = 0
  pro_size = 60000
  for i in range(pro_size):
    estimated_ans = output_answer(fully_connected_layer_2_TEST(fully_connected_layer_1_TEST(input_layer(i))))
    real_ans = Y[i]
    print("\r"+str(i+1),end="/60000...")
    time.sleep(0.00)
    if estimated_ans == real_ans:
      true_cnt += 1
    else:
      false_cnt += 1
  print("")
  print("Test is done.")
  print("Hit : ", true_cnt)
  print("Miss :", false_cnt)
  print("Accuracy is ", true_cnt/(true_cnt+false_cnt),".")

def first_question():
  while(True):
    i = input('Learn by Dropout algorithm, or test accuracy?(LEARN or TEST) ')
    if i == 'LEARN':
      mini_batch()
      final_input()
      break
    elif i == 'TEST':
      show_accuracy()
      break
    else:
      print("Try Again.") 

first_question()

