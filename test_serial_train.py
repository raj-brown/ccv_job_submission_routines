import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
#from mpi4py import MPI

np.random.seed(1234)
tf.set_random_seed(1234)

# Intialization of weights for Neural Netwroks
# Initial values of Biases are set to zeros.
def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0/(in_dim + out_dim))
    print ("Size for weight is ", size)
    return tf.Variable(tf.random_normal(shape=size, stddev = std))

# Neural Netwroks with Activation function wuth Tanh

def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

if __name__ == '__main__':
    #sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    N = 10001 # Number of trainning data
    x_data1 = np.linspace(-1, 0, N) #Training data for a particular rank.
    x_data2 = np.linspace(0, 1, N)
    x_data = [x_data1, x_data2]

    x_col=x_data
#   1D Input + 2 Layers with 16 Neurons Each + 1D Output
    layers = [1] + 2*[16] + [1]
    Li = len(layers)
    print("No. of layers are ", Li)
    Wi = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, Li)] ## Intializing the weightd for NN
    print ("Wi is ", Wi)
    bi = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, Li)] ## Initalizing biases by setting Zeros
    x_name = 'serial_x' # Variable name for training dictionary
    y_name = 'serial_y' # Varaible name for "     "   " for output
    x = tf.placeholder(tf.float32, shape=[None, 1], name=x_name) #Input Allocating the memory for input and output
    y = tf.placeholder(tf.float32, shape=[None, 1], name=y_name) # Output
    y_p = DNN(x, Wi, bi) # Prediction output from NN
    
    loss = tf.reduce_mean(tf.square(y_p - y)) # Error function
    
    train = tf.train.AdamOptimizer().minimize(loss) # Updating the Weights and Biases

    sess.run(tf.global_variables_initializer()) # Initializing the variable in tensorFlow

    #Feeding the training data to neural netwrok
    train_dict = {}
    x_name = 'serial_x'
    y_name = 'serial_y'
    x_col = np.reshape(x_col, [-1, 1])
    y_col = x_col**2
    train_dict['%s:0'%x_name] = x_col
    train_dict['%s:0'%y_name] = y_col

    Nmax = 10000 # Iteration counter
    start_time = time.perf_counter()
    start_clock = time.process_time()
    #train_dict = {x_train: X_col, y_train: Y_col} unhashable
    n = 0
# This part has to be offloaded to GPUs
    with tf.device('/device:GPU:0'):
        while n <= Nmax:
            y_pred, train_, loss_ = sess.run([y_p, train, loss], feed_dict=train_dict)
            n += 1
#        if n%100 == 0:
#            print('rank %d: n = %d, loss = %.3e'%(comm_rank, n, loss_))
    stop_time = time.perf_counter()
    stop_clock = time.process_time()
    print('Elapsed time (via time()): %f s'%(stop_time - start_time))
    print('Elapsed time (via clock()): %f s'%(stop_clock - start_clock))

