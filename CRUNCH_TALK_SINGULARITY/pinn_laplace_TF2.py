import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

np.random.seed(seed=1234)
tf.random.set_seed(1234)
tf.config.experimental.enable_tensor_float_32_execution(False)
#os.environ[‘TF_ENABLE_AUTO_MIXED_PRECISION’] = ‘1’



# Initalization of Network
def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal(shape=size, stddev = std))

# Neural Network 

def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

def train_vars(W, b):
    return W + b

@tf.function(jit_compile=True)
#@tf.function
def pdenn(X, W, b):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            u=DNN(X, W, b)
        u_x = tape1.gradient(u, X)
        del tape1
    u_xx = tape2.gradient(u_x, X)
    del tape2
    f = 4*tf.sin(2*np.pi*X)*np.pi*np.pi
    R = u_xx + f
    return R


@tf.function(jit_compile=True)
#@tf.function
def pdenn_first(X, W, b):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(X)
        u=DNN(X, W, b)
    u_x = tape1.gradient(u, X)
    del tape1
    f = 2*tf.cos(2*np.pi*X)*np.pi
    R = u_x - f
    return R


@tf.function(jit_compile=True)
#@tf.function
def train_step(W, b, x_0_train_tf, x_1_train_tf, x_train_tf, y_0_train_tf, y_1_train_tf, opt, order):
    with tf.GradientTape() as tape:
        tape.watch([W,b])
        y_0_nn = DNN(x_0_train_tf, W, b) 
        y_1_nn = DNN(x_1_train_tf, W, b)
        if order == 1: 
            R_nn = pdenn_first(x_train_tf, W, b)
        elif order == 2:
            R_nn = pdenn(x_train_tf,W,b)
        else:
            R_nn = pdenn(x_train_tf, W, b)
        y_pred = DNN(x_train_tf, W, b)
        loss =  tf.reduce_mean(tf.square(y_0_nn - y_0_train_tf)) + tf.reduce_mean(tf.square(y_1_nn - y_1_train_tf)) + tf.reduce_mean(tf.square(R_nn))
    grads = tape.gradient(loss, train_vars(W,b))
    opt.apply_gradients(zip(grads, train_vars(W,b)))
    return loss, y_pred



if __name__ == '__main__':
    N = 101 
    x_col = np.linspace(-1, 1, N).reshape((-1, 1))
    x_0 = np.array([-1]).reshape((-1, 1))
    x_1 = np.array([1]).reshape((-1, 1))
    y_0 = np.sin(2*np.pi*x_0)
    y_1 = np.sin(2*np.pi*x_1)
    layers = [1] + 4*[32] + [1]
    L = len(layers)
    W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
    b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)] 
    x_train_tf =   tf.convert_to_tensor(x_col, dtype=tf.float32)
    x_0_train_tf = tf.convert_to_tensor(x_0, dtype=tf.float32)
    y_0_train_tf = tf.convert_to_tensor(y_0, dtype=tf.float32)
    x_1_train_tf = tf.convert_to_tensor(x_1, dtype=tf.float32)
    y_1_train_tf = tf.convert_to_tensor(y_1, dtype=tf.float32)
    lr = 0.0001
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    Nmax = 30000 # Iteration counter
    n = 0
    #u=DNN(x_train_tf, W, b)
    #R = pdenn(x_train_tf, W, b)
    #print(f"x_tf: {u}")
    order = 1
    t1 = time.time()
    while n <= Nmax:
        t1 = time.time()
        loss_, y_pred = train_step(W, b, x_0_train_tf, x_1_train_tf, x_train_tf, y_0_train_tf, y_1_train_tf,optimizer, 2)
        t2 = time.time()
        print(f"Iteration is: {n} and loss is: {loss_}")
        n += 1
    print(f"Training took: {time.time() - t1} sec.")
    
    N_plot = 201
    xplot = np.linspace(-1, 1, N_plot).reshape((-1, 1))
    x_plot_tf = tf.convert_to_tensor(xplot, dtype=tf.float32) 
    y_pred_ = DNN(x_plot_tf, W, b)
    y_act = np.sin(2*np.pi*xplot)
    plt.plot(xplot, y_pred_, '-r', label="PINN Predicted")
    plt.plot(xplot, y_act, 'ob', label = "Actual Solution", markersize=4)
    plt.legend()
    plt.savefig("solutions_plot.png")
