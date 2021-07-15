import sys
sys.path.insert(0, '../PINNs-master/Utilities/')

import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, combinations
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random


nnn = 1349

np.random.seed(nnn)
tf.random.set_seed(nnn)
loss_graph =[]
val_graph = []
u_graph = []
f_graph = []

class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, val_set):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.y_u = X_u[:, 1:2]
        self.t_u = X_u[:, 2:3]

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.t_f = X_f[:, 2:3]

        self.u = u

        self.layers = layers

        self.x_val = val_set[:, 0:1]
        self.y_val = val_set[:, 1:2]
        self.t_val = val_set[:, 2:3]
        self.u_val = val_set[:, 3:4]
        
        self.n = 1

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.x_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.y_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_u.shape[1]])
        self.t_u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf)

        # Loss
        self.loss_u = (tf.reduce_mean(tf.square(self.u_tf - self.u_pred)))
        self.loss_f = (tf.reduce_mean(tf.square(self.f_pred)))
        self.loss = self.loss_u + self.loss_f

        # Validation
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.y_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_val.shape[1]])
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.u_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])

        self.u_val_pred = self.net_u(self.x_val_tf, self.y_val_tf, self.t_val_tf)
        self.f_val_pred = self.net_f(self.x_val_tf, self.y_val_tf, self.t_val_tf)

        self.loss_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.optimizer = self.opt.minimize(self.loss)

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return 1800*Y

    def net_u(self, x, y, t):
        u = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        return u
    
    def net_f(self, x, y, t):
        mu = 0.001
        Ct = 11e-9
        phi = 0.2
        B = 0.9
        k = 200*9.8692*10**(-16)

        nu = [k/(mu*B), 20/(24*60*60)/1000, (phi*Ct)/B]
        
        Q = get_Q(x, y)

        u = self.net_u(x, y, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_x = nu[0] * u_x
        u_y = nu[0] * u_y
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        f = u_xx + u_yy + Q * nu[1] - nu[2] * u_t
        return f*1E11

    def callback(self, loss, loss_val, loss_u, loss_f):
        print('It: %d, Loss: %e, Loss_val: %e' % (self.n, loss, loss_val))
        if self.n % 50 == 0:
            val_graph.append([self.n, loss_val])
            u_graph.append([self.n, loss_u])
            f_graph.append([self.n, loss_f])
        self.n = self.n+1

    def train(self,nIter):

        tf_dict = {self.x_u_tf: self.x_u, self.y_u_tf: self.y_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f,
                   self.x_val_tf: self.x_val, self.y_val_tf: self.y_val, self.t_val_tf: self.t_val, self.u_val_tf: self.u_val}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                val_loss = self.sess.run(self.loss_val, tf_dict)
                u_loss = self.sess.run(self.loss_u, tf_dict)
                f_loss = self.sess.run(self.loss_f, tf_dict)
                loss_graph.append([it, loss_value])
                val_graph.append([it, val_loss])
                u_graph.append([it, u_loss])
                f_graph.append([it, f_loss])
                print('It: %d, Loss: %.3e, Loss_val: %.3e, Loss_u: %.3e,Loss_f: %.3e, Time: %.2f' %(it, loss_value, val_loss,u_loss,f_loss, elapsed))

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.y_u_tf: X_star[:, 1:2], self.t_u_tf: X_star[:, 2:3]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.y_f_tf: X_star[:, 1:2], self.t_f_tf: X_star[:, 2:3]})

        return u_star, f_star


def plot_solution(X_star, u_star, index, string):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.title(string, fontsize=12)

def axisEqual3D(ax):
    extents = np.array(
        [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])  # getattr, object에 존재하는 속성 값 가져오는 함수
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}.jpg'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))

def get_Q(x,y):
    s1 = np.array(((x == 5) & (y == 5)), dtype=np.int)
    s2 = np.array(((x == 65) & (y == 65)), dtype=np.int)
    Q = s1-s2
    return Q

if __name__ == "__main__":
    layers = [3, 49*10, 1]

    data = scipy.io.loadmat('./Data/pressure.mat')

    t = np.arange(0, 30*60*4, 30*60)
    t = np.reshape(t, [-1, 1])
    x = data['x']  # [[1], [2], [3], ..., [n_x]]
    y = data['y']  # [[1], [2], [3], ..., [n_y]]
    
    Exact = data['usol'].T # [[P1, ..., Pn_x*n_y]_t0, ..., [P1, Pn_x*n_y]_tn] 
    Exact = Exact*1000 # kPa--> Pa
    
    n_t = np.shape(t)[0]
    n_x = 7
    n_y = 7
    X, Y, T = np.meshgrid(x, y, t)

    X_star = np.hstack((np.tile(x, (n_x*n_t,1)).flatten()[:,None],(np.tile(y,(n_t,n_y))).flatten()[:, None],sorted(T.flatten()[:, None])))
    u_star = Exact.flatten()[:, None]
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    ### 초기 조건
    # t_0의 전좌표에서의 압력 값
    xx1 = np.hstack((np.tile(x, (n_x, 1)).flatten()[:, None], np.tile(y, (1, n_y)).flatten()[:, None],np.tile(t[0:1], (n_x*n_y, 1)).flatten()[:, None]))  # 초기 1개의 시간에 대한 모든 좌표에서의 압력값
    uu1 = (Exact[0:1, :].T).flatten()[:, None]
    # t_1의 전좌표에서의 압력 값
    xx2 = np.hstack((np.tile(x, (n_x, 1)).flatten()[:, None], np.tile(y, (1, n_y)).flatten()[:, None],np.tile(t[1:2], (n_x*n_y, 1)).flatten()[:, None]))  # 초기 1개의 시간에 대한 모든 좌표에서의 압력값
    uu2 = (Exact[1:2, :].T).flatten()[:, None]

    ### 경계조건 2
    # u(0, 0, t)의 좌표와 압력값이 xx7 & uu7 (Inject Point)
    xx7 = np.hstack((np.tile(x[0],(1,n_t)).flatten()[:, None], np.tile(y[0],(1,n_t)).flatten()[:, None], t[0:n_t].flatten()[:, None]))
    uu7 = (Exact[0:n_t, 0:1].T).flatten()[:, None]
    # u(n_x, n_y, t)의 좌표와 압력값이 xx8 & uu8 (Produce Point)
    xx8 = np.hstack((np.tile(x[n_x-1],(1,n_t)).flatten()[:, None], np.tile(y[n_y-1],(1,n_t)).flatten()[:, None], t[0:n_t].flatten()[:, None]))
    uu8 = (Exact[0:n_t, -1:].T).flatten()[:, None] #u(end,end,t)

    N_u = n_x*n_y*1
    N_f = n_x*n_y*150
    
    X_u_train = np.vstack([xx1])
    X_f_train = X_star[:, :]
    u_train = np.vstack([uu1])


    ### training data validation data 8:2
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)

    idx_train = idx[0:round(N_u*0.8)]
    idx_val = idx[round(N_u*0.8):N_u]

    X_u_val = X_u_train[idx_val, :]
    u_val = u_train[idx_val, :]

    X_u_train = X_u_train[idx_train, :]
    u_train = u_train[idx_train, :]

    val_set = np.hstack([X_u_val, u_val])

    X_u_train = np.vstack([xx1, xx7,xx8])
    u_train = np.vstack([uu1,uu7,uu8])


    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, val_set)

    start_time = time.time()
    model.train(100000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, f_pred = model.predict(X_star)

    # error_u_norm = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    # error_u = np.linalg.norm(u_star - u_pred, 2)
    # error_f = np.linalg.norm(f_pred, 2)
    # print('Error u_norm %e' % (error_u_norm))
    # print('Error u: %e' % (error_u))
    # print('f_pred: %e' % (error_f))

    ### PLOT the Results
    plot_solution(X_star[(n_x*n_y*(n_t-1)):,0:2], (u_pred[(n_x*n_y*(n_t-1)):])/1000, 0, 'Predicted Pressure') # Predicted
    plot_solution(X_star[(n_x*n_y*(n_t-1)):,0:2], (u_star[(n_x*n_y*(n_t-1)):(n_x*n_y*n_t)])/1000, 1, 'True Pressure') # True
    
    plt.figure()
    plt.title('validation error', fontsize=12)
    val_np = np.array(val_graph)
    plt.plot(val_np[:, 0], val_np[:, 1])
    plt.xlabel('total iteration number')
    plt.ylabel('Loss')
    # plt.show(block=False)
    plt.show(block=False)

    plt.figure()
    plt.title('point u error', fontsize=12)
    u_np = np.array(u_graph)
    plt.plot(u_np[:, 0], u_np[:, 1])
    plt.xlabel('total iteration number', fontsize=10)
    plt.ylabel('Data based Loss', fontsize=10)
    # plt.show(block=False)
    plt.show(block=False)

    plt.figure()
    plt.title('PDE f error', fontsize=12)
    f_np = np.array(f_graph)
    plt.plot(f_np[:, 0], f_np[:, 1])
    plt.xlabel('total iteration number', fontsize=10)
    plt.ylabel('Physics based Loss', fontsize=10)
    # plt.show(block=False)
    plt.show()


