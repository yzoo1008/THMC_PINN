import sys

sys.path.insert(0, '../PINNs-master/Utilities/')

import tensorflow as tf
import os

tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
loss_graph = []
val_graph = []
u_graph = []
f_graph = []


class PhysicsInformedNN:
    def __init__(self, X_p, P, X_f, layers, lb, ub, X_val, P_val):

        self.lb = lb
        self.ub = ub

        self.x_p = X_p[:, 0:1]
        self.t_p = X_p[:, 1:2]
        self.p = P

        self.p_lb = P.min()
        self.p_ub = P.max()

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.layers = layers

        self.x_val = X_val[:, 0:1]
        self.t_val = X_val[:, 1:2]
        self.p_val = P_val

        self.n = 1

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.x_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_p.shape[1]])
        self.t_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_p.shape[1]])
        self.p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p.shape[1]])

        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.p_pred = self.net_p(self.x_p_tf, self.t_p_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss_p = (tf.reduce_mean(tf.square(self.p_tf - self.p_pred)))
        self.loss_f = (tf.reduce_mean(tf.square(self.f_pred)))
        self.loss = self.loss_p + self.loss_f
        # self.loss = self.loss_f

        # Validation
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.p_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_val.shape[1]])

        self.p_val_pred = self.net_p(self.x_val_tf, self.t_val_tf)
        self.f_val_pred = self.net_f(self.x_val_tf, self.t_val_tf)

        self.loss_val = tf.reduce_mean(tf.square(self.p_val_tf - self.p_val_pred))

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                                    use_locking=False, name='Adam')
        self.optimizer = self.opt.minimize(self.loss)

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                                               epsilon=1e-08, use_locking=False, name='Adam')
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
        return Y
        # return Y*10E4

    def net_p(self, x, t):
        p = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        p = (p + 1) / 2 * (self.p_ub - self.p_lb) + self.p_lb
        # p = p * (self.p_ub - self.p_lb) + self.p_lb
        return p

    def net_f(self, x, t):
        mu = 0.001
        Ct = 11e-9
        phi = 0.2
        B = 0.9
        k = 200 * 9.8692 * 10 ** (-16)

        nu = [k / (mu * B), (phi * Ct) / B]

        p = self.net_p(x, t)
        p_t = tf.gradients(p, t)[0]
        p_x = tf.gradients(p, x)[0]
        p_x = nu[0] * p_x
        p_xx = tf.gradients(p_x, x)[0]

        f = p_xx - nu[1] * p_t
        return f * 1E12

    def callback(self, loss, loss_val, loss_u, loss_f):
        print('It: %d, Loss: %e, Loss_val: %e' % (self.n, loss, loss_val))
        if self.n % 50 == 0:
            val_graph.append([self.n, loss_val])
            u_graph.append([self.n, loss_u])
            f_graph.append([self.n, loss_f])
        self.n = self.n + 1

    def train(self, nIter):
        tf_dict = {self.x_p_tf: self.x_p, self.t_p_tf: self.t_p, self.p_tf: self.p,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   self.x_val_tf: self.x_val, self.t_val_tf: self.t_val, self.p_val_tf: self.p_val}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                val_loss = self.sess.run(self.loss_val, tf_dict)
                u_loss = self.sess.run(self.loss_p, tf_dict)
                f_loss = self.sess.run(self.loss_f, tf_dict)
                loss_graph.append([it, loss_value])
                val_graph.append([it, val_loss])
                u_graph.append([it, u_loss])
                f_graph.append([it, f_loss])
                print('It: %d, Loss: %.3e, Loss_val: %.3e, Loss_u: %.3e,Loss_f: %.3e, Time: %.2f' % (
                it, loss_value, val_loss, u_loss, f_loss, elapsed))

    def predict(self, X_star):
        p_star = self.sess.run(self.p_pred,
                               {self.x_p_tf: X_star[:, 0:1], self.t_p_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred,
                               {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return p_star, f_star


def plot_solution(X_star, p_star, index, string):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 100
    x = np.linspace(lb[0], ub[0], nn)
    X, Y = np.meshgrid(x, [1])

    X_pts = np.array(X_star[:, 0:1].flatten()[:, None])
    X = np.tile(X_pts, (1, 2))

    Y_pts_0 = np.tile([0], (np.shape(X_star)[0], 1))
    Y_pts_1 = np.tile([1], (np.shape(X_star)[0], 1))
    Y = np.hstack([Y_pts_0, Y_pts_1])

    P_pts = np.reshape(p_star.flatten(), (-1, 1))
    P = np.tile(P_pts, (1, 2))

    plt.figure(index)
    plt.pcolor(X, Y, P, cmap='jet')
    plt.colorbar()
    plt.title(string, fontsize=12)
    plt.show()

def axisEqual3D(ax):
    extents = np.array(
        [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])  # getattr, object에 존재하는 속성 값 가져오는 함수
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def figsize(scale, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def newfig(width, nplots=1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    if crop == True:
        plt.savefig('{}.jpg'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))


if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('./Data/1D_Dirichlet.mat')

    x = data['x']  # [[1], [2], [3], ..., [n_x]]
    t = data['t']
    p = data['usol'].T  # [[P1, ..., Pn_x*n_y]_t0, ..., [P1, Pn_x*n_y]_tn]

    n_t = np.shape(p)[0]
    n_x = np.shape(p)[1]
    X, T = np.meshgrid(x, t)

    x_tile = np.tile(x, (n_t, 1)).flatten()[:, None] # (n_x * n_t, 1)
    t_tile = T.flatten()[:, None]
    X_star = np.hstack([x_tile, t_tile])
    P_star = p.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    ### 초기 조건
    # t_0의 전좌표에서의 압력 값
    xx1 = np.hstack([X[0:1, :].flatten()[:, None], T[0:1, :].flatten()[:, None]])
    pp1 = p[0:1, :].flatten()[:, None]
    # t_1의 전좌표에서의 압력 값
    xx2 = np.hstack([X[0:1, :].flatten()[:, None], T[1:2, :].flatten()[:, None]])
    pp2 = p[1:2, :].flatten()[:, None]

    ### 경계 조건
    # p(0, t)
    xx3 = np.hstack([X[:, 0:1].flatten()[:, None], T[:, 0:1].flatten()[:, None]])
    pp3 = p[:, 0:1].flatten()[:, None]
    # p(t, 0)
    xx4 = np.hstack([X[:, -1:].flatten()[:, None], T[:, -1:].flatten()[:, None]])
    pp4 = p[:, -1:].flatten()[:, None]

    N_u = n_x * 4
    N_f = n_x * 150

    X_p_train = np.vstack([xx1, xx2, xx3, xx4])
    P_p_train = np.vstack([pp1, pp2, pp3, pp4])
    X_f_train = X_star[:, :]

    # training data validation data 8:2
    idx = np.random.choice(X_p_train.shape[0], N_u, replace=False)

    idx_train = idx[0:round(N_u * 0.8)]
    idx_val = idx[round(N_u * 0.8):N_u]

    X_p_tr = X_p_train[idx_train, :]
    P_tr = P_p_train[idx_train, :]

    X_p_val = X_p_train[idx_val, :]
    P_val = P_p_train[idx_val, :]

    val_set = np.hstack([X_p_val, P_val])

    model = PhysicsInformedNN(X_p_tr, P_tr, X_f_train, layers, lb, ub, X_p_val, P_val)

    start_time = time.time()
    model.train(5000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    p_pred, f_pred = model.predict(X_star)

    # error_u_norm = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    # error_u = np.linalg.norm(u_star - u_pred, 2)
    # error_f = np.linalg.norm(f_pred, 2)
    # print('Error u_norm %e' % (error_u_norm))
    # print('Error u: %e' % (error_u))
    # print('f_pred: %e' % (error_f))

    ### PLOT the Results
    chk_idx = -(n_x * 6)
    plot_solution(X_star[-n_x:, 0:2], (p_pred[chk_idx:chk_idx+n_x]) / 1000, 0, 'Predicted Pressure')  # Predicted
    plot_solution(X_star[-n_x:, 0:2], (P_star[chk_idx:chk_idx+n_x]) / 1000, 1, 'True Pressure')  # True

    ###
    plt.figure()
    plt.title("t = 0s", fontsize=16)
    t_idx = 0
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.xlim([0, 100])
    plt.ylim([75, 175])
    plt.show()

    # plt.figure()
    # plt.title("t = 1400s", fontsize=16)
    # t_idx = 70
    # plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    # plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    # plt.xlabel('x (m)', fontsize=16)
    # plt.ylabel('p(x, t) (kPa)', fontsize=16)
    # plt.show()

    plt.figure()
    plt.title("t = 1600s", fontsize=16)
    t_idx = 80
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.xlim([0, 100])
    plt.ylim([75, 175])
    plt.show()

    plt.figure()
    plt.title("t = 1800s", fontsize=16)
    t_idx = 90
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.xlim([0, 100])
    plt.ylim([75, 175])
    plt.show()

    # plt.figure()
    # plt.title("t = 2000s", fontsize=16)
    # t_idx = 100
    # plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    # plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    # plt.xlabel('x (m)', fontsize=16)
    # plt.ylabel('p(x, t) (kPa)', fontsize=16)
    # plt.show()
    ###

    plt.figure()
    plt.title('validation error', fontsize=12)
    val_np = np.array(val_graph)
    plt.plot(val_np[:, 0], val_np[:, 1])
    plt.xlabel('total iteration number')
    plt.ylabel('Loss')
    plt.show(block=False)

    plt.figure()
    plt.title('point u error', fontsize=12)
    u_np = np.array(u_graph)
    plt.plot(u_np[:, 0], u_np[:, 1])
    plt.xlabel('total iteration number', fontsize=10)
    plt.ylabel('Data based Loss', fontsize=10)
    plt.show(block=False)

    plt.figure()
    plt.title('PDE f error', fontsize=12)
    f_np = np.array(f_graph)
    plt.plot(f_np[:, 0], f_np[:, 1])
    plt.xlabel('total iteration number', fontsize=10)
    plt.ylabel('Physics based Loss', fontsize=10)
    plt.show(block=False)


