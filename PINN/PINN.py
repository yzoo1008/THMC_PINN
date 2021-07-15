import sys
sys.path.insert(0, './Utilities/')
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from tensorflow import keras

nnn = 7777
# # 1D_Neumann
# W_PDE = 3E6
# LR = 0.0003
# 1D_Neumann_t_20
W_train = 1E1
W_PDE = 1E6
W_dP_dt = 1E8
LR = 0.01
# # 2D_Neumann
# W_PDE = 3E6
# LR = 0.0003
# LR = 0.001

np.random.seed(nnn)
tf.random.set_seed(nnn)
loss_graph = []
val_graph = []
u_graph = []
f_graph = []
dP_dt_graph = []


class PhysicsInformedNN:
    def __init__(self, X_p, P, Q_p, X_val, P_val, Q_val, X_f, Q_f, dP_dt_f1, dP_dt_f2,
                 layers, lb, ub,
                 obs_X, obs_P, obs_Q, obs_t, obs_t_idx,
                 XX, PP, QQ,
                 dim, n_x, n_y):

        self.dim = dim
        self.n_x = n_x
        self.n_y = n_y
        self.n_grid = n_x * n_y

        self.lb = lb
        self.ub = ub

        self.x_p = X_p[:, 0:1]
        self.y_p = X_p[:, 1:2]
        self.t_p = X_p[:, 2:3]
        self.p = P
        self.q_p = Q_p

        # self.p_lb = P.min()
        # self.p_ub = P.max()
        self.p_lb = PP.min()
        self.p_ub = PP.max()
        # self.p_lb = 10150 * 1000
        # self.p_ub = 9850 * 1000

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.t_f = X_f[:, 2:3]
        self.q_f = Q_f
        self.dP_dt_f1 = dP_dt_f1
        self.dP_dt_f2 = dP_dt_f2

        self.layers = layers

        self.x_val = X_val[:, 0:1]
        self.y_val = X_val[:, 1:2]
        self.t_val = X_val[:, 2:3]
        self.p_val = P_val
        self.q_val = Q_val

        self.obs_x = obs_X
        self.obs_q = obs_Q
        self.obs_p = obs_P
        self.obs_t = obs_t
        self.obs_t_idx = obs_t_idx

        self.XX = XX
        self.PP = PP
        self.QQ = QQ

        self.n = 1

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        # Training
        self.x_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_p.shape[1]])
        self.y_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_p.shape[1]])
        self.t_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_p.shape[1]])
        self.p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p.shape[1]])

        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.q_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_f.shape[1]])

        self.dP_dt_f1_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.dP_dt_f1.shape[0]])
        self.dP_dt_f2_tf = tf.compat.v1.placeholder(tf.float32, shape=[self.dP_dt_f2.shape[0]])

        self.p_pred = self.net_p(self.x_p_tf, self.y_p_tf, self.t_p_tf)
        self.f_pred, self.dP_dt_pred1, self.dP_dt_pred2 = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf, self.q_f_tf)

        # Loss - Training
        self.loss_p = (tf.reduce_mean(tf.square(self.p_tf - self.p_pred)))
        self.loss_f = (tf.reduce_mean(tf.square(self.f_pred)))
        self.loss_dP_dt1 = (tf.reduce_mean(tf.square(self.dP_dt_pred1 - self.dP_dt_f1_tf)))
        self.loss_dP_dt2 = (tf.reduce_mean(tf.square(self.dP_dt_pred2 - self.dP_dt_f2_tf)))
        self.loss_dP_dt = (self.loss_dP_dt1 + self.loss_dP_dt2) / 2.0
        self.loss = self.loss_p * W_train + self.loss_f * W_PDE + self.loss_dP_dt * W_dP_dt
        # self.loss = self.loss_p
        # self.loss = self.loss_f

        # Validation
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.y_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_val.shape[1]])
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.p_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_val.shape[1]])
        self.q_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_val.shape[1]])

        self.p_val_pred = self.net_p(self.x_val_tf, self.y_val_tf, self.t_val_tf)

        # Loss - Validation
        self.loss_val = tf.reduce_mean(tf.square(self.p_val_tf - self.p_val_pred))

        # Learning rate
        global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = LR
        self.learning_rate = tf.compat.v1.train.exponential_decay(initial_learning_rate,
                                                             global_step,
                                                             1000, 0.9, staircase=True)
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                               epsilon=1e-08, use_locking=False, name='Adam')
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, global_step=global_step)

        # # Optimizer
        # self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, beta1=0.9, beta2=0.999,
        #                                                        epsilon=1e-08, use_locking=False, name='Adam')
        # self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # Session & Init
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
            # H = tf.nn.leaky_relu(tf.add(tf.matmul(H, W), b), alpha=0.1)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_1D(self, X, weights, biases):
        num_layers = len(weights) + 1
        lb = np.array([self.lb[0], self.lb[2]])
        ub = np.array([self.ub[0], self.ub[2]])
        # H = 2.0 * (X - lb) / (ub - lb) - 1.0
        H = (X - lb) / (ub - lb)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_p(self, x, y, t):
        if self.dim == 2:
            p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
        elif self.dim == 1:
            p = self.neural_net_1D(tf.concat([x, t], 1), self.weights, self.biases)
        p = (p + 1) / 2 * (self.p_ub - self.p_lb) + self.p_lb
        return p

    def net_f(self, x, y, t, Q):
        mu = 0.001
        Ct = 11e-9
        phi = 0.2
        B = 0.9
        k = 200 * 9.8692 * 10 ** (-16)
        nu = [k / (mu * B), 20/(24*60*60)/1000, (phi * Ct) / B]

        p = self.net_p(x, y, t)

        n_grid = self.n_grid
        # p_t = (p[0:-n_grid] - p[n_grid:]) / (t[0:-n_grid] - t[n_grid:])
        p_t = tf.gradients(p, t)[0]
        dP_dt1 = p_t[0:-n_grid:n_grid]
        dP_dt2 = p_t[n_grid-1:-(n_grid-1):n_grid]
        p_x = tf.gradients(p, x)[0]
        p_xx = tf.gradients(p_x, x)[0]

        self.p_t = p_t
        self.p_xx = p_xx

        if self.dim == 2:
            p_y = tf.gradients(p, y)[0]
            p_yy = tf.gradients(p_y, y)[0]

            f = (p_xx + p_yy) + (nu[1]/nu[0])*Q - (nu[2]/nu[0])*p_t
            # f = nu[0]*(p_xx[0:-n_grid] + p_yy[0:-n_grid]) + nu[1] * Q[0:-n_grid] - nu[2] * p_t

        elif self.dim == 1:
            # f = p_xx + (nu[1]/nu[0])*Q - (nu[2]/nu[0])*p_t
            f = (nu[0]/nu[2])*p_xx + (nu[1]/nu[2])*Q - p_t
            # f = nu[0]*p_xx[0:-n_grid] + nu[1] * Q[0:-n_grid] - nu[2] * p_t

        return f, dP_dt1, dP_dt2

    def callback(self, loss, loss_val, loss_u, loss_f):
        print('It: %d, Loss: %e, Loss_val: %e' % (self.n, loss, loss_val))
        if self.n % 50 == 0:
            val_graph.append([self.n, loss_val])
            u_graph.append([self.n, loss_u])
            f_graph.append([self.n, loss_f])
        self.n = self.n + 1

    def train(self, nIter):
        tf_dict = {self.x_p_tf: self.x_p, self.y_p_tf: self.y_p, self.t_p_tf: self.t_p, self.p_tf: self.p,
                   self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f, self.q_f_tf: self.q_f,
                   self.x_val_tf: self.x_val, self.y_val_tf: self.y_val, self.t_val_tf: self.t_val,
                   self.p_val_tf: self.p_val, self.q_val_tf: self.q_val,
                   self.dP_dt_f1_tf: self.dP_dt_f1, self.dP_dt_f2_tf: self.dP_dt_f2}

        n_grid = self.n_grid
        start_time = time.time()
        for it in range(nIter):
            # pt = self.sess.run(self.p_t, tf_dict)
            # pxx = self.sess.run(self.p_xx, tf_dict)
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                val_loss = self.sess.run(self.loss_val, tf_dict)
                train_loss = self.sess.run(self.loss_p, tf_dict)
                PDE_loss = self.sess.run(self.loss_f, tf_dict)
                dP_dt_loss = self.sess.run(self.loss_dP_dt, tf_dict)

                loss_graph.append([it, loss_value])
                val_graph.append([it, val_loss])
                u_graph.append([it, train_loss])
                f_graph.append([it, PDE_loss])
                dP_dt_graph.append([it, dP_dt_loss])
                print('It: %d, Loss: %.3e, Loss_val: %.3e, Loss_p: %.3e, Loss_f: %.3e, Loss_dP_dt: %.3e, Time: %.2f' %
                      (it, loss_value, val_loss, train_loss, PDE_loss, dP_dt_loss, elapsed))

            if it % 500 == 0:
                if self.dim == 2:
                    PP, _ = self.predict(self.XX, self.QQ)
                    plot_trend(self.XX[:, 2:3], PP, self.PP, self.n_x, self.n_y, "Iter={0:d}".format(it))
                elif self.dim == 1:
                    PP, _ = self.predict(self.obs_x, self.obs_q)
                    plot_1D(self.obs_x[:, 0:1], self.obs_p, PP, self.n_x, self.n_y, "Iter={0:d}".format(it))
                # pred_p, _ = self.predict(self.obs_x, self.obs_q)
                # f_map = self.sess.run(self.f_pred, tf_dict)[n_grid * self.obs_t_idx:n_grid * (self.obs_t_idx + 1), 0:1]
                # plot_solution2(self.obs_p, pred_p, f_map,
                #                "t = " + str(self.obs_t) + "s, Iter={0:d}".format(it), self.n_x, self.n_y)

                # plot_solution(self.obs_p, pred_p, "t = " + str(self.obs_t) + "s, Iter={0:d}".format(it))
                # plot_solution2(f_map[49*200:49*(200+1), 0:1], "f map, Iter={0:d}".format(it))

    def predict(self, X_star, Q_star):
        p_star = self.sess.run(self.p_pred,
                               {self.x_p_tf: X_star[:, 0:1],
                                self.y_p_tf: X_star[:, 1:2],
                                self.t_p_tf: X_star[:, 2:3]})
        f_star = self.sess.run(self.f_pred,
                               {self.x_f_tf: X_star[:, 0:1],
                                self.y_f_tf: X_star[:, 1:2],
                                self.t_f_tf: X_star[:, 2:3],
                                self.q_f_tf: Q_star})

        return p_star, f_star


def plot_2D(p_star, p_pred, title, n_x, n_y):
    shape = [n_x, n_y]
    p_pred = np.reshape(p_pred / 1000, shape)
    p_star = np.reshape(p_star / 1000, shape)

    # v_min = 7000
    # v_max = 13000
    v_min = 9800
    v_max = 10200

    fig, ax = plt.subplots(1, 2)
    plt.suptitle(title, fontsize=16)
    ia = ax[0].imshow(p_star, cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_title("True")
    plt.colorbar(ia, ax=ax[0], fraction=0.05, pad=0.05)

    ib = ax[1].imshow(p_pred, cmap='jet', vmin=v_min, vmax=v_max)
    ax[1].set_title("Predict")
    plt.colorbar(ib, ax=ax[1], fraction=0.05, pad=0.05)

    plt.tight_layout()
    plt.show()


def plot_solution2(p_star, p_pred, f_map, title, n_x, n_y):
    shape = [n_x, n_y]
    p_pred = np.reshape(p_pred / 1000, shape)
    p_star = np.reshape(p_star / 1000, shape)
    f_map = np.reshape(f_map, shape)

    # v_min = 7000
    # v_max = 13000
    v_min = 9800
    v_max = 10200

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    plt.suptitle(title, fontsize=16)
    ia = ax[0].imshow(p_star, cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_title("True")
    plt.colorbar(ia, ax=ax[0], fraction=0.05, pad=0.05)

    ib = ax[1].imshow(p_pred, cmap='jet', vmin=v_min, vmax=v_max)
    ax[1].set_title("Predict")
    plt.colorbar(ib, ax=ax[1], fraction=0.05, pad=0.05)

    ia = ax[2].imshow(abs(f_map), cmap='jet', vmin=0)
    ax[2].set_title("PDE")
    plt.colorbar(ia, ax=ax[2], fraction=0.05, pad=0.05)

    plt.tight_layout()
    plt.show()


def plot_1D(X, p_true, p_pred, n_x, n_y, title):
    shape = [n_x, n_y]
    p_pred = np.reshape(p_pred / 1000.0, shape)
    p_true = np.reshape(p_true / 1000.0, shape)

    v_max = 13000
    v_min = 7000

    plt.figure()
    plt.title(title, fontsize=16)
    plt.plot(X, p_true, 'b-', linewidth=3.0)
    plt.plot(X, p_pred, 'r--', linewidth=3.0)
    plt.ylim([v_min, v_max])
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()


def plot_trend(t, p_pred, p_true, n_x, n_y, title):
    shape = [-1, n_x, n_y]
    p_pred = np.reshape(p_pred / 1000.0, shape)
    p_true = np.reshape(p_true / 1000.0, shape)
    t = np.reshape(t, shape)

    v_max = 10200
    v_min = 9800

    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    plt.suptitle(title, fontsize=16)
    ax[0, 0].plot(t[:, 0, 0], p_true[:, 0, 0], 'r-', linewidth=3.0)
    ax[0, 0].plot(t[:, 0, 0], p_pred[:, 0, 0], 'k--', linewidth=3.0)
    ax[0, 0].set_title("Injection")
    ax[0, 0].set_ylim([v_min, v_max])
    ax[0, 0].set_xlabel('t (s)', fontsize=16)
    ax[0, 0].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[0, 1].plot(t[:, 0, 3], p_true[:, 0, 3], 'r-', linewidth=3.0)
    ax[0, 1].plot(t[:, 0, 3], p_pred[:, 0, 3], 'k--', linewidth=3.0)
    ax[0, 1].set_title("(X, Y) = (0, 3)")
    ax[0, 1].set_ylim([v_min, v_max])
    ax[0, 1].set_xlabel('t (s)', fontsize=16)
    ax[0, 1].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[0, 2].plot(t[:, 0, 6], p_true[:, 0, 6], 'r-', linewidth=3.0)
    ax[0, 2].plot(t[:, 0, 6], p_pred[:, 0, 6], 'k--', linewidth=3.0)
    ax[0, 2].set_title("(X, Y) = (0, 6)")
    ax[0, 2].set_ylim([v_min, v_max])
    ax[0, 2].set_xlabel('t (s)', fontsize=16)
    ax[0, 2].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[1, 0].plot(t[:, 3, 0], p_true[:, 3, 0], 'r-', linewidth=3.0)
    ax[1, 0].plot(t[:, 3, 0], p_pred[:, 3, 0], 'k--', linewidth=3.0)
    ax[1, 0].set_title("(X, Y) = (3, 0)")
    ax[1, 0].set_ylim([v_min, v_max])
    ax[1, 0].set_xlabel('t (s)', fontsize=16)
    ax[1, 0].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[1, 1].plot(t[:, 3, 3], p_true[:, 3, 3], 'r-', linewidth=3.0)
    ax[1, 1].plot(t[:, 3, 3], p_pred[:, 3, 3], 'k--', linewidth=3.0)
    ax[1, 1].set_title("Middle")
    ax[1, 1].set_ylim([v_min, v_max])
    ax[1, 1].set_xlabel('t (s)', fontsize=16)
    ax[1, 1].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[1, 2].plot(t[:, 3, 6], p_true[:, 3, 6], 'r-', linewidth=3.0)
    ax[1, 2].plot(t[:, 3, 6], p_pred[:, 3, 6], 'k--', linewidth=3.0)
    ax[1, 2].set_title("(X, Y) = (3, 6)")
    ax[1, 2].set_ylim([v_min, v_max])
    ax[1, 2].set_xlabel('t (s)', fontsize=16)
    ax[1, 2].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[2, 0].plot(t[:, 6, 0], p_true[:, 6, 0], 'r-', linewidth=3.0)
    ax[2, 0].plot(t[:, 6, 0], p_pred[:, 6, 0], 'k--', linewidth=3.0)
    ax[2, 0].set_title("(X, Y) = (6, 0)")
    ax[2, 0].set_ylim([v_min, v_max])
    ax[2, 0].set_xlabel('t (s)', fontsize=16)
    ax[2, 0].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[2, 1].plot(t[:, 6, 3], p_true[:, 6, 3], 'r-', linewidth=3.0)
    ax[2, 1].plot(t[:, 6, 3], p_pred[:, 6, 3], 'k--', linewidth=3.0)
    ax[2, 1].set_title("(X, Y) = (6, 3)")
    ax[2, 1].set_ylim([v_min, v_max])
    ax[2, 1].set_xlabel('t (s)', fontsize=16)
    ax[2, 1].set_ylabel('p(x, t) (kPa)', fontsize=16)

    ax[2, 2].plot(t[:, 6, 6], p_true[:, 6, 6], 'r-', linewidth=3.0)
    ax[2, 2].plot(t[:, 6, 6], p_pred[:, 6, 6], 'k--', linewidth=3.0)
    ax[2, 2].set_title("Production")
    ax[2, 2].set_ylim([v_min, v_max])
    ax[2, 2].set_xlabel('t (s)', fontsize=16)
    ax[2, 2].set_ylabel('p(x, t) (kPa)', fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # layers = [3, 20, 20, 20, 20, 1]
    # layers = [2, 20, 20, 20, 20, 1]
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # layers = [2, 20, 40, 80, 80, 40, 20, 1]
    # layers = [3, 20, 40, 80, 80, 40, 20, 1]
    # layers = [2, 20, 40, 80, 160, 160, 80, 40, 20, 1]
    # layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    import GenData

    n_dim = 1
    # g = GenData.GenData(n_dim, './Data/2D_Neumann.mat')
    # g = GenData.GenData(n_dim, './Data/1D_Neumann.mat')
    g = GenData.GenData(n_dim, './Data/1D_Neumann_t_20.mat')
    n_I = 1
    n_B = -1
    obs_t_idx = 200
    X_p, P_p, Q_p = g.train_Data(n_I, n_B)
    # X_pde, Q_pde = g.PDE_Data(10, 10, 1800)
    X_pde, P_pde, Q_pde = g.train_Data(-1, 0)
    X_obs, P_obs, Q_obs = g.get_Data(obs_t_idx)

    X_p_train = X_p[:, :]
    P_p_train = P_p[:, :]
    Q_p_train = Q_p[:, :]

    X_f_train = X_pde[:, :]
    P_f_train = P_pde[:, :]
    Q_f_train = Q_pde[:, :]

    obs_X_pts = X_obs[:, :]
    obs_P_pts = P_obs[:, :]
    obs_Q_pts = Q_obs[:, :]

    n_x = g.n_x
    n_y = g.n_y
    t = g.t
    X = g.X[:, :]
    P = g.P[:, :]
    Q = g.Q[:, :]

    n_grid = n_x*n_y

    dP_dt_f1 = (P_f_train[0:-n_grid:n_grid, 0] - P_f_train[n_grid::n_grid, 0]) /\
              (X_f_train[0:-n_grid:n_grid, 2] - X_f_train[n_grid::n_grid, 2])
    dP_dt_f2 = (P_f_train[n_grid-1:-(n_grid-1):n_grid, 0] - P_f_train[2*n_grid-1::n_grid, 0]) / \
               (X_f_train[n_grid-1:-(n_grid-1):n_grid, 2] - X_f_train[2*n_grid-1::n_grid, 2])
    # training data validation data 8:2
    N_u = X_p_train.shape[0]
    idx = np.random.choice(X_p_train.shape[0], N_u, replace=False)
    # idx = np.arange(0, N_u, 1)

    idx_train = idx[0:round(N_u * 0.7)]
    idx_val = idx[round(N_u * 0.7):N_u]

    X_p_tr = X_p_train[idx_train, :]
    P_p_tr = P_p_train[idx_train, :]
    Q_p_tr = Q_p_train[idx_train, :]

    X_p_val = X_p_train[idx_val, :]
    P_p_val = P_p_train[idx_val, :]
    Q_p_val = Q_p_train[idx_val, :]

    ### PINN Model
    model = PhysicsInformedNN(X_p_tr, P_p_tr, Q_p_tr,
                              X_p_val, P_p_val, Q_p_val,
                              X_f_train, Q_f_train, dP_dt_f1, dP_dt_f2,
                              layers, X.min(0), X.max(0),
                              obs_X_pts, obs_P_pts, obs_Q_pts, t[obs_t_idx], obs_t_idx,
                              X, P, Q,
                              n_dim, n_x, n_y)

    start_time = time.time()
    model.train(3000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    p_pred, f_pred = model.predict(X[:, :], Q[:, :])

    ### PLOT
    # for t_idx in [0, 10, 30, 50, 100, 150, 300, 400, 480]:
    #     plot_2D(P[n_grid * t_idx:n_grid * (t_idx + 1)],
    #             p_pred[n_grid * t_idx:n_grid * (t_idx + 1)],
    #             "t = " + str(t[t_idx]), n_x, n_y)
    # for t_idx in [0, 10, 20, 30, 40]:
    #     plot_1D(X[n_grid * t_idx:n_grid * (t_idx + 1), 0:1],
    #               P[n_grid * t_idx:n_grid * (t_idx + 1)],
    #               p_pred[n_grid * t_idx:n_grid * (t_idx + 1)],
    #             n_x, n_y, "t = " + str(t[t_idx]))
    for t_idx in [0, 10, 20, 40, 50, 100, 200, 300, 400, 456]:
        plot_1D(X[n_grid * t_idx:n_grid * (t_idx + 1), 0:1],
                P[n_grid * t_idx:n_grid * (t_idx + 1)],
                p_pred[n_grid * t_idx:n_grid * (t_idx + 1)],
                n_x, n_y, "t = " + str(t[t_idx]))
    ###

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    val_np = np.array(val_graph)
    ax[0].plot(val_np[:, 0], val_np[:, 1])
    ax[0].set_title("Validation Loss", fontsize=18)
    ax[0].set_xlabel('Iterations', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)

    u_np = np.array(u_graph)
    ax[1].plot(u_np[:, 0], u_np[:, 1])
    ax[1].set_title("Training Loss", fontsize=18)
    ax[1].set_xlabel('Iterations', fontsize=16)
    ax[1].set_ylabel('Loss', fontsize=16)

    f_np = np.array(f_graph)
    ax[2].plot(f_np[:, 0], f_np[:, 1])
    ax[2].set_title('PDE Loss', fontsize=18)
    ax[2].set_xlabel('Iterations', fontsize=16)
    ax[2].set_ylabel('Loss', fontsize=16)

    dP_dt_np = np.array(dP_dt_graph)
    ax[3].plot(dP_dt_np[:, 0], dP_dt_np[:, 1])
    ax[3].set_title('dP_dt Loss', fontsize=18)
    ax[3].set_xlabel('Iterations', fontsize=16)
    ax[3].set_ylabel('Loss', fontsize=16)

    plt.tight_layout()
    plt.show()

