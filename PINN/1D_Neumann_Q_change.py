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

np.random.seed(nnn)
tf.random.set_seed(nnn)
loss_graph = []
val_graph = []
u_graph = []
f_graph = []


class PhysicsInformedNN:
    def __init__(self, X_p, P, Q_p, X_val, P_val, Q_val, X_f, Q_f, layers, lb, ub, obs_X, obs_Q, obs_P):

        self.lb = lb
        self.ub = ub

        self.x_p = X_p[:, 0:1]
        self.t_p = X_p[:, 1:2]
        self.p = P
        self.q_p = Q_p

        self.p_lb = P.min()
        self.p_ub = P.max()

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.q_f = Q_f

        self.layers = layers

        self.x_val = X_val[:, 0:1]
        self.t_val = X_val[:, 1:2]
        self.p_val = P_val
        self.q_val = Q_val

        self.obs_x = obs_X
        self.obs_q = obs_Q
        self.obs_p = obs_P

        self.n = 1

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        # Training
        self.x_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_p.shape[1]])
        self.t_p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_p.shape[1]])
        self.p_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p.shape[1]])

        self.x_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.q_f_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_f.shape[1]])

        self.p_pred = self.net_p(self.x_p_tf, self.t_p_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf, self.q_f_tf)

        # Loss - Training
        self.loss_p = (tf.reduce_mean(tf.square(self.p_tf - self.p_pred)))
        self.loss_f = (tf.reduce_mean(tf.square(self.f_pred)))
        self.loss = self.loss_p + self.loss_f

        # Validation
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.p_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_val.shape[1]])
        self.q_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_val.shape[1]])

        self.p_val_pred = self.net_p(self.x_val_tf, self.t_val_tf)
        self.f_val_pred = self.net_f(self.x_val_tf, self.t_val_tf, self.q_val_tf)

        # Loss - Validation
        self.loss_val = tf.reduce_mean(tf.square(self.p_val_tf - self.p_val_pred))

        # Optimizer
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0003, beta1=0.9, beta2=0.999,
                                                               epsilon=1e-08, use_locking=False, name='Adam')
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # self.optimizer = keras.optimizers.Adam(0.001, clipnorm=1.0)
        # self.train_op_Adam = self.optimizer.minimize(self.loss)

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
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_p(self, x, t):
        p = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        p = (p + 1) / 2 * (self.p_ub - self.p_lb) + self.p_lb
        return p

    def net_f(self, x, t, Q):
        mu = 0.001
        Ct = 11e-9
        phi = 0.2
        B = 0.9
        k = 200 * 9.8692 * 10 ** (-16)
        nu = [k / (mu * B), 120/(24*60*60)/1000, (phi * Ct) / B]

        p = self.net_p(x, t)
        p_t = tf.gradients(p, t)[0]
        p_x = tf.gradients(p, x)[0]
        p_x = nu[0] * p_x
        p_xx = tf.gradients(p_x, x)[0]

        f = p_xx + nu[1] * Q - nu[2] * p_t
        # return f * 8E11
        # return f * 3E11
        return f * 3.5E11

    def callback(self, loss, loss_val, loss_u, loss_f):
        print('It: %d, Loss: %e, Loss_val: %e' % (self.n, loss, loss_val))
        if self.n % 50 == 0:
            val_graph.append([self.n, loss_val])
            u_graph.append([self.n, loss_u])
            f_graph.append([self.n, loss_f])
        self.n = self.n + 1

    def train(self, nIter):
        tf_dict = {self.x_p_tf: self.x_p, self.t_p_tf: self.t_p, self.p_tf: self.p,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f, self.q_f_tf: self.q_f,
                   self.x_val_tf: self.x_val, self.t_val_tf: self.t_val,
                   self.p_val_tf: self.p_val, self.q_val_tf: self.q_val}

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
                print('It: %d, Loss: %.3e, Loss_val: %.3e, Loss_p: %.3e, Loss_f: %.3e, Time: %.2f' % (
                it, loss_value, val_loss, u_loss, f_loss, elapsed))
            if it % 500 == 0:
                pred_p, _ = self.predict(self.obs_x, self.obs_q)

                plt.figure()
                plt.title("t = 18,000s, Iter={0:d}".format(it), fontsize=16)
                plt.plot(self.obs_x[:, 0:1], self.obs_p / 1000, 'b-', linewidth=3.0)
                plt.plot(self.obs_x[:, 0:1], pred_p / 1000, 'r--', linewidth=3.0)
                plt.xlabel('x (m)', fontsize=16)
                plt.ylabel('p(x, t) (kPa)', fontsize=16)
                plt.show()

    def predict(self, X_star, Q_star):
        p_star = self.sess.run(self.p_pred,
                               {self.x_p_tf: X_star[:, 0:1], self.t_p_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred,
                               {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2], self.q_f_tf: Q_star})

        return p_star, f_star


def get_Q(x, Q):
    inj = np.array((x == 5), dtype=np.int)
    prd = np.array((x == 395), dtype=np.int)
    return Q*(inj-prd)


def plot_solution(X_star, p_star, index, string):
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


if __name__ == "__main__":
    layers = [2, 20, 20, 20, 20, 1]

    # data = scipy.io.loadmat('./Data/1D_Neumann_Q_change.mat')
    data = scipy.io.loadmat('./Data/1D_Neumann.mat')

    x = data['x'] + 5.0  # [[1], [2], [3], ..., [n_x]]
    t = data['t']
    p = data['usol'].T * 1000.0  # [[P1, ..., Pn_x*n_y]_t0, ..., [P1, Pn_x*n_y]_tn]

    n_t = np.shape(p)[0]
    n_x = np.shape(p)[1]
    X, T = np.meshgrid(x, t)

    x_tile = np.tile(x, (n_t, 1)).flatten()[:, None] # (n_x * n_t, 1)
    t_tile = T.flatten()[:, None]
    X_star = np.hstack([x_tile, t_tile])
    P_star = p.flatten()[:, None]
    Q_star = np.vstack([get_Q(X_star[0:20*n_x, 0:1], 1), get_Q(X_star[20*n_x:41*n_x, 0:1], 1)])

    lb = X_star.min(0)
    ub = X_star.max(0)

    ### 초기 조건
    # t_0 ~ t_(n_pts-1)의 전좌표에서의 압력 값
    n_pts = 2
    xx1 = np.hstack([X[0:n_pts, :].flatten()[:, None], T[0:n_pts, :].flatten()[:, None]])
    pp1 = p[0:n_pts, :].flatten()[:, None]

    N_u = n_x * n_pts

    X_p_train = np.vstack([xx1])
    P_p_train = np.vstack([pp1])
    Q_p_train = np.vstack([get_Q(X_p_train[:, 0:1], 1)])

    X_f_train = X_star[:, :]
    Q_f_train = Q_star[:, :]

    # training data validation data 8:2
    idx = np.random.choice(X_p_train.shape[0], N_u, replace=False)

    idx_train = idx[0:round(N_u * 0.8)]
    idx_val = idx[round(N_u * 0.8):N_u]

    X_p_tr = X_p_train[idx_train, :]
    P_tr = P_p_train[idx_train, :]
    Q_p_tr = Q_p_train[idx_train, :]

    X_p_val = X_p_train[idx_val, :]
    P_val = P_p_train[idx_val, :]
    Q_p_val = Q_p_train[idx_val, :]

    # 임의의 t에서 iteration 에 따른 결과 변화 관측.
    obs_X_pts = X_star[n_x*30:n_x*(30+1), 0:2]
    obs_Q_pts = Q_star[n_x*30:n_x*(30+1), 0:1]
    obs_P_pts = P_star[n_x*30:n_x*(30+1), 0:1]

    model = PhysicsInformedNN(X_p_tr, P_tr, Q_p_tr,
                              X_p_val, P_val, Q_p_val,
                              X_f_train, Q_f_train,
                              layers, lb, ub,
                              obs_X_pts, obs_Q_pts, obs_P_pts)

    start_time = time.time()
    model.train(8000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    p_pred, f_pred = model.predict(X_star, Q_star)

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
    v_max = 13000
    v_min = 7000

    plt.figure()
    plt.title("t = 0s", fontsize=16)
    t_idx = 0
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.ylim([v_min, v_max])
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 6,000s", fontsize=16)
    t_idx = 10
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.ylim([v_min, v_max])
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 12,000s", fontsize=16)
    t_idx = 20
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.ylim([v_min, v_max])
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 18,000s", fontsize=16)
    t_idx = 30
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.ylim([v_min, v_max])
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 24,000s", fontsize=16)
    t_idx = 40
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.ylim([v_min, v_max])
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()
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


