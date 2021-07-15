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
W_PDE = 1E12

np.random.seed(nnn)
tf.random.set_seed(nnn)
loss_graph = []
val_graph = []
u_graph = []
f_graph = []


class PhysicsInformedNN:
    def __init__(self, X_p, P, Q_p, X_val, P_val, Q_val, X_f, Q_f, layers, lb, ub, obs_X, obs_P, obs_Q, obs_t):

        self.lb = lb
        self.ub = ub

        self.x_p = X_p[:, 0:1]
        self.y_p = X_p[:, 1:2]
        self.t_p = X_p[:, 2:3]
        self.p = P
        self.q_p = Q_p

        self.p_lb = P.min()
        self.p_ub = P.max()

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.t_f = X_f[:, 2:3]
        self.q_f = Q_f

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

        self.p_pred = self.net_p(self.x_p_tf, self.y_p_tf, self.t_p_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf, self.q_f_tf)

        # Loss - Training
        self.loss_p = (tf.reduce_mean(tf.square(self.p_tf - self.p_pred)))
        self.loss_f = (tf.reduce_mean(tf.square(self.f_pred)))
        self.loss = self.loss_p + self.loss_f
        # self.loss = self.loss_p

        # Validation
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.y_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_val.shape[1]])
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.p_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_val.shape[1]])
        self.q_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_val.shape[1]])

        self.p_val_pred = self.net_p(self.x_val_tf, self.y_val_tf, self.t_val_tf)
        # self.f_val_pred = self.net_f(self.x_val_tf, self.y_val_tf, self.t_val_tf, self.q_val_tf)

        # Loss - Validation
        self.loss_val = tf.reduce_mean(tf.square(self.p_val_tf - self.p_val_pred))

        # Optimizer
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0003, beta1=0.9, beta2=0.999,
                                                               epsilon=1e-08, use_locking=False, name='Adam')
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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

    def net_p(self, x, y, t):
        p = self.neural_net(tf.concat([x, y, t], 1), self.weights, self.biases)
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

        # self.xxxx = x
        # self.yyyy = y
        # self.tttt = t
        # self.pppp = p

        n_grid = 7*7
        # self.dttt = (t[0:-n_grid] - t[n_grid:])
        self.p_t = (p[0:-n_grid] - p[n_grid:]) / (t[0:-n_grid] - t[n_grid:])

        # p_t = tf.gradients(p, t)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_x = nu[0] * p_x
        p_y = nu[0] * p_y
        p_xx = tf.gradients(p_x, x)[0]
        p_yy = tf.gradients(p_y, y)[0]
        self.p_xx = p_xx[0:-n_grid]
        self.p_yy = p_yy[0:-n_grid]

        # f = p_xx + p_yy + nu[1] * Q - nu[2] * p_t
        f = self.p_xx + self.p_yy + nu[1] * Q[0:-n_grid] - nu[2] * self.p_t
        # self.ffff = f * 1E12
        return f * W_PDE

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
                   self.p_val_tf: self.p_val, self.q_val_tf: self.q_val}

        start_time = time.time()
        for it in range(nIter):
            # xxx = self.sess.run(self.xxxx, tf_dict)
            # yyy = self.sess.run(self.yyyy, tf_dict)
            # ttt = self.sess.run(self.tttt, tf_dict)
            # dtt = self.sess.run(self.dttt, tf_dict)
            # ppp = self.sess.run(self.pppp, tf_dict)
            # ppt = self.sess.run(self.p_t, tf_dict)
            # fff = self.sess.run(self.ffff, tf_dict)
            # # ppxx = self.sess.run(self.pp_xx, tf_dict)
            # # ppyy = self.sess.run(self.pp_yy, tf_dict)

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

            if it % 200 == 0:
                pred_p, _ = self.predict(self.obs_x, self.obs_q)
                f_map = self.sess.run(self.f_pred, tf_dict)
                # plot_solution(self.obs_p, pred_p, "t = " + str(self.obs_t) + "s, Iter={0:d}".format(it))
                # plot_solution2(f_map[49*200:49*(200+1), 0:1], "f map, Iter={0:d}".format(it))
                plot_solution2(self.obs_p, pred_p, f_map[49 * 200:49 * (200 + 1), 0:1], "t = " + str(self.obs_t) + "s, Iter={0:d}".format(it))

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


def get_Q(x, y, Q):
    inj = np.array((x == 5) & (y == 5), dtype=np.int)
    prd = np.array((x == 65) & (y == 65), dtype=np.int)
    return Q*(inj-prd)


def plot_solution(p_star, p_pred, title):
    p_pred = np.reshape(p_pred / 1000, [7, 7])
    p_star = np.reshape(p_star / 1000, [7, 7])

    fig, ax = plt.subplots(1, 2)
    plt.suptitle(title, fontsize=16)
    ia = ax[0].imshow(p_star, cmap='jet', vmin=10200, vmax=9800)
    # ia = ax[0].imshow(p_star, cmap='jet')
    ax[0].set_title("True")
    plt.colorbar(ia, ax=ax[0], fraction=0.05, pad=0.05)

    ib = ax[1].imshow(p_pred, cmap='jet', vmin=10200, vmax=9800)
    # ib = ax[1].imshow(p_pred, cmap='jet')
    ax[1].set_title("Predict")
    plt.colorbar(ib, ax=ax[1], fraction=0.05, pad=0.05)

    plt.tight_layout()
    plt.show()


def plot_solution2(p_star, p_pred, f_map, title):
    p_pred = np.reshape(p_pred / 1000, [7, 7])
    p_star = np.reshape(p_star / 1000, [7, 7])
    f_map = np.reshape(f_map / W_PDE, [7, 7])

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    plt.suptitle(title, fontsize=16)
    ia = ax[0].imshow(p_star, cmap='jet', vmin=10200, vmax=9800)
    ax[0].set_title("True")
    plt.colorbar(ia, ax=ax[0], fraction=0.05, pad=0.05)

    ib = ax[1].imshow(p_pred, cmap='jet', vmin=10200, vmax=9800)
    ax[1].set_title("Predict")
    plt.colorbar(ib, ax=ax[1], fraction=0.05, pad=0.05)

    ia = ax[2].imshow(abs(f_map), cmap='jet', vmin=0)
    ax[2].set_title("PDE")
    plt.colorbar(ia, ax=ax[2], fraction=0.05, pad=0.05)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # layers = [3, 20, 20, 20, 20, 1]
    layers = [3, 20, 40, 80, 80, 40, 20, 1]
    # layers = [3, 20, 40, 80, 160, 160, 80, 40, 20, 1]
    # layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('./Data/2D_Neumann.mat')

    # x = np.arange(5, 65 + 10, 10)
    # y = np.arange(5, 65 + 10, 10)
    x = data['X_star'][:, 0]
    y = data['X_star'][:, 1]
    t = np.arange(0, 864000 + 1800, 1800)
    p = data['p_star'].T * 1000.0  # [[P1, ..., Pn_x*n_y]_t0, ..., [P1, Pn_x*n_y]_tn]

    n_t = np.shape(p)[0]
    n_x = 7
    n_y = 7
    X, Y, T = np.meshgrid(x, y, t)
    x_tile = np.tile(x, (n_t, 1)).flatten()[:, None]
    y_tile = np.tile(y, (n_t, 1)).flatten()[:, None]
    t_tile = np.tile(t, (n_x*n_y, 1)).T.flatten()[:, None]

    X_star = np.hstack([x_tile, y_tile, t_tile])
    P_star = p.flatten()[:, None]
    Q_star = np.vstack([get_Q(X_star[:, 0:1], X_star[:, 1:2], 1)])

    lb = X_star.min(0)
    ub = X_star.max(0)

    n_grid = n_x*n_y

    ### 초기 조건
    # t_0 ~ t_(n_pts-1)의 전좌표에서의 압력 값
    n_pts = 50
    x_I = np.hstack([X_star[0:(n_pts*n_grid), 0].flatten()[:, None],
                     X_star[0:(n_pts*n_grid), 1].flatten()[:, None],
                     X_star[0:(n_pts*n_grid), 2].flatten()[:, None]])
    p_I = P_star[0:(n_pts*n_grid), 0].flatten()[:, None]
    q_I = Q_star[0:(n_pts*n_grid), 0].flatten()[:, None]

    # x_I2 = np.hstack([X_star[(480*n_grid):((480+1)*n_grid), 0].flatten()[:, None],
    #                  X_star[(480*n_grid):((480+1)*n_grid), 1].flatten()[:, None],
    #                  X_star[(480*n_grid):((480+1)*n_grid), 2].flatten()[:, None]])
    # p_I2 = P_star[(480*n_grid):((480+1)*n_grid), 0].flatten()[:, None]
    # q_I2 = Q_star[(480*n_grid):((480+1)*n_grid), 0].flatten()[:, None]

    # # Boundary Condition
    n_t = 0  # 481
    # # P((x_0, y_0), t)
    # x_B = np.hstack([X_star[0:(n_t*n_grid):n_grid, 0].flatten()[:, None],
    #                  X_star[0:(n_t*n_grid):n_grid, 1].flatten()[:, None],
    #                  X_star[0:(n_t*n_grid):n_grid, 2].flatten()[:, None]])
    # p_B = P_star[0:(n_t*n_grid):n_grid, 0].flatten()[:, None]
    # q_B = Q_star[0:(n_t*n_grid):n_grid, 0].flatten()[:, None]
    # # P((x_n, y_n), t)
    # x_B2 = np.hstack([X_star[n_grid-1:(n_t+1)*n_grid-1:n_grid, 0].flatten()[:, None],
    #                  X_star[n_grid-1:(n_t+1)*n_grid-1:n_grid, 1].flatten()[:, None],
    #                  X_star[n_grid-1:(n_t+1)*n_grid-1:n_grid, 2].flatten()[:, None]])
    # p_B2 = P_star[n_grid-1:(n_t+1)*n_grid-1:n_grid, 0].flatten()[:, None]
    # q_B2 = Q_star[n_grid-1:(n_t+1)*n_grid-1:n_grid, 0].flatten()[:, None]
    #
    # X_p_train = np.vstack([x_I, x_B, x_B2])
    # P_p_train = np.vstack([p_I, p_B, p_B2])
    # Q_p_train = np.vstack([q_I, q_B, q_B2])

    X_p_train = np.vstack([x_I])
    P_p_train = np.vstack([p_I])
    Q_p_train = np.vstack([q_I])

    # X_p_train = np.vstack([x_I, x_I2])
    # P_p_train = np.vstack([p_I, p_I2])
    # Q_p_train = np.vstack([q_I, q_I2])

    ### For f
    x_space = 10
    y_space = 10
    t_space = 600

    n_x_f = int((65-5)/x_space) + 1
    n_y_f = int((65-5)/y_space) + 1
    n_t_f = int((864000-0)/t_space) + 1

    x_f = np.tile(np.arange(5, 65 + x_space, x_space), (1, n_y_f)).flatten()[:, None]
    y_f = np.tile(np.arange(5, 65 + y_space, y_space), (n_x_f, 1)).T.flatten()[:, None]
    t_f = np.arange(0, 864000 + t_space, t_space)

    x_tile_f = np.tile(x_f, (n_t_f, 1)).flatten()[:, None]
    y_tile_f = np.tile(y_f, (n_t_f, 1)).flatten()[:, None]
    t_tile_f = np.tile(t_f, (n_x_f*n_y_f, 1)).T.flatten()[:, None]

    X_star_f = np.hstack([x_tile_f, y_tile_f, t_tile_f])
    Q_star_f = np.vstack([get_Q(X_star_f[:, 0:1], X_star_f[:, 1:2], 1)])
    ###
    #
    # # f_loss는 전체 grid에 대해 계산
    X_f_train = X_star_f[:, :]
    Q_f_train = Q_star_f[:, :]
    # X_f_train = X_star[:, :]
    # Q_f_train = Q_star[:, :]

    # training data validation data 8:2
    N_u = n_pts * n_grid + n_t
    # idx = np.random.choice(X_p_train.shape[0], N_u, replace=False)
    idx = np.arange(0, N_u, 1)

    idx_train = idx[0:round(N_u * 0.7)]
    idx_val = idx[round(N_u * 0.7):N_u]

    X_p_tr = X_p_train[idx_train, :]
    P_tr = P_p_train[idx_train, :]
    Q_p_tr = Q_p_train[idx_train, :]

    X_p_val = X_p_train[idx_val, :]
    P_val = P_p_train[idx_val, :]
    Q_p_val = Q_p_train[idx_val, :]

    # 임의의 t에서 iteration 에 따른 결과 변화 관측.
    obs_t_idx = 200
    obs_X_pts = X_star[n_grid*obs_t_idx:n_grid*(obs_t_idx+1), 0:3]
    obs_P_pts = P_star[n_grid*obs_t_idx:n_grid*(obs_t_idx+1), 0:1]
    obs_Q_pts = Q_star[n_grid*obs_t_idx:n_grid*(obs_t_idx+1), 0:1]

    model = PhysicsInformedNN(X_p_tr, P_tr, Q_p_tr,
                              X_p_val, P_val, Q_p_val,
                              X_f_train, Q_f_train,
                              layers, lb, ub,
                              obs_X_pts, obs_P_pts, obs_Q_pts, t[obs_t_idx])

    start_time = time.time()
    model.train(2000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    p_pred, f_pred = model.predict(X_star, Q_star)

    ### PLOT
    for t_idx in [0, 10, 30, 50, 100, 150, 300, 400, 480]:
        plot_solution(P_star[n_grid * t_idx:n_grid * (t_idx + 1)],
                      p_pred[n_grid * t_idx:n_grid * (t_idx + 1)],
                      "t = " + str(t[t_idx]))
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
    plt.plot(f_np[:, 0], f_np[:, 1]/(W_PDE**2))
    plt.xlabel('total iteration number', fontsize=10)
    plt.ylabel('Physics based Loss', fontsize=10)
    plt.show(block=False)

    pass


