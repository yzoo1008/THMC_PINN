import sys
sys.path.insert(0, './Utilities/')
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import time
import GenData


# np.random.seed(1234)
# tf.random.set_seed(1234)
loss_graph = []
train_graph = []
val_graph = []
pde_graph = []


class PhysicsInformedNN:
    def __init__(self,dim, n_x, n_y):

        self.dim = dim
        self.n_x = n_x
        self.n_y = n_y
        self.n_grid = n_x * n_y
        self.n = 1
        self.layers = [2, 20, 40, 40, 40, 40, 40, 20, 1]

    def load_data(self, X_tr, P_tr, Q_tr,
                  X_val, P_val, Q_val,
                  X_pde, P_pde, Q_pde,
                  X, P, Q):
        if self.dim == 1:
            self.lb = np.array([X_pde[:, 0].min(0), X_pde[:, 2].min(0)])
            self.ub = np.array([X_pde[:, 0].max(0), X_pde[:, 2].max(0)])
        elif self.dim == 2:
            self.lb = X_pde.min(0)
            self.ub = X_pde.max(0)
        self.p_lb = P_pde.min(0)
        self.p_ub = P_pde.max(0)

        # Train set
        self.x_tr = X_tr[:, 0:1]
        self.y_tr = X_tr[:, 1:2]
        self.t_tr = X_tr[:, 2:3]
        self.p_tr = P_tr[:, :]
        self.q_tr = Q_tr[:, :]

        # Validation set
        self.x_val = X_val[:, 0:1]
        self.y_val = X_val[:, 1:2]
        self.t_val = X_val[:, 2:3]
        self.p_val = P_val[:, :]
        self.q_val = Q_val[:, :]

        # PDE set
        self.x_pde = X_pde[:, 0:1]
        self.y_pde = X_pde[:, 1:2]
        self.t_pde = X_pde[:, 2:3]
        self.p_pde = P_pde[:, :]
        self.q_pde = Q_pde[:, :]

        # All data points
        self.X = X
        self.P = P
        self.Q = Q

        # Initialize network
        self.weights, self.biases = self.initialize_network(self.layers)

        # Session
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        )

        # Tensors
        # Training set
        self.x_tr_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_tr.shape[1]])
        self.y_tr_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_tr.shape[1]])
        self.t_tr_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_tr.shape[1]])
        self.p_tr_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_tr.shape[1]])
        self.q_tr_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_tr.shape[1]])
        # Validation set
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.y_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_val.shape[1]])
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.p_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_val.shape[1]])
        self.q_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_val.shape[1]])
        # PDE set
        self.x_pde_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x_pde.shape[1]])
        self.y_pde_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.y_pde.shape[1]])
        self.t_pde_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t_pde.shape[1]])
        self.p_pde_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p_pde.shape[1]])
        self.q_pde_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.q_pde.shape[1]])

        # Prediction
        self.p_pred_tr = self.get_p(self.x_tr_tf, self.y_tr_tf, self.t_tr_tf)
        self.p_pred_val = self.get_p(self.x_val_tf, self.y_val_tf, self.t_val_tf)
        self.pde_err = self.get_pde_err(self.x_pde_tf, self.y_pde_tf, self.t_pde_tf, self.q_pde_tf)

        # Loss - Training
        self.loss_p_tr = tf.reduce_mean(tf.square(self.p_pred_tr - self.p_tr_tf))
        self.loss_pde = (tf.reduce_mean(tf.square(self.pde_err)))
        self.loss = self.loss_p_tr + self.loss_pde

        # Loss - Validation
        self.loss_p_val = tf.reduce_mean(tf.square(self.p_pred_val - self.p_val_tf))
        self.loss_val = self.loss_p_val + self.loss_pde

        # Optimizer
        g_step = tf.Variable(0, trainable=False)
        self.lr = tf.compat.v1.train.exponential_decay(
            0.0001, g_step, 1000, 0.96, staircase=True
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam'
        )
        self.optimize = self.optimizer.minimize(self.loss, global_step=g_step)

        # # Optimizer - Pressure
        # g_step_p = tf.Variable(0, trainable=False)
        # self.lr_p = tf.compat.v1.train.exponential_decay(
        #     0.0003, g_step_p, 1000, 0.96, staircase=True
        # )
        # self.optimizer_p = tf.compat.v1.train.AdamOptimizer(
        #     learning_rate=self.lr_p, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam'
        # )
        # self.optimize_p = self.optimizer_p.minimize(self.loss_p_tr, global_step=g_step_p)

        # Session & Init
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_network(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for ly in range(0, num_layers - 1):
            w = self.xavier_init(size=[layers[ly], layers[ly + 1]])
            b = tf.Variable(tf.zeros([1, layers[ly + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(w)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def network(self, x, weights, biases):
        num_layers = len(weights) + 1
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        # h = (x - self.lb) / (self.ub - self.lb)
        for ly in range(0, num_layers - 2):
            w = weights[ly]
            b = biases[ly]
            h = tf.tanh(tf.add(tf.matmul(h, w), b))
            # h = tf.nn.leaky_relu(tf.add(tf.matmul(h, w), b), alpha=0.1)
        w = weights[-1]
        b = biases[-1]
        y = tf.add(tf.matmul(h, w), b)
        return y

    def get_p(self, x, y, t):
        if self.dim == 1:
            p = self.network(tf.concat([x, t], 1), self.weights, self.biases)
        elif self.dim == 2:
            p = self.network(tf.concat([x, y, t], 1), self.weights, self.biases)

        # p = (p + 1.0) * (self.p_ub - self.p_lb) / 2.0 + self.p_lb
        p = p * (self.p_ub - self.p_lb) + self.p_lb
        return p

    def get_pde_err(self, x, y, t, q):
        mu = 0.001
        Ct = 11e-9
        phi = 0.2
        B = 0.9
        k = 200 * 9.8692 * 10 ** (-16)
        nu = [k / (mu * B), 20 / (24 * 60 * 60) / 1000, (phi * Ct) / B]

        p = self.get_p(x, y, t)

        # p_t = tf.gradients(p, t)[0]
        # # n_grid = self.n_grid
        # # p_t = (p[0:-n_grid] - p[n_grid:]) / (t[0:-n_grid] - t[n_grid:])
        # p_x = tf.gradients(p, x)[0]
        # p_xx = tf.gradients(p_x, x)[0]
        # # self.p_xx = p_xx

        p_x = np.gradient(p, axis=0)
        p_t = np.gradient(p, axis=1)
        p_xx = np.gradient(p_x, axis=0)

        if self.dim == 1:
            pde_err = nu[0] * p_xx + nu[1] * q - nu[2] * p_t
            # pde_err = (nu[0] / nu[2]) * p_xx + (nu[1] / nu[2]) * q - p_t
            # pde_err = (nu[0] / nu[2]) * p_xx[0:-n_grid] + (nu[1] / nu[2]) * q[0:-n_grid] - p_t
            # self.pde_eq_0 = (nu[0] / nu[2]) * p_xx
            # self.pde_eq_1 = (nu[1] / nu[2]) * q
            # self.pde_eq_2 = p_t
        elif self.dim == 2:
            p_y = tf.gradients(p, y)[0]
            p_yy = tf.gradients(p_y, y)[0]
            pde_err = (nu[0] / nu[2]) * (p_xx + p_yy) + (nu[1] / nu[2]) * q - p_t
        # return pde_err * 2E2
        # return pde_err * 3E13
        return pde_err * 1E9

    def train(self, n_iter):
        tf_dict = {
            self.x_tr_tf: self.x_tr, self.y_tr_tf: self.y_tr, self.t_tr_tf: self.t_tr,
            self.p_tr_tf: self.p_tr, self.q_tr_tf: self.q_tr,
            self.x_val_tf: self.x_val, self.y_val_tf: self.y_val, self.t_val_tf: self.t_val,
            self.p_val_tf: self.p_val, self.q_val_tf: self.q_val,
            self.x_pde_tf: self.x_pde, self.y_pde_tf: self.y_pde, self.t_pde_tf: self.t_pde,
            self.p_pde_tf: self.p_pde, self.q_pde_tf: self.q_pde
        }

        st_time = time.time()
        for iter in range(n_iter):
            self.sess.run(self.optimize, tf_dict)
            if iter % 50 == 0:
                elapsed_time = time.time()-st_time
                loss = self.sess.run(self.loss, tf_dict)
                val_loss = self.sess.run(self.loss_val, tf_dict)
                p_loss = self.sess.run(self.loss_p_tr, tf_dict)
                pde_loss = self.sess.run(self.loss_pde, tf_dict)

                loss_graph.append([iter, loss])
                val_graph.append([iter, val_loss])
                train_graph.append([iter, p_loss])
                pde_graph.append([iter, pde_loss])

                print('It: %d, Loss: %.3e, Loss_val: %.3e, Loss_p_tr: %.3e, Loss_pde: %.3e, Time: %.2f'
                      % (iter, loss, val_loss, p_loss, pde_loss, elapsed_time))

            if iter % 500 == 0:
                # pde_eq_0 = self.sess.run(self.pde_eq_0, tf_dict)
                # pde_eq_1 = self.sess.run(self.pde_eq_1, tf_dict)
                # pde_eq_2 = self.sess.run(self.pde_eq_2, tf_dict)
                # plot_pde_err(pde_eq_0, pde_eq_1, pde_eq_2)

                t_point = 1
                if self.dim == 1:
                    x = self.X[self.n_grid * t_point:self.n_grid * (t_point + 1), :]
                    p = self.P[self.n_grid * t_point:self.n_grid * (t_point + 1), 0]
                    predict = self.predict(x)
                    plot1d(x[:, 0], p, predict, self.n_x, self.n_y, self.p_lb, self.p_ub, "iter={0:d}".format(iter))
                elif self.dim == 1:
                    x = self.X[self.n_grid * t_point:self.n_grid * (t_point + 1), :]
                    p = self.P[self.n_grid * t_point:self.n_grid * (t_point + 1), 0]
                    predict = self.predict(x)
                    plot2d(p, predict, self.n_x, self.n_y, self.p_lb, self.p_ub, "iter={0:d}".format(iter))

    def predict(self, x):
        pressure = self.sess.run(self.p_pred_tr,
                                 {self.x_tr_tf: x[:, 0:1],
                                  self.y_tr_tf: x[:, 1:2],
                                  self.t_tr_tf: x[:, 2:3]})
        return pressure


def plot1d(x, predict, true, nx, ny, v_min, v_max, title):
    predict = np.reshape(predict / 1000.0, [nx, ny])
    true = np.reshape(true / 1000.0, [nx, ny])

    plt.figure()
    plt.title(title, fontsize=16)
    plt.plot(x, true, 'b--', linewidth=3.0)
    plt.plot(x, predict, 'r--', linewidth=3.0)
    plt.ylim([v_min / 1000.0, v_max / 1000.0])
    plt.xlabel('X (m)', fontsize=16)
    plt.ylabel('P(x, t) (kPa)', fontsize=16)
    plt.show()


def plot2d(predict, true, nx, ny, v_min, v_max, title):
    predict = np.reshape(predict / 1000, [nx, ny])
    true = np.reshape(true / 1000, [nx, ny])

    fig, ax = plt.subplots(1, 2)
    plt.suptitle(title, fontsize=16)
    ia = ax[0].imshow(true, cmap='jet', vmin=v_min, vmax=v_max)
    ax[0].set_title("True")
    plt.colorbar(ia, ax=ax[0], fraction=0.05, pad=0.05)

    ib = ax[1].imshow(predict, cmap='jet', vmin=v_min, vmax=v_max)
    ax[1].set_title("Predict")
    plt.colorbar(ib, ax=ax[1], fraction=0.05, pad=0.05)

    plt.tight_layout()
    plt.show()


def plot_pde_err(pde1, pde2, pde3):
    fig, ax = plt.subplots(1, 4)
    ax[0].boxplot([np.reshape(pde1, [-1])], sym="b*")
    ax[0].set_xticklabels(['p_xx'])
    ax[1].boxplot([np.reshape(pde2, [-1])], sym="b*")
    ax[1].set_xticklabels(['q'])
    ax[2].boxplot([np.reshape(pde3, [-1])], sym="b*")
    ax[2].set_xticklabels(['p_t'])
    ax[3].boxplot([np.reshape(pde1+pde2+pde3, [-1])], sym="b*")
    ax[3].set_xticklabels(['pde_loss'])
    plt.suptitle('Multiple box plots of PDE equations')
    # plt.xticks([1, 2, 3], ['p_xx', 'q', 'p_t'])
    plt.show()


if __name__ == "__main__":
    n_dim = 1
    g = GenData.GenData(n_dim, './Data/1D_Neumann_t_20.mat')
    # g = GenData.GenData(n_dim, './Data/1D_Neumann.mat')

    n_t = 11

    X_train, P_train, Q_train = g.train_data(1, n_t)
    X_pde, P_pde, Q_pde = g.pde_data(np.arange(10))#[1, 2])

    n_x = g.n_x
    n_y = g.n_y
    t = g.t
    X = g.X[:, :]
    P = g.P[:, :]
    Q = g.Q[:, :]

    n_grid = n_x*n_y

    # training data validation data 8:2
    N_tr = X_train.shape[0]
    idx = np.random.choice(N_tr, N_tr, replace=False)
    # idx = np.arange(0, N_u, 1)

    idx_train = idx[:]#idx[0:round(N_tr * 0.7)]
    idx_val = idx[round(N_tr * 0.7):N_tr]

    X_tr = X_train[idx_train, :]
    P_tr = P_train[idx_train, :]
    Q_tr = Q_train[idx_train, :]

    X_val = X_train[idx_val, :]
    P_val = P_train[idx_val, :]
    Q_val = Q_train[idx_val, :]

    ### PINN Model
    model = PhysicsInformedNN(n_dim, n_x, n_y)
    model.load_data(X_tr, P_tr, Q_tr,
                    X_val, P_val, Q_val,
                    X_pde, P_pde, Q_pde,
                    X, P, Q)
    start_time = time.time()
    model.train_pressure(10000)
    model.train(10000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    P_pred = model.predict(X[:, :])

    ### PLOT
    for t_idx in np.arange(0, n_t, n_t<5 if 1 else int(n_t/5)):
        plot1d(X[n_grid * t_idx:n_grid * (t_idx + 1), 0:1],
               P[n_grid * t_idx:n_grid * (t_idx + 1)],
               P_pred[n_grid * t_idx:n_grid * (t_idx + 1)],
               n_x, n_y, P.min(0), P.max(0),
               "t = " + str(t[t_idx]) + "s")

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    loss_np = np.array(loss_graph)
    ax[0].plot(loss_np[:, 0], loss_np[:, 1])
    ax[0].set_title("Loss", fontsize=18)
    ax[0].set_xlabel('Iterations', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)

    val_np = np.array(val_graph)
    ax[1].plot(val_np[:, 0], val_np[:, 1])
    ax[1].set_title("Validation Loss", fontsize=18)
    ax[1].set_xlabel('Iterations', fontsize=16)
    ax[1].set_ylabel('Loss', fontsize=16)

    train_np = np.array(train_graph)
    ax[2].plot(train_np[:, 0], train_np[:, 1])
    ax[2].set_title('Training Loss', fontsize=18)
    ax[2].set_xlabel('Iterations', fontsize=16)
    ax[2].set_ylabel('Loss', fontsize=16)

    pde_np = np.array(pde_graph)
    ax[3].plot(pde_np[:, 0], pde_np[:, 1])
    ax[3].set_title("PDE Loss", fontsize=18)
    ax[3].set_xlabel('Iterations', fontsize=16)
    ax[3].set_ylabel('Loss', fontsize=16)

    plt.tight_layout()
    plt.show()

    ###