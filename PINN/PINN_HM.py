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
        self.layers = [2, 40, 40, 40, 40, 40, 1]

    def load_data(self, X_tr, P_tr, Q_tr,
                  X_val, P_val, Q_val,
                  X_pde, P_pde, Q_pde,
                  X, P, Q):
        if self.dim == 1:
            self.lb = np.array([X[:, 0].min(0), X[:, 2].min(0)])
            self.ub = np.array([X[:, 0].max(0), X[:, 2].max(0)])
        elif self.dim == 2:
            self.lb = X.min(0)
            self.ub = X.max(0)
        self.p_lb = P.min(0)
        self.p_ub = P.max(0)

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
        self.k = tf.Variable([200.0], dtype=tf.float32)
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
        global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = 0.01
        self.learning_rate = tf.compat.v1.train.exponential_decay(
            initial_learning_rate, global_step, 1000, 0.9, staircase=True
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam'
        )
        self.optimize = self.optimizer.minimize(self.loss, global_step=global_step)

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

        p = p * (self.p_ub - self.p_lb) + self.p_lb
        return p

    def get_pde_err(self, x, y, t, q):
        mu = 0.001
        Ct = 11e-9
        phi = 0.2
        B = 0.9
        k = self.k * 9.8692 * 10 ** (-16)
        nu = [k / (mu * B), 20 / (24 * 60 * 60) / 1000, (phi * Ct) / B]

        p = self.get_p(x, y, t)

        p_t = tf.gradients(p, t)[0]
        p_x = tf.gradients(p, x)[0]
        p_xx = tf.gradients(p_x, x)[0]

        if self.dim == 1:
            pde_err = (nu[0] / nu[2]) * p_xx + (nu[1] / nu[2]) * q - p_t
        elif self.dim == 2:
            p_y = tf.gradients(p, y)[0]
            p_yy = tf.gradients(p_y, y)[0]
            pde_err = (nu[0] / nu[2]) * (p_xx + p_yy) + (nu[1] / nu[2]) * q - p_t

        return pde_err

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
                k = self.sess.run(self.k)

                loss_graph.append([iter, loss])
                val_graph.append([iter, val_loss])
                train_graph.append([iter, p_loss])
                pde_graph.append([iter, pde_loss])

                print('It: %d, Loss: %.3e, Loss_val: %.3e, Loss_p_tr: %.3e, Loss_pde: %.3e, k: %.3e, Time: %.2f'
                      % (iter, loss, val_loss, p_loss, pde_loss, k, elapsed_time))

            if iter % 500 == 0:
                if self.dim == 1:
                    x = self.X[self.n_grid * 200:self.n_grid * (200 + 1), :]
                    p = self.P[self.n_grid * 200:self.n_grid * (200 + 1), 0]
                    predict = self.predict(x)
                    plot1d(x[:, 0], p, predict, self.n_x, self.n_y, self.p_lb, self.p_ub, "iter={0:d}".format(iter))
                elif self.dim == 2:
                    x = self.X[self.n_grid * 200:self.n_grid * (200 + 1), :]
                    p = self.P[self.n_grid * 200:self.n_grid * (200 + 1), 0]
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
    plt.plot(x, true, 'b-', linewidth=3.0)
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


if __name__ == "__main__":
    n_dim = 1
    g = GenData.GenData(n_dim, './Data/1D_Neumann_t_20.mat')

    X_train, P_train, Q_train = g.train_Data(-1, 0)
    # X_train_t0, P_train_t0, Q_train_t0 = g.train_Data(100, 0)
    X_pde, P_pde, Q_pde = g.train_Data(-1, 0)

    # X_train = np.vstack([X_train, X_train_t0, X_train_t0, X_train_t0, X_train_t0, X_train_t0])
    # P_train = np.vstack([P_train, P_train_t0, P_train_t0, P_train_t0, P_train_t0, P_train_t0])
    # Q_train = np.vstack([Q_train, Q_train_t0, Q_train_t0, Q_train_t0, Q_train_t0, Q_train_t0])

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

    idx_train = idx[0:round(N_tr * 0.7)]
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
                    X, P, Q,)
    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    P_pred = model.predict(X[:, :])

    ### PLOT
    for t_idx in [0, 10, 20, 40, 50, 100, 200, 300, 400, 456]:
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