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

rng = np.random

np.random.seed(nnn)
tf.random.set_seed(nnn)
loss_graph = []
val_graph = []
u_graph = []
f_graph = []


class PhysicsInformedNN:
    def __init__(self,
                 X_p, P, Q_p,
                 X_val, P_val, Q_val,
                 X_f, Q_f,
                 layers, lb, ub,
                 obs_X, obs_P, obs_Q):

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

        nn_model = self.network(2, [20, 20, 20, 20], 1)
        nn_model.summary()

        p_hat = nn_model(X_p)
        loss = tf.reduce_mean(tf.square(p_hat - P))

        optimizer = tf.keras.optimizers.Adam()




        exit()

    def swish(self, x):
        return x*tf.math.sigmoid(x)

    def network(self, num_inputs=2, layers=[20, 20, 20, 20], num_outputs=1):
        inputs = tf.keras.layers.Input(shape=(num_inputs, ))
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation='tanh', kernel_initializer='he_normal')(x)
        outputs = tf.keras.layers.Dense(num_outputs, kernel_initializer='he_normal')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model


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
    n_pts = 10
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
                              obs_X_pts, obs_P_pts, obs_Q_pts)

    start_time = time.time()
    model.train(0)
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
    plt.figure()
    plt.title("t = 0s", fontsize=16)
    t_idx = 0
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 6,000s", fontsize=16)
    t_idx = 10
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 12,000s", fontsize=16)
    t_idx = 20
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 18,000s", fontsize=16)
    t_idx = 30
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
    plt.xlabel('x (m)', fontsize=16)
    plt.ylabel('p(x, t) (kPa)', fontsize=16)
    plt.show()

    plt.figure()
    plt.title("t = 24,000s", fontsize=16)
    t_idx = 40
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], P_star[n_x*t_idx:n_x*(t_idx+1)]/1000, 'b-', linewidth=3.0)
    plt.plot(X_star[n_x*t_idx:n_x*(t_idx+1), 0:1], p_pred[n_x*t_idx:n_x*(t_idx+1)]/1000, 'r--', linewidth=3.0)
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


