import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import GenData_2
import time


# Functions
def minmax_scaling(X_sc, X):
    max_v = np.max(X, axis=0)
    min_v = np.min(X, axis=0)
    return 2 * (X_sc - min_v) / (max_v - min_v) - 1


def minmax_scaling_inv(X_sc, X):
    max_v = np.max(X, axis=0)
    min_v = np.min(X, axis=0)
    return (X_sc + 1) / 2 * (max_v - min_v) + min_v


def plot1d(x, predict, true, nx, ny, v_min, v_max, title, save_path=None):
    predict = np.reshape(predict / 1000.0, [nx, ny])
    true = np.reshape(true / 1000.0, [nx, ny])

    plt.figure()
    plt.title(title, fontsize=16)
    plt.plot(x, true, 'b--', linewidth=3.0)
    plt.plot(x, predict, 'r--', linewidth=3.0)
    plt.ylim([v_min / 1000.0, v_max / 1000.0])
    plt.xlabel('X (m)', fontsize=16)
    plt.ylabel('P(x, t) (kPa)', fontsize=16)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


# Loss Graph
train_loss_graph = []
val_loss_graph = []
p_loss_graph = []
pde_loss_graph = []

# Load Dataset
seed_number = 1234
tf.random.set_seed(seed_number)

n_dim = 1
datafile = './Data/1D_Neumann.mat'
g = GenData_2.GenData(n_dim, datafile)
n_I = 20
n_B = 41
x_p, y_p, t_p, p_p, q_p = g.train_data(n_I, n_B)
x_pde, y_pde, t_pde, p_pde, q_pde = g.pde_data(np.arange(n_B).tolist())

# Divide the dataset.
N_tr = x_p.shape[0]
idx = np.random.choice(N_tr, N_tr, replace=False)

idx_train = idx[:]  # idx[0:round(N_tr * 0.7)]
idx_val = idx[round(N_tr * 0.7):N_tr]

X_tr = x_p[idx_train, :]
Y_tr = y_p[idx_train, :]
T_tr = t_p[idx_train, :]
P_tr = p_p[idx_train, :]
Q_tr = q_p[idx_train, :]

X_val = x_p[idx_val, :]
Y_val = y_p[idx_val, :]
T_val = t_p[idx_val, :]
P_val = p_p[idx_val, :]
Q_val = q_p[idx_val, :]

X_pde = x_pde[:, :]
Y_pde = y_pde[:, :]
T_pde = t_pde[:, :]
P_pde = p_pde[:, :]
Q_pde = q_pde[:, :]

# Min-Max scaling.
ub = np.max(np.hstack([X_tr, T_tr]), axis=0)
lb = np.min(np.hstack([X_tr, T_tr]), axis=0)
p_ub = np.max(P_tr, axis=0)
p_lb = np.min(P_tr, axis=0)
W = (p_ub - p_lb) * 2

# tf.data.Dataset
batch_size = -1
shuffle = False
# train_ds = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr, T_tr, P_tr, Q_tr)).shuffle(10000).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val, T_val, P_val, Q_val)).shuffle(10000).batch(batch_size)
# pde_ds = tf.data.Dataset.from_tensor_slices((X_pde, Y_pde, T_pde, P_pde, Q_pde)).batch(batch_size)

# Model
input_ = tf.keras.layers.Input(shape=(2,))
x = tf.keras.layers.Lambda(lambda X: 2.0 * (X - lb) / (ub - lb) - 1.0)(input_)
x = tf.keras.layers.Dense(20, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(40, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(80, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(40, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(20, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(1, activation=None, kernel_initializer="glorot_normal")(x)
output_ = tf.keras.layers.Lambda(lambda X: (X + 1.0) / 2.0 * (p_ub - p_lb) + p_lb)(x)
model = tf.keras.models.Model(input_, output_)
model.summary()

# Optimizer
lr = 0.001
optimizer = tf.keras.optimizers.Adam(lr)


# Functions - Tensorflow
@tf.function
def get_p_err(p, p_pred):
    p_f64 = tf.cast(p, dtype=tf.float64)
    p_pred_f64 = tf.cast(p_pred, dtype=tf.float64)
    p_err = p_f64 - p_pred_f64
    return p_err


@tf.function
def get_pde_err(d2p_dx2, dp_dx, dp_dt, Q):
    mu = 0.001
    Ct = 11e-9
    phi = 0.2
    B = 0.9
    k = 200 * 9.8692 * 10 ** (-16)
    nu = [k, 120 / (24 * 60 * 60) / 1000 * (mu * B), phi * Ct * mu]

    nu_0 = tf.constant(nu[0], dtype=tf.float64)
    nu_1 = tf.constant(nu[1], dtype=tf.float64)
    nu_2 = tf.constant(nu[2], dtype=tf.float64)
    q_tr = tf.dtypes.cast(Q, tf.float64)

    pde_err = nu_0 * d2p_dx2 + nu_1 * q_tr - nu_2 * dp_dt
    return pde_err


@tf.function
def get_diff(X, T):
    with tf.GradientTape(persistent=True) as t_x:
        t_x.watch(X)
        with tf.GradientTape(persistent=True) as t_x2:
            t_x2.watch(X)
            P = model(tf.stack([X, T], axis=1))
        dp_dx = t_x2.gradient(P, X)
    d2p_dx2 = t_x.gradient(dp_dx, X)

    with tf.GradientTape(persistent=True) as t_t:
        t_t.watch(T)
        P = model(tf.stack([X, T], axis=1))
    dp_dt = t_t.gradient(P, T)

    return d2p_dx2, dp_dx, dp_dt


@tf.function
def train_step(X, T, P, X_pde, T_pde, Q_pde):
    # 미분을 위한 GradientTape을 적용합니다.
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        P_pred = model(tf.stack([X, T], axis=1))
        d2p_dx2, dp_dx, dp_dt = get_diff(X_pde, T_pde)
        # 2. Loss 계산
        p_err = get_p_err(P, P_pred) / W
        loss_p = tf.math.reduce_mean(tf.math.square(p_err))
        pde_err = get_pde_err(d2p_dx2, dp_dx, dp_dt, Q_pde)
        loss_pde = tf.math.reduce_mean(tf.math.square(pde_err))

        loss = loss_p + loss_pde

    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)

    # 4. 오차역전파(Back propagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, gradients, loss_p, loss_pde, dp_dx, dp_dt, d2p_dx2


@tf.function
def test_step(X, T, P):
    # 1. 예측 (prediction)
    P_pred = model(tf.stack([X, T], axis=1))
    # 2. Loss 계산
    p_err = get_p_err(P, P_pred) / W
    loss = tf.math.reduce_mean(tf.math.square(p_err))
    return loss


# Main
EPOCHS = 200

n_x = g.n_x
n_y = g.n_y
t = g.t

for epoch in range(EPOCHS):
    train_loss = 0.0
    val_loss = 0.0
    p_loss = 0.0
    pde_loss = 0.0

    loss, grad, loss_p, loss_pde, dp_dx, dp_dt, d2p_dx2 = train_step(X_tr, T_tr, P_tr, X_pde, T_pde, Q_pde)
    train_loss += loss.numpy()
    p_loss += loss_p.numpy()
    pde_loss += loss_pde.numpy()

    loss = test_step(X_val, T_val, P_val)
    val_loss += loss.numpy()

    # Graph
    train_loss_graph.append([epoch + 1, train_loss])
    val_loss_graph.append([epoch + 1, val_loss])
    p_loss_graph.append([epoch + 1, p_loss])
    pde_loss_graph.append([epoch + 1, pde_loss])

    if epoch % 100 == 0:
        print('Epoch: {}, Training Loss: {:.3e}, Validation Loss: {:.3e}, P_loss: {:.3e}, PDE_loss: {:.3e}'
              .format(epoch + 1, train_loss, val_loss, p_loss, pde_loss))

        obs_t_idx = 40
        X_obs, Y_obs, T_obs, P_obs, Q_obs = g.grid_data_for_specific_t(obs_t_idx)
        P_res = model(np.hstack([X_obs, T_obs]))
        plot1d(X_obs[:, 0], P_res, P_obs, n_x, n_y, 7E6, 13E6, "Epoch: {}, t: {}s".format(epoch + 1, t[obs_t_idx]))

# Visualization
now = time.localtime()
dir_name = '{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}' \
    .format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
dir_path = os.path.join('./Results', dir_name)
os.mkdir(dir_path)
print(dir_name)

for t_idx in [0, 5, 10, 20, 30, 40]:
    X_obs, Y_obs, T_obs, P_obs, Q_obs = g.grid_data_for_specific_t(t_idx)
    P_res = model(np.hstack([X_obs, T_obs]))
    plot1d(X_obs[:, 0], P_res, P_obs, n_x, n_y, 7E6, 13E6, "t={}s".format(t[t_idx]),
           os.path.join(dir_path, 't={}s'.format(t[t_idx]) + '.png'))

fig, ax = plt.subplots(1, 4, figsize=(20, 5))

np_train_loss = np.array(train_loss_graph)
ax[0].plot(np_train_loss[:, 0], np_train_loss[:, 1])
ax[0].set_title("Training Loss", fontsize=18)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Loss', fontsize=16)

np_val_loss = np.array(val_loss_graph)
ax[1].plot(np_val_loss[:, 0], np_val_loss[:, 1])
ax[1].set_title("Validation Loss", fontsize=18)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)

np_p_loss = np.array(p_loss_graph)
ax[2].plot(np_p_loss[:, 0], np_p_loss[:, 1])
ax[2].set_title('P Loss', fontsize=18)
ax[2].set_xlabel('Epochs', fontsize=16)
ax[2].set_ylabel('Loss', fontsize=16)

np_pde_loss = np.array(pde_loss_graph)
ax[3].plot(np_pde_loss[:, 0], np_pde_loss[:, 1])
ax[3].set_title("PDE Loss", fontsize=18)
ax[3].set_xlabel('Epochs', fontsize=16)
ax[3].set_ylabel('Loss', fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'Loss Graph.png'))
plt.show()

# Info file
f = open(os.path.join(dir_path, 'info.txt'), 'w')
f.write('n_dim: {}\n'.format(n_dim))
f.write('File name: {}\n'.format(datafile))
f.write('n_I: {}\n'.format(n_I))
f.write('n_B: {}\n'.format(n_B))
f.write('Loss P weight: {}\n'.format(W))
f.write('Epochs: {}\n'.format(EPOCHS))
f.write('Learning rate: {}\n'.format(lr))
f.write('Seed number: {}\n'.format(seed_number))
f.write('Batch size: {}\n'.format(batch_size))
f.write('Shuffle: {}\n'.format(shuffle))
model.summary(print_fn=lambda x: f.write(x + '\n'))
f.close()
