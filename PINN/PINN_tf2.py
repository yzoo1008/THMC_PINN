import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import time
import GenData

n_dim = 1
g = GenData.GenData(n_dim, './Data/1D_Neumann.mat')
n_I = 20
n_B = 41
X_p, P_p, Q_p = g.train_data(n_I, n_B)
X_pde, P_pde, Q_pde = g.pde_data(np.arange(41))

X_p_train = X_p[:, :]
P_p_train = P_p[:, :]
Q_p_train = Q_p[:, :]

X_f_train = X_pde[:, :]
Q_f_train = Q_pde[:, :]

n_x = g.n_x
n_y = g.n_y
t = g.t
X = g.X[:, :]
P = g.P[:, :]
Q = g.Q[:, :]

n_grid = n_x*n_y

# training data validation data 8:2
N_u = n_I * n_grid + n_B * 2
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

# Dataset
max_X = np.max(X_p_train, axis=0)
min_X = np.min(X_p_train, axis=0)
max_Y = np.max(P_p_train, axis=0)
min_Y = np.min(P_p_train, axis=0)

X_tr = 2 * (X_p_tr - min_X) / (max_X - min_X) - 1
Y_tr = 2 * (P_p_tr - min_Y) / (max_Y - min_Y) - 1
Q_tr = Q_p_tr

X_val = 2 * (X_p_val - min_X) / (max_X - min_X) - 1
Y_val = 2 * (P_p_val - min_Y) / (max_Y - min_Y) - 1
Q_val = Q_p_val
##
train_ds = tf.data.Dataset.from_tensor_slices((X_tr, Y_tr, Q_tr)).shuffle(10000).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val, Q_val)).batch(32)

# Model
input_ = tf.keras.layers.Input(shape=(3,))
x = tf.keras.layers.Dense(20, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(input_)
x = tf.keras.layers.Dense(40, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(80, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(40, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
x = tf.keras.layers.Dense(20, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
output_ = tf.keras.layers.Dense(1, activation=None, kernel_initializer="glorot_normal")(x)
model = tf.keras.models.Model(input_, output_)
model.summary()

optimizer = tf.keras.optimizers.Adam(0.0003)

# train_loss = tf.keras.metrics.Mean()
# test_loss = tf.keras.metrics.Mean()

@tf.function
def loss_function(y, y_hat):
    return tf.keras.losses.MSE(y, y_hat)

@tf.function
def PDE_loss_function(X, Y, T, Q):
    mu = 0.001
    Ct = 11e-9
    phi = 0.2
    B = 0.9
    k = 200 * 9.8692 * 10 ** (-16)
    nu = [k / (mu * B), 20 / (24 * 60 * 60) / 1000, (phi * Ct) / B]

    X_tf = tf.Variable(X)
    Y_tf = tf.Variable(Y)
    T_tf = tf.Variable(T)

    XYT = tf.concat([X_tf, Y_tf, T_tf], axis=1)
    with tf.GradientTape(persistent=True) as t:
        with tf.GradientTape(persistent=True) as t2:
            P = model(XYT)

        dP_dX = t2.gradient(P, X_tf)
        dP_dY = t2.gradient(P, Y_tf)
        dP_dT = t2.gradient(P, T_tf)
    d2P_dX2 = t.gradient(dP_dX, X_tf)
    d2P_dY2 = t.gradient(dP_dY, Y_tf)

    return nu[0] * (d2P_dX2 + d2P_dY2) + nu[1] * Q - nu[2] * dP_dT


@tf.function
def grads(X):
    with tf.GradientTape(persistent=True) as t:
        with tf.GradientTape(persistent=True) as t2:
            P = model(X)

        dP_dX = t2.gradient(P, X)
    # d2P_dX2 = t.gradient(dP_dX, X)
    return dP_dX, P
    # return dP_dX, d2P_dX2


@tf.function
def train_step(X, Y):
    # 미분을 위한 GradientTape을 적용합니다.
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        predictions = model(X)
        # 2. Loss 계산
        loss = loss_function(Y, predictions)

    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)

    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss 업데이트 합니다.
    # train_loss(loss)
    return loss, gradients


@tf.function
def test_step(X, Y):
    # 1. 예측 (prediction)
    predictions = model(X)
    # 2. Loss 계산
    loss = loss_function(Y, predictions)

    # Test셋에 대해서는 gradient를 계산 및 backpropagation 하지 않습니다.

    # loss 업데이트 합니다.
    # test_loss(loss)
    return loss


EPOCHS = 10

n_tr = len(train_ds)
n_val = len(val_ds)

for epoch in range(EPOCHS):
    train_loss = 0.0
    val_loss = 0.0
    for xyt, p_tr, q_tr in train_ds:
        loss, grad = train_step(xyt, p_tr)
        # loss_f = PDE_loss_function(xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3], q_tr)
        grd, ppp = grads(xyt)
        print(grd, ppp)
        train_loss += sum(loss)/n_tr
        # print(loss_f)

    for xyt_val, p_val, q_val in val_ds:
        loss = test_step(xyt_val, p_val)
        val_loss += sum(loss)/n_val

    template = 'Epoch: {}, 학습 손실: {:.6f}, 검증 손실: {:.6f}'
    print (template.format(epoch+1, train_loss, val_loss))



for t_idx in [100, 200, 400]:
    X_obs, P_obs, Q_obs = g.get_Data(t_idx)

    P_hat = model(2 * (X_obs[:, :] - min_X) / (max_X - min_X) - 1)
    P_res = ((P_hat + 1) / 2) * (max_Y - min_Y) + min_Y

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    plt.suptitle("{}s".format(t[t_idx]), fontsize=12)
    ib = ax[0].imshow(np.reshape(P_obs/1000, [n_x, n_y]), cmap='jet', vmin=10200, vmax=9800)
    ax[0].set_title("True")
    plt.colorbar(ib, ax=ax[0], fraction=0.05, pad=0.05)

    ia = ax[1].imshow(np.reshape(P_res/1000, [n_x, n_y]), cmap='jet', vmin=10200, vmax=9800)
    ax[1].set_title("Predict")
    plt.colorbar(ia, ax=ax[1], fraction=0.05, pad=0.05)
    plt.show()


