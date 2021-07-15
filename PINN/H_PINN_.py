import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import GenData


# Load Dataset
n_dim = 1
g = GenData.GenData(n_dim, './Data/1D_Neumann.mat')
n_I = 20
n_B = 41
X_p, P_p, Q_p = g.train_data(n_I, n_B)
X_pde, P_pde, Q_pde = g.pde_data(np.arange(n_B).tolist())

# Divide the dataset.
N_tr = X_p.shape[0]
idx = np.random.choice(N_tr, N_tr, replace=False)

idx_train = idx[:]#idx[0:round(N_tr * 0.7)]
idx_val = idx[round(N_tr * 0.7):N_tr]

X_tr = X_p[idx_train, :]
P_tr = P_p[idx_train, :]
Q_tr = Q_p[idx_train, :]

X_val = X_p[idx_val, :]
P_val = P_p[idx_val, :]
Q_val = Q_p[idx_val, :]

# Min-Max scaling.
max_X = np.max(X_tr, axis=0)
min_X = np.min(X_tr, axis=0)
max_P = np.max(P_tr, axis=0)
min_P = np.min(P_tr, axis=0)

X_tr_sc = 2 * (X_tr - min_X) / (max_X - min_X) - 1
Y_tr_sc = 2 * (P_tr - min_P) / (max_P - min_P) - 1
Q_tr_sc = Q_tr

X_val_sc = 2 * (X_val - min_X) / (max_X - min_X) - 1
Y_val_sc = 2 * (P_val - min_P) / (max_P - min_P) - 1
Q_val_sc = Q_val

# tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_tr_sc, Y_tr_sc, Q_tr_sc)).shuffle(10000).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_val_sc, Y_val_sc, Q_val_sc)).batch(32)

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

# Optimizer
optimizer = tf.keras.optimizers.Adam(0.0003)

@tf.function
def loss_function(y, y_hat):
    return tf.keras.losses.MSE(y, y_hat)

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

    # 4. 오차역전파(Back propagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, gradients


@tf.function
def test_step(X, Y):
    # 1. 예측 (prediction)
    predictions = model(X)
    # 2. Loss 계산
    loss = loss_function(Y, predictions)
    return loss

EPOCHS = 10

n_tr = len(train_ds)
n_val = len(val_ds)

for epoch in range(EPOCHS):
    train_loss = 0.0
    val_loss = 0.0
    for xyt, p_tr, q_tr in train_ds:
        loss, grad = train_step(xyt, p_tr)
        train_loss += sum(loss)/n_tr

    for xyt_val, p_val, q_val in val_ds:
        loss = test_step(xyt_val, p_val)
        val_loss += sum(loss)/n_val

    template = 'Epoch: {}, 학습 손실: {:.6f}, 검증 손실: {:.6f}'
    print(template.format(epoch+1, train_loss, val_loss))


# Visualization
n_x = g.n_x
n_y = g.n_y
t = g.t
for t_idx in [10, 20, 40]:
    X_obs, P_obs, Q_obs = g.grid_data_for_specific_t(t_idx)

    P_hat = model(2 * (X_obs[:, :] - min_X) / (max_X - min_X) - 1)
    P_res = ((P_hat + 1) / 2) * (max_P - min_P) + min_P

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    plt.suptitle("{}s".format(t[t_idx]), fontsize=12)
    ib = ax[0].imshow(np.reshape(P_obs/1000, [n_x, n_y]), cmap='jet', vmin=10200, vmax=9800)
    ax[0].set_title("True")
    plt.colorbar(ib, ax=ax[0], fraction=0.05, pad=0.05)

    ia = ax[1].imshow(np.reshape(P_res/1000, [n_x, n_y]), cmap='jet', vmin=10200, vmax=9800)
    ax[1].set_title("Predict")
    plt.colorbar(ia, ax=ax[1], fraction=0.05, pad=0.05)
    plt.show()