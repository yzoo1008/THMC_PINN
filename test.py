import tensorflow as tf
import numpy as np



x_tr = np.arange(100)
y_tr = 2*x_tr^2 + 3*x_tr + 3

x_ts = np.arange(100, 105)
y_ts = 2*x_ts^2 + 3*x_ts + 3

@tf.function
def cal_f(x, y):
    y_x = tf.gradient(y, x)
    f = y_x - 2*x + 3
    return f

x_tf = tf.Variable()
y_tf = tf.Variable()
f = cal_f(x,y)

model = Sequential()
model.add(Dense(1, activation='relu', input_dim=1))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(optimizer='adam', loss='mse', matrics=['accuracy'])

model.fit(x_tr, y_tr)


model.evaluate(x_ts, y_ts)


