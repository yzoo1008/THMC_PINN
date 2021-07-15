import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


nnn = 1337 #이 숫자를 바꾸어주며 초기 뉴럴 네트워크의 설정을 바꿈
np.random.seed(nnn)
tf.set_random_seed(nnn)


val_graph = []
u_graph = []
f_graph = []

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, Q, Val_set):
        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]  # 0에는 위치 정보, x 가 들어있고 1에는 시간정도 t가 들어있다.
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers
        self.nu = nu
        self.Q = Q
        self.x_val = Val_set[:, 0:1]
        self.t_val = Val_set[:, 1:2]
        self.u_val = Val_set[:, 2:3]
        self.XT_val = Val_set[:, 0:2]
        self.n = 1

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.Q_tf = tf.placeholder(tf.float32, shape=[None, self.Q.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf, self.Q_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred))



        self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.u_val_tf = tf.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])
        self.XT_val_tf = tf.placeholder(tf.float32, shape=[None, self.XT_val.shape[1]])
        self.u_val_pred = self.net_u(self.x_val_tf, self.t_val_tf)
        self.f_val_pred = self.net_f2(self.x_val_tf, self.t_val_tf)

        # Val loss
        self.loss_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))
                        # tf.reduce_mean(tf.square(self.f_val_pred))


        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000, # 최대 iteration 횟수
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,     #음... 메모리에 저장해두는 정도? hessian 값을???
                                                                         'maxls': 50,      # Maximum number of line search steps per iteration
                                                                         'ftol': 1.0 * np.finfo(float).eps})  #the iteration stop 조건
                                                                                                           #float information  .eps : 표현 가능한 가장 작은 값을 돌려준다.  #Adam해볼까...


         # tf session
        init = tf.global_variables_initializer()
        self.sess.run(init)

# 맨 처음 초기값 알아서 잘 결정
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

# initialize 하는 방법 중에 하나. 자비애
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

#뉴럴 네트워크 만들기.
    def neural_net(self, X, weights, biases):  #X값 넣어주면 뉴럴넷 통해서 Y 출력해줌.
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 #scaling 값들을 일정 범위 내로 정리해주기./ -1~1
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return 1400*Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t, Q):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_x = self.nu[0] * u_x
        u_xx = tf.gradients(u_x, x)[0]

        f = u_xx + Q * self.nu[1] - self.nu[2] * u_t
        return f*6500000000000
        # return f

    def net_f2(self, x, t):
        Q = step_function(x)
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_x = self.nu[0] * u_x
        u_xx = tf.gradients(u_x, x)[0]

        f = u_xx + Q * self.nu[1] - self.nu[2] * u_t
        return f*6500000000000
        # return f

    # Loss 출력하는 함수.
    def callback(self, loss, loss_val, loss_u, loss_f):
        print('It: %d, Loss: %e, Loss_val: %e' % (self.n, loss, loss_val))
        if self.n % 100 ==0:
            val_graph.append([self.n, loss_val])
            u_graph.append([self.n, loss_u])
            f_graph.append([self.n, loss_f])
        self.n = self.n+1

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,self.x_val_tf: self.x_val, self.t_val_tf: self.t_val,
                   self.u_val_tf: self.u_val,self.Q_tf: self.Q}

        self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss, self.loss_val, self.loss_u, self.loss_f],
                                    loss_callback=self.callback)


    def predict(self, X_star, Q):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2], self.Q_tf: Q[:, 0:1]})

        return u_star, f_star


def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}.jpg'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))


def step_function(x):
    s1 = np.array(x == 0 , dtype=np.int)
    s2 = np.array(x == 390, dtype=np.int)
    Q = s1-s2
    return Q


if __name__ == "__main__":
    mu = 0.001
    Ct = 0.000000011
    phi = 0.2
    B = 0.9
    k = 200*9.8692*10**(-16)

    nu = [k/mu/B, 5/3600/1000/1000/10, phi*Ct/B] # 압력을 나누어준 만큼 나누어야한다.
    noise = 0.0

    layers = [2, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/1D_Neumann.mat')


    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    Exact = Exact/10 # make it kPa 에다가 10으로 더 나눠줌. 즉, 총 10000 으로 나누어줌 그래서 nu[1]도 나누어줌.



    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

#Boundary condition and initial condition

    xx1 = np.hstack(((X[0:10, :].T).flatten()[:, None], (T[0:10, :].T).flatten()[:, None]))
    uu1 = (Exact[0:10, :].T).flatten()[:, None] #  u(x,0)





######################################### N_u 최적화 하고 싶으면 이 값을 조정하면 된다.
    N_u = [80]
    N_f = [4000] #원본코드에서는 쓰였으나 여기선 쓰지 않음.


    for i in range(len(N_u)):
        for j in range(len(N_f)):



            X_u_train = np.vstack([xx1])

            # X_f_train = lb + (ub - lb) * lhs(2, N_f[j]) #원본코드에서 쓰였던 N_f
            X_f_train = X_star[:, :]


            u_train = np.vstack([uu1])
            Q = step_function(X_f_train[:,0:1])




            ################################### 경계조건을 주가해보자., 일부 앞부분만!
            aa = 10
            xx7 = np.hstack((X[0:aa, 0:1], T[0:aa, 0:1])) #
            uu7 = Exact[0:aa, 0:1] #u(0,t)  여기서 inject
            #
            xx8 = np.hstack((X[0:aa, -1:], T[0:aa, -1:]))
            uu8 = Exact[0:aa, -1:] #u(end,t)  여기서 produce

            idy = np.random.choice(xx7.shape[0], 6, replace=False)
            xx7 = xx7[idy, :]
            xx8 = xx8[idy, :]
            uu7 = uu7[idy, :]
            uu8 = uu8[idy, :]

            ### training data validation data 8:2
            idx = np.random.choice(X_u_train.shape[0], N_u[i], replace=False)

            idx_train = idx[0:round(N_u[i]*0.8)]
            idx_val = idx[round(N_u[i]*0.8):N_u[i]]

            X_u_val = X_u_train[idx_val, :]
            u_val = u_train[idx_val, :]

            X_u_train = X_u_train[idx_train, :]
            u_train = u_train[idx_train, :]
            Val_set = np.hstack([X_u_val, u_val])

            X_u_train = np.vstack([X_u_train, xx7, xx8])
            u_train = np.vstack([u_train, uu7, uu8])


            model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, Q, Val_set)

            start_time = time.time()
            model.train()
            elapsed = time.time() - start_time
            print('Training time: %.4f' % (elapsed))

            u_pred, f_pred = model.predict(X_star, Q)

            error_u_norm = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
            error_u = np.linalg.norm(u_star - u_pred, 2)
            error_f = np.linalg.norm(f_pred, 2)
            print('Error u_norm %e' % (error_u_norm))
            print('Error u: %e' % (error_u))
            print('f_pred: %e' % (error_f))

            U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
            Error = np.abs(Exact - U_pred)



            ######################################################################
            ############################# Plotting ###############################
            ######################################################################

            fig, ax = plt.subplots(nrows=2, ncols=3, )

            ####### Row 0: u(t,x) ##################
            ax = plt.subplot(211)
            h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                          extent=[t.min(), t.max(), x.min(), x.max()],
                          origin='lower', aspect='auto', vmin=600, vmax=1400)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            v = np.linspace(600, 1400, 5)
            fig.colorbar(h, cax=cax, ticks=v)


            ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
                    clip_on=False)

            L1 = 0
            # L1 = 20
            L2 = 30
            L3 = 38

            line = np.linspace(x.min(), x.max(), 2)[:, None]
            ax.plot(t[L1] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax.plot(t[L2] * np.ones((2, 1)), line, 'w-', linewidth=1)
            ax.plot(t[L3] * np.ones((2, 1)), line, 'w-', linewidth=1)

            ax.set_xlabel('t (sec)')
            ax.set_ylabel('x (m)')
            ax.legend(frameon=False, loc='best')
            ax.set_title('$u(t,x)$', fontsize=12)

            #
            # ####### Row 1: u(t,x) slices ##################
            # gs1 = gridspec.GridSpec(1, 3)
            # gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
            #
            #

        #왼쪽 아래 그림 그리기ㅣㅣㅣㅣ
            ax = plt.subplot(234)
            ax.plot(x, Exact[L1, :], 'b-', linewidth=2, label='Exact')
            ax.plot(x, U_pred[L1, :], 'r--', linewidth=2, label='Prediction')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('u(t,x) (kPa/10)')
            # ax.set_title('$t = 12000$', fontsize=10)
            ax.set_title('$t = 0$', fontsize=10)
            ax.axis('auto')
            ax.set_xlim([0, 400])
            # ax.set_ylim([600, 1400])


        #중간 아래 그림 그리기
            ax = plt.subplot(235)
            ax.plot(x, Exact[L2, :], 'b-', linewidth=2, label='Exact')
            ax.plot(x, U_pred[L2, :], 'r--', linewidth=2, label='Prediction')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('u(t,x) (kPa/10)')
            ax.axis('auto')
            ax.set_xlim([0, 400])
            ax.set_ylim([600, 1400])
            ax.set_title('$t = 18000$', fontsize=10)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

        #우측 아래 그림 그리기
            ax = plt.subplot(236)
            ax.plot(x, Exact[L3, :], 'b-', linewidth=2, label='Exact')
            ax.plot(x, U_pred[L3, :], 'r--', linewidth=2, label='Prediction')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('u(t,x) (kPa/10)')
            ax.axis('auto')
            ax.set_xlim([0, 400])
            ax.set_ylim([600, 1400])
            ax.set_title('$t = 22800$', fontsize=10)


        #     그림 안겹치게 하기. plt.tight_layout()
            plt.tight_layout()

            nums = [i, j]
            savefig('../figures/flow_equation_test_change_exercise{0}{1}'.format(*nums))

            plt.show()
        plt.figure()
        plt.title('validation error', fontsize=12)
        val_np = np.array(val_graph)
        plt.plot(val_np[:, 0], val_np[:, 1])
        plt.xlabel('total iteration number')
        plt.ylabel('Loss')
        plt.show(block=False)

        plt.figure()
        plt.title('training data based error', fontsize=12)
        u_np = np.array(u_graph)
        plt.plot(u_np[:, 0], u_np[:, 1])
        plt.xlabel('total iteration number', fontsize=10)
        plt.ylabel('Data based Loss', fontsize=10)
        plt.show(block=False)

        plt.figure()
        plt.title('PDE based error', fontsize=12)
        f_np = np.array(f_graph)
        plt.plot(f_np[:, 0], f_np[:, 1])
        plt.xlabel('total iteration number', fontsize=10)
        plt.ylabel('Physics based Loss', fontsize=10)
        plt.show(block=False)