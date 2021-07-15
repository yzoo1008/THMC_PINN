import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(12346)
tf.set_random_seed(12346)

val_graph = []
u_graph = []
f_graph = []

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, Val_set):
        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]  # 0에는 위치 정보, x 가 들어있고 1에는 시간정도 t가 들어있다.
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1] #임의로 뽑은 그 밑에 N_f 10000개 중에서 뽑은 임의의 지점에서의 위치와 시간.
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers
        self.nu = nu
        self.n = 1
        self.x_val = Val_set[:, 0:1]
        self.t_val = Val_set[:, 1:2]
        self.u_val = Val_set[:, 2:3]
        self.XT_val = Val_set[:, 0:2]

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

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred))

        # Loss_Val
        self.x_val_tf = tf.placeholder(tf.float32, shape=[None, self.x_val.shape[1]])
        self.t_val_tf = tf.placeholder(tf.float32, shape=[None, self.t_val.shape[1]])
        self.u_val_tf = tf.placeholder(tf.float32, shape=[None, self.u_val.shape[1]])
        self.XT_val_tf = tf.placeholder(tf.float32, shape=[None, self.XT_val.shape[1]])
        self.u_val_pred = self.net_u(self.x_val_tf, self.t_val_tf)
        self.f_val_pred = self.net_f(self.x_val_tf, self.t_val_tf)

        self.loss_val = tf.reduce_mean(tf.square(self.u_val_tf - self.u_val_pred))


        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        # tf session
        init = tf.global_variables_initializer()
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
        Y = 150*tf.add(tf.matmul(H, W), b) #스케일링으로 줄어든 범위를 다시 늘려줘야한다.
        return Y


    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

#f 계산하기. PDE 잘 만족하는지 체크.
    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = self.nu * u_t - u_xx

        return f*10000 #error_u와 크기를 맞춰줘야한다.

#Loss 출력하는 함수.
    def callback(self, loss, loss_val, loss_u, loss_f):
        print('It: %d, Loss: %e, Loss_val: %e' % (self.n, loss, loss_val))
        if self.n % 100 ==0:
            val_graph.append([self.n, loss_val])
            u_graph.append([self.n, loss_u])
            f_graph.append([self.n, loss_f])
        self.n = self.n+1

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f, self.x_val_tf: self.x_val, self.t_val_tf: self.t_val,
                   self.u_val_tf: self.u_val}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_val, self.loss_u, self.loss_f],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

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
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))


if __name__ == "__main__":
    k = 200*9.8692*10**(-16)
    mu = 0.001
    Ct = 0.000000011
    phi = 0.2
    nu = mu * phi * Ct / k #약 11.1457869
    noise = 0.0

    N_u = 50
    N_f = 3000
    layers = [2, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/1D_Dirichlet.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T/1000

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

#Boundary condition and initial condition
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T)) #초기시간의 (x,0) 값들
    uu1 = Exact[0:1, :].T #  u(x,0)


    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1] #u(0,t)

    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:] #u(end,t)  end = 100 for given data.

#바운더리 컨티션 만족하도록 train
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lhs(N_f, )
    X_f_train = X_star[:, :]
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    #validation
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)

    idx_train = idx[0:round(N_u * 0.8)]
    idx_val = idx[round(N_u * 0.8):N_u]

    X_u_val = X_u_train[idx_val, :]
    u_val = u_train[idx_val, :]

    X_u_train = X_u_train[idx_train, :]
    u_train = u_train[idx_train, :]
    Val_set = np.hstack([X_u_val, u_val])


    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, Val_set)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, f_pred = model.predict(X_star)

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
                  origin='lower', aspect='auto', vmin=100, vmax=150)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    v = np.linspace(100, 150, 3)
    fig.colorbar(h, cax=cax, ticks=v)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[71] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[81] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[91] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$u(t,x)$', fontsize=12)

#왼쪽 아래 그림 그리기ㅣㅣㅣㅣ
    ax = plt.subplot(234)
    # ax.plot(x, Exact[71, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[71, :], 'r--', linewidth=2, label='Prediction')
    ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    # ax.set_title('$t = 1400$', fontsize=10)
    ax.set_title('$t = 0$', fontsize=10)
    ax.axis('auto')
    ax.set_xlim([0, 100])
    ax.set_ylim([75, 175])

#중간 아래 그림 그리기
    ax = plt.subplot(235)
    ax.plot(x, Exact[81, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[81, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('auto')
    ax.set_xlim([0, 100])
    ax.set_ylim([75, 175])
    ax.set_title('$t = 1600$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

#우측 아래 그림 그리기
    ax = plt.subplot(236)
    ax.plot(x, Exact[91, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[91, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('auto')
    ax.set_xlim([0, 100])
    ax.set_ylim([75, 175])
    ax.set_title('$t = 1800$', fontsize=10)
    plt.tight_layout()

    savefig('../figures/flow_equation_multiply150')
    plt.show(block=False)

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
    plt.show()


