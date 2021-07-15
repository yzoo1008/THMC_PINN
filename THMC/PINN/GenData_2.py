import numpy as np
import scipy.io


class GenData:
    def __init__(self, dim, file):
        self.dim = dim
        self.file = file
        self.X, self.P, self.Q, self.n_x, self.n_y, self.n_t = self.load_data()

    def load_data(self):
        data = scipy.io.loadmat(self.file)
        if self.dim == 1:
            x = data['x'][:, 0] + 5.0
            t = data['t'][:, 0]
            p = data['usol'].T * 1000.0

            n_x = np.shape(np.unique(x))[0]
            n_t = np.shape(t)[0]

            y = np.tile([0], (n_x, ))

            n_y = np.shape(np.unique(y))[0]

        elif self.dim == 2:
            x = data['X_star'][:, 0]
            y = data['X_star'][:, 1]
            t = data['t'][:, 0]
            p = data['p_star'].T * 1000.0

            n_x = np.shape(np.unique(x))[0]
            n_y = np.shape(np.unique(y))[0]
            n_t = np.shape(t)[0]

        self.x = x
        self.y = y
        self.t = t

        x_tile = np.tile(x, (n_t, 1)).flatten()[:, None]
        y_tile = np.tile(y, (n_t, 1)).flatten()[:, None]
        t_tile = np.tile(t, (n_x * n_y, 1)).T.flatten()[:, None]

        X_star = np.hstack([x_tile, y_tile, t_tile])
        P_star = p.flatten()[:, None]
        Q_star = np.vstack([self.get_Q(X_star[:, 0:1], X_star[:, 1:2], 1, x[0], y[0], x[-1], y[-1])])

        return X_star, P_star, Q_star, n_x, n_y, n_t

    def get_Q(self, x, y, Q, I_x, I_y, P_x, P_y):
        inj = np.array((x == I_x) & (y == I_y), dtype=np.int)
        prd = np.array((x == P_x) & (y == P_y), dtype=np.int)
        return Q*(inj-prd)

    def I_cond(self, X, P, Q, n_x, n_y, n_pts):
        # t_0 ~ t_(n_pts-1)의 전좌표에서의 압력 값
        n_grid = n_x * n_y
        if n_pts >= 0:
            x_I = np.hstack([X[0:(n_pts * n_grid), 0].flatten()[:, None],
                             X[0:(n_pts * n_grid), 1].flatten()[:, None],
                             X[0:(n_pts * n_grid), 2].flatten()[:, None]])
            p_I = P[0:(n_pts * n_grid), 0].flatten()[:, None]
            q_I = Q[0:(n_pts * n_grid), 0].flatten()[:, None]

            return x_I, p_I, q_I
        else:
            x_I = np.hstack([X[:, 0].flatten()[:, None],
                             X[:, 1].flatten()[:, None],
                             X[:, 2].flatten()[:, None]])
            p_I = P[:, 0].flatten()[:, None]
            q_I = Q[:, 0].flatten()[:, None]

            return x_I, p_I, q_I

    def B_cond(self, X, P, Q, n_x, n_y, n_t):
        # P((x_0, y_0), t)
        n_grid = n_x * n_y
        x_B = np.hstack([X[0:(n_t*n_grid):n_grid, 0].flatten()[:, None],
                         X[0:(n_t*n_grid):n_grid, 1].flatten()[:, None],
                         X[0:(n_t*n_grid):n_grid, 2].flatten()[:, None]])
        p_B = P[0:(n_t*n_grid):n_grid, 0].flatten()[:, None]
        q_B = Q[0:(n_t*n_grid):n_grid, 0].flatten()[:, None]

        # P((x_n, y_n), t)
        x_B2 = np.hstack([X[n_grid-1:(n_t+1)*n_grid-1:n_grid, 0].flatten()[:, None],
                         X[n_grid-1:(n_t+1)*n_grid-1:n_grid, 1].flatten()[:, None],
                         X[n_grid-1:(n_t+1)*n_grid-1:n_grid, 2].flatten()[:, None]])
        p_B2 = P[n_grid-1:(n_t+1)*n_grid-1:n_grid, 0].flatten()[:, None]
        q_B2 = Q[n_grid-1:(n_t+1)*n_grid-1:n_grid, 0].flatten()[:, None]

        return np.vstack([x_B, x_B2]), np.vstack([p_B, p_B2]), np.vstack([q_B, q_B2])

    def grid_data_for_specific_t(self, t):
        # 임의의 t에서의 값
        n_grid = self.n_x * self.n_y
        X = self.X[n_grid * t:n_grid * (t + 1), 0:1]
        Y = self.X[n_grid * t:n_grid * (t + 1), 1:2]
        T = self.X[n_grid * t:n_grid * (t + 1), 2:3]
        P = self.P[n_grid * t:n_grid * (t + 1), 0:1]
        Q = self.Q[n_grid * t:n_grid * (t + 1), 0:1]
        return X, Y, T, P, Q

    def well_data_for_specific_t(self, t):
        n_grid = self.n_x * self.n_y
        inj_x = self.X[n_grid * t:n_grid * t + 1, 0:3]
        inj_p = self.P[n_grid * t:n_grid * t + 1, 0:1]
        inj_q = self.Q[n_grid * t:n_grid * t + 1, 0:1]

        prd_x = self.X[n_grid * (t + 1) - 1:n_grid * (t + 1), 0:3]
        prd_p = self.P[n_grid * (t + 1) - 1:n_grid * (t + 1), 0:1]
        prd_q = self.Q[n_grid * (t + 1) - 1:n_grid * (t + 1), 0:1]

        X = np.vstack([inj_x, prd_x])
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        p = np.vstack([inj_p, prd_p])
        q = np.vstack([inj_q, prd_q])
        return x, y, t, p, q

    def train_data(self, n_I, n_B):
        x_I, p_I, q_I = self.I_cond(self.X, self.P, self.Q, self.n_x, self.n_y, n_I)
        x_B, p_B, q_B = self.B_cond(self.X, self.P, self.Q, self.n_x, self.n_y, n_B)
        if n_B == 0:
            return x_I[:, 0:1], x_I[:, 1:2], x_I[:, 2:3], p_I, q_I
        if n_I == 0:
            return x_B[:, 0:1], x_B[:, 1:2], x_B[:, 2:3], p_B, q_B
        X = np.vstack([x_I, x_B])
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        return x, y, t, np.vstack([p_I, p_B]), np.vstack([q_I, q_B])

    def pde_data(self, t_list):
        X, Y, T, P, Q = self.grid_data_for_specific_t(t_list[0])
        if len(t_list) <= 1:
            return X, Y, T, P, Q
        for t in t_list:
            x, y, t, p, q = self.grid_data_for_specific_t(t)
            X = np.vstack([X, x])
            Y = np.vstack([Y, y])
            T = np.vstack([T, t])
            P = np.vstack([P, p])
            Q = np.vstack([Q, q])
        return X, Y, T, P, Q

    # def PDE_Data(self, x_space, y_space, t_space):
    #     x_max = max(self.x)
    #     x_min = min(self.x)
    #     y_max = max(self.y)
    #     y_min = min(self.y)
    #     t_max = max(self.t)
    #     t_min = min(self.t)
    #
    #     n_x_f = int((x_max - x_min) / x_space) + 1
    #     n_y_f = int((y_max - y_min) / y_space) + 1
    #     n_t_f = int((t_max - t_min) / t_space) + 1
    #
    #     x_f = np.tile(np.arange(x_min, x_max + x_space, x_space), (1, n_y_f)).flatten()[:, None]
    #     y_f = np.tile(np.arange(y_min, y_max + y_space, y_space), (n_x_f, 1)).T.flatten()[:, None]
    #     t_f = np.arange(t_min, t_max + t_space, t_space)
    #
    #     x_tile_f = np.tile(x_f, (n_t_f, 1)).flatten()[:, None]
    #     y_tile_f = np.tile(y_f, (n_t_f, 1)).flatten()[:, None]
    #     t_tile_f = np.tile(t_f, (n_x_f * n_y_f, 1)).T.flatten()[:, None]
    #
    #     X_f = np.hstack([x_tile_f, y_tile_f, t_tile_f])
    #     Q_f = np.vstack([self.get_Q(X_f[:, 0:1], X_f[:, 1:2], 1, x_min, y_min, x_max, y_max)])
    #
    #     return X_f, Q_f
