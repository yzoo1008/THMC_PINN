import GenData
g = GenData.GenData(1, './Data/1D_Neumann.mat')
# g = GenData.GenData(2, './Data/2D_Neumann.mat')
X_tr, P_tr, Q_tr =g.trainData(10, 10)
X_pde, Q_pde = g.PDE_Data(10, 10, 600)

pass