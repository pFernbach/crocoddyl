import numpy as np
import time
import pinocchio as pin
import crocoddyl
from crocoddyl import ActivationModelQuadraticBarrier
from crocoddyl.utils import LQRModelDerived
import matplotlib
#matplotlib.use("Qt4agg")
import matplotlib.pyplot as plt
from math import floor

DT = 0.05
H = 0.83
G = 9.81
SIZE_FEET = 0.05

class ActionModelCOP(LQRModelDerived):
    """
    Define an action model corresponding to the MPC formulation of weiber 2010
    state = [cx, cy, dcx, dcy, ddcx, ddcy, pfx, pfy] c: COM position, pf: current position of the foot
    control = [dddcx, dddcy, , pfx, pfy] pf: next position of the foot (only used in the the switching model)
    """
    def __init__(self, v_mean, weight_control_regularization=0, weight_com_velocity=1, weight_cop_centering=1e-6, h=H, g=G, dt=0.05):
        LQRModelDerived.__init__(self, 8, 4)
        # other cost function not from the LQR formulation (eg. inequalities represented with quadratic barriers)
        self.activation_cop_constraints = ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(np.array([0.]), np.array([SIZE_FEET * SIZE_FEET])))
        self.cop_constraints_data = self.activation_cop_constraints.createData()
        # define some selection matrices used to get individual component from the state/control vector
        self.Sp = np.zeros([2, 8])
        self.Sv = np.zeros([2, 8])
        self.Sa = np.zeros([2, 8])
        self.Sc = np.zeros([2, 8])
        self.Sj = np.zeros([2, 4])
        self.Sp[0, 0] = 1
        self.Sp[1, 1] = 1
        self.Sv[0, 2] = 1
        self.Sv[1, 3] = 1
        self.Sa[0, 4] = 1
        self.Sa[1, 5] = 1
        self.Sc[0, 6] = 1
        self.Sc[1, 7] = 1
        self.Sj[0, 0] = 1
        self.Sj[1, 1] = 1
        # Z = Pz * X (COP computation)
        self.Pz = np.zeros([2, 8])
        self.Pz[0, 0] = 1
        self.Pz[0, 4] = h / g
        self.Pz[1, 1] = 1
        self.Pz[1, 5] = h / g
        self.Pz[0, 6] = -1
        self.Pz[1, 7] = -1


        # Define transition model matrixes
        Fx = np.identity(8)
        Fx[0, 2] = dt
        Fx[0, 4] = dt * dt / 2.
        Fx[1, 3] = dt
        Fx[1, 5] = dt * dt / 2.
        Fx[2, 4] = dt
        Fx[3, 5] = dt
        self.Fx = Fx
        Fu = np.zeros([8, 4])
        Fu[0, 0] = dt**3 / 6.
        Fu[1, 1] = dt**3 / 6.
        Fu[2, 0] = dt*dt / 2.
        Fu[3, 1] = dt*dt / 2.
        Fu[4, 0] = dt
        Fu[5, 1] = dt
        self.Fu = Fu


        # Define cost matrixes:
        # quadratic cost part related to regularization of U:
        Luu = np.zeros([4, 4])
        Luu[:2, :2] = Luu[:2, :2] + np.identity(2) * weight_control_regularization
        self.Luu = Luu
        # quadratic cost related to cop placement:
        Lxx = self.Pz.transpose() @ self.Pz * weight_cop_centering
        # quadratic cost related to the mean velocity:
        Lxx += self.Sv.transpose() @ self.Sv * weight_com_velocity
        self.Lxx = Lxx
        # no other part in the quadratic part cost function:
        self.Lxu = np.zeros([8, 4])

        # linear part of the cost function:
        lx =  np.zeros(8)
        # cost for following the mean velocity:
        lx += -self.Pv.transpose() @ v_mean * weight_com_velocity
        self.lx = lx
        self.lu = np.zeros(4)

    def calc(self, data, x, u=None):
        super().calc( data, x, u)
        # Add COP constraints :
        Pf = x[6:]
        r = np.linalg.norm(self.Pz @ x - Pf) ** 2
        # print("Calc residual cop constraint : ", r)
        self.activation_cop_constraints.calc(self.cop_constraints_data, np.array([r]))
        # print("Calc activation cost : ",  self.cop_constraints_data.a)
        # print("Cost before constraints : ", data.cost)
        data.cost += self.cop_constraints_data.a * 1e6
        # print("Cost after constraints : ", data.cost)

        # Add kinematics constraints:
        #TODO

    def calcDiff(self, data, x, u=None):
        print("here diff")
        super().calcDiff(data, x, u)
        # Add COP constraints :
        Pf = x[6:]
        r = np.linalg.norm(self.Pz @ x - Pf) ** 2
        self.activation_cop_constraints.calcDiff(self.cop_constraints_data, np.array([r]))
        Rx = 2. * np.dot(self.Sv.transpose() @ self.Sp - self.Sv.transpose() @ self.Sa * (H/G) - self.Sv.transpose() @ self.Sc, x)
        Ru = 2. * np.dot(self.Sj.transpose() @ self.Sa * (H*H/(G*G)) + self.Sj.transpose() @ self.Sc * H / G + self.Sj.transpose() @ self.Sp * H / G, x)
        # print("RU shape : ", Ru.shape)
        # print("Rx shape : ", Rx.shape)
        Lx = data.Lx
        Lx += Rx.transpose() * self.cop_constraints_data.Ar * 1e6
        data.Lx = Lx
        Lu = data.Lu
        Lu += Ru.transpose() * self.cop_constraints_data.Ar * 1e6
        data.Lu = Lu

        # Add kinematics constraints:
        #TODO


class ActionModelCOPSwitch(ActionModelCOP):
    """
    Define an action model corresponding to the MPC formulation of weiber 2010
    state = [cx, cy, dcx, dcy, ddcx, ddcy, pfx, pfy] c: COM position, pf: current position of the foot
    control = [dddcx, dddcy, , pfx, pfy] pf: next position of the foot (only used in the the switching model)
    """
    def __init__(self, v_mean, weight_control_regularization=0, weight_com_velocity=1, weight_cop_centering=1e-6, h=H, g=G, dt=0.05):
        ActionModelCOP.__init__(self, v_mean, weight_control_regularization, weight_com_velocity, weight_cop_centering, h, g, dt)
        # change transition model : next contact from control and not from state
        self.Fx[6, 6] = 0
        self.Fx[7, 7] = 0
        self.Fu[6, 2] = 1
        self.Fu[7, 3] = 1




def plotCoP(xs):
    # format pinocchio stdVector to numpy array:
    n = len(xs)
    cop_t = np.zeros([2, n])
    Pz = np.zeros([2, 6])
    # Z = Pz * X (COP computation)
    Pz[0, 0] = 1
    Pz[0, 4] = H / G
    Pz[1, 1] = 1
    Pz[1, 5] = H / G
    for i in range(n):
        cop_t[:, i] = Pz @ xs[i][:6]
    # build a times vector
    times = np.array([i * DT for i in range(n)])
    fig = plt.figure("CoP(xy)")
    ax = fig.gca()
    plt.plot(cop_t[0, :].T, cop_t[1, :].T)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.axis('equal')
    plt.grid(True)

    fig, ax = plt.subplots(2, 1)
    fig.canvas.set_window_title("COP trajectory")
    fig.suptitle("COP trajectory", fontsize=20)
    labels = ["COP position X", "COP position Y"]
    colors = ['r', 'g']
    for i in range(2):
        ax_sub = ax[i]
        ax_sub.plot(times.T, cop_t[i, :].T, color=colors[i])
        ax_sub.set_xlabel('time (s)')
        ax_sub.set_ylabel(labels[i])
        ax_sub.yaxis.grid()
        ax_sub.xaxis.grid()

def plotCOM(xs):
    # format pinocchio stdVector to numpy array:
    n = len(xs)
    X_t = np.zeros([6, n])
    for i in range(n):
        X_t[:, i] = xs[i][:6]
    # build a times vector
    times = np.array([i * DT for i in range(n)])
    labels = ["x (m)", "y (m)", "dx (m/s)", "dy (m/s)", "ddx (m/s^2)", "ddy (m/s^2)"]
    colors = ['r', 'g']
    fig, ax = plt.subplots(3, 2)
    fig.canvas.set_window_title("COM trajectory")
    fig.suptitle("COM trajectory", fontsize=20)
    for i in range(3):  # line = pos,vel,acc
        for j in range(2):  # col = x,y,z
            ax_sub = ax[i, j]
            ax_sub.plot(times.T, X_t[i * 2 + j, :].T, color=colors[j])
            ax_sub.set_xlabel('time (s)')
            ax_sub.set_ylabel(labels[i * 2 + j])
            ax_sub.yaxis.grid()
            ax_sub.xaxis.grid()

def plotJerk(us):
    # format pinocchio stdVector to numpy array:
    n = len(us)
    U_t = np.zeros([2, n])
    for i in range(n):
        U_t[:, i] = us[i][:2]
    # build a times vector
    times = np.array([i * DT for i in range(n)])
    fig, ax = plt.subplots(2, 1)
    fig.canvas.set_window_title("COM Jerk trajectory")
    fig.suptitle("COM Jerk  trajectory", fontsize=20)
    labels = ["COM Jerk X", "COM Jerk Y"]
    colors = ['r', 'g']
    for i in range(2):
        ax_sub = ax[i]
        ax_sub.plot(times.T, U_t[i, :].T, color=colors[i])
        ax_sub.set_xlabel('time (s)')
        ax_sub.set_ylabel(labels[i])
        ax_sub.yaxis.grid()
        ax_sub.xaxis.grid()

"""
NUM_STEPS = 6
DURATION_DS = 0.1
DURATION_SS = 0.9

STEP_LENGHT = 0.1
n_ds = int(floor(DURATION_DS / DT))
n_ss = int(floor(DURATION_SS / DT))
current_x = 0.
current_t = 0.

running_models = []
for i in range(NUM_STEPS):
    model = ActionModelCOP([current_x, Y_OFFSET[i % 2], ])
    running_models += [model] * (n_ds + n_ss)
    current_x += STEP_LENGHT

current_x -= STEP_LENGHT
final_ds_model = ActionModelCOP([current_x, 0.])
running_models += [final_ds_model] * 20

"""

Y_OFFSET = [-0.1, 0.1]
V_MEAN = np.array([0.1, 0.])

running_model = ActionModelCOP(V_MEAN)
final_model = ActionModelCOP(V_MEAN)

# Formulating the optimal control problem
x0 = np.matrix([0., Y_OFFSET[0], 0, 0, 0, 0, 0, Y_OFFSET[0]]).T
problem = crocoddyl.ShootingProblem(x0, [running_model] * 30 , final_model)

# Creating the DDP solver for this OC problem, defining a logger
t_start = time.time()
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()
t_tot = time.time() - t_start
print("computation time of crocoddyl: " +str(t_tot * 1000.) +" ms" )
# plot :
plotCoP(ddp.xs)
plotCOM(ddp.xs)
plotJerk(ddp.us)
plt.show()