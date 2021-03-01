import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl import ActionModelLQR
import matplotlib
#matplotlib.use("Qt4agg")
import matplotlib.pyplot as plt

DT = 0.05
H = 0.83
G = 9.81

class ActionModelCOP(ActionModelLQR):
    """
    Define an action model corresponding to the MPC formulation of kajita 2007
    state = [cx, cy, dcx, dcy, ddcx, ddcy]
    control = [dddcx, dddcy]
    """
    def __init__(self, cop_target, weight_control_regularization=1e-6, weight_state_regularization=0, weight_cop=1e3, h=H, g=G, dt=0.05):
        ActionModelLQR.__init__(self, 6, 2)
        # Define transition model matrixes
        Fx = np.identity(6)
        Fx[0, 2] = dt
        Fx[0, 4] = dt * dt / 2.
        Fx[1, 3] = dt
        Fx[1, 5] = dt * dt / 2.
        Fx[2, 4] = dt
        Fx[3, 5] = dt
        self.Fx = Fx
        Fu = np.zeros([6, 2])
        Fu[0, 0] = dt**3 / 6.
        Fu[1, 1] = dt**3 / 6.
        Fu[2, 0] = dt*dt / 2.
        Fu[3, 1] = dt*dt / 2.
        Fu[4, 0] = dt
        Fu[5, 1] = dt
        self.Fu = Fu

        # Define cost matrixes:
        Pz = np.zeros([2, 6])
        # Z = Pz * X (COP computation)
        Pz[0, 0] = 1
        Pz[0, 4] = h / g
        Pz[1, 1] = 1
        Pz[1, 5] = h / g

        # quadratic cost part related to regularization of U:
        self.Luu = np.identity(2) * weight_control_regularization
        # quadratic cost related to cop placement:
        Lxx = Pz.transpose() @ Pz * weight_cop
        # regularization of velocity and acceleration
        Lxx[2:, 2:] = Lxx[2:, 2:] + np.identity(4) * weight_state_regularization
        self.Lxx = Lxx
        # no other part in the quadratic part cost function:
        self.Lxu = np.zeros([6, 2])

        # linear part of the cost function:
        self.lx = Pz.transpose() @ cop_target
        self.lu = np.zeros(2)


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
        cop_t[:, i] = Pz @ xs[i]
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
        X_t[:, i] = xs[i]
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
        U_t[:, i] = us[i]
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
