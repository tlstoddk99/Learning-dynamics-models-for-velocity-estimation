import matplotlib.pyplot as plt
from utils.state_wrapper import STATE_DEF_LIST, StateWrapper
import torch


def plot_state(t_data, x_data, x_sim, list_of_states):
    wx = StateWrapper(x_data)
    wx_res = StateWrapper(x_sim)
    for state in list_of_states:
        plt.subplot(len(list_of_states), 1, list_of_states.index(state) + 1)
        plt.title(state)
        plt.plot(t_data, getattr(wx, state), label=state)
        plt.plot(t_data, getattr(wx_res, state),
                 label=f'{state}_sim', linestyle='--')
        plt.legend()
        plt.grid()


def plot_tire_model(tire_model):

    slip_ratio = torch.linspace(-1.0, 1.0, 100)
    Fy_f = tire_model.Fx(slip_ratio).detach().numpy()

    slip_angle = torch.linspace(-0.7, 0.7, 100)
    Fy_f = tire_model.Fy_f(slip_angle).detach().numpy()
    Fy_r = tire_model.Fy_r(slip_angle).detach().numpy()

    plt.subplot(2, 1, 1)
    plt.title('Fx')
    plt.plot(slip_ratio, Fy_f, label='Fy_f')
    plt.legend()
    plt.grid()
    plt.xlabel('slip ratio [-]')
    plt.ylabel('Fx [N]')

    plt.subplot(2, 1, 2)
    plt.title('Fy')
    plt.plot(slip_angle, Fy_f, label='Fy_f')
    plt.plot(slip_angle, Fy_r, label='Fy_r')
    plt.xlabel('slip angle [rad]')
    plt.ylabel('Fy [N]')
    plt.legend()
    plt.grid()


def plot_with_covariance(t_data, x_data, x_data_est, P_over_time, batch=0):
    plt.subplot(4, 1, 1)
    plt.plot(t_data, x_data_est[batch, :, 0], label="vx est")
    plt.plot(t_data, x_data[batch, :, 0], label="vx")
    plt.fill_between(
        t_data,
        x_data_est[batch, :, 0] - P_over_time[batch, :, 0],
        x_data_est[batch, :, 0] + P_over_time[batch, :, 0],
        alpha=0.5,
    )
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t_data, x_data_est[batch, :, 1], label="vy est")
    plt.plot(t_data, x_data[batch, :, 1], label="vy")
    plt.fill_between(
        t_data,
        x_data_est[batch, :, 1] - P_over_time[batch, :, 1],
        x_data_est[batch, :, 1] + P_over_time[batch, :, 1],
        alpha=0.5,
    )
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t_data, x_data_est[batch, :, 2], label="r est")
    plt.plot(t_data, x_data[batch, :, 2], label="r")
    plt.fill_between(
        t_data,
        x_data_est[batch, :, 2] - P_over_time[batch, :, 2],
        x_data_est[batch, :, 2] + P_over_time[batch, :, 2],
        alpha=0.5,
    )
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t_data, x_data_est[batch, :, 3], label="omega_wheels est")
    plt.plot(t_data, x_data[batch, :, 3], label="omega_wheels")
    plt.fill_between(
        t_data,
        x_data_est[batch, :, 3] - P_over_time[batch, :, 3],
        x_data_est[batch, :, 3] + P_over_time[batch, :, 3],
        alpha=0.5,
    )
    plt.legend()

    plt.show()
