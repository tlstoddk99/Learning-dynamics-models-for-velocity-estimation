import numpy as np

class VehicleParams:
    def __init__(self):
        self.m = 5.1
        self.g = 9.81
        self.I_z = 0.46
        self.L = 0.33
        self.lr = 0.115
        self.lf = self.L - self.lr
        self.Cd0 = 0.1
        self.Cd1 = 0.01
        self.Cd2 = 0.1
        self.R = 0.05
        self.I_e = 0.2
        self.K_fi = 0.90064745
        self.b1 = 0.304115174
        self.b0 = 0.50421894
        self.eps = 1e-6

        # 정적 하중 분포 계산
        self.Fn_f = self.m * self.g * self.lr / self.L
        self.Fn_r = self.m * self.g * self.lf / self.L

class PacejkaTireModel:
    def __init__(self):
        self.p = VehicleParams()

        # Pacejka tire parameters
        self.B_f = 5.94814182
        self.C_f = 2.27540553
        self.D_f = 0.8095089

        self.B_r = 5.94814182
        self.C_r = 2.27540553
        self.D_r = 0.8095089

        self.long_B = 2.8
        self.long_C = 1.2
        self.long_mu_tire = 0.891585168

    def slip_angle_front_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = x
        return np.arctan((v_y + self.p.lf * r) / (v_x + self.p.eps)) - delta

    def slip_angle_rear_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = x
        return np.arctan((v_y - self.p.lr * r) / (v_x + self.p.eps))

    def slip_ratio_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = x
        return (omega_wheels - v_x) / (v_x + self.p.eps)

    def slip_ratio_front_func(self, x):
        v_x, v_y, r, omega_wheels, friction, delta, Iq = x
        v_front = v_x * np.cos(delta) + (v_y + r * self.p.lr) * np.sin(delta)
        return (omega_wheels - v_front) / (v_front + self.p.eps)

    def Fy_f(self, slip_angle):
        return -self.p.Fn_f * self.D_f * np.sin(self.C_f * np.arctan(self.B_f * slip_angle))

    def Fy_r(self, slip_angle):
        return -self.p.Fn_r * self.D_r * np.sin(self.C_r * np.arctan(self.B_r * slip_angle))

    def Fx(self, slip_ratio):
        return self.p.m * self.p.g * self.long_mu_tire * np.sin(self.long_C * np.arctan(self.long_B * slip_ratio))

    def forward(self, x):
        Fy_f_ = self.Fy_f(self.slip_angle_front_func(x))
        Fy_r_ = self.Fy_r(self.slip_angle_rear_func(x))
        Fx_ = self.Fx(self.slip_ratio_func(x))

        Fx_f_ = Fx_ * self.p.lr / (self.p.lf + self.p.lr)
        Fx_r_ = Fx_ * self.p.lf / (self.p.lf + self.p.lr)

        return np.array([Fy_f_, Fy_r_, Fx_f_, Fx_r_])

class SingleTrackPacejkaModel:
    def __init__(self):
        self.tire_model = PacejkaTireModel()
        self.p = VehicleParams()

    def forward(self, t, x):
        p = self.p
        v_x, v_y, r, omega_wheels, friction, delta, Iq = x

        Fy_f, Fy_r, Fx_f, Fx_r = self.tire_model.forward(x)

        F_drag = p.Cd0 * np.sign(v_x) + \
                 p.Cd1 * v_x + \
                 p.Cd2 * v_x ** 2

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * np.cos(delta) -
                               Fy_f * np.sin(delta) - F_drag + p.m * v_y * r)

        v_y_dot = 1.0 / p.m * (Fx_f * np.sin(delta) +
                               Fy_r + Fy_f * np.cos(delta) - p.m * v_x * r)

        r_dot = 1.0 / p.I_z * (
            (Fx_f * np.sin(delta) + Fy_f * np.cos(delta)) * p.lf - Fy_r * p.lr)

        omega_wheels_dot = p.R / p.I_e * (p.K_fi * Iq - p.R * Fx_f - p.R * Fx_r
                                          - omega_wheels * p.b1 - np.sign(omega_wheels) * p.b0)

        return np.stack([
            v_x_dot,
            v_y_dot,
            r_dot,
            omega_wheels_dot,
            np.zeros_like(friction),
            np.zeros_like(delta),
            np.zeros_like(Iq)
        ], axis=-1)


def observation(model, x):
    x_dot = model.forward(0.0, x)
    v_x, v_y, r, omega_wheels, friction, delta, Iq = x
    v_x_dot, v_y_dot, r_dot, omega_wheels_dot, *_ = x_dot
    a_x = v_x_dot - r * v_y
    a_y = v_y_dot + r * v_x
    return np.stack([a_x, a_y, r, omega_wheels], axis=-1)
