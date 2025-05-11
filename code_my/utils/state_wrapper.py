import torch
from typing import Dict
from collections import namedtuple

STATE_DEF_LIST = ['v_x', 'v_y', 'r', 'omega_wheels', 'friction',
                  'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu']
STATE_DEF_LIST_SHORT = STATE_DEF_LIST[:7]


WX = namedtuple('WX', ['v_x', 'v_y', 'r', 'omega_wheels',
                       'friction', 'delta', 'Iq'])


@torch.jit.script
def StateWrapper(x):
    return WX(v_x=x[..., 0], v_y=x[..., 1], r=x[..., 2], omega_wheels=x[..., 3],
              friction=x[..., 4], delta=x[..., 5], Iq=x[..., 6])
