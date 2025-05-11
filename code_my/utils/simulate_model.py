import torch
from utils.state_wrapper import STATE_DEF_LIST_SHORT
from copy import deepcopy
import torchdiffeq as ode


def simulate_model(model, x_data, dt, states_to_simulate,
                   solver_settitngs={'method': 'dopri5', 'rtol': 1e-5, 'atol': 1e-6}):
    with torch.no_grad():

        x0 = x_data[0].clone().detach().unsqueeze(0)
        x_res = x_data.clone().detach() * 0.0

        states_idx = [i for i, state in enumerate(STATE_DEF_LIST_SHORT) if state not in states_to_simulate]

        for i in range(x_data.shape[0]):
            x0[0, states_idx] = x_data[i, states_idx]

            x_res[i] = deepcopy(x0[0])
            x0 = ode.odeint(model, x0, dt, **solver_settitngs)[-1]
            
            if i % 100 == 0:            
                print(f'\r{i+1}/{x_data.shape[0]}', end='')

    return x_res
