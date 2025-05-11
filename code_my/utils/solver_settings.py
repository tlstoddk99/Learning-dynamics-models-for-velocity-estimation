import torchdiffeq as ode
import torch


def get_solver_settings(args):
    return {"method": args.common_solver_method,
            "rtol": args.common_solver_rtol,
            "atol": args.common_solver_atol}


def get_ode_solve_method(args, model, fit_parameters):
    solver_settings = get_solver_settings(args)

    def ode_solve_adjoint(x, dt): return ode.odeint_adjoint(
        model, x, dt, **solver_settings, adjoint_params=tuple(fit_parameters))[-1]

    def ode_solve_naive(x, dt): return ode.odeint(
        model, x, dt, **solver_settings)[-1]

    if args.common_solver_backprop_adjoint:
        return ode_solve_adjoint
    else:
        return ode_solve_naive
