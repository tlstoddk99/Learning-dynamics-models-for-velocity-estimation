import pandas as pd
import torch


def save_as_dataframe(x_ukf: torch.tensor, x_gt, P, Ts=0.01):
    df_list = []
    for i in range(x_ukf.shape[0]):
        df = pd.DataFrame(columns=['vx_ukf', 'vy_ukf', 'r_ukf',
                          'omega_e_ukf', 'vx_gt', 'vy_gt', 'r_gt', 'omega_e_gt'])
        df['vx_ukf'] = x_ukf[i, :, 0].detach().numpy()
        df['vy_ukf'] = x_ukf[i, :, 1].detach().numpy()
        df['r_ukf'] = x_ukf[i, :, 2].detach().numpy()
        df['omega_e_ukf'] = x_ukf[i, :, 3].detach().numpy()
        df['friction_ukf'] = x_ukf[i, :, 4].detach().numpy()

        df['vx_gt'] = x_gt[i, :, 0].detach().numpy()
        df['vy_gt'] = x_gt[i, :, 1].detach().numpy()
        df['r_gt'] = x_gt[i, :, 2].detach().numpy()
        df['omega_e_gt'] = x_gt[i, :, 3].detach().numpy()
        df['friction_gt'] = x_gt[i, :, 4].detach().numpy()

        df['vx_e'] = df['vx_ukf'] - df['vx_gt']
        df['vy_e'] = df['vy_ukf'] - df['vy_gt']
        df['r_e'] = df['r_ukf'] - df['r_gt']
        df['omega_e_e'] = df['omega_e_ukf'] - df['omega_e_gt']
        df['friction_e'] = df['friction_ukf'] - df['friction_gt']

        df['P_vx'] = P[i, :, 0, 0].detach().numpy()
        df['P_vy'] = P[i, :, 1, 1].detach().numpy()
        df['P_r'] = P[i, :, 2, 2].detach().numpy()
        df['P_omega_e'] = P[i, :, 3, 3].detach().numpy()
        df['P_friction'] = P[i, :, 4, 4].detach().numpy()
        df_list.append(df)

    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    df['time'] = df.index * Ts
    return df
