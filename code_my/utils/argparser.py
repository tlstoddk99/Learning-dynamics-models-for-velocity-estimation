import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Vehicle Dynamics Model Training")

    # Common parameters

    parser.add_argument("--common_solver_method", type=str, default="rk4",
                        help="Solver method (default: 'rk4')", choices=["euler", "rk4", "dopri5"])

    parser.add_argument("--common_solver_atol", type=float, default=1e-6,
                        help="Solver absolute tolerance (default: 1e-6)")

    parser.add_argument("--common_solver_rtol", type=float, default=1e-5,
                        help="Solver relative tolerance (default: 1e-5)")

    parser.add_argument("--common_solver_backprop_adjoint", type=bool, default=False,
                        help="Solver backprop adjoint (default: False)")

    parser.add_argument("--common_loss_fn", type=str, default="MSELoss",
                        help="Loss function (default: 'MSELoss')", choices=["MSELoss", "HUBBER"])

    parser.add_argument("--common_precision", type=int, default=32,
                        help="Precision (default: 32)", choices=[32, 64])

    parser.add_argument("--common_dataset_path", type=str, default="code/opti_test/hoons_all_train_and_val.csv",
                        help="Dataset path")

    parser.add_argument("--common_test_run_id", nargs='+', type=int,
                        help="Test run IDs (default: [val_id])", default=[7, 6, 23])

    parser.add_argument("--common_downsample_all", type=int, default=1,
                        help="Downsample all (default: 1)")

    parser.add_argument("--common_dataset_scaler", type=float, default=1.0,
                        help="Dataset scaler all (default: 1)")

    parser.add_argument("--common_loader_workers", type=int, default=3,
                        help="Loader workers (default: 4)")

    parser.add_argument("--common_device", type=str, default="cpu",
                        help="Device (default: 'cpu')", choices=["cpu", "cuda"])

    parser.add_argument("--common_Ts_mult", type=int, default=1,
                        help="Ts multiplier (default: 1)")

    # Classic Model parameters
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Random seed (default: 42)")

    parser.add_argument("--base_optimizer", type=str, default="AdamW",
                        help="Optimizer (default: 'AdamW')", choices=["Adam", "AdamW", "RMSProp"])

    parser.add_argument("--base_weight_decay", type=float, default=0.0,
                        help="Weight decay (default: 1e-5)")

    parser.add_argument("--base_tire_force_reg", type=float, default=0.0,
                        help="Tire force regularization (default: 0.0)")

    parser.add_argument("--base_epochs", type=int, default=5000,
                        help="Epochs (default: 1000)")

    parser.add_argument("--base_lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")

    parser.add_argument("--base_batch_size", type=int, default=2048,
                        help="Batch size (default: 1024)")

    parser.add_argument("--base_states_weights", nargs='+', type=float, default=[0.2225, 0.5064, 0.1566, 0.1145, 0.0, 0.0, 0.0],
                        help="States weights (default: [0.2225, 0.5064, 0.1566, 0.1145, 0.0, 0.0])")

    parser.add_argument("--base_tire_model", type=str, default="neural",
                        help="Tire model (default: 'Pacejka')", choices=["pacejka", "neural", "neural_const_friction", "neural_sr"])

    # Residual Model parameters
    parser.add_argument("--res_enable", type=int, default=0,
                        help="1 - model+res, 0 model")

    parser.add_argument("--res_activation", type=str, default="Sigmoid",
                        help="Activation function (default: 'Sigmoid')", choices=["ReLU", "Sigmoid", "Tanh", "Softpuls", "SiLU"])

    parser.add_argument("--res_layer_count", type=int, default=3,
                        help="Layer count [counting all MLP layers] (default: 3)")

    parser.add_argument("--res_mlp_size", type=int, default=0,
                        help="Hidden size (default: 256)")

    parser.add_argument("--res_input_normalization", type=int, default=0,
                        help="Input normalization  0")

    parser.add_argument("--res_seed", type=int, default=42,
                        help="Random seed (default: 42)")

    parser.add_argument("--res_optimizer", type=str, default="AdamW",
                        help="Optimizer (default: 'AdamW')", choices=["Adam", "AdamW", "RMSProp"])

    parser.add_argument("--res_epochs", type=int, default=1000,
                        help="Epochs (default: 1000)")

    parser.add_argument("--res_lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")

    parser.add_argument("--res_batch_size", type=int, default=2048,
                        help="Batch size (default: 1024)")

    parser.add_argument("--res_save_interval", type=int, default=20,
                        help="Save interval (default: 10)")

    parser.add_argument("--res_states_weights", nargs='+', type=float, default=[0.2225, 0.5064, 0.1566, 0.1145, 0.0, 0.0, 0.0],
                        help="States weights (default: [0.2225, 0.5064, 0.1566, 0.1145, 0.0, 0.0, 0.0])")

    # UKF parameters
    parser.add_argument("--ukf_res_model_epoch", type=int, default=-1,
                        help="Epochs (default: -1: best model) other values: epoch number of training res model,"
                        "must be modulo of res_save_interval")

    parser.add_argument("--ukf_seed", type=int, default=42,
                        help="Random seed (default: 42)")

    parser.add_argument("--ukf_optimizer", type=str, default="AdamW",
                        help="Optimizer (default: 'AdamW')", choices=["Adam", "AdamW", "RMSProp"])

    parser.add_argument("--ukf_lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")

    parser.add_argument("--ukf_epochs", type=int, default=1000,
                        help="Epochs (default: 1000)")

    parser.add_argument("--ukf_train_model", type=int, default=1,
                        help="Train model (default: 1) - if 0, only noise model will be trained")

    parser.add_argument("--ukf_batch_size", type=int, default=256,
                        help="Batch size (default: 100)")

    parser.add_argument("--ukf_start_loss", type=int, default=0,
                        help="Start loss size (default: 0)")

    parser.add_argument("--ukf_sequence_length", type=int, default=500,
                        help="Sequence length for training (default: 200)")

    parser.add_argument("--ukf_test_sequence_length", type=int, default=1000,
                        help="Sequence length for testing (default: 1000)")

    parser.add_argument("--ukf_grad_clip", type=float, default=1e3,
                        help="Gradient clipping (default: 1e3 - disabled)")

    parser.add_argument("--ukf_noise_model", type=str, default="hetero",
                        help="Noise model (default: 'diagonal')", choices=["diagonal", "crosscov", "hetero"])

    parser.add_argument("--ukf_states_weights", nargs='+', type=float, default=[0.2225, 0.5064, 0.1566, 0.1145, 1.0],
                        help="States weights (default: [0.2225, 0.5064, 0.1566, 0.1145])")

    parser.add_argument("--ukf_send_df_interval", type=int, default=50,
                        help="Send interval (default: 10)")

    parser.add_argument("--ukf_q_entropy_lb", type=float, default=-9.0,
                        help="Q entropy (default: -9.0)")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config_dict = vars(args)

    for key, value in config_dict.items():
        print(f"{key}: {value}")
        # if 'ukf' in key:
        #     print(f"UKF in key: {key} and value: {value}")
