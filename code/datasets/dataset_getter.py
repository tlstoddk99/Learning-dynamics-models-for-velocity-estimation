import torch
from datasets.optitrack import OptitrackDataset
from datasets.optitrack_sequential import OptitrackDatasetSequential
from utils.state_wrapper import STATE_DEF_LIST


def get_loaders(args, mode):
    batch_size = getattr(args, f"{mode}_batch_size")

    train_dataset = OptitrackDataset(
        csv_file=args.common_dataset_path,
        subsample_all=args.common_downsample_all,
        Ts_multiplier=args.common_Ts_mult,
        test_run_id=args.common_test_run_id,
        test=False,
        dtype=torch.float32 if args.common_precision == 32 else torch.float64,
        dataset_scaler=args.common_dataset_scaler,
        device=torch.device(args.common_device),
    )

    test_dataset = OptitrackDataset(
        csv_file=args.common_dataset_path,
        subsample_all=args.common_downsample_all,
        Ts_multiplier=args.common_Ts_mult,
        test_run_id=args.common_test_run_id,
        test=True,
        dtype=torch.float32 if args.common_precision == 32 else torch.float64,
        device=torch.device(args.common_device)
    )
    assert train_dataset.state_def() == STATE_DEF_LIST, "state definitions must match"

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.common_loader_workers
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.common_loader_workers
    )
    return train_data_loader, test_data_loader, train_dataset, test_dataset


def get_sequence_loaders(args, mode):
    batch_size = getattr(args, f"{mode}_batch_size")

    train_dataset = OptitrackDatasetSequential(
        csv_file=args.common_dataset_path,
        subsample_all=args.common_downsample_all,
        Ts_multiplier=args.common_Ts_mult,
        check_new_run=True,
        test_run_id=args.common_test_run_id,
        test=False,
        dtype=torch.float32 if args.common_precision == 32 else torch.float64,
        device=torch.device(args.common_device),
        dataset_scaler=args.common_dataset_scaler,
        sequence_length=args.ukf_sequence_length
    )

    test_dataset = OptitrackDatasetSequential(
        csv_file=args.common_dataset_path,
        subsample_all=args.common_downsample_all,
        Ts_multiplier=args.common_Ts_mult,
        check_new_run=True,
        test_run_id=args.common_test_run_id,
        test=True,
        dtype=torch.float32 if args.common_precision == 32 else torch.float64,
        device=torch.device(args.common_device),
        sequence_length=args.ukf_test_sequence_length
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.common_loader_workers
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.common_loader_workers
    )

    return train_data_loader, test_data_loader, train_dataset, test_dataset
