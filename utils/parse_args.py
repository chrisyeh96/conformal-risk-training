import argparse
from collections.abc import Sequence

import torch


def parse_args(
    commands: Sequence[str],
    lr: bool = True,
    l2reg: bool = True,
    datasets: Sequence[str] = ()
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        'command', choices=commands,
        help=f'command to run, one of {commands}')
    p.add_argument(
        '--alpha', type=float, nargs='+',
        default=(0.01, 0.05, 0.1, 0.2),
        help='risk level in (0, 1)')

    if lr:
        p.add_argument(
            '--lr', type=float, nargs='+',
            help='learning rate (unused for command "best_hp")')
    if l2reg:
        p.add_argument(
            '--l2reg', type=float, nargs='+',
            help='L2 regularization strength (unused for command "best_hp")')
    if len(datasets) > 1:
        p.add_argument(
            '--dataset', choices=datasets, required=True,
            help='dataset')

    p.add_argument(
        '--shuffle', action='store_true',
        help='shuffle the dataset before splitting into train/calib/test')
    p.add_argument(
        '--future-temp', action='store_true',
        help='whether to include future temperature features')
    p.add_argument(
        '--multiprocess', type=int, default=1,
        help='number of processes to use for multiprocessing')
    p.add_argument(
        '--tag', default='',
        help='tag to append to the model name')
    p.add_argument(
        '--device', default='cpu',
        help='either "cpu", "cuda", or "cuda:<device_id>"')

    args = p.parse_args()

    for alpha in args.alpha:
        if not (0 < alpha < 1):
            raise ValueError(f'alpha must be in (0, 1), got {alpha}')

    if args.tag != '':
        args.tag = f'_{args.tag}'

    return args


def check_args_device(device: str) -> None:
    """Checks if requested device is available."""
    if device.startswith('cuda'):
        # get number of available cuda devices
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise ValueError('CUDA is not available.')

        if device.startswith('cuda:'):
            device_id = int(device.split(':')[1])
            if device_id > num_gpus:
                raise ValueError(f'CUDA device {device_id} is not available. Only found '
                                 f'{num_gpus} devices.')
    elif device == 'cpu':
        pass
    else:
        raise ValueError(f'Invalid device: {device}. Must be "cpu", "cuda", or "cuda:#".')
