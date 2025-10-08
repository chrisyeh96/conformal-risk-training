import argparse
from collections.abc import Iterable, Mapping
from collections import defaultdict
import functools
import io
import itertools
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm

from models.mlp import MLP, run_epoch_mlp, train_mlp
from storage.data import get_loaders, get_tensors, get_train_calib_split
from storage.problems import (
    StorageConstants, StorageProblemNonRobust, StorageProblemLambda,
    StorageProblemLambdaParameterized)
from utils.parse_args import check_args_device
from utils.multiprocess import run_parallel

Device = str | torch.device

INPUT_DIM = 101  # including future_temp
Y_DIM = 24
MAX_PRETRAIN_EPOCHS = 500
MAX_FINETUNE_EPOCHS = 100
BATCH_SIZE = 400
PSEUDOCAILB_SIZE = 200
SEEDS = range(10)
LOG_PRICES = False
LABEL_NOISE = 20

STORAGE_CONSTS = [
    StorageConstants(lam=0.1, eps=.05),
    # StorageConstants(lam=1, eps=.5),
    # StorageConstants(lam=10, eps=5),
    # StorageConstants(lam=35, eps=15)
]


def cvar(x: np.ndarray, q: float) -> float:
    # TODO: check numpy's quantile function
    return np.mean(x[x >= np.quantile(x, q)]).item()


def get_optimal_task_losses(
    shuffle: bool, label_noise: float, split: str, const: StorageConstants
) -> dict[str, np.ndarray]:
    """
    Returns a dict with keys:
        task_loss, z_in, z_out, z_net
    """
    tensors, y_info = get_tensors(
        shuffle=shuffle, log_prices=LOG_PRICES, label_noise=label_noise)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info

    prob = StorageProblemNonRobust(T=24, y_mean=y_mean, y_std=y_std, const=const)

    result = defaultdict(list)
    task_losses = []
    for y in tensors[f'Y_{split}'].numpy():  # type: ignore
        task_loss = prob.solve(y).value
        task_losses.append(task_loss)
        for k, v in prob.primal_vars.items():
            assert isinstance(v.value, np.ndarray)
            result[k].append(v.value)

    result = {k: np.concatenate(v) for k, v in result.items()}
    result['task_loss'] = np.array(task_losses)
    return result


def save_preds(
    seed: int, shuffle: bool, future_temp: bool, label_noise: float,
    const: StorageConstants, saved_ckpt_fmt: str, out_dir: str, device: Device,
    tag: str = ''
) -> None:
    """
    Saves the predictions of the model on the test split to a npz file.
    """
    tensors, y_info = get_tensors(
        shuffle=shuffle, log_prices=LOG_PRICES, future_temp=future_temp,
        label_noise=label_noise)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)

    model = MLP(input_dim=tensors['X_test'].shape[1], y_dim=Y_DIM)
    saved_ckpt_path = os.path.join(out_dir, saved_ckpt_fmt.format(seed=seed))
    model.load_state_dict(torch.load(saved_ckpt_path, weights_only=True))

    prob = StorageProblemNonRobust(T=24, y_mean=y_mean, y_std=y_std, const=const)

    results = run_epoch_mlp(
        model, loaders['test'], prob=prob, return_z=True, device=device)

    save_path = os.path.join(out_dir, f'preds{tag}_s{seed}.npz')
    tqdm.write(f'Saving predictions to {save_path}')
    np.savez_compressed(save_path, allow_pickle=False, **results)


def get_best_hp_for_seed(
    seed: int, shuffle: bool, future_temp: bool, label_noise: float, device: Device,
    out_dir: str, tag: str = '', saved_ckpt_fmt: str = '', **train_kwargs: Any
) -> tuple[list[tuple[float, float, int, float]], str]:
    tensors, _ = get_tensors(
        shuffle=shuffle, log_prices=LOG_PRICES, future_temp=future_temp,
        label_noise=label_noise)
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)

    best_val_loss = np.inf
    best_model = None

    lrs = 10. ** np.arange(-4, -1.4, 0.5)
    l2regs = [1e-4]

    pbar = tqdm(total=len(lrs) * len(l2regs))
    losses = []
    for lr, l2reg in itertools.product(lrs, l2regs):
        model = MLP(input_dim=tensors['X_test'].shape[1], y_dim=Y_DIM)
        if saved_ckpt_fmt != '':
            saved_ckpt_path = os.path.join(out_dir, saved_ckpt_fmt.format(seed=seed))
            model.load_state_dict(torch.load(saved_ckpt_path, weights_only=True))

        lr_scheduler_kwargs = dict(factor=0.1, patience=10)
        result = train_mlp(
            model, loaders, max_epochs=MAX_PRETRAIN_EPOCHS, lr=lr, l2reg=l2reg,
            lr_schedule_kwargs=lr_scheduler_kwargs, cutoff=20,
            return_best_model=True, device=device, show_pbar=True, **train_kwargs)
        losses.append((lr.item(), l2reg, seed, result['val_loss']))

        if result['val_loss'] < best_val_loss:
            best_val_loss = result['val_loss']
            best_model = model.cpu()

        pbar.update(1)

    assert best_model is not None
    ckpt_path = os.path.join(out_dir, f'mlp{tag}_s{seed}.pt')
    torch.save(best_model.state_dict(), ckpt_path)

    return losses, f'best val loss: {best_val_loss:.3f}'


def train_mse(
    shuffle: bool, future_temp: bool, device: Device, out_dir: str, label_noise: float,
    tag: str = '', saved_ckpt_fmt: str = '', seeds: Iterable[int] = SEEDS,
    multiprocess: int = 1, **train_kwargs: Any
) -> None:
    """Trains a MLP regression model on the storage problem with MSE loss.

    Performs hyperparameter tuning of learning rate and L2 regularization strength.
    Saves a CSV file to "{out_dir}/hyperparams{tag}.csv" with columns:
        lr, l2reg, seed, loss
    Saves the best model for each seed to
        "{out_dir}/mlp_s{seed}{tag}.pt".

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        future_temp: whether to include future temperature features
        device: 'cpu' or 'cuda'
        out_dir: where to save the results
        label_noise: stddev of Gaussian noise added to the labels
        tag: additional tag to append to CSV and saved model filenames
        saved_ckpt_fmt: format string for name of existing model checkpoint in
            out_dir, takes {seed} as an argument, e.g. 'mlp_s{seed}.pt'. Leave
            empty to train from scratch.
        seeds: list of seeds to use for training
        multiprocess: # of seeds to run in parallel
        train_kwargs: passed to `train_mlp`
    """
    func = functools.partial(
        get_best_hp_for_seed, shuffle=shuffle, future_temp=future_temp,
        label_noise=label_noise, device=device, out_dir=out_dir, tag=tag,
        saved_ckpt_fmt=saved_ckpt_fmt, **train_kwargs)
    kwargs_list = [dict(seed=s) for s in seeds]
    results = run_parallel(func, kwargs_list, workers=multiprocess)

    losses = []
    for result in results:
        if result is not None:
            losses.extend(result)

    df = pd.DataFrame(losses, columns=['lr', 'l2reg', 'seed', 'loss'])
    csv_path = os.path.join(out_dir, f'hyperparams{tag}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved results to {csv_path}')


def get_zs(
    prob: StorageProblemNonRobust, preds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = preds.shape[0]
    z_in = np.zeros((N, Y_DIM))
    z_out = np.zeros((N, Y_DIM))
    z_net = np.zeros((N, Y_DIM))
    for i, pred in enumerate(preds):
        prob.solve(pred)
        z_in[i] = prob.primal_vars['z_in'].value
        z_out[i] = prob.primal_vars['z_out'].value
        z_net[i] = prob.primal_vars['z_net'].value
    return z_in, z_out, z_net


def get_lams(
    tensors_dict: Mapping[str, Tensor], model: MLP, prob: StorageProblemNonRobust, device: Device,
    alphas: Iterable[float], deltas: Iterable[float]
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        alpha, delta, lambda, t, task loss, CVaR
    """
    X_train = tensors_dict['X_train'].to(device, non_blocking=True)
    X_val = tensors_dict['X_calib'].to(device, non_blocking=True)
    Y_train_np = tensors_dict['Y_train'].cpu().numpy()
    Y_val_np = tensors_dict['Y_calib'].cpu().numpy()

    # get decision variables
    with torch.no_grad():
        model.eval().to(device)
        pred_np_train = model(X_train).cpu().numpy()
        pred_np_val = model(X_val).cpu().numpy()
    z_in_train, z_out_train, z_net_train = get_zs(prob, pred_np_train)
    z_in_val, z_out_val, z_net_val = get_zs(prob, pred_np_val)

    max_lambda_prob_var_t = StorageProblemLambda(
        T=Y_DIM, const=prob.const,
        y=Y_train_np, y_mean=prob.y_mean, y_std=prob.y_std,
        z_in=z_in_train, z_out=z_out_train, z_net=z_net_train, quad=False, t_fixed=False)

    max_lambda_prob_fixed_t = StorageProblemLambda(
        T=Y_DIM, const=prob.const,
        y=Y_val_np, y_mean=prob.y_mean, y_std=prob.y_std,
        z_in=z_in_val, z_out=z_out_val, z_net=z_net_val, quad=False, t_fixed=True)

    rows = []
    for alpha, delta in itertools.product(alphas, deltas):
        max_lambda_prob_var_t.solve(alpha, delta)
        assert max_lambda_prob_var_t.t.value is not None
        t = max_lambda_prob_var_t.t.value.item()

        λ = max_lambda_prob_fixed_t.solve(alpha, delta, t=t)
        if λ == 0.:
            task_loss = 0.
            cvar_value = 0.
        else:
            task_loss = np.mean(max_lambda_prob_fixed_t.task_loss())
            cvar_value = cvar(max_lambda_prob_fixed_t.fis.value, q=delta)

        rows.append({
            'alpha': alpha, 'delta': delta,
            'lambda': λ, 't': t,
            'task loss': task_loss, 'CVaR': cvar_value
        })

    return pd.DataFrame(rows).set_index(['alpha', 'delta'])


def crc(
    shuffle: bool, future_temp: bool, label_noise: float, const: StorageConstants,
    alphas: Iterable[float], deltas: Iterable[float], seed: int, saved_ckpt_fmt: str,
    device: Device
) -> list[dict[str, float]]:
    """
    Post-hoc CRC. Always call this function within a torch.no_grad() context.

    Returns:
        list of dicts, one per (alpha, delta) pair, with keys:
            seed, alpha, delta, lambda, task loss, cvar
    """
    tensors, y_info = get_tensors(
        shuffle=shuffle, log_prices=LOG_PRICES, future_temp=future_temp,
        label_noise=label_noise)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)

    prob = StorageProblemNonRobust(T=Y_DIM, y_mean=y_mean, y_std=y_std, const=const)

    # load the model
    model = MLP(input_dim=tensors['X_test'].shape[1], y_dim=Y_DIM)
    saved_ckpt_path = saved_ckpt_fmt.format(seed=seed)
    model.load_state_dict(torch.load(saved_ckpt_path, weights_only=True))
    model.eval().to(device)

    calib_df = get_lams(
        tensors_dict=tensors_cv, model=model,
        prob=prob, device=device, alphas=alphas, deltas=deltas)

    # use lambdas on test set
    with torch.no_grad():
        model.eval().to(device)
        pred_np = model(tensors['X_test'].to(device)).cpu().numpy()  # type: ignore
    z_in, z_out, z_net = get_zs(prob, preds=pred_np)
    y_test = tensors['Y_test'].cpu().numpy()  # type: ignore
    rows = []
    for alpha, delta in itertools.product(alphas, deltas):
        λ = calib_df.loc[(alpha, delta), 'lambda']
        task_losses = prob.task_loss(z_in * λ, z_out * λ, z_net * λ, y=y_test, is_standardized=True)
        financial_losses = prob.financial_loss(z_in * λ, z_out * λ, y=y_test, is_standardized=True)
        assert isinstance(task_losses, np.ndarray)
        assert isinstance(financial_losses, np.ndarray)
        rows.append({
            'seed': seed, 'alpha': alpha, 'delta': delta, 'lambda': λ,
            'task loss': np.mean(task_losses).item(),
            'task losses': task_losses,
            'financial losses': financial_losses,
            'cvar': cvar(financial_losses, q=delta)
        })

    return rows


def e2ecrc(
    shuffle: bool, future_temp: bool, label_noise: float, const: StorageConstants,
    alpha: float, delta: float, seed: int, saved_ckpt_fmt: str, device: Device,
    lr: float, mse_loss_frac: float, max_epochs: int, batch_size: int, num_cal: int,
    savedir: str
) -> tuple[dict[str, Any], str]:
    """
    Conformal Risk Training

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        future_temp: whether to include future temperature features
        label_noise: stddev of Gaussian noise added to the labels
        const: StorageConstants object
        alpha: risk threshold
        delta: quantile level
        seed: random seed
        saved_ckpt_fmt: format string for name of existing model checkpoint in
            savedir, takes {seed} as an argument, e.g. 'mlp_s{seed}.pt'
        device: 'cpu' or 'cuda'
        lr: learning rate
        mse_loss_frac: weight for the MSE loss in [0, 1],
            (1-mse_loss_frac) is the weight for the task loss
        max_epochs: number of epochs to train for
        batch_size: batch size for training
        num_cal: number of calibration samples to use
        savedir: where to save the results
    """
    assert mse_loss_frac < 1, 'mse_loss_frac must be strictly < 1'
    MSELoss = torch.nn.MSELoss(reduction='mean')

    tensors, y_info = get_tensors(
        shuffle=shuffle, log_prices=LOG_PRICES, future_temp=future_temp,
        label_noise=label_noise)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info
    y_mean_tch = torch.from_numpy(y_mean)
    y_std_tch = torch.from_numpy(y_std)
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=batch_size)

    X_val = tensors_cv['X_calib'].to(device)
    Y_val = tensors_cv['Y_calib'].numpy()

    prob_z = StorageProblemNonRobust(T=Y_DIM, y_mean=y_mean, y_std=y_std, const=const)
    cvxpylayer_z = prob_z.get_cvxpylayer()

    prob_lambda = StorageProblemLambdaParameterized(
        T=Y_DIM, const=const, y_mean=y_mean, y_std=y_std, N=PSEUDOCAILB_SIZE,
        alpha=alpha, delta=delta, quad=False)
    prob_lambda_val = StorageProblemLambdaParameterized(
        T=Y_DIM, const=const, y_mean=y_mean, y_std=y_std, N=X_val.shape[0],
        alpha=alpha, delta=delta, quad=False)
    cvxpylayer_lambda = prob_lambda.get_cvxpylayer()

    # load the model
    model = MLP(input_dim=tensors['X_test'].shape[1], y_dim=Y_DIM)
    saved_ckpt_path = saved_ckpt_fmt.format(seed=seed)
    model.load_state_dict(torch.load(saved_ckpt_path, weights_only=True))
    model.to(device)

    # we keep L2 regularization fixed at 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10)
    current_lr = lr

    result: dict[str, Any] = {
        # metadata
        'seed': seed,
        'alpha': alpha,
        'delta': delta,
        'lr': lr,
        'max_epochs': max_epochs,

        # actual results
        'train_task_losses': [],  # avg loss per epoch
        'train_lams': [],  # avg lambda per epoch
        'val_task_losses': [],  # avg fpr per epoch
        'val_lams': [],  # avg lambda per epoch
        'best_epoch': 0,
        'val_task_loss': np.inf,  # lowest task loss on val set
        'val_lam': np.inf,  # lambda corresponding to lowest task loss on val set

        # these results get filled in later:
        # 'test_task_loss'
        # 'test_cvar'
        # 'test_mse'
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()
    msg_prefix = f'α{alpha:.2f} δ{delta:.2f} lr{lr:.2g} s{seed}'

    pbar = tqdm(range(max_epochs))
    pbar.set_description(f'{msg_prefix} e0')
    for epoch in pbar:
        model.train()

        train_task_losses = []
        train_λs = []
        for x, y in loaders['train']:
            batch_size = x.shape[0]
            if batch_size <= num_cal:
                tqdm.write(f'Batch size {batch_size} is too small, skipping')

            x = x.to(device, non_blocking=True)
            preds = model(x)
            z_in, z_out, z_net = cvxpylayer_z(preds.cpu())

            # calculate lambda using only the calibration half of the batch.
            # cvxpylayers requires DPP programs, which cannot involve any multiplication
            # between parameters. Therefore, we have to do some preprocessing ourselves.
            y_cal_unstd = y[:num_cal] * y_std_tch + y_mean_tch  # shape [num_cal, T]
            coeff_lin = torch.sum(y_cal_unstd * (z_in[:num_cal] - z_out[:num_cal]), dim=1)
            λ, _ = cvxpylayer_lambda(coeff_lin)
            train_λs.append(λ.item())

            # calculate task loss on each example in rest of the batch
            task_loss = prob_z.task_loss_torch(
                y[num_cal:], is_standardized=True,
                solution=(z_in[num_cal:] * λ, z_out[num_cal:] * λ, z_net[num_cal:] * λ)
            ).mean()
            train_task_losses.append(task_loss.item())
            loss = task_loss

            if mse_loss_frac > 0:
                # calculate MSE loss on entire batch
                mse_loss = MSELoss(preds, y.to(device))
                loss = mse_loss_frac * mse_loss + (1 - mse_loss_frac) * task_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute results on validation set
        # due to the small dataset size, we use the validation set for both
        # choosing lambda and computing task loss
        with torch.no_grad():
            model.eval()
            val_preds = model(X_val).cpu().numpy()
        z_in, z_out, z_net = get_zs(prob_z, val_preds)
        val_λ = prob_lambda_val.solve(y=Y_val, z_in=z_in, z_out=z_out, z_net=z_net)
        val_task_loss = prob_z.task_loss(
            z_in * val_λ, z_out * val_λ, z_net * val_λ, y=Y_val, is_standardized=True
        ).mean().item()

        # store results
        result['train_task_losses'].append(np.mean(train_task_losses))
        result['train_lams'].append(np.mean(train_λs))
        result['val_task_losses'].append(val_task_loss)
        result['val_lams'].append(val_λ)

        msg = (f'{msg_prefix} e{epoch}: '
               f'train task loss: {np.mean(train_task_losses):.3g} ± {np.std(train_task_losses):.3g}, '
               f'train λ: {np.mean(train_λs):.3g} ± {np.std(train_λs):.3g}, '
               f'val task loss: {val_task_loss:.3f}, val λ: {val_λ:.3g}')
        pbar.set_description(msg)

        steps_since_decrease += 1

        if val_task_loss < result['val_task_loss']:
            result['best_epoch'] = epoch
            result['val_task_loss'] = val_task_loss
            result['val_lam'] = val_λ
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

        lr_scheduler.step(val_task_loss)
        if lr_scheduler.get_last_lr()[0] < current_lr:
            current_lr = lr_scheduler.get_last_lr()[0]
            tqdm.write(f'{msg_prefix} finished epoch {epoch}, reducing lr to {current_lr}')

    # load best model
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))

    # evaluate on test
    with torch.no_grad():
        test_result = run_epoch_mlp(
            model, loaders['test'], prob=prob_z, λ=result['val_lam'],
            return_z=False, device=device)
        result['test_mse'] = test_result['loss'].mean().item()
        result['test_task_loss'] = test_result['task_loss'].mean().item()
        result['test_cvar'] = cvar(test_result['financial_loss'], q=delta)

    msg = (f'{msg_prefix} best epoch {result["best_epoch"]}, '
           f'test cvar: {result["test_cvar"]:.3f}, test task loss: {result["test_task_loss"]:.3f}, '
           f'λ: {result["val_lam"]:.3g}')

    # save results
    basename = f'a{alpha:.2f}_delta{delta:.2f}_lr{lr:.2g}_s{seed}'
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, f'{basename}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # save model
    ckpt_path = os.path.join(savedir, f'{basename}.pt')
    torch.save(model.cpu().state_dict(), ckpt_path)

    return result, msg


def finetune_task_loss(
    shuffle: bool, future_temp: bool, label_noise: float, const: StorageConstants,
    alphas: Iterable[float], deltas: Iterable[float], seed: int, saved_ckpt_fmt: str,
    device: Device, lr: float, mse_loss_frac: float, max_epochs: int, batch_size: int,
    savedir: str
) -> tuple[dict[str, Any], str]:
    """
    Fine-tune model using task loss.

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        future_temp: whether to include future temperature features
        label_noise: stddev of Gaussian noise added to the labels
        const: StorageConstants object
        alphas: list of risk thresholds
        deltas: list of quantile levels
        seed: random seed
        saved_ckpt_fmt: format string for name of existing model checkpoint in
            savedir, takes {seed} as an argument, e.g. 'mlp_s{seed}.pt'
        device: 'cpu' or 'cuda'
        lr: learning rate
        mse_loss_frac: weight for the MSE loss in [0, 1],
            (1-mse_loss_frac) is the weight for the task loss
        max_epochs: number of epochs to train for
        batch_size: batch size for training
        savedir: where to save the results
    """
    assert mse_loss_frac < 1, 'mse_loss_frac must be strictly < 1'
    MSELoss = torch.nn.MSELoss(reduction='mean')

    tensors, y_info = get_tensors(
        shuffle=shuffle, log_prices=LOG_PRICES, future_temp=future_temp,
        label_noise=label_noise)
    assert isinstance(y_info, tuple)
    y_mean, y_std = y_info
    tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
    loaders = get_loaders(tensors_cv, batch_size=batch_size)

    X_val = tensors_cv['X_calib'].to(device)
    Y_val = tensors_cv['Y_calib'].numpy()

    prob = StorageProblemNonRobust(T=Y_DIM, y_mean=y_mean, y_std=y_std, const=const)
    cvxpylayer = prob.get_cvxpylayer()

    # load the model
    model = MLP(input_dim=tensors['X_test'].shape[1], y_dim=Y_DIM)
    saved_ckpt_path = saved_ckpt_fmt.format(seed=seed)
    model.load_state_dict(torch.load(saved_ckpt_path, weights_only=True))
    model.to(device)

    # we keep L2 regularization fixed at 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10)
    current_lr = lr

    result: dict[str, Any] = {
        # metadata
        'alphas': alphas,
        'deltas': deltas,
        'seed': seed,
        'lr': lr,
        'max_epochs': max_epochs,

        # actual results
        'train_task_losses': [],  # avg loss per epoch
        'val_task_losses': [],  # avg loss per epoch
        'best_epoch': 0,
        'val_task_loss': np.inf,  # lowest task loss on val set

        # these results get filled in later:
        # 'test_task_loss'
        # 'test_cvar'
        # 'test_mse'
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()
    msg_prefix = f'lr{lr:.2g} s{seed}'

    pbar = tqdm(range(max_epochs))
    pbar.set_description(f'{msg_prefix} e0')
    for epoch in pbar:
        model.train()

        train_task_losses = []
        for x, y in loaders['train']:
            x = x.to(device, non_blocking=True)
            preds = model(x)
            z_in, z_out, z_net = cvxpylayer(preds.cpu())

            # calculate task loss on each example in rest of the batch
            task_loss = prob.task_loss_torch(
                y, is_standardized=True, solution=(z_in, z_out, z_net)
            ).mean()
            train_task_losses.append(task_loss.item())
            loss = task_loss

            if mse_loss_frac > 0:
                # calculate MSE loss on entire batch
                mse_loss = MSELoss(preds, y.to(device, non_blocking=True))
                loss = mse_loss_frac * mse_loss + (1 - mse_loss_frac) * task_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute results on validation set
        # due to the small dataset size, we use the validation set for both
        # choosing lambda and computing task loss
        with torch.no_grad():
            model.eval()
            val_preds = model(X_val).cpu().numpy()
        z_in, z_out, z_net = get_zs(prob, val_preds)
        val_task_loss = prob.task_loss(
            z_in, z_out, z_net, y=Y_val, is_standardized=True
        ).mean().item()

        # store results
        train_task_loss = np.mean(train_task_losses)
        result['train_task_losses'].append(train_task_loss)
        result['val_task_losses'].append(val_task_loss)

        msg = (f'{msg_prefix} e{epoch}: '
               f'train task loss: {train_task_loss:.3f}, '
               f'val task loss: {val_task_loss:.3f}')
        pbar.set_description(msg)

        steps_since_decrease += 1

        if val_task_loss < result['val_task_loss']:
            result['best_epoch'] = epoch
            result['val_task_loss'] = val_task_loss
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

        lr_scheduler.step(val_task_loss)
        if lr_scheduler.get_last_lr()[0] < current_lr:
            current_lr = lr_scheduler.get_last_lr()[0]
            tqdm.write(f'{msg_prefix} finished epoch {epoch}, reducing lr to {current_lr}')

    # load best model
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))

    # compute results on validation set
    calib_df = get_lams(
        tensors_dict=tensors_cv, model=model,
        prob=prob, device=device, alphas=alphas, deltas=deltas)
    alpha_deltas = calib_df.index.tolist()
    result['alpha_deltas'] = alpha_deltas
    result['val_lams'] = calib_df['lambda'].tolist()
    result['val_task_losses'] = calib_df['task loss'].tolist()
    result['val_cvars'] = calib_df['CVaR'].tolist()

    # evaluate on test
    with torch.no_grad():
        model.eval().to(device)
        pred_np = model(tensors['X_test'].to(device)).cpu().numpy()  # type: ignore
    z_in, z_out, z_net = get_zs(prob, preds=pred_np)
    y_test = tensors['Y_test'].cpu().numpy()  # type: ignore
    result['test_task_losses'] = []
    result['test_cvars'] = []
    for alpha, delta in alpha_deltas:
        λ = calib_df.loc[(alpha, delta), 'lambda']
        task_losses = prob.task_loss(z_in * λ, z_out * λ, z_net * λ, y=y_test, is_standardized=True)
        financial_losses = prob.financial_loss(z_in * λ, z_out * λ, y=y_test, is_standardized=True)
        assert isinstance(task_losses, np.ndarray)
        assert isinstance(financial_losses, np.ndarray)
        result['test_task_losses'].append(np.mean(task_losses).item())
        result['test_cvars'].append(cvar(financial_losses, q=delta))

    msg = (f'{msg_prefix} best epoch {result["best_epoch"]}')

    # save results
    basename = f'lr{lr:.2g}_s{seed}'
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, f'{basename}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # save model
    ckpt_path = os.path.join(savedir, f'{basename}.pt')
    torch.save(model.cpu().state_dict(), ckpt_path)

    return result, msg


def parse_args(commands: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        'command', choices=commands,
        help=f'command to run, one of {commands}')
    p.add_argument(
        '--alpha', type=float, nargs='+',
        help='CVaR threshold')
    p.add_argument(
        '--delta', type=float, nargs='+',
        help='quantile level for CVaR')
    p.add_argument(
        '--shuffle', action='store_true',
        help='shuffle the dataset before splitting into train/calib/test')
    p.add_argument(
        '-s', '--seeds', type=int, nargs='+', default=SEEDS,
        help='seeds to use')
    p.add_argument(
        '--future-temp', action='store_true',
        help='whether to include future temperature features')
    p.add_argument(
        '--lr', type=float, nargs='+',
        help='learning rate')
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
    check_args_device(args.device)
    return args


def main(args: argparse.Namespace) -> None:
    alphas = args.alpha
    deltas = args.delta
    shuffle = args.shuffle
    seeds = args.seeds
    device = args.device

    tag = args.tag
    if args.tag != '':
        tag = f'_{tag}'

    if shuffle:
        out_dir = f'out/storage_mlp{tag}_shuffle/'
    else:
        out_dir = f'out/storage_mlp{tag}/'
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'optimal':
        results = get_optimal_task_losses(
            shuffle=shuffle, label_noise=LABEL_NOISE, split='test',
            const=STORAGE_CONSTS[0])
        save_path = os.path.join(out_dir, 'optimal_task_loss.npz')
        print('Saving optimal task loss to:', save_path)
        np.savez_compressed(save_path, allow_pickle=False, **results)

    elif args.command == 'pretrain':
        train_mse(
            shuffle=shuffle, future_temp=args.future_temp,
            device=device, out_dir=out_dir, label_noise=LABEL_NOISE,
            tag=tag, seeds=seeds, multiprocess=args.multiprocess)

    elif args.command == 'savepreds':
        for seed in tqdm(seeds):
            save_preds(
                seed=seed, shuffle=shuffle, future_temp=args.future_temp,
                label_noise=LABEL_NOISE, const=STORAGE_CONSTS[0],
                saved_ckpt_fmt='mlp_s{seed}.pt', out_dir=out_dir,
                device=device, tag=tag)

    elif args.command == 'crc':
        saved_ckpt_fmt = os.path.join(out_dir, 'mlp_s{seed}.pt')
        with torch.no_grad():
            all_rows: list[dict[str, float]] = []
            for s in tqdm(seeds):
                crc_results = crc(
                    seed=s, shuffle=shuffle, future_temp=args.future_temp,
                    label_noise=LABEL_NOISE, const=STORAGE_CONSTS[0],
                    alphas=alphas, deltas=deltas,
                    saved_ckpt_fmt=saved_ckpt_fmt, device=device)
                all_rows.extend(crc_results)
            # save results to file
            df = pd.DataFrame(all_rows)

            all_task_losses = {
                (row['alpha'], row['delta'], row['seed']): row['task losses']
                for row in all_rows
            }
            with open(os.path.join(out_dir, 'crc_task_losses.pkl'), 'wb') as f:
                pickle.dump(all_task_losses, f)

            all_financial_losses = {
                (row['alpha'], row['delta'], row['seed']): row['financial losses']
                for row in all_rows
            }
            with open(os.path.join(out_dir, 'crc_financial_losses.pkl'), 'wb') as f:
                pickle.dump(all_financial_losses, f)

            del df['task losses']
            del df['financial losses']
            df.to_csv(os.path.join(out_dir, 'crc.csv'), index=False)

    elif args.command == 'e2ecrc':
        savedir = os.path.join(out_dir, f'e2ecrc{tag}')
        func = functools.partial(
            e2ecrc, shuffle=shuffle, future_temp=args.future_temp,
            label_noise=LABEL_NOISE, const=STORAGE_CONSTS[0],
            saved_ckpt_fmt=os.path.join(out_dir, 'mlp_s{seed}.pt'), device=device,
            mse_loss_frac=0.1, max_epochs=MAX_FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
            num_cal=PSEUDOCAILB_SIZE, savedir=savedir)
        kwargs_list = [
            dict(alpha=alpha, delta=delta, seed=s, lr=lr)
            for alpha, delta, s, lr in itertools.product(alphas, deltas, seeds, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    elif args.command == 'finetune_taskloss':
        savedir = os.path.join(out_dir, f'finetune_taskloss{tag}')
        func = functools.partial(
            finetune_task_loss, shuffle=shuffle, future_temp=args.future_temp,
            label_noise=LABEL_NOISE, const=STORAGE_CONSTS[0],
            alphas=alphas, deltas=deltas,
            saved_ckpt_fmt=os.path.join(out_dir, 'mlp_s{seed}.pt'), device=device,
            mse_loss_frac=0.1, max_epochs=MAX_FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
            savedir=savedir)
        kwargs_list = [
            dict(seed=s, lr=lr)
            for s, lr in itertools.product(seeds, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == "__main__":
    commands = ('optimal', 'pretrain', 'savepreds', 'crc', 'e2ecrc', 'finetune_taskloss')
    args = parse_args(commands)
    main(args)
