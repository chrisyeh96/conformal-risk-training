from __future__ import annotations

import argparse
from collections.abc import Iterable
import functools
import io
import itertools
import json
import os
from typing import Any

import PIL.Image
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from polyps import pranet_utils
from polyps.dataloader import get_loaders, PolypDataset
from polyps.PraNet_Res2Net import PraNet
from utils.multiprocess import run_parallel
from utils.parse_args import check_args_device

Device = str | torch.device
Scalar = float | np.number

SEEDS = range(10)
OUT_DIR = 'out/polyps'
MIN_PSEUDOCALIB_SIZE = 200
MAX_EPOCHS = 100
BATCH_SIZE = 400


def calc_fnr(
    preds: Tensor, masks: Tensor, lam: Scalar | Iterable[Scalar]
) -> float | tuple[float, ...]:
    """
    Calculate the false negative rate (FNR) for a given set of predictions and masks.

    Args:
        preds: predicted probabilities of shape [n, 1, h, w]
        masks: ground truth masks of shape [n, 1, h, w]
        lam: threshold(s) for positive predictions

    Returns:
        fnr: false negative rate(s) for corresponding to lam
    """
    N = len(preds)
    preds = preds[:, 0]  # shape [n, h, w]
    masks = masks[:, 0]  # shape [n, h, w]

    # Alternative equivalent code:
    # only care about true-positive pixels
    # tp_per_img = masks.sum(dim=(-2, -1))
    # fnr_per_image = ((preds < lam) & masks).sum(dim=(-2, -1)) / tp_per_img
    # fnr = fnr_per_image.mean()

    weights = masks / masks.sum(dim=(-2, -1), keepdim=True)
    flat_preds = preds[masks]
    flat_weights = weights[masks]

    if isinstance(lam, Iterable):
        return tuple(
            ((flat_preds < l) * flat_weights).sum().item() / N
            for l in lam
        )
    else:
        return ((flat_preds < lam) * flat_weights).sum().item() / N


def calc_fpr(
    preds: Tensor, masks: Tensor, lam: Scalar | Iterable[Scalar]
) -> float | tuple[float, ...]:
    """
    Calculate the false positive rate (FPR) for a given set of predictions and masks.

    Args:
        preds: predicted probabilities of shape [n, 1, h, w]
        masks: ground truth masks of shape [n, 1, h, w]
        lam: threshold(s) for positive predictions

    Returns:
        fpr: false positive rate(s) for corresponding to lam
    """
    N = len(preds)

    neg_masks = ~masks
    weights_neg = neg_masks / neg_masks.sum(dim=(-2, -1), keepdim=True)
    flat_preds_neg = preds[neg_masks]
    flat_weights_neg = weights_neg[neg_masks]

    assert np.isclose(flat_weights_neg.sum().item(), N)

    if isinstance(lam, Iterable):
        return tuple(
            ((flat_preds_neg >= l) * flat_weights_neg).sum().item() / N
            for l in lam
        )
    else:
        return ((flat_preds_neg >= lam) * flat_weights_neg).sum().item() / N


def fpr_loss(preds: Tensor, masks: Tensor, lam: Tensor, T: float) -> Tensor:
    """Compute an approximate false positive rate (FPR) loss.

    Args:
        preds: predicted probabilities, shape [n, 1, h, w]
        masks: ground truth masks, shape [n, 1, h, w], type bool
        lam: threshold for positive predictions
        T: temperature for sigmoid function

    Returns:
        fpr_loss: approximate false positive rate loss
    """
    N = len(preds)

    with torch.no_grad():
        neg_masks = ~masks
        weights_neg = neg_masks / neg_masks.sum(dim=(-2, -1), keepdim=True)
        flat_weights_neg = weights_neg[neg_masks]
        # assert np.isclose(flat_weights_neg.sum().item(), N)

    approx_fps = torch.sigmoid((preds[neg_masks] - lam) / T)
    return (approx_fps * flat_weights_neg).sum() / N


def get_lams(
    preds: Tensor, masks: Tensor, alphas: Iterable[float]
) -> list[float]:
    """
    Get the lambda values for the given predictions and masks.

    Args:
        preds: predicted probabilities of shape [n, 1, h, w]
        masks: ground truth masks of shape [n, 1, h, w]
        alphas: risk thresholds

    Returns:
        lams: lambda values for the corresponding alphas
    """
    N = len(preds)

    # only care about true-positive pixels
    weights = masks / masks.sum(dim=(-2, -1), keepdim=True)
    flat_preds = preds[masks]
    flat_weights = weights[masks]
    assert np.isclose(N, flat_weights.sum().item())

    sort_inds = torch.argsort(flat_preds, descending=True)
    cum_weights = flat_weights[sort_inds].cumsum(dim=0)

    lam_inds = [int(torch.nonzero(cum_weights > (1-a)*(N+1))[0].item()) for a in alphas]
    return flat_preds[sort_inds][lam_inds].cpu().numpy().tolist()


def run_epoch(
    model: PraNet,
    loader: torch.utils.data.DataLoader,
    device: Device,
) -> tuple[Tensor, Tensor]:
    """
    Args:
        model: PraNet model
        loader: DataLoader for the test dataset
        device: device to run the model on

    Returns:
        preds: predicted probabilities, shape [n, 1, h, w], on given device
        masks: ground truth masks, shape [n, 1, h, w], type bool, on given device
    """
    model.eval().to(device)

    all_preds = []
    all_masks = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        _, _, _, preds = model(images)
        preds.sigmoid_()

        all_preds.append(preds)
        all_masks.append(masks)

    if len(all_preds) == 1:
        preds = all_preds[0]
        masks = all_masks[0]
    else:
        preds = torch.cat(all_preds)
        masks = torch.cat(all_masks)
    return preds, masks


def save_preds_to_png(ckpt_path: str, seed: int, device: Device, out_dir: str) -> None:
    """
    Save model predictions on the test set to .npy and .png files.

    Saves files to
        {out_dir}/{dataset_name}_{image_filename}.png
        {out_dir}_raw/{dataset_name}_{image_filename}.npy
    """
    os.makedirs(out_dir, exist_ok=True)

    out_dir_raw = f'{out_dir}_raw'
    os.makedirs(out_dir_raw, exist_ok=True)

    # Load the model
    model = PraNet()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # Load the dataset
    loaders = get_loaders(splits=('test',), seed=seed)
    preds, _ = run_epoch(model, loader=loaders['test'], device=device)  # shape [n, 1, h, w]

    subset_ds = loaders['test'].dataset
    assert isinstance(subset_ds, torch.utils.data.Subset)
    ds = subset_ds.dataset
    indices = subset_ds.indices
    assert isinstance(ds, PolypDataset)

    for pred, idx in tqdm(zip(preds, indices)):
        img_path = ds.image_paths[idx]
        img_path_split = img_path.rsplit('/')
        ds_name = img_path_split[-3]
        img_name = img_path_split[-1]
        img_root = os.path.splitext(img_name)[0]

        # save raw prediction
        np.save(os.path.join(out_dir_raw, f'{ds_name}_{img_root}.npy'), pred.cpu().numpy())

        # reshape raw prediction to original image size and save as PNG
        with PIL.Image.open(img_path) as img:
            width, height = img.size

        pred = torch.nn.functional.interpolate(pred[None], size=(height, width), mode='bilinear')
        pred = pred[0, 0]  # shape [h, w]
        pred = np.round(pred.cpu().numpy() * 255).astype(np.uint8)

        pil_image = PIL.Image.fromarray(pred)
        pil_image.save(os.path.join(out_dir, f'{ds_name}_{img_name}'))
        pil_image.close()


def crc(
    alphas: Iterable[float], seed: int, ckpt_path: str, device: Device
) -> list[dict[str, float]]:
    """
    Post-hoc CRC. Always call this function within a torch.no_grad() context.
    """
    # Load the model
    model = PraNet()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # Load the dataset
    loaders = get_loaders(splits=('val', 'test'), seed=seed)

    # get lambdas from the calibration set
    preds, masks = run_epoch(model, loader=loaders['val'], device=device)  # shape [n, 1, h, w]
    lams = get_lams(preds, masks, alphas)

    # use lambdas on test set
    preds, masks = run_epoch(model, loader=loaders['test'], device=device)  # shape [n, 1, h, w]
    fnrs = calc_fnr(preds, masks, lam=lams)
    fprs = calc_fpr(preds, masks, lam=lams)
    assert isinstance(fnrs, tuple)
    assert isinstance(fprs, tuple)

    rows = [
        {'seed': seed, 'alpha': alpha, 'lambda': lam, 'fnr': fnr, 'fpr': fpr}
        for alpha, lam, fnr, fpr in zip(alphas, lams, fnrs, fprs)
    ]
    return rows


def e2ecrc(
    alpha: float, seed: int, ckpt_path: str, device: Device,
    lr: float, max_epochs: int, batch_size: int, savedir: str,
    T: float = 0.1
) -> tuple[dict[str, Any], str]:
    """
    Args:
        alpha: risk threshold
        seed: random seed for splitting val/test
        ckpt_path: path to model checkpoint file
        device: device to run the model on
        lr: learning rate
        max_epochs: maximum # of epochs to train
        batch_size: batch size
        T: temperature for sigmoid function in FPR loss
    """
    # initialize pre-trained model and freeze ResNet backbone
    tqdm.write(f'Initializing pre-trained model: {ckpt_path}')
    model = PraNet()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.resnet.requires_grad_(False)
    model.to(device).train()

    loaders = get_loaders(batch_size=batch_size, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # PraNet paper did not use L2 regularization

    result: dict[str, Any] = {
        # metadata
        'seed': seed,
        'alpha': alpha,
        'lr': lr,
        'train_fpr_T': T,
        'max_epochs': max_epochs,

        # actual results
        'train_fpr_losses': [],  # avg fpr loss per epoch
        'train_lams': [],  # avg lambda per epoch
        'val_fprs': [],  # avg fpr per epoch
        'val_lams': [],  # avg lambda per epoch
        'best_epoch': 0,
        'val_fpr': np.inf,  # lowest FPR on val set
        'val_lam': np.inf,  # lambda corresponding to lowest FPR on val set

        # these results get filled in later:
        # 'test_fpr'
        # 'test_fnr'
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()
    msg_prefix = f'a{alpha:.2f} lr{lr:.2g} s{seed}'

    pbar = tqdm(range(max_epochs))
    pbar.set_description(f'{msg_prefix} e0')
    for epoch in pbar:
        train_fprs = []
        train_lams = []
        for images, masks in loaders['train']:
            if len(images) <= MIN_PSEUDOCALIB_SIZE:
                tqdm.write(f'Batch size {len(images)} is too small, skipping')
                continue

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            _, _, _, preds = model(images)
            preds.sigmoid_()

            # get lam from the calibration set
            num_cal = max(len(images) // 2, MIN_PSEUDOCALIB_SIZE)
            preds_cal = preds[:num_cal]
            masks_cal = masks[:num_cal]
            flat_preds = preds_cal[masks_cal]

            with torch.no_grad():
                weights = masks_cal / masks_cal.sum(dim=(-2, -1), keepdim=True)
                flat_weights = weights[masks_cal]

                sort_inds = torch.argsort(flat_preds, descending=True)
                cum_weights = flat_weights[sort_inds].cumsum(dim=0)
                lam_ind = torch.nonzero(cum_weights > (1-alpha)*(num_cal+1))[0].item()
                assert isinstance(lam_ind, int)

                # true lambda
                # lam = flat_preds[sort_inds][lam_ind]

                # use up to 0.5% of true-positive pixels
                half_num_pixels = int(np.ceil(len(flat_preds) / 400))
                lam_inds = range(max(0, lam_ind - half_num_pixels), min(len(flat_preds), lam_ind + half_num_pixels))

            lam = flat_preds[sort_inds][lam_inds].mean()
            train_lams.append(lam.item())

            # compute false positive rate loss on the prediction set
            loss = fpr_loss(preds[num_cal:], masks[num_cal:], lam=lam, T=T)
            train_fprs.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute results on validation set
        with torch.no_grad():
            preds, masks = run_epoch(model, loader=loaders['val'], device=device)
            val_lam = get_lams(preds, masks, alphas=(alpha,))[0]
            val_fpr = calc_fpr(preds, masks, lam=val_lam)
            assert isinstance(val_fpr, float)

        # store results
        result['train_fpr_losses'].append(np.mean(train_fprs))
        result['train_lams'].append(np.mean(train_lams))
        result['val_fprs'].append(val_fpr)
        result['val_lams'].append(val_lam)

        msg = (f'{msg_prefix} e{epoch}: '
               f'train FPR loss: {np.mean(train_fprs):.3g} ± {np.std(train_fprs):.3g}, '
               f'train lam: {np.mean(train_lams):.3g} ± {np.std(train_lams):.3g}, '
               f'val FPR: {val_fpr:.3f}, val lam: {val_lam:.3g}')
        pbar.set_description(msg)

        steps_since_decrease += 1

        if val_fpr < result['val_fpr']:
            result['best_epoch'] = epoch
            result['val_fpr'] = val_fpr
            result['val_lam'] = val_lam
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

    # load best model
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))

    # evaluate on test
    with torch.no_grad():
        preds, masks = run_epoch(model, loader=loaders['test'], device=device)  # shape [n, 1, h, w]
        result['test_fpr'] = calc_fpr(preds, masks, lam=result['val_lam'])
        result['test_fnr'] = calc_fnr(preds, masks, lam=result['val_lam'])

    msg = (f'{msg_prefix} best epoch {result["best_epoch"]}, '
           f'test FNR: {result["test_fnr"]:.3f}, test FPR: {result["test_fpr"]:.3f}, '
           f'lam: {result["val_lam"]:.3g}')

    # save results
    basename = f'a{alpha:.2f}_lr{lr:.2g}_s{seed}'
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, f'{basename}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # save model
    ckpt_path = os.path.join(savedir, f'{basename}.pt')
    torch.save(model.cpu().state_dict(), ckpt_path)

    return result, msg


def train_base(
    alphas: Iterable[float], seed: int, ckpt_path: str, device: Device,
    lr: float, max_epochs: int, savedir: str
) -> tuple[dict[str, Any], str]:
    """
    """
    # these values come from the PraNet repo
    GRADIENT_CLIP_MARGIN = 0.5
    LR_DECAY_RATE = 0.1
    LR_DECAY_EPOCH = 50
    BATCH_SIZE = 16

    # initialize pre-trained model and freeze ResNet backbone
    tqdm.write(f'Initializing pre-trained model: {ckpt_path}')
    model = PraNet()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.resnet.requires_grad_(False)
    model.to(device).train()

    loaders = get_loaders(batch_size=BATCH_SIZE, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # PraNet paper did not use L2 regularization

    result: dict[str, Any] = {
        # metadata
        'alphas': alphas,
        'seed': seed,
        'lr': lr,
        'max_epochs': max_epochs,

        # actual results
        'train_losses': [],  # avg loss per epoch
        'val_losses': [],
        'best_epoch': 0,
        'val_loss': np.inf,  # lowest loss on val set

        # these results get filled in later:
        # 'val_lams'
        # 'val_fprs'
        # 'val_fnrs'
        # 'test_fprs'
        # 'test_fnrs'
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()
    msg_prefix = f'lr{lr:.2g} s{seed}'

    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]

    pbar = tqdm(range(max_epochs))
    pbar.set_description(f'{msg_prefix} e0')
    for epoch in pbar:
        pranet_utils.adjust_lr(optimizer, epoch+1, LR_DECAY_RATE, LR_DECAY_EPOCH)

        train_losses = []
        for images, masks in loaders['train']:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
            orig_size = images.shape[-1]

            for rate in size_rates:
                trainsize = int(round(orig_size*rate/32)*32)
                if rate != 1:
                    scaled_images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    with torch.no_grad():
                        scaled_masks = F.upsample(masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                else:
                    scaled_images = images
                    scaled_masks = masks

                loss = pranet_utils.pranet_loss(model, scaled_images, scaled_masks)
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                pranet_utils.clip_gradient(optimizer, GRADIENT_CLIP_MARGIN)
                optimizer.step()

        # compute results on validation set
        with torch.no_grad():
            total_val_loss = 0.
            for images, masks in loaders['val']:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device=device, dtype=torch.float32, non_blocking=True)
                val_loss = pranet_utils.pranet_loss(model, images, masks).item()
                total_val_loss += val_loss * len(images)
            val_loss = total_val_loss / len(loaders['val'].dataset)

        # store results
        train_loss = np.mean(train_losses)
        result['train_losses'].append(train_loss)
        result['val_losses'].append(val_loss)

        msg = (f'{msg_prefix} e{epoch}: '
               f'train loss: {train_loss:.3g}, val loss: {val_loss:.3g}')
        pbar.set_description(msg)

        steps_since_decrease += 1

        if val_loss < result['val_loss']:
            result['best_epoch'] = epoch
            result['val_loss'] = val_loss
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

    # load best model
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))

    # compute results on validation set
    with torch.no_grad():
        preds, masks = run_epoch(model, loader=loaders['val'], device=device)
        val_lams = get_lams(preds, masks, alphas=alphas)
        val_fnrs = calc_fnr(preds, masks, lam=val_lams)
        val_fprs = calc_fpr(preds, masks, lam=val_lams)
        result['val_lams'] = val_lams
        result['val_fnrs'] = val_fnrs
        result['val_fprs'] = val_fprs

    # evaluate on test
    with torch.no_grad():
        preds, masks = run_epoch(model, loader=loaders['test'], device=device)  # shape [n, 1, h, w]
        result['test_fprs'] = calc_fpr(preds, masks, lam=val_lams)
        result['test_fnrs'] = calc_fnr(preds, masks, lam=val_lams)

    msg = f'{msg_prefix} best epoch {result["best_epoch"]}'

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
        help='risk threshold')
    p.add_argument(
        '-s', '--seeds', type=int, nargs='+', default=SEEDS,
        help='seeds to use')
    p.add_argument(
        '--lr', type=float, nargs='+',
        help='learning rate')
    p.add_argument(
        '--ckpt-path', default='polyps/PraNet-19.pth',
        help='path to model checkpoint file')
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
    seeds = args.seeds
    ckpt_path = args.ckpt_path
    device = args.device

    tag = args.tag
    if args.tag != '':
        tag = f'_{tag}'

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.command == 'crc':
        with torch.no_grad():
            all_rows = []
            for s in tqdm(seeds):
                results = crc(
                    alphas=alphas, seed=s, ckpt_path=ckpt_path, device=device)
                all_rows.extend(results)
            # save results to file
            df = pd.DataFrame(all_rows)
            df.to_csv(os.path.join(OUT_DIR, 'crc.csv'), index=False)

    elif args.command == 'savepreds':
        with torch.no_grad():
            for s in seeds:
                out_dir = os.path.join(OUT_DIR, f'preds{tag}_s{s}')
                save_preds_to_png(
                    ckpt_path=ckpt_path, seed=s, device=device, out_dir=out_dir)

    elif args.command == 'e2ecrc':
        savedir = os.path.join(OUT_DIR, f'e2ecrc{tag}')
        func = functools.partial(
            e2ecrc, ckpt_path=ckpt_path, device=device, max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE, T=0.1, savedir=savedir)
        kwargs_list = [
            dict(alpha=alpha, seed=s, lr=lr)
            for alpha, s, lr in itertools.product(alphas, seeds, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    elif args.command == 'trainbase':
        savedir = os.path.join(OUT_DIR, f'trainbase{tag}')
        func = functools.partial(
            train_base, alphas=alphas, ckpt_path=ckpt_path, device=device,
            max_epochs=MAX_EPOCHS, savedir=savedir)
        kwargs_list = [
            dict(seed=s, lr=lr)
            for s, lr in itertools.product(seeds, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == "__main__":
    commands = ('crc', 'savepreds', 'e2ecrc', 'trainbase')
    args = parse_args(commands)
    main(args)
