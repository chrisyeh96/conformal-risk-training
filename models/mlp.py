from collections.abc import Mapping
from collections import defaultdict
import io
from typing import Any

import numpy as np
import torch.utils.data
from torch import nn, Tensor
from tqdm.auto import tqdm

from problems.protocols import NonRobustProblemProtocol

Device = str | torch.device


class MLP(nn.Module):
    """
    Args:
        input_dim: dimension of each input example x
        y_dim: dimension of each label y
        n_hidden_layers: # of hidden layers
    """
    def __init__(self, input_dim: int, y_dim: int, n_hidden_layers: int = 3):
        super().__init__()
        self.y_dim = y_dim
        act = nn.LeakyReLU()

        layers = [
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            act
        ]
        for _ in range(n_hidden_layers - 1):
            layers.extend([
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                act
            ])
        layers.append(nn.Linear(256, y_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: shape [..., input_dim]

        Returns:
            loc: shape [..., y_dim], prediction
        """
        return self.net(x)


def run_epoch_mlp(
    model: MLP,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    prob: NonRobustProblemProtocol | None = None,
    λ: float = 1,
    return_z: bool = False,
    device: Device = 'cpu'
) -> dict[str, np.ndarray]:
    """Runs 1 epoch of training or evaluation with a MLP regressor and MSE loss.

    Args:
        model: model to train
        loader: data loader
        optimizer: optimizer to use for training, None for evaluation
        prob: optional problem to solve
        λ: adjustment to decision vector
        return_z: whether to return decision vectors, requires `prob` to be provided
        device: either 'cpu' or 'cuda'

    Returns:
        result: dict, maps each key to an array of shape [num_examples, ...]
            - 'loss': loss for each example, always present
            - 'task_loss': task loss for each example. Present if `prob` is not None
            - <primal var name>: decision variable for each example. Present if `prob`
                is not None and `return_z` is True
    """
    if optimizer is None:
        model.eval().to(device)
    else:
        model.train().to(device)

    loss_fn = nn.MSELoss(reduction='none')
    losses = []
    task_losses: list[float] = []
    financial_losses: list[float] = []
    zs: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if prob is not None:
            y_np_batch = y.detach().numpy()
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.detach().cpu().numpy())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        # calculate task loss if `prob` is given
        if prob is not None:
            with torch.no_grad():
                pred_np_batch = pred.cpu().numpy()
                for y_np, pred_np in zip(y_np_batch, pred_np_batch):
                    prob.solve(pred_np)
                    if return_z:
                        for k, v in prob.primal_vars.items():
                            assert v.value is not None
                            zs[k].append(v.value)
                    task_loss = prob.task_loss_np(y_np, is_standardized=True, scale=λ)
                    financial_loss = prob.financial_loss_np(y_np, is_standardized=True, scale=λ)
                    task_losses.append(task_loss)
                    financial_losses.append(financial_loss)

    result = {'loss': np.concatenate(losses)}
    if prob is not None:
        result['task_loss'] = np.array(task_losses)
        result['financial_loss'] = np.array(financial_losses)
        if return_z:
            for k, v in zs.items():
                result[k] = np.stack(v)
    return result


def train_mlp(
    model: MLP,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    max_epochs: int,
    lr: float,
    l2reg: float,
    cutoff: int = 20,
    lr_schedule_kwargs: Mapping[str, Any] | None = None,
    show_pbar: bool = False,
    return_best_model: bool = False,
    device: Device = 'cpu'
) -> dict[str, Any]:
    """Trains a MLP regressor with MSE loss.

    Uses early-stopping based on loss on the calibration set.

    Args:
        model: model to train
        loaders: maps split to dataloader
        max_epochs: maximum number of epochs to train
        lr: learning rate
        l2reg: L2 regularization strength
        freeze: names of parts of model to freeze during training
        cutoff: number of epochs without improvement to stop training
        lr_schedule_kwargs: kwargs for ReduceLROnPlateau.
            If None, no learning rate schedule is used.
        show_pbar: if True, show a progress bar
        return_best_model: if True, return the model with the best validation loss,
            otherwise returns the model from the last training epoch
        device: either 'cpu' or 'cuda'

    Returns:
        result: dict of performance metrics
            'train_losses': training loss at each epoch
            'val_losses': validation loss at each epoch
            'best_epoch': best epoch
            'val_loss': best validation loss
    """
    model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)

    if lr_schedule_kwargs is None:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **lr_schedule_kwargs)
    current_lr = lr

    result: dict[str, Any] = {
        'train_losses': [],
        'val_losses': [],
        'best_epoch': 0,
        'val_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    pbar = tqdm(total=max_epochs) if show_pbar else None
    for epoch in range(max_epochs):
        # train
        train_result = run_epoch_mlp(
            model, loaders['train'], optimizer=optimizer, device=device)
        train_loss = train_result['loss'].mean()
        result['train_losses'].append(train_loss)

        # calculate loss on calibration set
        with torch.no_grad():
            val_result = run_epoch_mlp(model, loaders['calib'], device=device)
            val_loss = val_result['loss'].mean()
            result['val_losses'].append(val_loss)

        if pbar is not None:
            msg = f'Train MSE: {train_loss:.3g}, Calib MSE: {val_loss:.3g}'
            pbar.set_description(msg)
            pbar.update()

        steps_since_decrease += 1

        if val_loss < result['val_loss']:
            result['best_epoch'] = epoch
            result['val_loss'] = val_loss
            steps_since_decrease = 0
            buffer.seek(0)
            torch.save(model.state_dict(), buffer)

        if steps_since_decrease > cutoff:
            break

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)
            if lr_scheduler.get_last_lr()[0] < current_lr:
                current_lr = lr_scheduler.get_last_lr()[0]
                tqdm.write(f'Finished epoch {epoch}, reducing lr to {current_lr}')

    # load best model
    if return_best_model:
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))

    return result
