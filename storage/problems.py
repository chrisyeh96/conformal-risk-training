"""
This module implements the battery storage problems.
"""
from collections.abc import Sequence
from typing import NamedTuple

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
from torch import Tensor

from problems.nonrobust import NonRobustProblem
from problems.protocols import NonRobustProblemProtocol

TOL = 1e-7  # constraint violation tolerance


class StorageConstants(NamedTuple):
    lam: float = 0.1  # ε^{flex} parameter in paper
    eps: float = 0.05  # ε^{ramp} parameter in paper
    eff: float = 0.9  # γ parameter in paper
    c_in: float = 0.5
    c_out: float = 0.2
    B: float = 1  # B parameter in paper


DEFAULT_CONSTANTS = StorageConstants()


class StorageProblemBase:
    """
    Base class for the battery storage problem. This base class should be subclassed
    to implement specific uncertainty constraints. The base class defines the primal problem
    variables, constraints, and the task loss.
    """
    # instance variables
    const: StorageConstants
    constraints: list[cp.Constraint]
    f_tilde: cp.Expression | float
    Fz: cp.Expression
    primal_vars: dict[str, cp.Variable]

    # to be implemented in subclass
    prob: cp.Problem
    y_mean: np.ndarray
    y_std: np.ndarray
    params: dict[str, cp.Parameter]

    def __init__(self, T: int, const: StorageConstants = DEFAULT_CONSTANTS):
        self.const = const
        lam, eps, eff, c_in, c_out, B = self.const

        z_net = cp.Variable(T, name='z_net')
        z_in = cp.Variable(T, name='z_in', nonneg=True)
        z_out = cp.Variable(T, name='z_out', nonneg=True)
        self.primal_vars = {
            'z_in': z_in,
            'z_out': z_out,
            'z_net': z_net,
        }

        self.constraints = [
            # SOC constraints
            -B/2 <= z_net, z_net <= B/2,
            z_net == cp.cumsum(eff * z_in - z_out),

            # ramp constraints
            z_in <= c_in,
            z_out <= c_out,
        ]

        self.Fz = z_in - z_out

        self.f_tilde = (
            lam * cp.norm2(z_net)**2
            + eps * cp.norm2(z_in)**2
            + eps * cp.norm2(z_out)**2
        )

    def check_constraints(
        self, z_in: np.ndarray, z_out: np.ndarray, z_net: np.ndarray
    ) -> None:
        """Check constraint satisfaction for the given primal variables."""
        # backup primal variables
        primal_vars_old = {k: var.value for k, var in self.primal_vars.items()}

        # check constraint satisfaction
        self.primal_vars['z_in'].value = z_in
        self.primal_vars['z_out'].value = z_out
        self.primal_vars['z_net'].value = z_net
        for constr in self.constraints:
            if constr.shape == ():
                assert constr.violation() < TOL
            else:
                assert (constr.violation() < TOL).all()

        # restore primal variables
        for k, var in self.primal_vars.items():
            var.value = primal_vars_old[k]

    def task_loss(
        self, z_in: np.ndarray, z_out: np.ndarray, z_net: np.ndarray,
        y: np.ndarray, is_standardized: bool
    ) -> np.ndarray | np.floating:
        """Computes task loss for a single example or batch of examples.

        Args:
            z_in: shape [..., T], charging energy
            z_out: shape [..., T], discharging energy
            z_net: shape [..., T], net energy
            y: shape [..., T], energy price
            is_standardized: whether y is standardized
        """
        lam, eps, eff, c_in, c_out, B = self.const

        # check constraint satisfaction
        assert (-TOL <= z_in).all() and (z_in <= c_in + TOL).all()
        assert (-TOL <= z_out).all() and (z_out <= c_out + TOL).all()
        assert (-B/2 - TOL <= z_net).all() and (z_net <= B/2 + TOL).all()
        assert np.max(np.abs(z_net - np.cumsum(eff * z_in - z_out, axis=-1))) <= TOL

        if is_standardized:
            y = y * self.y_std + self.y_mean

        task_loss = (
            np.sum(y * (z_in - z_out), axis=-1)
            + lam * np.linalg.norm(z_net, axis=-1)**2
            + eps * np.linalg.norm(z_in, axis=-1)**2
            + eps * np.linalg.norm(z_out, axis=-1)**2
        )
        return task_loss

    def task_loss_np(self, y: np.ndarray, is_standardized: bool,
                     scale: float = 1.) -> float:
        """Computes task loss for a single example.

        Args:
            y: shape [T], energy price
            is_standardized: whether y is standardized
            scale: scaling factor for z_in and z_out

        Returns:
            task_loss: task loss for one example
        """
        assert self.prob.value is not None, 'Problem must be solved first'
        if scale == 1 and is_standardized and np.array_equal(self.params['y'].value, y):  # type: ignore
            return self.prob.value  # type: ignore

        z_in = self.primal_vars['z_in'].value * scale  # type: ignore
        z_out = self.primal_vars['z_out'].value * scale  # type: ignore
        z_net = self.primal_vars['z_net'].value * scale  # type: ignore

        assert isinstance(z_in, np.ndarray) and isinstance(z_out, np.ndarray) and isinstance(z_net, np.ndarray)
        return self.task_loss(z_in, z_out, z_net, y=y, is_standardized=is_standardized).item()

    def task_loss_torch(
        self, y: Tensor, is_standardized: bool, solution: Sequence[Tensor]
    ) -> Tensor:
        """Computes task loss for a single example or a batch of examples.

        Args:
            y: shape [..., T], energy price
            is_standardized: whether y is standardized
            solution: tuple of (z_in, z_out, z_state, and additional dual variables)

        Returns:
            task_loss: shape [...], task loss for each example
        """
        z_in, z_out, z_net = solution[:3]

        lam = self.const.lam
        eps = self.const.eps

        if is_standardized:
            y = y * torch.from_numpy(self.y_std) + torch.from_numpy(self.y_mean)

        task_loss = (
            torch.sum(y * (z_in - z_out), dim=-1)
            + lam * torch.norm(z_net, dim=-1)**2
            + eps * torch.norm(z_in, dim=-1)**2
            + eps * torch.norm(z_out, dim=-1)**2
        )
        assert task_loss.shape == y.shape[:-1]
        return task_loss

    def financial_loss(
        self, z_in: np.ndarray, z_out: np.ndarray,
        y: np.ndarray, is_standardized: bool
    ) -> np.ndarray | np.floating:
        """Computes financial loss for a single example or batch of examples.

        Args:
            z_in: shape [..., T], charging energy
            z_out: shape [..., T], discharging energy
            y: shape [..., T], energy price
            is_standardized: whether y is standardized
        """
        # check constraint satisfaction
        assert (-TOL <= z_in).all() and (z_in <= self.const.c_in + TOL).all()
        assert (-TOL <= z_out).all() and (z_out <= self.const.c_out + TOL).all()

        if is_standardized:
            y = y * self.y_std + self.y_mean

        financial_loss = np.sum(y * (z_in - z_out), axis=-1)
        return financial_loss

    def financial_loss_np(self, y: np.ndarray, is_standardized: bool,
                          scale: float = 1.) -> float:
        """Computes financial loss for a single example.

        Args:
            y: shape [T], energy price
            is_standardized: whether y is standardized
            scale: scaling factor for z_in and z_out

        Returns:
            financial_loss: financial loss for one example
        """
        assert self.prob.value is not None, 'Problem must be solved first'

        z_in = self.primal_vars['z_in'].value * scale  # type: ignore
        z_out = self.primal_vars['z_out'].value * scale  # type: ignore
        assert isinstance(z_in, np.ndarray) and isinstance(z_out, np.ndarray)

        return self.financial_loss(z_in, z_out, y=y, is_standardized=is_standardized).item()


class StorageProblemNonRobust(StorageProblemBase, NonRobustProblem, NonRobustProblemProtocol):
    def __init__(self, T: int, y_mean: np.ndarray, y_std: np.ndarray,
                 const: StorageConstants = DEFAULT_CONSTANTS):
        StorageProblemBase.__init__(self, T=T, const=const)
        NonRobustProblem.__init__(self, y_dim=T, y_mean=y_mean, y_std=y_std, Fz=self.Fz)
        NonRobustProblemProtocol.__init__(self)


class StorageProblemLambda:
    def __init__(
        self, T: int, const: StorageConstants,
        y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray,
        z_in: np.ndarray, z_out: np.ndarray, z_net: np.ndarray,
        quad: bool, t_fixed: bool
    ):
        """
        Args:
            T: time horizon
            const: storage constants
            y: shape [N, T], energy price (standardized)
            y_mean: shape [T], mean of y
            y_std: shape [T], std of y
            z_in: shape [N, T], charging energy
            z_out: shape [N, T], discharging energy
            z_net: shape [N, T], net energy
            quad: whether to include quadratic terms in the loss
            t_fixed: whether to treat t as fixed or an optimization variable
        """
        self.T = T
        self.const = const
        self.y = y
        self.y_mean = y_mean
        self.y_std = y_std
        self.z_in = z_in
        self.z_out = z_out
        self.z_net = z_net
        self.quad = quad
        self.t_fixed = t_fixed
        N = y.shape[0]

        alpha = cp.Parameter(name='alpha')
        tail_factor = cp.Parameter(name='tail_factor', nonneg=True)
        λ = cp.Variable(name='λ', nonneg=True)

        # t must be >= B(λmin) = 0, which is why we use nonneg=True
        if t_fixed:
            t = cp.Parameter(name='t', nonneg=True)
        else:
            t = cp.Variable(name='t', nonneg=True)

        self.alpha = alpha
        self.tail_factor = tail_factor
        self.λ = λ
        self.t = t

        y_unstd = y * y_std + y_mean
        coeff_lin = np.sum(y_unstd * (z_in - z_out), axis=-1)

        if not quad:
            fis = coeff_lin * λ  # shape [N]
            fbar = 100 * λ

        else:
            εflex = self.const.lam
            εramp = self.const.eps
            c_in = self.const.c_in
            c_out = self.const.c_out
            C = self.const.B

            coeff_quad = (
                εflex * np.linalg.norm(z_net, axis=-1)**2
                + εramp * (np.linalg.norm(z_in, axis=-1)**2 + np.linalg.norm(z_out, axis=-1)**2)
            )
            fis = coeff_lin * λ + coeff_quad * λ**2  # shape [N]
            fbar = 100 * λ + (εflex * C**2/4 + εramp * (c_in**2 + c_out**2)) * T * λ**2

        h = t + tail_factor / (N+1) * (
            cp.maximum(0, fbar - t) + cp.sum(cp.maximum(0, fis - t))
        )
        self.fis = fis

        self.prob = cp.Problem(
            objective=cp.Maximize(λ),
            constraints=[h <= alpha, λ <= 1, t <= alpha])

    def task_loss(self) -> np.ndarray:
        assert self.prob.value is not None, 'Problem must be solved first'
        assert self.fis.value is not None
        assert self.λ.value is not None

        if self.quad:
            return self.fis.value
        else:
            εflex = self.const.lam
            εramp = self.const.eps

            coeff_quad = (
                εflex * np.linalg.norm(self.z_net, axis=-1)**2
                + εramp * (np.linalg.norm(self.z_in, axis=-1)**2
                           + np.linalg.norm(self.z_out, axis=-1)**2)
            )
            return self.fis.value + coeff_quad * self.λ.value**2

    def solve(self, alpha: float, δ: float, t: float | None = None) -> float:
        self.alpha.value = alpha  # type: ignore
        self.tail_factor.value = 1. / (1 - δ)  # type: ignore
        if self.t_fixed:
            assert t is not None, 't must be provided when t_fixed is True'
            self.t.value = t

        try:
            self.prob.solve(solver=cp.CLARABEL)

            if 'user_limit' in self.prob.status:
                print(f'Problem status: {self.prob.status}. Re-running with SCS solver...')
                self.prob.solve(solver=cp.SCS)
        except cp.error.SolverError as e:
            print('Solver error:', e)
            # print('Re-running with verbose output...')
            # self.prob.solve(solver=cp.CLARABEL, verbose=True)
            print('Re-running with SCS solver...')
            self.prob.solve(solver=cp.SCS)
        if self.prob.status != 'optimal':
            print('Problem status:', self.prob.status)
            if 'infeasible' in self.prob.status:
                return 0.  # λ_min = 0
        return self.prob.value


class StorageProblemLambdaParameterized:
    def __init__(
        self, T: int, const: StorageConstants,
        y_mean: np.ndarray, y_std: np.ndarray, N: int,
        alpha: float, delta: float, quad: bool = False
    ):
        """
        Args:
            T: time horizon
            const: storage constants
            y_mean: shape [T], mean of y
            y_std: shape [T], std of y
            N: number of samples
            alpha: CVaR risk threshold
            delta: CVaR quantile level
            quad: whether to include quadratic terms in the loss
        """
        self.const = const
        self.y_mean = y_mean
        self.y_std = y_std
        self.T = T
        self.quad = quad

        coeff_lin = cp.Parameter((N,), name='coeff_lin')
        self.params = {'coeff_lin': coeff_lin}

        λ = cp.Variable(name='λ', nonneg=True)
        t = cp.Variable(name='t', nonneg=True)  # t must be >= B(λmin) = 0
        self.vars = {
            'λ': λ,
            't': t,
        }

        εflex = self.const.lam
        εramp = self.const.eps
        c_in = self.const.c_in
        c_out = self.const.c_out
        C = self.const.B

        if not quad:
            fis = coeff_lin * λ  # shape [N]
            fbar = 100 * λ
        else:
            z_in_norm2 = cp.Parameter((N,), name='z_in_norm2', nonneg=True)
            z_out_norm2 = cp.Parameter((N,), name='z_out_norm2', nonneg=True)
            z_net_norm2 = cp.Parameter((N,), name='z_net_norm2', nonneg=True)
            self.params.update({
                'z_in_norm2': z_in_norm2,
                'z_out_norm2': z_out_norm2,
                'z_net_norm2': z_net_norm2,
            })
            coeff_quad = εflex * z_net_norm2 + εramp * (z_in_norm2 + z_out_norm2)
            fis = coeff_lin * λ + coeff_quad * λ**2  # shape [N]
            fbar = 100 * λ + (εflex * C**2/4 + εramp * (c_in**2 + c_out**2)) * T * λ**2

        h = t + 1./((1-delta) * (N+1)) * (
            cp.maximum(0, fbar - t) + cp.sum(cp.maximum(0, fis - t))
        )
        self.fis = fis

        self.prob = cp.Problem(
            objective=cp.Maximize(λ),
            constraints=[h <= alpha, λ <= 1, t <= alpha])

    def solve(
        self, y: np.ndarray, z_in: np.ndarray, z_out: np.ndarray, z_net: np.ndarray,
    ) -> float:
        y_unstd = y * self.y_std + self.y_mean
        self.params['coeff_lin'].value = np.sum(y_unstd * (z_in - z_out), axis=1)  # shape [N]
        if self.quad:
            self.params['z_in_norm2'].value = np.sum(z_in ** 2, axis=1)
            self.params['z_out_norm2'].value = np.sum(z_out ** 2, axis=1)
            self.params['z_net_norm2'].value = np.sum(z_net ** 2, axis=1)

        self.prob.solve(solver=cp.CLARABEL)
        if self.prob.status != 'optimal':
            print('Problem status:', self.prob.status)
        return self.prob.value.item()

    def get_cvxpylayer(self) -> CvxpyLayer:
        return CvxpyLayer(
            self.prob,
            parameters=list(self.params.values()),
            variables=list(self.vars.values())
        )
