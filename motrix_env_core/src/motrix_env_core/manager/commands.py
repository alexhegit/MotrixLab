# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

import motrix_env_core.manager.context as context
from motrix_env_core.config import configclass
from motrix_env_core.manager.kernel_data import validate_kernel_data
from motrix_env_core.manager.metrics import collect_metrics, materialize_metrics

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class CommandTerm(abc.ABC):
    """Environment-local kernel-data command pipeline with persistent runtime state."""

    command: torch.Tensor

    @abc.abstractmethod
    def update(self, ctx: context.ManagerContext) -> None: ...

    @abc.abstractmethod
    def advance(self, ctx: context.ManagerContext) -> None: ...

    @abc.abstractmethod
    def reset(self, ctx: context.ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None: ...


@configclass(kw_only=True)
class CommandCfg(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ctx: context.BuildContext) -> CommandTerm:
        raise NotImplementedError


@configclass
class ManagerCommandsCfg:
    def to_dict(self) -> dict[str, CommandCfg]:
        term_cfgs: dict[str, CommandCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, CommandCfg):
                raise TypeError(
                    f"Manager command {term_field.name!r} must be a CommandCfg, got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


class CommandManager:
    """Build and own the configured command terms and their metrics."""

    def __init__(
        self,
        cfgs: dict[str, CommandCfg],
        build_context: Callable[[dict[str, CommandTerm]], context.BuildContext],
        num_envs: int,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        self.cfgs = cfgs
        self.terms: dict[str, CommandTerm] = {}
        for name, cfg in cfgs.items():
            term = cfg(build_context({}))
            term = validate_kernel_data(
                materialize_metrics(term, num_envs, context=f"Manager command {name!r}"),
                context=f"Manager command {name!r}",
            )
            self.terms[name] = term
        for name, term in self.terms.items():
            for metric_name, value in collect_metrics(
                term, num_envs, context=f"Manager command {name!r}", context_path=("commands", name)
            ):
                if metric_name in metrics:
                    raise ValueError(f"Duplicate per-environment metric name: {metric_name!r}")
                metrics[metric_name] = value


__all__ = ["CommandCfg", "CommandManager", "CommandTerm", "ManagerCommandsCfg"]
