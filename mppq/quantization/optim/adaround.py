from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from mppq.executor import TorchExecutor, TorchQuantizeDelegator
from mppq.executor.base import BaseGraphExecutor
from mppq.ir.base.graph import BaseGraph
from mppq.ir.base.opdef import Variable
from mppq.ir.base.quantize import QuantableOperation, QuantableVariable
from mppq.quant import (
    QuantizationProperty,
    QuantizationStates,
    TensorQuantizationConfig,
)
from mppq.quantization.algorithm.training import TrainableBlock
from mppq.quantization.measure import torch_mean_square_error
from mppq.quantization.optim.base import OPTIM_ALGORITHMS
from mppq.quantization.optim.training import LSQDelegator, TrainingBasedPass


class TimeDecay:
    """A helper class computing time decay."""

    def __init__(
        self,
        t_max: int,
        decay: float = 0.2,
        beta_start: float = 20,
        beta_end: float = 2,
    ):
        self.t_max = t_max
        self.start_decay = decay * t_max
        self.start_b = beta_start
        self.end_b = beta_end

    def __call__(self, t):
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + 0.5 * (self.start_b - self.end_b) * (
            1 + np.cos(rel_t * np.pi)
        )


class AdaroundRegTerm(torch.nn.Module):
    """Adaround Reg Term is a part of Adaround optimization algorithm.
    This term represents the difference between a fp32 value and its quantized counter-part.
        We use a same implementation as proposed in Adaround paper.
    Args:
        torch ([type]): [description]
    """

    def __init__(
        self,
        max_iter: int = 20000,
        zeta: float = 1.1,
        gamma: float = -0.1,
        alpha: float = 0.01,
        beta: float = 20,
        warm_ratio: float = 0.2,
    ):
        self.max_iter = max_iter
        self.zeta = zeta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.warm_ratio = warm_ratio
        self.temp_anneal = TimeDecay(self.max_iter, self.warm_ratio)
        super().__init__()

    def rectified_sigmoid(self, r: torch.Tensor) -> torch.Tensor:
        return ((self.zeta - self.gamma) * torch.sigmoid(r) + self.gamma).clamp(0, 1)

    def forward(self, r: torch.Tensor, iters: int) -> torch.Tensor:
        if iters < self.max_iter * self.warm_ratio:
            round_loss = torch.tensor(0, device=r.device)
        else:
            self.beta = self.temp_anneal(iters)
            round_loss = (
                self.alpha
                * (
                    1
                    - torch.pow((self.rectified_sigmoid(r) - 0.5).abs() * 2, self.beta)
                ).sum()
            )
        return round_loss


class AdaRoundDelegator(TorchQuantizeDelegator):
    def __init__(
        self,
        var: Variable,
        config: TensorQuantizationConfig,
        steps: int,
    ) -> None:
        assert isinstance(var, QuantableVariable)
        self.reg = AdaroundRegTerm(max_iter=steps)
        self.config = config
        self.var = var
        self.is_parameter = self.var.is_parameter
        self.rounding = self.initiate_rounding(
            value=self.var.value, config=self.config, zeta=1.1, gamma=-0.1
        )

        if not self.var.is_parameter:
            raise TypeError(
                f"Can not create adaround delegator with variable {var.name}, "
                "Adaround delegator works only with parameter."
            )
        if self.config.state == QuantizationStates.PASSIVE:
            raise TypeError(
                f"Can not create adaround delegator with variable {var.name}, "
                "Adaround delegator can not work with passive parameter."
            )
        self.param_backup = None
        if self.is_parameter:
            self.param_backup = self.var.value.clone()

    @staticmethod
    def initiate_rounding(
        value: torch.Tensor, config: TensorQuantizationConfig, zeta: float, gamma: float
    ) -> torch.Tensor:
        with torch.no_grad():
            scale = config.scale
            assert scale is not None
            if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
                shape = [
                    1 if axis != config.channel_axis else -1
                    for axis in range(value.ndim)
                ]
                scale = scale.view(shape)

            rounding = (value / scale) - (value / scale).floor()
            rounding = -torch.log((zeta - gamma) / (rounding - gamma) - 1)
            rounding = torch.zeros_like(rounding).copy_(rounding)
            rounding.requires_grad = True
        return rounding

    def trainable_tensors(self) -> List[torch.Tensor]:
        tensors = [self.rounding]
        return tensors

    def finalize(self) -> None:
        weight, scale, offset = self.var.value, self.config.scale, self.config.offset
        assert scale is not None and offset is not None
        if self.config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            shape = [
                1 if axis != self.config.channel_axis else -1
                for axis in range(weight.ndim)
            ]
            scale = scale.view(shape)
            offset = offset.view(shape)
        weight = (weight / scale).floor() + (self.rounding >= 0).float()
        weight = torch.clamp(
            weight + offset, self.config.quant_min, self.config.quant_max
        )
        weight = (weight - offset) * scale
        self.var.value = weight

    def withdraw(self) -> None:
        with torch.no_grad():
            if self.param_backup is not None:
                self.var.value.copy_(self.param_backup)

    def __call__(
        self, tensor: torch.Tensor, config: TensorQuantizationConfig
    ) -> torch.Tensor:
        scale = config.scale
        offset = config.offset
        assert scale is not None and offset is not None
        if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
            shape = [
                1 if axis != config.channel_axis else -1 for axis in range(tensor.ndim)
            ]
            scale = scale.view(shape)
            offset = offset.view(shape)
        tensor = (tensor / scale).floor() + self.reg.rectified_sigmoid(self.rounding)
        tensor = torch.clamp(tensor + offset, config.quant_min, config.quant_max)
        tensor = (tensor - offset) * scale
        return tensor

    def regularization_loss(self, step: int) -> torch.Tensor:
        return self.reg.forward(r=self.rounding, iters=step)


@OPTIM_ALGORITHMS.register()
class AdaroundPass(TrainingBasedPass):
    """Blockwise Reconstruction Pass, perform adaround block by block, if you
    specify interested_layers in the setting, then only block which contains
    any of operations specified in interested_layers will be optimized,
    otherwise all searched blocks will be optimized. A standard procedure is,
    first turn all training-based optimization passes off in your quantization
    setting and run a plain quantization, then use error analysis tool(provided
    by ppq) to analysis snr error or cosine similarities of every layer, choose
    names of those with significant snr or poor similarities as your
    interested_layers, then turn on this pass and do optimization. In case you
    have no idea which layers should be selected as interested_layers, simply
    leave it as blank and all blocks will be tuned. Note that you could control
    the maximum number of operations in a block by setting
    OPTIM_ADVOPT_GRAPH_MAXSIZE in ppq.core.common, and by default every block
    will be trained for 300 epochs, which takes certain long time. The
    optimization goal of every block is.
                Loss = LpNormLoss(y, y^) + lambda * rounding_loss(v)
    where y is the output of the current block running in fp32 mode, and y^ is the
    output of the current block running in quant mode, lambda is a hyperparameter
    adjusting scales of rounding loss, and v is the element-wise rounding
    parameter applied to weights of every computing op in the block.
    """

    def __init__(
        self,
        name: str = "Block-wise Adaround Reconstruction",
        interested_layers: List[str] = [],
        is_scale_trainable: bool = False,
        steps: int = 5000,
        lr: float = 1e-3,
        gamma: float = 1.0,
        collecting_device: str = "cuda",
        block_size: int = 4,
    ) -> None:
        super().__init__(name=name)
        self.interested_layers = interested_layers
        self.lr = lr
        self.gamma = gamma
        self.steps = steps
        self.block_size = block_size
        self.collecting_device = collecting_device
        self.is_scale_trainable = is_scale_trainable
        self.loss_fn = torch_mean_square_error

    def tune_block_weight_scale(
        self,
        block: TrainableBlock,
        steps: int = 900,
        loss_fn: Callable = torch_mean_square_error,
    ) -> None:
        # before we tune weight roundings and activation scales, we optimize weight
        # scale by minimizing MSE(W, W^), 900 epochs would be enough in this
        # non-overfit setting.
        # Note that this is usually unnecessary in 8 bit quantization, but we do it
        # anyway and the loss checking procedure makes sure we always obtain no worse
        # results.
        for op in block.rps:
            if op.is_computing_op and isinstance(op, QuantableOperation):
                c, v = op.input_quant_config[1], op.inputs[1]
                delegator = LSQDelegator(config=c, var=v, is_parameter_trainable=False)
                params = delegator.trainable_tensors()
                if len(params) == 0:
                    continue

                optimizer = torch.optim.Adam(params, lr=self.lr)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [int(steps / 2), int(steps * 2 / 3)]
                )

                initial_loss = loss_fn(delegator(tensor=v.value, config=c), v.value)
                for _ in range(steps):
                    optimizer.zero_grad()
                    loss = loss_fn(delegator(tensor=v.value, config=c), v.value)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                post_loss = loss_fn(delegator(tensor=v.value, config=c), v.value)

                if post_loss > initial_loss:
                    delegator.withdraw()

    def finetune(
        self,
        steps: int,
        learning_rate: float,
        block: TrainableBlock,
        executor: TorchExecutor,
        qt_inputs: List[Dict[str, torch.Tensor]],
        fp_outputs: List[Dict[str, torch.Tensor]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Tuple[float, float]:

        # step - 1: enable gradient for training.
        self.enable_block_gradient(block)

        # record pre training loss.
        pre_loss = self.compute_block_loss(
            block=block,
            qt_inputs=qt_inputs,
            fp_outputs=fp_outputs,
            executor=executor,
            loss_fn=self.loss_fn,
        )

        # tune block weight scale
        self.tune_block_weight_scale(block=block)

        # collect trainable params
        trainable_params, delegators = [], {}
        for op in block.rps:
            if not isinstance(op, QuantableOperation):
                continue

            # register quant delegator
            for cfg, var in op.config_with_variable:
                if cfg.state not in {
                    QuantizationStates.ACTIVATED,
                    QuantizationStates.PASSIVE,
                }:
                    continue
                if var.is_parameter and cfg.state not in {QuantizationStates.PASSIVE}:
                    delegator = AdaRoundDelegator(config=cfg, var=var, steps=steps)
                    trainable_params.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator
                elif self.is_scale_trainable:
                    delegator = LSQDelegator(
                        config=cfg, var=var, is_offset_trainable=False
                    )
                    trainable_params.extend(delegator.trainable_tensors())
                    executor.register_quantize_delegate(config=cfg, delegator=delegator)
                    delegators[cfg] = delegator

        # check if empty.
        tensors = [tensor for tensor in trainable_params if tensor.requires_grad]
        if len(tensors) == 0:
            for cfg, delegator in delegators.items():
                executor.remove_quantize_delegate(config=cfg)
            return 0, 0

        # initialize optimizer.
        if optimizer is None:
            optimizer = torch.optim.Adam(tensors, lr=learning_rate)

        dataset_length = len(qt_inputs)
        if dataset_length == 0:
            raise ValueError("Dataset is empty.")

        # step 2 - training procedure
        for idx in tqdm(range(steps), desc="# Tuning Procedure "):
            qt_input, fp_output = (
                qt_inputs[idx % dataset_length],
                fp_outputs[idx % dataset_length],
            )

            # forward
            optimizer.zero_grad()
            feed_dict = {k: v.to(executor._device) for k, v in qt_input.items()}
            output_names = [name for name in fp_output]

            qt_output = executor.partial_graph_forward(
                operations=block.rps, feed_dict=feed_dict, output_names=output_names
            )

            # compute loss
            loss = 0.0
            for idx, name in enumerate(output_names):
                loss += self.loss_fn(
                    qt_output[idx], fp_output[name].to(executor._device)
                )

            # collect reg terms
            for delegator in delegators.values():
                if isinstance(delegator, AdaRoundDelegator):
                    loss += delegator.regularization_loss(idx) * self.gamma

            # backward from loss
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # step - 3: record post training loss
        post_loss = self.compute_block_loss(
            block=block,
            qt_inputs=qt_inputs,
            fp_outputs=fp_outputs,
            executor=executor,
            loss_fn=self.loss_fn,
        )

        # check and withdraw
        for cfg, delegator in delegators.items():
            if post_loss > pre_loss:
                delegator.withdraw()
            else:
                delegator.finalize()
            executor.remove_quantize_delegate(config=cfg)

        # disable gradient for evaluation.
        self.disable_block_gradient(block)

        # clear cache
        torch.cuda.empty_cache()
        return pre_loss, post_loss

    def optimize(
        self,
        graph: BaseGraph,
        dataloader: Optional[Iterable] = None,
        executor: Optional[BaseGraphExecutor] = None,
        collate_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        assert isinstance(executor, TorchExecutor)
        assert dataloader is not None
        blocks = self.split_graph_into_blocks(
            graph=graph,
            executing_order=executor._executing_order,
            blocksize=self.block_size,
            interested_layers=self.interested_layers,
        )

        # do per-block finetune
        for block_idx, block in enumerate(blocks):
            # collect data for training
            qt_inputs, fp_outputs = self.collect(
                graph=graph,
                block=block,
                executor=executor,
                dataloader=dataloader,
                collate_fn=collate_fn,
                collecting_device=self.collecting_device,
            )

            print(
                f"# Block [{block_idx + 1} / {len(blocks)}]: "
                f"[{block.sp.name} -> {block.ep.name}]"
            )
            pre_loss, post_loss = self.finetune(
                steps=self.steps,
                learning_rate=self.lr,
                block=block,
                qt_inputs=qt_inputs,
                fp_outputs=fp_outputs,
                executor=executor,
            )
            print(
                f"# Tuning Finished  : ({pre_loss:.4f} -> {min(pre_loss, post_loss):.4f}) [Block Loss]"
            )
            print("")  # blank line
