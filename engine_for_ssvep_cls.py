# --------------------------------------------------------
# SSVEP classification (40-class) using CLS token on LaBraM backbone
# ---------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils
from contextlib import nullcontext


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps: int = 0,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    ch_names=None,
                    args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    criterion = nn.CrossEntropyLoss()
    grad_accum = getattr(args, "gradient_accumulation_steps", 1) if args is not None else 1

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        if samples.ndim == 3:
            samples = samples.unsqueeze(2)  # (B, chan, 1, time)

        # reshape time axis into patches of 200 to match patch embed
        patch_len = 200
        time_len = samples.shape[-1]
        usable_len = (time_len // patch_len) * patch_len
        if usable_len == 0:
            raise ValueError(f"time length {time_len} too short for patch length {patch_len}")
        if usable_len != time_len:
            samples = samples[..., :usable_len]
        num_patches = usable_len // patch_len

        bsz, n_chan, n_subband, _ = samples.shape
        samples = samples.reshape(bsz, n_chan, n_subband, num_patches, patch_len)
        samples = samples.reshape(bsz, n_chan * n_subband, num_patches, patch_len)

        # expand channel indices for subbands to align pos_embed length
        if ch_names is not None:
            base_chans = utils.get_input_chans(ch_names)  # [cls + channels]
            # duplicate channel indices for each subband
            input_chans = [base_chans[0]] + [c for c in base_chans[1:] for _ in range(n_subband)]
        else:
            input_chans = None
        targets = targets.to(device, non_blocking=True)

        sync_needed = args is not None and getattr(args, "distributed", False) and (step + 1) % grad_accum != 0
        my_context = model.no_sync if sync_needed else nullcontext
        with my_context():
            with torch.cuda.amp.autocast():
                outputs = model(samples, input_chans=input_chans, classification=True)
                loss = criterion(outputs, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                sys.exit(1)

            loss = loss / grad_accum
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % grad_accum == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % grad_accum == 0:
                optimizer.zero_grad()

        torch.cuda.synchronize()
        acc = (outputs.max(-1)[1] == targets).float().mean().item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(acc=acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device, header='Test:', ch_names=None):
    criterion = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.float().to(device, non_blocking=True) / 100
        if samples.ndim == 3:
            samples = samples.unsqueeze(2)

        patch_len = 200
        time_len = samples.shape[-1]
        usable_len = (time_len // patch_len) * patch_len
        if usable_len == 0:
            raise ValueError(f"time length {time_len} too short for patch length {patch_len}")
        if usable_len != time_len:
            samples = samples[..., :usable_len]
        num_patches = usable_len // patch_len

        bsz, n_chan, n_subband, _ = samples.shape
        samples = samples.reshape(bsz, n_chan, n_subband, num_patches, patch_len)
        samples = samples.reshape(bsz, n_chan * n_subband, num_patches, patch_len)

        if ch_names is not None:
            base_chans = utils.get_input_chans(ch_names)
            input_chans = [base_chans[0]] + [c for c in base_chans[1:] for _ in range(n_subband)]
        else:
            input_chans = None
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples, input_chans=input_chans, classification=True)
            loss = criterion(outputs, targets)

        acc = (outputs.max(-1)[1] == targets).float().mean().item()
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc, n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
