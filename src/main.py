# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional, cast

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#import db2_hf_bert as hf_bert_module
import db2_mosaic_bert as mosaic_bert_module
import db2_text_data as text_data_module
from composer import Trainer, algorithms, Callback
from composer.callbacks import (LRMonitor, MemoryMonitor,
                                OptimizerMonitor, RuntimeEstimator,
                                SpeedMonitor)
from composer.loggers import WandBLogger
from composer.loggers import FileLogger, Logger
from composer.optim import DecoupledAdamW
from composer.utils import checkpoint as composer_checkpoint
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from composer.core import State
import os, torch
from composer.utils import dist

if not dist.is_initialized():
    dist.initialize_dist(device='gpu' if torch.cuda.is_available() else 'cpu')

# Ensure this process is bound to its GPU
if torch.cuda.is_available():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if dist.get_global_rank() == 0:
        print(f"WORLD_SIZE={dist.get_world_size()}  LOCAL_RANK={local_rank}  CUDA_DEVICE={torch.cuda.current_device()}")

class BestLossCheckpoint(Callback):
    """Saves checkpoint whenever eval LanguageCrossEntropy improves."""
    def __init__(self, save_folder: str, filename: str = 'best-loss-checkpoint.pt'):
        self.save_folder = save_folder
        self.filename    = filename
        self.best_loss   = float('inf')

    def eval_end(self, state: State, logger: Logger) -> None:
        current_loss = float('inf')
        if dist.get_global_rank() == 0:
            for key, val in state.eval_metric_values.items():
                if 'LanguageCrossEntropy' in key:
                    current_loss = float(val)
                    break
        loss_tensor = torch.tensor(current_loss, dtype=torch.float64, device=torch.cuda.current_device())
        torch.distributed.broadcast(loss_tensor, src=0)
        current_loss = loss_tensor.item()
        
        if current_loss >= self.best_loss:
            # No improvement — all ranks skip, no collective imbalance
            if dist.get_global_rank() == 0:
                print(f"[BestLossCheckpoint] Loss {current_loss:.6f} "
                    f"(best: {self.best_loss:.6f}) — no improvement.")
            return
        
        self.best_loss = current_loss

        os.makedirs(self.save_folder, exist_ok=True)
        save_path = os.path.join(self.save_folder, self.filename)

        # This is a collective — all ranks must call it simultaneously
        composer_checkpoint.save_checkpoint(
            state=state,
            filename=save_path,
        )

        if dist.get_global_rank() == 0:
            print(
                f"[BestLossCheckpoint] ✓ New best loss {current_loss:.6f} "
                f" -> {save_path}"
            )

def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} '
            'as a result, the batch size would be truncated, please adjust `global_batch_size` '
            f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f'WARNING: device_train_microbatch_size > device_train_batch_size, '
                f'will be reduced from {device_microbatch_size} -> {device_train_batch_size}.'
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size
    # Safely set `device_eval_batch_size` if not provided by user
    if 'device_eval_batch_size' not in cfg:
        if cfg.device_train_microbatch_size == 'auto':
            cfg.device_eval_batch_size = 1
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if 'wandb' in cfg.get('loggers', {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    elif name == 'best_loss_checkpoint': # best checkpoint is created with this callback
        return BestLossCheckpoint(save_folder=kwargs['save_folder'],
            filename=kwargs.get('filename', 'best-loss-checkpoint.pt'))
    #elif name == 'health_checker':
        #return HealthChecker(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name, kwargs):
    if name == 'wandb': #aded wandb logging
        return WandBLogger(**kwargs)
    elif name == 'file_logger':
        return FileLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                                  alpha_f=cfg.alpha_f)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup,
                                         alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')


def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(model.parameters(),
                              lr=cfg.lr,
                              betas=cfg.betas,
                              eps=cfg.eps,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')

# check text_data file to know more
def build_dataloader(cfg, tokenizer, device_batch_size):
    if cfg.name == 'text': 
        return text_data_module.build_text_dataloader(cfg, tokenizer, device_batch_size)
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')

def build_model(cfg: DictConfig):
    if cfg.name == 'hf_bert':
        return hf_bert_module.create_hf_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    elif cfg.name == 'mosaic_bert':
        return mosaic_bert_module.create_mosaic_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get('pretrained_checkpoint', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            gradient_checkpointing=cfg.get('gradient_checkpointing', None))
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def main(cfg: DictConfig,
         return_trainer: bool = False,
         do_train: bool = True) -> Optional[Trainer]:
    print('Training using config: ')
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(
        cfg.train_loader,
        model.tokenizer,
        cfg.global_train_batch_size // dist.get_world_size(),
    )
    print('Building eval loader...')
    global_eval_batch_size = cfg.get('global_eval_batch_size',
                                     cfg.global_train_batch_size)
    eval_loader = build_dataloader(
        cfg.eval_loader,
        model.tokenizer,
        global_eval_batch_size // dist.get_world_size(),
    )

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in cfg.get('loggers', {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in cfg.get('callbacks', {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in cfg.get('algorithms', {}).items()
    ]

    if cfg.get('run_name') is None:
        cfg.run_name = os.environ.get('COMPOSER_RUN_NAME', 'bert')

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        device_train_microbatch_size=cfg.get('device_train_microbatch_size',
                                             'auto'),
        save_folder=cfg.get('save_folder', './model_pretrain/'),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep',
                                             -1),
        save_overwrite=cfg.get('save_overwrite', False),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
        python_log_level=cfg.get('python_log_level', None),
    )

    print('Logging config...')
    log_config(cfg)

    if do_train:
        for name, param in model.named_parameters():
            if 'bias' in name and param.grad is not None:
                print(f"{name}: grad_norm={param.grad.norm():.6f}, "
                f"param_norm={param.norm():.6f}, "
                f"ratio={param.grad.norm()/param.norm():.6f}")
        total_params = sum(p.numel() for p in model.parameters())
        optimizer_params = sum(p.numel() for pg in optimizer.param_groups 
                       for p in pg['params'])
        print(f"model params: {total_params}, optimizer params: {optimizer_params}")
        print('Starting training...')
        trainer.fit()

        # Log the resolved microbatch size to WandB (useful when auto is used)
        if 'wandb' in cfg.get('loggers', {}):
            try:
                import wandb
                if wandb.run:
                    resolved_mbs = trainer.state.device_train_microbatch_size
                    wandb.config.update({
                        'resolved_device_train_microbatch_size': resolved_mbs
                    }, allow_val_change=True)
                    print(f'Resolved device_train_microbatch_size: {resolved_mbs}')
            except Exception as e:
                print(f'Could not log resolved microbatch size: {e}')

    if return_trainer:
        return trainer

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg) 
    main(cfg)