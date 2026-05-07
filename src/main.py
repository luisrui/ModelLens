from __future__ import annotations
import argparse
import datetime
import logging
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from types import SimpleNamespace
from module.utils.general_util import read_yaml

from module.utils.general_util import set_random_seed, read_yaml, setup_logger
from module.model.MLP import *
from module.model.Loss import *
from module.procedure import *
from module.data.loader import *
from module.model.loader import *

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, None)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _init_distributed(args) -> None:
    use_ddp = bool(getattr(args, "use_ddp", False))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.is_distributed = bool(use_ddp and world_size > 1)
    args.world_size = world_size
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not args.is_distributed:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(f"cuda:{args.local_rank}")
        backend = "nccl"
    else:
        args.device = torch.device("cpu")
        backend = "gloo"
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)


def _make_logger(log_file: str, is_main: bool) -> logging.Logger:
    if is_main:
        return setup_logger(log_file)
    logger = logging.getLogger(f"two_tower.rank_non_main")
    logger.handlers = []
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def load_checkpoint_compatible(model: torch.nn.Module, checkpoint_path: str) -> tuple[int, int, int]:
    """
    Load checkpoint with backward compatibility for architecture changes.

    Returns:
        (num_exact_loaded, num_shape_adapted, num_skipped)
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    ckpt_state = raw.get("model", raw) if isinstance(raw, dict) else raw
    if not isinstance(ckpt_state, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt_state)}")

    base_model = _unwrap_model(model)
    cur_state = base_model.state_dict()
    exact_loaded = 0
    shape_adapted = 0
    skipped = 0

    for key, old_val in ckpt_state.items():
        if key not in cur_state:
            skipped += 1
            continue

        new_val = cur_state[key]
        if new_val.shape == old_val.shape:
            cur_state[key] = old_val
            exact_loaded += 1
            continue

        # Shape-adapt by copying the overlapping block and keeping the rest as init.
        if new_val.ndim == old_val.ndim:
            merged = new_val.clone()
            overlap = tuple(min(a, b) for a, b in zip(new_val.shape, old_val.shape))
            index = tuple(slice(0, n) for n in overlap)
            merged[index] = old_val[index]
            cur_state[key] = merged
            shape_adapted += 1
        else:
            skipped += 1

    base_model.load_state_dict(cur_state, strict=False)
    return exact_loaded, shape_adapted, skipped


def main(args):
    args.use_ddp = bool(getattr(args, "use_ddp", False)) or _env_flag("USE_DDP", False)
    if args.use_ddp:
        args.use_data_parallel = False
    args.ddp_find_unused_parameters = bool(getattr(args, "ddp_find_unused_parameters", False)) or _env_flag(
        "DDP_FIND_UNUSED_PARAMETERS", False
    )

    _init_distributed(args)
    set_random_seed(args.seed + int(getattr(args, "rank", 0)))

    # Backward-compatible device parsing:
    # allow values like 0 / "0" / "1" in yaml and map to cuda:{id}.
    if not bool(getattr(args, "is_distributed", False)):
        device_raw = str(args.device).strip()
        if device_raw.isdigit():
            if torch.cuda.is_available():
                device_raw = f"cuda:{device_raw}"
            else:
                device_raw = "cpu"
        primary_device = torch.device(device_raw)
        args.device = primary_device

    is_main = int(getattr(args, "rank", 0)) == 0
    if bool(getattr(args, "is_distributed", False)) and not is_main:
        args.use_wandb = False

    # Load train, val, test datasets and dataloaders
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, test_ood_dataset, test_ood_loader = get_dataset_and_dataloader(args)
    
    args.num_models = train_dataset.get_num_models()
    args.num_tasks = train_dataset.get_num_tasks()
    args.num_metrics = train_dataset.get_num_metrics() if hasattr(train_dataset, "get_num_metrics") else 1
    if getattr(args, "use_dataset_id_as_desp", False):
        args.num_datasets = train_dataset.get_num_global_datasets()
    args.unknown_metric_id = getattr(train_dataset, "metric2id", {}).get("unknown_metric", 0)
    # Size buckets are needed whenever the size embedding is used — either by
    # the prior head (use_size_prior) or by the backbone (use_size_feature,
    # default True). Only zero out when both are off.
    _need_size_emb = bool(args.use_size_prior) or bool(getattr(args, "use_size_feature", True))
    args.num_size_buckets = train_dataset.get_num_size_buckets() if _need_size_emb else 0
    args.num_families = train_dataset.get_num_families() if getattr(args, "use_family_prior", False) else 0
    if getattr(args, "use_ms_spider_repr", False):
        ms_dim = getattr(train_dataset, "ms_repr_dim", 0)
        if ms_dim and ms_dim > 0:
            args.ms_repr_dim = int(ms_dim)
    
    train_func, test_func, procedure_name = load_procedure_by_name(args)    
    log_dir = f"log/{procedure_name}/{args.data_name}/{args.trail_name}"
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
    logger = _make_logger(os.path.join(log_dir, "train.log"), is_main=is_main)
    
    # 5) Init model/opt
    device = torch.device(args.device)
    if is_main:
        logger.info(f"Initializing model class: {args.model_name}")
    model = build_model_by_name(args.model_name, args).to(device)

    if bool(getattr(args, "is_distributed", False)):
        model = DDP(
            model,
            device_ids=[args.local_rank] if args.device.type == "cuda" else None,
            output_device=args.local_rank if args.device.type == "cuda" else None,
            find_unused_parameters=bool(getattr(args, "ddp_find_unused_parameters", False)),
        )
        if is_main:
            logger.info(
                f"Enabled DDP (world_size={args.world_size}, local_rank={args.local_rank}, rank={args.rank})"
            )

    use_data_parallel = bool(getattr(args, "use_data_parallel", False)) and not bool(getattr(args, "is_distributed", False))
    if use_data_parallel and torch.cuda.is_available():
        cfg_ids = getattr(args, "device_ids", None)
        if isinstance(cfg_ids, (list, tuple)) and len(cfg_ids) > 0:
            device_ids = [int(x) for x in cfg_ids]
        else:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) >= 2:
            output_id = int(device_ids[0])
            args.device = torch.device(f"cuda:{output_id}")
            model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_id)
            if is_main:
                logger.info(f"Enabled DataParallel on device_ids={device_ids} (primary=cuda:{output_id})")
        else:
            if is_main:
                logger.warning(
                    "use_data_parallel=True but fewer than 2 CUDA devices are available. "
                    "Falling back to single-GPU training."
                )

    criterion = get_loss_by_name(args)
    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(params, lr=args.learning_rate)
    
    if args.checkpoint_path:
        exact, adapted, skipped = load_checkpoint_compatible(model, args.checkpoint_path)
        if is_main:
            logger.info(
                f"Loaded checkpoint from {args.checkpoint_path} "
                f"(exact={exact}, adapted={adapted}, skipped={skipped})"
            )
        
    best_model_path = None
    if args.is_train:
        best_model_path = train_func(args, model, train_loader, val_loader, criterion, optim, logger, procedure_name=procedure_name)
    if bool(getattr(args, "is_distributed", False)):
        dist.barrier()

    if is_main:
        if args.is_train and best_model_path:
            exact, adapted, skipped = load_checkpoint_compatible(model, best_model_path)
            logger.info(
                f"Loaded best checkpoint from {best_model_path} "
                f"(exact={exact}, adapted={adapted}, skipped={skipped})"
            )

        only_ood_eval = bool(getattr(args, "only_ood_eval", False))
        if not only_ood_eval:
            results = test_func(args, model, test_loader, logger, mode="in-domain", procedure_name=procedure_name)
        if args.is_ood:
            results = test_func(args, model, test_ood_loader, logger, mode="ood", procedure_name=procedure_name)
        elif only_ood_eval:
            logger.warning("only_ood_eval=True but is_ood=False; no test split was evaluated.")

    if bool(getattr(args, "is_distributed", False)):
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--config", type=str, default="../configs/vicuna-7b/args.yaml", help="the relative path of argments file")
    # parse.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parse.parse_args()
    yaml_args = read_yaml(path=args.config)
    yaml_args.update(vars(args))
    args = SimpleNamespace(**yaml_args)
    main(args)
