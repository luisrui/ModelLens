# module/model/loader.py
import importlib, pkgutil
from types import SimpleNamespace
from typing import Any
from module.model import registry as R
import module.model as model_pkg  
from module.model.Loss import *

# Ensure all model modules are imported to trigger decorators
def _ensure_all_model_modules_imported():
    """Import all model modules in module.model to trigger decorators."""
    for m in pkgutil.iter_modules(model_pkg.__path__):
        importlib.import_module(f"{model_pkg.__name__}.{m.name}")

def build_model_by_name(name: str, args: Any):
    # 1) Ensure all model modules are imported to avoid "not registered"
    _ensure_all_model_modules_imported()
    # 2) Get class from registry
    cls = R.get_model_class(name)
    return cls(args)

def get_loss_by_name(args: Any):
    if args.loss_type == "ensemble":
        criterion = JointCriterion(
            args,
            ListwisePointwiseLoss(args, temperature=1.0, use_l2_reg=True, max_grad_norm=5.0),
            PairwisePointwiseLoss(args, use_sample_weight=True, use_l2_reg=True, max_grad_norm=5.0),
            lambda_list=args.lambda_list,
            lambda_pair=args.lambda_pair,
            max_grad_norm=5.0,
        )
    elif args.loss_type == "pairwise":
        criterion = PairwiseBPRLoss(args, use_sample_weight=True, use_l2_reg=True, max_grad_norm=5.0)
    elif args.loss_type == "listwise":
        criterion = ListwiseRankLoss(args, temperature=1.0, use_l2_reg=True, max_grad_norm=5.0)
    elif args.loss_type == "pairwise_pointwise":
        criterion = PairwisePointwiseLoss(args, use_sample_weight=True, use_l2_reg=True, max_grad_norm=5.0)
    elif args.loss_type == "listwise_pointwise":
        criterion = ListwisePointwiseLoss(args, temperature=1.0, use_l2_reg=True, max_grad_norm=5.0)
    elif args.loss_type == "listwise_pairwise":
        criterion = JointCriterion(
            args,
            ListwiseRankLoss(args, temperature=1.0, use_l2_reg=True, max_grad_norm=5.0),
            PairwiseBPRLoss(args, use_sample_weight=True, use_l2_reg=True, max_grad_norm=5.0),
            lambda_list=args.lambda_list,
            lambda_pair=args.lambda_pair,
            max_grad_norm=5.0,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    return criterion