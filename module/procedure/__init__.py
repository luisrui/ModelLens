from .train_pairwise import *
from .train_listwise import *
from .train import *


def load_procedure_by_name(args):
    """Dispatch the (train_fn, test_fn, procedure_tag) for a given config.

    ModelLens supports a single backbone family (MLP and its
    metric-conditioned variants) and three loss regimes from the paper:
    listwise, pairwise, and the full ensemble (listwise + pairwise +
    pointwise).  Combinations such as ``listwise_pointwise`` and
    ``pairwise_pointwise`` reproduce ablation results in Section 4.4.
    """
    name = args.model_name
    mlp_family = {"MLP", "MLPMetric", "MLPMetricFull"}
    loss_type = args.loss_type

    if name not in mlp_family:
        raise ValueError(
            f"Unknown model name: {name}. Supported: {sorted(mlp_family)}"
        )

    if loss_type in ("ensemble", "listwise_pairwise"):
        return train_mlp, test_mlp, "mlp"
    if loss_type == "pairwise":
        return train_mlp_pairwise, test_mlp, "mlp"
    if loss_type == "listwise":
        return train_mlp_listwise, test_mlp, "mlp"
    if loss_type == "pairwise_pointwise":
        return train_mlp_pairPointwise, test_mlp, "mlp"
    if loss_type == "listwise_pointwise":
        return train_mlp_listPointwise, test_mlp, "mlp"
    raise ValueError(f"Unknown loss type: {loss_type}")
