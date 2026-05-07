import numpy as np
import random
import torch
import yaml
import os
import logging
from tqdm import tqdm
import pathlib

def set_random_seed(seed=2020):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)
    return dict

class TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
            

def setup_logger(log_path: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("two_tower")
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate printing

    # if duplicate call, avoid adding handler
    if logger.handlers:
        return logger

    # console (use tqdm)
    ch = TqdmHandler()
    ch.setLevel(level)

    # file
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def load_keys(path="/scr/users/ruicai/ModelProfile/data/keys.txt") -> dict[str, str]:
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Key file not found: {p}")
    
    kv: dict[str,str] = {}
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            if "=" not in s:
                raise ValueError(f"Line {i} missing '=': {s}")
            k, v = s.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv

def _to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    else:
        return batch
    
def format_eval_summary(summary: dict, topk_list, tag: str = "Test", epoch: int | None = None) -> str:
    """
    Format the summary dictionary of evaluate_ranking function into a multi-line string for logger.info.
    tag: "Valid" / "Test"
    epoch: If it is validation stage, you can pass epoch, display on the first line; if it is test stage, you can pass None.
    """
    lines = []

    # Top overview line
    if epoch is not None:
        head = (f"[{tag}] epoch={epoch}  "
                f"datasets={summary.get('datasets_evaluated', 'NA')}  "
                f"mean_weighted_tau={summary.get('mean_weighted_tau', float('nan')):.4f}")
    else:
        head = (f"[{tag}]  "
                f"datasets={summary.get('datasets_evaluated', 'NA')}  "
                f"mean_weighted_tau={summary.get('mean_weighted_tau', float('nan')):.4f}")
    lines.append(head)

    # Table header
    lines.append(f"[{tag}]   K   mean_ndcg    mean_hit   mean_recall   #datasets")

    # Each K line
    for k in topk_list:
        ndcg = summary.get(f"mean_ndcg@{k}", float('nan'))
        hit  = summary.get(f"mean_hit@{k}", float('nan'))
        recall = summary.get(f"mean_recall@{k}", float('nan'))
        n_ds = summary.get(f"num_datasets_for_k@{k}", 0)
        lines.append(
            f"[{tag}] {k:4d}  {ndcg:10.4f}  {hit:9.4f}  {recall:9.4f}  {n_ds:9d}"
        )

    return "\n".join(lines)