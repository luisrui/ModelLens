import pandas as pd
from typing import Dict, Tuple, Any, List
import numpy as np
import torch
import scipy.stats
from pathlib import Path
import re


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _normalize_dataset_token(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _expand_target_dataset_tokens(names: List[str]) -> set[str]:
    alias = {
        "cars": ["cars", "cars196", "cars186", "cars-196", "cars-186", "stanfordcars", "stanford cars"],
        "pets": ["pets", "pet", "oxfordiiitpets", "oxfordiiitpet", "oxford-iiit pets", "oxford-iiit pet", "oxfordpets"],
        "pet": ["pets", "pet", "oxfordiiitpets", "oxfordiiitpet", "oxford-iiit pets", "oxford-iiit pet", "oxfordpets"],
        "flowers102": ["flowers102", "flowers-102", "oxford 102 flowers", "oxford 102 flower"],
        "food101": ["food101", "food-101"],
        "renderedsst2": ["renderedsst2", "rendered sst2"],
        "stl10": ["stl10", "stl-10"],
        "cifar10": ["cifar10", "cifar-10"],
        "cifar100": ["cifar100", "cifar-100"],
        "caltech101": ["caltech101", "caltech-101"],
    }
    out: set[str] = set()
    for n in names:
        key = _normalize_dataset_token(n)
        if not key:
            continue
        out.add(key)
        for a in alias.get(key, []):
            out.add(_normalize_dataset_token(a))
    return out


def _target_dataset_token_map(names: List[str]) -> Dict[str, set[str]]:
    alias = {
        "cars": ["cars", "cars196", "cars186", "cars-196", "cars-186", "stanfordcars", "stanford cars"],
        "pets": ["pets", "pet", "oxfordiiitpets", "oxfordiiitpet", "oxford-iiit pets", "oxford-iiit pet", "oxfordpets"],
        "pet": ["pets", "pet", "oxfordiiitpets", "oxfordiiitpet", "oxford-iiit pets", "oxford-iiit pet", "oxfordpets"],
        "flowers102": ["flowers102", "flowers-102", "oxford 102 flowers", "oxford 102 flower"],
        "food101": ["food101", "food-101"],
        "renderedsst2": ["renderedsst2", "rendered sst2"],
        "stl10": ["stl10", "stl-10"],
        "cifar10": ["cifar10", "cifar-10"],
        "cifar100": ["cifar100", "cifar-100"],
        "caltech101": ["caltech101", "caltech-101"],
    }
    out: Dict[str, set[str]] = {}
    for n in names:
        key = _normalize_dataset_token(n)
        toks = {key} if key else set()
        for a in alias.get(key, []):
            toks.add(_normalize_dataset_token(a))
        out[str(n)] = toks
    return out


def _resolve_dataset_id(ds, ds_name: str):
    # exact match first
    if ds_name in ds.dataset2id:
        return ds.dataset2id[ds_name]

    # Try canonical token + alias-expanded tokens.
    token_candidates = {_normalize_dataset_token(ds_name)}
    try:
        token_candidates.update(_expand_target_dataset_tokens([str(ds_name)]))
    except Exception:
        pass

    # prebuilt map from dataset module
    token2id = getattr(ds, "dataset_token2id", None)
    if isinstance(token2id, dict):
        for tok in token_candidates:
            if tok in token2id:
                return token2id[tok]

    # fallback build on the fly
    for k, v in ds.dataset2id.items():
        if _normalize_dataset_token(k) in token_candidates:
            return v
    return None


def _resolve_metric_id(ds, ds_name: str, g: pd.DataFrame | None = None):
    metric2id = getattr(ds, "metric2id", None)
    if not isinstance(metric2id, dict) or len(metric2id) == 0:
        return None

    metric_name = None
    if g is not None and "metric" in g.columns and len(g) > 0:
        metric_name = str(g["metric"].iloc[0])
    if metric_name is None and "||" in str(ds_name):
        metric_name = str(ds_name).split("||", 1)[1]
    if metric_name is None:
        metric_name = "unknown_metric"

    if metric_name in metric2id:
        return int(metric2id[metric_name])

    tok = _normalize_dataset_token(metric_name)
    token2id = getattr(ds, "metric_token2id", None)
    if isinstance(token2id, dict) and tok in token2id:
        return int(token2id[tok])

    return int(metric2id.get("unknown_metric", 0))


def _build_all_model_size_ids(ds, all_model_names: List[str], device):
    """
    Build size bucket ids for all models with safe fallback.
    If dataset does not provide modelid2bucket, return all-zero ids.
    """
    model2id = getattr(ds, "model2id", {})
    modelid2bucket = getattr(ds, "modelid2bucket", None)
    if isinstance(modelid2bucket, dict) and len(modelid2bucket) > 0:
        vals = []
        for m in all_model_names:
            mid = model2id.get(m, None)
            if mid is None:
                vals.append(0)
            else:
                vals.append(int(modelid2bucket.get(mid, 0)))
        return torch.as_tensor(vals, dtype=torch.long, device=device)
    return torch.zeros(len(all_model_names), dtype=torch.long, device=device)


def _forward_with_optional_metric(model, *args, metric_ids=None):
    model_ref = _unwrap_model(model)
    if metric_ids is not None:
        try:
            return model_ref(*args, metric_ids=metric_ids)
        except TypeError:
            pass
    return model_ref(*args)


def _score_matrix_with_optional_metric(model, *args, metric_ids=None, **kwargs):
    model_ref = _unwrap_model(model)
    if metric_ids is not None:
        try:
            return model_ref.score_matrix(*args, metric_ids=metric_ids, **kwargs)
        except TypeError:
            pass
    return model_ref.score_matrix(*args, **kwargs)


def _canonical_backbone_token(name: str) -> str:
    tok = _normalize_dataset_token(name)
    # handle common aliases in target model files
    alias = {
        "inceptionv3": "inceptionv3",
        "inception3": "inceptionv3",
        "mnasnet10": "mnasnet10",
        "mnasnet1_0": "mnasnet10",
        "mnasnet10py": "mnasnet10",
        "mobilenetv2": "mobilenetv2",
        "densenet121": "densenet121",
        "densenet169": "densenet169",
        "densenet201": "densenet201",
    }
    if tok in alias:
        return alias[tok]
    if tok.startswith("resnet") and tok[6:].isdigit():
        return f"resnet{tok[6:]}"
    if tok.startswith("densenet") and tok[8:].isdigit():
        return f"densenet{tok[8:]}"
    return tok


def _normalize_model_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _model_alias_candidates(name: str) -> List[str]:
    n = str(name).strip().lower()
    if not n:
        return []
    cands = [
        n,
        n.replace("-", ""),
        n.replace("_", ""),
        n.replace(" ", ""),
        re.sub(r"[^a-z0-9]+", "", n),
        n.replace("-", "_"),
        n.replace("_", "-"),
    ]
    if "/" in n:
        base = n.split("/")[-1]
        cands.extend([
            base,
            base.replace("-", ""),
            base.replace("_", ""),
            re.sub(r"[^a-z0-9]+", "", base),
        ])
    out = []
    seen = set()
    for c in cands:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _build_model_alias_lookup(model2id: Dict[str, int]) -> Dict[str, str]:
    alias = {}
    for mk in model2id.keys():
        for a in _model_alias_candidates(mk):
            if a not in alias:
                alias[a] = mk
    return alias


def _resolve_model_key_eval(
    raw_model_name: str,
    model2id: Dict[str, int],
    alias_lookup: Dict[str, str],
) -> str | None:
    for a in _model_alias_candidates(raw_model_name):
        if a in model2id:
            return a
        m = alias_lookup.get(a)
        if m is not None and m in model2id:
            return m
    return None


def _load_target_models_generic(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target model file not found: {path}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) == 0:
        raise ValueError(f"target model file is empty: {path}")
    return lines


def _get_ms_scores_for_tokens(ds, ds_id: int, tokens: List[str]) -> np.ndarray | None:
    ms_scores_map = getattr(ds, "ms_scores_by_dsid", None)
    if not isinstance(ms_scores_map, dict):
        return None
    if ds_id not in ms_scores_map:
        return None
    raw_scores = ms_scores_map[ds_id]
    if isinstance(raw_scores, torch.Tensor):
        raw_scores = raw_scores.detach().cpu().numpy()
    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)

    # Preferred: align by exported model_list.
    model_list_map = getattr(ds, "ms_model_list_by_dsid", None)
    if isinstance(model_list_map, dict) and ds_id in model_list_map:
        names = model_list_map[ds_id]
        if len(names) == len(raw_scores):
            tok2score = {}
            for n, s in zip(names, raw_scores):
                t = _canonical_backbone_token(str(n))
                if t not in tok2score:
                    tok2score[t] = float(s)
            arr = []
            for t in tokens:
                if t not in tok2score:
                    return None
                arr.append(tok2score[t])
            return np.asarray(arr, dtype=float)

    # Fallback: assume same order as target tokens.
    if len(raw_scores) >= len(tokens):
        return np.asarray(raw_scores[: len(tokens)], dtype=float)
    return None


def _load_target_backbones(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target model file not found: {path}")
    labels = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tokens = [_canonical_backbone_token(x) for x in labels]
    # de-dup while keeping order
    out, seen = [], set()
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@torch.no_grad()
def evaluate_ranking_mlp_fixed_backbones(
    model,
    data_loader,
    device,
    topk_list,
    target_models_file: str,
    target_datasets: List[str] | None = None,
    task_name: str = "Image Classification",
    strict_target_datasets: bool = False,
):
    """
    Validation for product tuning:
    evaluate only fixed backbone set (e.g. 10 pretrain models) on fixed datasets.
    """
    model.eval()
    ds = data_loader.dataset
    origin = ds.origin_data

    target_tokens = _load_target_backbones(target_models_file)
    if len(target_tokens) < 2:
        raise ValueError(f"Need at least 2 target backbones in {target_models_file}, got {len(target_tokens)}")

    # token -> model2id key
    token_to_modelkey: Dict[str, str] = {}
    for mk in ds.model2id.keys():
        tok = _canonical_backbone_token(mk)
        if tok not in token_to_modelkey:
            token_to_modelkey[tok] = mk
    missing_tokens = [t for t in target_tokens if t not in token_to_modelkey]
    if missing_tokens:
        raise ValueError(f"Missing target backbones in model2id: {missing_tokens}")

    all_model_names = [ds.id2model[i] for i in range(ds.get_num_models())]
    all_model_size_ids = _build_all_model_size_ids(ds, all_model_names, device)
    use_family_prior = getattr(ds, "use_family_prior", False)
    if use_family_prior and hasattr(ds, "modelid2family"):
        all_model_family_ids = torch.as_tensor(
            [ds.modelid2family[ds.model2id[m]] for m in all_model_names], dtype=torch.long, device=device
        )
    else:
        all_model_family_ids = None
    model_cache = _unwrap_model(model).build_model_cache(
        all_model_names=all_model_names,
        all_model_size_ids=all_model_size_ids,
        all_model_family_ids=all_model_family_ids,
        device=device,
    )

    wanted_ds_tokens = None
    target_dataset_map = None
    if target_datasets:
        wanted_ds_tokens = _expand_target_dataset_tokens(list(target_datasets))
        target_dataset_map = _target_dataset_token_map(list(target_datasets))

    task_id = ds.task2id.get(task_name, None)
    if task_id is None:
        raise ValueError(f"task_name '{task_name}' not found in dataset task2id")

    per_dataset_stats = {}
    taus_fused = []
    NDCG = {k: [] for k in topk_list}
    HIT = {k: [] for k in topk_list}
    RECALL = {k: [] for k in topk_list}
    COUNT = {k: 0 for k in topk_list}

    if "metric" in origin.columns:
        grouped = origin.groupby(["dataset", "metric"], sort=False)
    else:
        grouped = [((ds_name, None), g_ds) for ds_name, g_ds in origin.groupby("dataset", sort=False)]

    for (ds_name, metric_name), g_ds in grouped:
        ds_base_name = str(ds_name).split("||", 1)[0]
        ds_tok = _normalize_dataset_token(ds_base_name)
        if wanted_ds_tokens is not None and ds_tok not in wanted_ds_tokens:
            continue

        g_task = g_ds[g_ds["task"] == task_name]
        if len(g_task) == 0:
            # strict mode for product tuning; skip datasets without target task
            continue
        g = g_task.dropna(subset=["value"]).groupby("model", as_index=False)["value"].mean()
        if len(g) < 2:
            continue

        gt_map = {}
        for _, row in g.iterrows():
            tok = _canonical_backbone_token(str(row["model"]))
            if tok in target_tokens and tok not in gt_map:
                gt_map[tok] = float(row["value"])
        if len(gt_map) < 2:
            continue

        # keep fixed order from target list, but only use tokens existing in GT
        cur_tokens = [t for t in target_tokens if t in gt_map]
        gt_values = np.asarray([gt_map[t] for t in cur_tokens], dtype=float)
        gt_ranks = _to_rank(gt_values, descending=True)

        ds_lookup_name = (
            f"{ds_name}||{metric_name}"
            if (metric_name is not None and getattr(ds, "use_dataset_metric_key", True))
            else str(ds_name)
        )
        ds_id = _resolve_dataset_id(ds, ds_lookup_name)
        if ds_id is None:
            continue
        ds_emb = torch.tensor(ds.dataset_vecs[ds_id], dtype=torch.float, device=device).unsqueeze(0)
        if getattr(ds, "use_ms_spider_repr", False):
            ms_repr = ds.ms_repr_by_dsid.get(ds_id, None)
            ms_repr_batch = ms_repr.to(device).unsqueeze(0) if ms_repr is not None else None
        else:
            ms_repr_batch = None

        gt_model_keys = [token_to_modelkey[t] for t in cur_tokens]
        gt_model_ids = [ds.model2id[k] for k in gt_model_keys]
        metric_id = _resolve_metric_id(ds, ds_lookup_name, g_task)
        task_ids = torch.tensor([task_id] * len(gt_model_ids), dtype=torch.long, device=device)
        metric_ids = (
            torch.tensor([metric_id] * len(gt_model_ids), dtype=torch.long, device=device)
            if metric_id is not None
            else None
        )
        ds_emb_per = ds_emb.expand(len(gt_model_ids), -1)
        if hasattr(ds, "modelid2bucket") and isinstance(ds.modelid2bucket, dict):
            size_ids = torch.as_tensor(
                [int(ds.modelid2bucket.get(mid, 0)) for mid in gt_model_ids],
                dtype=torch.long,
                device=device,
            )
        else:
            size_ids = torch.zeros(len(gt_model_ids), dtype=torch.long, device=device)
        family_ids = None
        if getattr(ds, "use_family_prior", False) and hasattr(ds, "modelid2family"):
            family_ids = torch.as_tensor([ds.modelid2family[mid] for mid in gt_model_ids], dtype=torch.long, device=device)
        if ms_repr_batch is not None:
            ms_repr_per = ms_repr_batch.expand(len(gt_model_ids), -1, -1)
        else:
            ms_repr_per = None
        inter_map = getattr(ds, "ms_interaction_by_dsid", {}).get(ds_id, {})
        if len(inter_map) > 0:
            dim = int(getattr(ds, "ms_repr_dim", 1024))
            zero_vec = torch.zeros(dim, dtype=torch.float32)
            ms_interaction = torch.stack([inter_map.get(mid, zero_vec) for mid in gt_model_ids], dim=0).to(device)
        else:
            ms_interaction = None

        model_out = _forward_with_optional_metric(
            model,
            task_ids,
            ds_emb_per,
            torch.as_tensor(gt_model_ids, dtype=torch.long, device=device),
            gt_model_keys,
            size_ids,
            family_ids,
            ms_repr_per,
            ms_interaction,
            metric_ids=metric_ids,
        )
        if isinstance(model_out, tuple):
            pred_scores = model_out[0].detach().cpu().numpy()
        else:
            pred_scores = model_out.detach().cpu().numpy()
        pred_ranks = _to_rank(pred_scores, descending=True)
        tau_fused = scipy.stats.weightedtau(gt_ranks, pred_ranks, rank=None).correlation

        aligned_df = pd.DataFrame(
            {
                "backbone_token": cur_tokens,
                "true": gt_values,
                "true_rank": gt_ranks,
                "pred": pred_scores,
                "pred_rank": pred_ranks,
                "rank_diff": gt_ranks - pred_ranks,
            }
        ).sort_values(["pred_rank", "true_rank"])
        stats = {
            "num_models": int(len(cur_tokens)),
            "weightedtau": None if tau_fused is None else float(tau_fused),
            "aligned": aligned_df,
        }
        for k in topk_list:
            eff_k = min(k, len(cur_tokens))
            ndcg_k = _ndcg_at_k(gt_values, pred_scores, eff_k)
            hit_k = _topk_hit(gt_values, pred_scores, eff_k)
            gt_top_idx = np.argsort(-gt_values, kind="stable")[:eff_k]
            pred_top_idx = np.argsort(-pred_scores, kind="stable")[:eff_k]
            inter = np.intersect1d(pred_top_idx, gt_top_idx)
            recall_k = len(inter) / float(eff_k) if eff_k > 0 else float("nan")
            stats[f"ndcg@{k}"] = float(ndcg_k)
            stats[f"hit@{k}"] = float(hit_k)
            stats[f"recall@{k}"] = float(recall_k)
            if len(cur_tokens) >= k:
                if not np.isnan(ndcg_k):
                    NDCG[k].append(float(ndcg_k))
                if not np.isnan(hit_k):
                    HIT[k].append(float(hit_k))
                if not np.isnan(recall_k):
                    RECALL[k].append(float(recall_k))
                COUNT[k] += 1
        per_dataset_stats[(task_name, ds_lookup_name)] = stats
        if tau_fused is not None and not np.isnan(tau_fused):
            taus_fused.append(float(tau_fused))

    summary = {
        "datasets_evaluated": len(per_dataset_stats),
        "mean_weighted_tau": float(np.mean(taus_fused)) if taus_fused else float("nan"),
        "eval_protocol": "fixed_backbones",
        "target_models_file": str(target_models_file),
    }
    if target_dataset_map is not None:
        eval_tokens = {
            _normalize_dataset_token(str(ds_name).split("||", 1)[0]) for (_, ds_name) in per_dataset_stats.keys()
        }
        covered = []
        missing = []
        for req_name, req_tokens in target_dataset_map.items():
            if len(eval_tokens.intersection(req_tokens)) > 0:
                covered.append(req_name)
            else:
                missing.append(req_name)
        summary["target_datasets_requested"] = list(target_dataset_map.keys())
        summary["target_datasets_evaluated"] = covered
        summary["target_datasets_missing"] = missing
        if strict_target_datasets and len(missing) > 0:
            raise ValueError(
                f"Strict target dataset eval enabled, but missing datasets in validation: {missing}. "
                f"covered={covered}"
            )
    for k in topk_list:
        summary[f"mean_ndcg@{k}"] = float(np.mean(NDCG[k])) if NDCG[k] else float("nan")
        summary[f"mean_hit@{k}"] = float(np.mean(HIT[k])) if HIT[k] else float("nan")
        summary[f"num_datasets_for_k@{k}"] = int(COUNT[k])
        summary[f"mean_recall@{k}"] = float(np.mean(RECALL[k])) if RECALL[k] else float("nan")
    return summary, per_dataset_stats


@torch.no_grad()
def evaluate_ranking_mlp_target_models_all_datasets(
    model,
    data_loader,
    device,
    topk_list,
    target_models_file: str,
    task_name: str | None = "Image Classification",
    strict_target_models: bool = False,
    min_models_per_dataset: int = 2,
):
    """
    Evaluate ranking quality on ALL existing datasets, but only over a target model subset.
    Intended for "new model generalization" evaluation.
    """
    model.eval()
    ds = data_loader.dataset
    origin = ds.origin_data

    target_labels = _load_target_models_generic(target_models_file)
    alias_lookup = _build_model_alias_lookup(ds.model2id)
    resolved_targets: List[Tuple[str, str]] = []
    unresolved_targets: List[str] = []
    seen_keys = set()
    for label in target_labels:
        key = _resolve_model_key_eval(label, ds.model2id, alias_lookup)
        if key is None:
            unresolved_targets.append(label)
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        resolved_targets.append((label, key))
    if strict_target_models and unresolved_targets:
        raise ValueError(f"Unresolved target models: {unresolved_targets}")
    if len(resolved_targets) < 2:
        raise ValueError(
            f"Need at least 2 resolved target models, got {len(resolved_targets)} "
            f"(requested={len(target_labels)})."
        )

    target_keys = [k for _, k in resolved_targets]
    target_id_set = {ds.model2id[k] for k in target_keys}

    all_model_names = [ds.id2model[i] for i in range(ds.get_num_models())]
    all_model_size_ids = _build_all_model_size_ids(ds, all_model_names, device)
    use_family_prior = getattr(ds, "use_family_prior", False)
    if use_family_prior and hasattr(ds, "modelid2family"):
        all_model_family_ids = torch.as_tensor(
            [ds.modelid2family[ds.model2id[m]] for m in all_model_names],
            dtype=torch.long,
            device=device,
        )
    else:
        all_model_family_ids = None
    model_cache = _unwrap_model(model).build_model_cache(
        all_model_names=all_model_names,
        all_model_size_ids=all_model_size_ids,
        all_model_family_ids=all_model_family_ids,
        device=device,
    )

    per_dataset_stats = {}
    taus_fused = []
    NDCG = {k: [] for k in topk_list}
    HIT = {k: [] for k in topk_list}
    RECALL = {k: [] for k in topk_list}
    COUNT = {k: 0 for k in topk_list}

    if "metric" in origin.columns:
        grouped = origin.groupby(["task", "dataset", "metric"], sort=False)
    else:
        grouped = origin.groupby(["task", "dataset"], sort=False)

    for key, g in grouped:
        if len(key) == 3:
            task_name_row, ds_name, metric_name = key
        else:
            task_name_row, ds_name = key
            metric_name = None
        if task_name is not None and str(task_name_row) != str(task_name):
            continue
        g = g.dropna(subset=["value"]).groupby("model", as_index=False)["value"].mean()
        if len(g) < min_models_per_dataset:
            continue

        values_by_mid: Dict[int, List[float]] = {}
        model_key_by_mid: Dict[int, str] = {}
        for _, row in g.iterrows():
            model_key = _resolve_model_key_eval(str(row["model"]), ds.model2id, alias_lookup)
            if model_key is None:
                continue
            mid = ds.model2id[model_key]
            if mid not in target_id_set:
                continue
            values_by_mid.setdefault(mid, []).append(float(row["value"]))
            model_key_by_mid[mid] = model_key

        if len(values_by_mid) < min_models_per_dataset:
            continue

        target_mids_sorted = [ds.model2id[k] for k in target_keys if ds.model2id[k] in values_by_mid]
        if len(target_mids_sorted) < min_models_per_dataset:
            continue

        gt_values = np.asarray(
            [float(np.mean(values_by_mid[mid])) for mid in target_mids_sorted],
            dtype=float,
        )
        gt_ranks = _to_rank(gt_values, descending=True)

        task_id = ds.task2id.get(task_name_row, None)
        if task_id is None:
            continue
        ds_lookup_name = (
            f"{ds_name}||{metric_name}"
            if (metric_name is not None and getattr(ds, "use_dataset_metric_key", True))
            else str(ds_name)
        )
        ds_id = _resolve_dataset_id(ds, ds_lookup_name)
        if ds_id is None:
            continue

        ds_emb = torch.tensor(ds.dataset_vecs[ds_id], dtype=torch.float, device=device).unsqueeze(0)
        if getattr(ds, "use_ms_spider_repr", False):
            ms_repr = ds.ms_repr_by_dsid.get(ds_id, None)
            ms_repr_batch = ms_repr.to(device).unsqueeze(0) if ms_repr is not None else None
        else:
            ms_repr_batch = None

        task_ids = torch.tensor([task_id], dtype=torch.long, device=device)
        metric_id = _resolve_metric_id(ds, ds_lookup_name, g)
        metric_ids = (
            torch.tensor([metric_id], dtype=torch.long, device=device)
            if metric_id is not None
            else None
        )
        scores_all = (
            _score_matrix_with_optional_metric(
                model,
                task_ids,
                ds_emb,
                model_cache,
                ms_repr_batch=ms_repr_batch,
                metric_ids=metric_ids,
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
        )
        pred_scores = scores_all[target_mids_sorted]
        pred_ranks = _to_rank(pred_scores, descending=True)
        tau = scipy.stats.weightedtau(gt_ranks, pred_ranks, rank=None).correlation

        num_models = len(target_mids_sorted)
        per_ds_ndcg = {}
        per_ds_hit = {}
        per_ds_recall = {}
        for k in topk_list:
            eff_k = min(k, num_models)
            ndcg_k = _ndcg_at_k(gt_values, pred_scores, eff_k)
            hit_k = _topk_hit(gt_values, pred_scores, eff_k)
            gt_top_idx = np.argsort(-gt_values, kind="stable")[:eff_k]
            pred_top_idx = np.argsort(-pred_scores, kind="stable")[:eff_k]
            inter = np.intersect1d(pred_top_idx, gt_top_idx)
            recall_k = len(inter) / float(eff_k) if eff_k > 0 else float("nan")
            per_ds_ndcg[k] = float(ndcg_k)
            per_ds_hit[k] = float(hit_k)
            per_ds_recall[k] = float(recall_k)
            if num_models >= k:
                if not np.isnan(ndcg_k):
                    NDCG[k].append(float(ndcg_k))
                if not np.isnan(hit_k):
                    HIT[k].append(float(hit_k))
                if not np.isnan(recall_k):
                    RECALL[k].append(float(recall_k))
                COUNT[k] += 1

        aligned_df = pd.DataFrame({
            "model": [model_key_by_mid[mid] for mid in target_mids_sorted],
            "true": gt_values,
            "true_rank": gt_ranks,
            "pred": pred_scores,
            "pred_rank": pred_ranks,
            "rank_diff": gt_ranks - pred_ranks,
        }).sort_values(["pred_rank", "true_rank"])

        stats = {
            "num_models": int(num_models),
            "weightedtau": None if tau is None else float(tau),
            "aligned": aligned_df,
        }
        for k in topk_list:
            stats[f"ndcg@{k}"] = per_ds_ndcg[k]
            stats[f"hit@{k}"] = per_ds_hit[k]
            stats[f"recall@{k}"] = per_ds_recall[k]

        per_dataset_stats[(task_name_row, ds_lookup_name)] = stats
        if tau is not None and not np.isnan(tau):
            taus_fused.append(float(tau))

    summary = {
        "datasets_evaluated": len(per_dataset_stats),
        "mean_weighted_tau": float(np.mean(taus_fused)) if taus_fused else float("nan"),
        "eval_protocol": "target_models_all_datasets",
        "target_models_file": str(target_models_file),
        "target_models_requested": int(len(target_labels)),
        "target_models_resolved": int(len(resolved_targets)),
        "target_models_unresolved": unresolved_targets,
        "task_filter": task_name,
    }
    for k in topk_list:
        summary[f"mean_ndcg@{k}"] = float(np.mean(NDCG[k])) if NDCG[k] else float("nan")
        summary[f"mean_hit@{k}"] = float(np.mean(HIT[k])) if HIT[k] else float("nan")
        summary[f"num_datasets_for_k@{k}"] = int(COUNT[k])
        summary[f"mean_recall@{k}"] = float(np.mean(RECALL[k])) if RECALL[k] else float("nan")
    return summary, per_dataset_stats

@torch.no_grad()
def evaluate_ranking_mlp(
    model,
    data_loader,
    device,
    topk_list
):
    """
    Evaluate (task-wise full model ranking):
      1) Build the cache for all models (name/ID encoding, size embedding, log q_s)
      2) For each dataset t, compute scores_all[t, :] = \tilde{s}(m,t)
      3) Align with GT (aggregated value), compute ranking metrics
    Return: summary, per_dataset_stats
    """
    model.eval()
    ds = data_loader.dataset  

    origin = ds.origin_data 
    if "metric" in origin.columns:
        per_td_group = list(origin.groupby(["task", "dataset", "metric"], sort=False))
    else:
        per_td_group = list(origin.groupby(["task", "dataset"], sort=False))

    per_dataset_stats = {}
    WTAU_weighted_sum, WTAU_weighted_pairs = 0.0, 0
    NDCG = {k: [] for k in topk_list}
    HIT = {k: [] for k in topk_list}
    RECALL = {k: [] for k in topk_list}
    COUNT = {k: 0 for k in topk_list}

    all_model_names = list(ds.model2id.keys())

    all_model_names = [ds.id2model[i] for i in range(ds.get_num_models())]
    all_model_size_ids = _build_all_model_size_ids(ds, all_model_names, device)

    # Family IDs for evaluation (if available)
    use_family_prior = getattr(ds, "use_family_prior", False)
    if use_family_prior and hasattr(ds, "modelid2family"):
        all_model_family_ids = torch.as_tensor(
            [ds.modelid2family[ds.model2id[m]] for m in all_model_names],
            dtype=torch.long,
            device=device,
        )
    else:
        all_model_family_ids = None

    model_cache = _unwrap_model(model).build_model_cache(
        all_model_names=all_model_names,
        all_model_size_ids=all_model_size_ids,
        all_model_family_ids=all_model_family_ids,
        device=device,
    )

    for key, g in per_td_group:
        if len(key) == 3:
            task_name, ds_name, metric_name = key
        else:
            task_name, ds_name = key
            metric_name = None
        g = g.dropna(subset=["value"]).groupby("model", as_index=False)["value"].mean()
        if len(g) < 2:
            continue
        
        gt_model_names_all = g["model"].tolist()
        gt_model_ids = [ds.model2id[m] for m in gt_model_names_all if m in ds.model2id]
        if len(gt_model_ids) < 2:
            continue

        # Filter model names to only those that exist in ds.model2id
        gt_model_names = [ds.id2model[mid] for mid in gt_model_ids]
        num_models = len(gt_model_names)

        gt_values = g.set_index("model").loc[gt_model_names, "value"].to_numpy()
        gt_ranks = _to_rank(gt_values, descending=True)

        task_id = ds.task2id[task_name]
        ds_lookup_name = (
            f"{ds_name}||{metric_name}"
            if (metric_name is not None and getattr(ds, "use_dataset_metric_key", True))
            else str(ds_name)
        )
        ds_id = _resolve_dataset_id(ds, ds_lookup_name)
        if ds_id is None:
            continue
        ds_emb = torch.tensor(
            ds.dataset_vecs[ds_id], dtype=torch.float, device=device
        ).unsqueeze(0)  # [1, D]
        if getattr(ds, "use_ms_spider_repr", False):
            ms_repr = ds.ms_repr_by_dsid.get(ds_id, None)
            if ms_repr is not None:
                ms_repr_batch = ms_repr.to(device).unsqueeze(0)  # [1, 10, D_ms]
            else:
                ms_repr_batch = None
        else:
            ms_repr_batch = None

        task_ids = torch.tensor([task_id], dtype=torch.long, device=device)
        metric_id = _resolve_metric_id(ds, ds_lookup_name, g)
        metric_ids = (
            torch.tensor([metric_id], dtype=torch.long, device=device)
            if metric_id is not None
            else None
        )
        scores_all = (
            _score_matrix_with_optional_metric(
                model,
                task_ids,
                ds_emb,
                model_cache,
                ms_repr_batch=ms_repr_batch,
                metric_ids=metric_ids,
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
        )  # [M]

        pred_scores = scores_all[gt_model_ids]
        pred_ranks = _to_rank(pred_scores, descending=True)

        tau = scipy.stats.weightedtau(gt_ranks, pred_ranks, rank=None).correlation
        
        per_ds_ndcg = {}
        per_ds_hit = {}
        per_ds_recall = {}        
        per_ds_correct_cnt = {}

        for k in topk_list:
            eff_k = min(k, num_models)
            
            ndcg_k = _ndcg_at_k(gt_values, pred_scores, eff_k)
            hit_k = _topk_hit(gt_values, pred_scores, eff_k)

            per_ds_ndcg[k] = float(ndcg_k)
            per_ds_hit[k] = float(hit_k)

            gt_top_idx = np.argsort(-gt_values, kind="stable")[:eff_k]
            pred_top_idx = np.argsort(-pred_scores, kind="stable")[:eff_k]
            inter = np.intersect1d(pred_top_idx, gt_top_idx)
            correct_cnt = len(inter)
            recall_k = correct_cnt / float(eff_k)
            
            per_ds_correct_cnt[k] = int(correct_cnt)
            per_ds_recall[k] = float(recall_k)
            
            if num_models >= k:
                if not np.isnan(ndcg_k):
                    NDCG[k].append(float(ndcg_k))
                if not np.isnan(hit_k):
                    HIT[k].append(float(hit_k))
                if not np.isnan(recall_k):
                    RECALL[k].append(float(recall_k))
                COUNT[k] += 1
        
        aligned_df = pd.DataFrame({
            "model": gt_model_names,
            "true": gt_values,
            "true_rank": gt_ranks,
            "pred": pred_scores,
            "pred_rank": pred_ranks,
            "rank_diff": gt_ranks - pred_ranks,
        }).sort_values(["pred_rank", "true_rank"])

        stats = {
            "num_models": int(num_models),
            "weightedtau": None if tau is None else float(tau),
            "aligned": aligned_df,
        }
        for k in topk_list:
            stats[f"ndcg@{k}"] = per_ds_ndcg[k]
            stats[f"hit@{k}"] = per_ds_hit[k]
            stats[f"recall@{k}"] = per_ds_recall[k]

        per_dataset_stats[(task_name, ds_lookup_name)] = stats

        if tau is not None and not np.isnan(tau):
            n_pairs = num_models * (num_models - 1) // 2
            WTAU_weighted_sum += float(tau) * n_pairs
            WTAU_weighted_pairs += n_pairs
    
    mean_tau_weighted = (
        float(WTAU_weighted_sum / WTAU_weighted_pairs)
        if WTAU_weighted_pairs > 0
        else float("nan")
    )
        
    summary = {
        "datasets_evaluated": len(per_dataset_stats),
        "mean_weighted_tau": mean_tau_weighted,
    }

    for k in topk_list:
        summary[f"mean_ndcg@{k}"] = float(np.mean(NDCG[k])) if NDCG[k] else float("nan")
        summary[f"mean_hit@{k}"]  = float(np.mean(HIT[k]))  if HIT[k] else float("nan")
        summary[f"num_datasets_for_k@{k}"] = int(COUNT[k])
        summary[f"mean_recall@{k}"] = float(np.mean(RECALL[k])) if RECALL[k] else float("nan")
        
    return summary, per_dataset_stats
    
@torch.no_grad()
def evaluate_ranking_two_tower(
    model, 
    data_loader, 
    device, 
    topk: int = 10,
    margin_eps: float = 0.02):
    """
    For each dataset in valid_data_loader.dataset, compute a score for every model,
    rank models by the score, and compare with ground-truth ranking aggregated from
    triplet metric values. Returns summary and per-dataset results.
    """
    model.eval()
    ds = data_loader.dataset  
    # Build GT per-dataset model value (mean) -> ranks
    origin = ds.origin_data  # pandas DataFrame with columns: dataset, model, value
    per_ds_group = list(origin.groupby("dataset", sort=False))

    WTAU, NDCG, HIT = [], [], []
    per_dataset_stats: Dict[str, Dict[str, Any]] = {}

    all_model_names = list(ds.model2id.keys())
    all_model_size_ids = _build_all_model_size_ids(ds, all_model_names, device)
    for ds_name, g in per_ds_group:
        g = g.dropna(subset=["value"]).groupby("model", as_index=False)["value"].mean()
        if len(g) < 2:
            continue

        # GT arrays aligned by model ids that appear in GT for this dataset
        gt_model_names = g["model"].tolist()
        gt_model_ids = [ds.model2id[m] for m in gt_model_names if m in ds.model2id]
        if len(gt_model_ids) < 2:
            continue

        gt_values = g.set_index("model").loc[[ds.id2model[mid] for mid in gt_model_ids], "value"].to_numpy()
        gt_ranks = _to_rank(gt_values, descending=True)

        # Pred scores for ALL models for this dataset, then subset to GT ids
        ds_id = _resolve_dataset_id(ds, ds_name)
        if ds_id is None:
            continue
        ds_emb = torch.tensor(ds.dataset_vecs[ds_id], dtype=torch.float, device=device).unsqueeze(0)
        
        scores_all = model.score_matrix(ds_emb, all_model_names, all_model_size_ids).detach().cpu().numpy().squeeze()
        
        pred_scores = scores_all[gt_model_ids]
        pred_ranks = _to_rank(pred_scores, descending=True)
        
        # Calculate the metrics
        wtau = scipy.stats.weightedtau(gt_ranks, pred_ranks, rank=None).correlation
        # pacc = _pair_accuracy(gt_values, pred_scores, margin_eps=margin_eps)
        ndcg = _ndcg_at_k(gt_values, pred_scores, topk)
        hitk  = _topk_hit(gt_values, pred_scores, topk)
        
        per_dataset_stats[ds_name] = {
            "num_models": int(len(g)),
            "weightedtau": None if wtau is None else float(wtau),
            # "pair_accuracy": None if (pacc is np.nan) else float(pacc),
            f"ndcg@{topk}": float(ndcg),
            f"hit@{topk}": float(hitk),
            "aligned": pd.DataFrame({
                "model": gt_model_names,
                "true": gt_values,
                "true_rank": gt_ranks,
                "pred": pred_scores,
                "pred_rank": pred_ranks,
                "rank_diff": gt_ranks - pred_ranks,     
            }).sort_values(["pred_rank", "true_rank"])
        }
        
        if wtau is not None and not np.isnan(wtau): WTAU.append(float(wtau))
        # if not np.isnan(pacc): PACC.append(float(pacc))
        if not np.isnan(ndcg): NDCG.append(float(ndcg))
        if not np.isnan(hitk): HIT.append(float(hitk))
    
    summary = {
        "datasets_evaluated": len(per_dataset_stats),
        "mean_weighted_tau": float(np.mean(WTAU)) if WTAU else float("nan"),
        # "mean_pairacc": float(np.mean(PACC)) if PACC else float("nan"),
        f"mean_ndcg@{topk}": float(np.mean(NDCG)) if NDCG else float("nan"),
        f"mean_hit@{topk}": float(np.mean(HIT)) if HIT else float("nan"),
    }
    return summary, per_dataset_stats

def _to_rank(values: np.ndarray, descending: bool = True) -> np.ndarray:
    order = np.argsort(-values) if descending else np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def _dcg(gains: np.ndarray, k: int) -> float:
    gains = gains[:k]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum((2.0**gains - 1.0) * discounts))

def _ndcg_at_k(y_true_scores: np.ndarray,
               y_pred_scores: np.ndarray,
               k: int) -> float:
    """
    Use the normalized y_true_scores as gain (can also be replaced with discrete correlation).
    """
    k = min(k, len(y_true_scores))
    if k == 0:
        return 0.0
    # Normalize the true scores to [0,1]
    mn, mx = float(y_true_scores.min()), float(y_true_scores.max())
    rel = (y_true_scores - mn) / (mx - mn + 1e-9)

    # Predicted ranking
    pred_order = np.argsort(-y_pred_scores)  # desc
    rel_pred = rel[pred_order]
    dcg = _dcg(rel_pred, k)

    # Ideal ranking
    ideal_order = np.argsort(-y_true_scores)
    rel_ideal = rel[ideal_order]
    idcg = _dcg(rel_ideal, k) + 1e-9
    return float(dcg / idcg)

def _pair_accuracy(y_true_scores: np.ndarray,
                   y_pred_scores: np.ndarray,
                   margin_eps: float = 0.02) -> float:
    """
    Pairwise comparison accuracy within the same dataset.
    margin_eps>0 can ignore pairs with very small true score difference to reduce noise (e.g. 0.005/0.01).
    """
    n = len(y_true_scores)
    if n < 2:
        return np.nan
    correct, total = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dy = y_true_scores[i] - y_true_scores[j]
            if abs(dy) <= margin_eps:
                continue
            ds = y_pred_scores[i] - y_pred_scores[j]
            total += 1
            correct += int((dy > 0 and ds > 0) or (dy < 0 and ds < 0))
    return float(correct / total) if total > 0 else np.nan

def _topk_hit(y_true_scores: np.ndarray,
                       y_pred_scores: np.ndarray,
                       k: int) -> float:
    """
    Return (Hit@K).
    """
    best_idx = int(np.argmax(y_true_scores))
    pred_top_idx = np.argsort(-y_pred_scores)[:k]
    return float(best_idx in pred_top_idx)
