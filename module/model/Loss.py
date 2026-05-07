import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import math


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _call_model_with_optional_metric(
    model: nn.Module,
    task_ids: torch.Tensor,
    ds_emb: torch.Tensor,
    model_ids: torch.Tensor,
    model_names,
    size_ids: torch.Tensor | None,
    family_ids: torch.Tensor | None,
    ms_repr: torch.Tensor | None,
    ms_interaction: torch.Tensor | None,
    metric_ids: torch.Tensor | None,
):
    model_ref = _unwrap_model(model)
    if metric_ids is not None:
        try:
            return model_ref(
                task_ids,
                ds_emb,
                model_ids,
                model_names,
                size_ids,
                family_ids,
                ms_repr,
                ms_interaction,
                metric_ids=metric_ids,
            )
        except TypeError:
            pass
    return model_ref(
        task_ids,
        ds_emb,
        model_ids,
        model_names,
        size_ids,
        family_ids,
        ms_repr,
        ms_interaction,
    )

class JointCriterion(nn.Module):
    """
    Put listwise+pointwise and pairwise+pointwise into one criterion,
    so that train_func still only calls: loss, stats = criterion(model, batch)
    
    batch structure from EnsembleLoader:
        batch = {
            "list": batch_list,
            "pair": batch_pair,
        }
    """
    def __init__(self, args, list_criterion, pair_criterion,
                 lambda_list: float = 1.0,
                 lambda_pair: float = 1.0,
                 max_grad_norm: float | None = 5.0):
        super().__init__()
        self.list_criterion = list_criterion
        self.pair_criterion = pair_criterion
        self.lambda_list = lambda_list
        self.lambda_pair = lambda_pair
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def clip_grads(self, model: nn.Module):
        # unified grad clipping, can use your own max_grad_norm,
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def forward(self, model: nn.Module, batch):
        """
        batch: dict{"list": batch_list, "pair": batch_pair}
        return: total_loss, stats (merge list / pair stats)
        """
        batch_list = batch["list"]
        batch_pair = batch["pair"]

        # 1) listwise + pointwise
        loss_list, stats_list = self.list_criterion(model, batch_list)

        # 2) pairwise + pointwise
        loss_pair, stats_pair = self.pair_criterion(model, batch_pair)

        # 3) combine total loss
        loss_total = self.lambda_list * loss_list + self.lambda_pair * loss_pair

        # 4) combine stats (for wandb / log)
        stats = {}
        if stats_list is not None:
            for k, v in stats_list.items():
                stats[f"list/{k}"] = v
        if stats_pair is not None:
            for k, v in stats_pair.items():
                stats[f"pair/{k}"] = v

        stats["loss_total"] = float(loss_total.detach().item())
        stats["loss_list"]  = float(loss_list.detach().item())
        stats["loss_pair"]  = float(loss_pair.detach().item())

        return loss_total, stats
    
class PairwisePointwiseLoss(nn.Module):
    """
    Pairwise BPR + Pointwise z-score Regression.
    """
    def __init__(
        self,
        args,
        use_sample_weight: bool = True,
        use_l2_reg: bool = False,
        max_grad_norm: float | None = 5.0,
    ):
        super().__init__()
        self.use_sample_weight = use_sample_weight
        self.use_l2_reg = use_l2_reg
        self.max_grad_norm = max_grad_norm
        
        self.point_loss_weight = getattr(args, "point_loss_weight", 1.0)
        self.lambda_mono = getattr(args, "lambda_mono", 0.0)

    @torch.no_grad()
    def clip_grads(self, model: nn.Module):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def forward(self, model: nn.Module, batch):
        """
        batch: (
            task_ids, ds_emb,
            pos_ids, pos_name, pos_size_ids, pos_family_ids, pos_z,
            neg_ids, neg_name, neg_size_ids, neg_family_ids, neg_z,
            weight
        )

        model(
          task_ids, ds_emb, model_ids, model_names, size_ids, family_ids
        ) -> (score, z_pred)
        """
        (
            task_ids,
            ds_emb,
            ms_repr,
            pos_ms_interaction,
            neg_ms_interaction,
            pos_metric_ids, pos_ids, pos_names, pos_size_ids, pos_family_ids, pos_z,
            neg_metric_ids, neg_ids, neg_names, neg_size_ids, neg_family_ids, neg_z,
            weight,
        ) = batch

        s_pos, z_pos_pred = _call_model_with_optional_metric(
            model, task_ids, ds_emb, pos_ids, pos_names, pos_size_ids, pos_family_ids, ms_repr, pos_ms_interaction, pos_metric_ids
        )  # [B], [B]
        s_neg, z_neg_pred = _call_model_with_optional_metric(
            model, task_ids, ds_emb, neg_ids, neg_names, neg_size_ids, neg_family_ids, ms_repr, neg_ms_interaction, neg_metric_ids
        )  # [B], [B]

        # =======================
        # 1) Pairwise BPR loss
        # =======================
        delta = s_pos - s_neg                  # [B]
        pair_loss = F.softplus(-delta)         # = -log σ(delta)
        if self.use_sample_weight and (weight is not None):
            pair_loss = pair_loss * weight     # [B]
        pair_loss = pair_loss.mean()           # scalar

        # =======================
        # 2) Pointwise z-score loss
        # =======================
        point_loss_pos = (z_pos_pred - pos_z) ** 2          # [B]
        point_loss_neg = (z_neg_pred - neg_z) ** 2          # [B]
        point_loss = point_loss_pos + point_loss_neg        # [B]

        if self.use_sample_weight and (weight is not None):
            point_loss = point_loss * weight
        point_loss = point_loss.mean()        # scalar

        model_ref = _unwrap_model(model)
        if self.use_l2_reg and hasattr(model_ref, "l2_reg") and model_ref.l2_reg > 0:
            l2_term = torch.tensor(0., device=pair_loss.device)
            for p in model.parameters():
                if p.requires_grad:
                    l2_term = l2_term + p.pow(2).sum()
            pair_loss = pair_loss + model_ref.l2_reg * l2_term
        
        loss = pair_loss + self.point_loss_weight * point_loss

        stats = {
            "loss_pair": float(pair_loss.detach().cpu()),
            "loss_point": float(point_loss.detach().cpu()),
            "s_pos_mean": float(s_pos.detach().mean().cpu()),
            "s_neg_mean": float(s_neg.detach().mean().cpu()),
            "margin_mean": float(delta.detach().mean().cpu()),
        }

        return loss, stats

class PairwiseBPRLoss(nn.Module):
    """
    Pairwise Bayesian Personalized Ranking Loss
    """
    def __init__(self, args, use_sample_weight: bool = True, use_l2_reg: bool = False, max_grad_norm: float | None = 5.0):
        super().__init__()
        self.use_sample_weight = use_sample_weight
        self.use_l2_reg = use_l2_reg
        self.max_grad_norm = max_grad_norm
        self.lambda_mono = args.lambda_mono if hasattr(args, "lambda_mono") else 0.0

    @torch.no_grad()
    def clip_grads(self, model: nn.Module):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def forward(self, model: nn.Module, batch):
        """
        batch: (task_id, dataset_emb, pos_model_id, pos_model_name, pos_model_size_id, pos_family_id, pos_z, neg_model_id, neg_model_name, neg_model_size_id, neg_family_id, neg_z, weight)
        model.score(dataset_emb, model_ids) -> [B]
        """
        (
            task_ids,
            ds_emb,
            ms_repr,
            pos_ms_interaction,
            neg_ms_interaction,
            pos_metric_ids, pos_ids, pos_names, pos_size_ids, pos_family_ids, pos_z,
            neg_metric_ids, neg_ids, neg_names, neg_size_ids, neg_family_ids, neg_z,
            weight,
        ) = batch

        s_pos = _call_model_with_optional_metric(
            model, task_ids, ds_emb, pos_ids, pos_names, pos_size_ids, pos_family_ids, ms_repr, pos_ms_interaction, pos_metric_ids
        )  # [B] or ([B], [B])
        s_neg = _call_model_with_optional_metric(
            model, task_ids, ds_emb, neg_ids, neg_names, neg_size_ids, neg_family_ids, ms_repr, neg_ms_interaction, neg_metric_ids
        )  # [B] or ([B], [B])
        if isinstance(s_pos, tuple):
            s_pos = s_pos[0]
        if isinstance(s_neg, tuple):
            s_neg = s_neg[0]
        delta = s_pos - s_neg

        pair_loss = F.softplus(-delta)  # = -log σ(delta)

        if self.use_sample_weight and (weight is not None):
            pair_loss = pair_loss * weight

        loss = pair_loss.mean()

        model_ref = _unwrap_model(model)
        if self.use_l2_reg and hasattr(model_ref, "l2_reg") and model_ref.l2_reg > 0:
            l2_term = torch.tensor(0., device=loss.device)
            for p in model.parameters():
                if p.requires_grad:
                    l2_term = l2_term + p.pow(2).sum()
            loss = loss + model_ref.l2_reg * l2_term

        # size_bucket_order = list(range(model.num_size_buckets))[:-1]
        # mono_penalty = 0.0
        # if self.lambda_mono > 0.0 and size_bucket_order is not None:
        #     with torch.no_grad():
        #         pass
        #     t = model._encode_dataset(ds_emb)          # [B, D]
        #     t_proj = model.W_prior(t)                 # [B, P]
        #     S = model.size_emb.weight                 # [K, P]
        #     prior_all = (t_proj @ S.T)                # [B, K]

        #     idx = torch.tensor(size_bucket_order, device=prior_all.device)  # e.g. [0,1,2,3,4,5]
        #     ordered = prior_all.index_select(dim=1, index=idx)              # [B, K]
        #     diffs = ordered[:, :-1] - ordered[:, 1:]                        # b_k - b_{k+1}
        #     mono_penalty = F.relu(diffs).mean()                              # 只惩罚“逆单调”部分

        # loss = loss + self.lambda_mono * mono_penalty
    
        stats = {
            "loss_pair": pair_loss.mean().item(),
            "s_pos_mean": s_pos.mean().item(),
            "s_neg_mean": s_neg.mean().item(),
            "margin_mean": delta.mean().item(),
        }
        return loss, stats

class ListwisePointwiseLoss(nn.Module):
    """
    List-wise pointwise loss over per-dataset candidate sets.
    """
    def __init__(
        self,
        args,
        temperature: float = 1.0,
        use_l2_reg: bool = False,
        max_grad_norm: float | None = 5.0,
    ):
        super().__init__()
        self.tau = getattr(args, "listwise_temperature", temperature)
        self.use_l2_reg = use_l2_reg
        self.max_grad_norm = max_grad_norm

        self.point_loss_weight = getattr(args, "point_loss_weight", 1.0)
        self.lambda_mono = getattr(args, "lambda_mono", 0.0)
        
    @torch.no_grad()
    def clip_grads(self, model: nn.Module):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
    def forward(self, model: nn.Module, batch):
        # --------- unpack batch ---------
        task_ids      = batch["task_ids"]       # [B]
        ds_ids        = batch["ds_ids"]         # [B]
        dataset_embs  = batch["dataset_embs"]   # [B, D_data]
        ms_repr_batch = batch.get("ms_repr_batch", None)  # [B, 10, D_ms] or None
        ms_interactions_batch = batch.get("ms_interactions_batch", None)  # [N_total, D_ms] or None
        model_ids     = batch["model_ids"]      # [N_total]
        metric_ids    = batch.get("metric_ids", None)  # [N_total] or None
        size_buckets  = batch.get("size_buckets", None)  # [N_total] or None
        family_ids    = batch.get("family_ids", None)    # [N_total] or None
        values        = batch["values"]         # [N_total]  (z-value / normalized metric)
        ds_ptr        = batch["ds_ptr"]         # [B+1]
        model_names   = batch["model_names"]    # list, len = N_total

        device = dataset_embs.device
        task_ids     = task_ids.to(device)
        ds_ids       = ds_ids.to(device)
        dataset_embs = dataset_embs.to(device)
        model_ids    = model_ids.to(device)
        if metric_ids is not None:
            metric_ids = metric_ids.to(device)
        values       = values.to(device)
        ds_ptr       = ds_ptr.to(device)
        if size_buckets is not None:
            size_buckets = size_buckets.to(device)
        if family_ids is not None:
            family_ids = family_ids.to(device)
        if ms_repr_batch is not None:
            ms_repr_batch = ms_repr_batch.to(device)
        if ms_interactions_batch is not None:
            ms_interactions_batch = ms_interactions_batch.to(device)

        B = dataset_embs.size(0)       # num datasets in this batch
        N_total = model_ids.size(0)    # total (d,m) pairs in this batch

        # === construct dataset embedding for each (task,dataset,model) pair ===
        ds_index_for_each_model = torch.empty(N_total, dtype=torch.long, device=device)
        for b in range(B):
            start = ds_ptr[b].item()
            end   = ds_ptr[b + 1].item()
            ds_index_for_each_model[start:end] = b  # [N_total]

        dataset_emb_per_model = dataset_embs[ds_index_for_each_model]  # [N_total, D_data]
        task_ids_per_model    = task_ids[ds_index_for_each_model]      # [N_total]
        if ms_repr_batch is not None:
            ms_repr_per_model = ms_repr_batch[ds_index_for_each_model]  # [N_total, 10, D_ms]
        else:
            ms_repr_per_model = None

        scores, z_pred = _call_model_with_optional_metric(
            model,
            task_ids_per_model,        # [N_total]
            dataset_emb_per_model,     # [N_total, D_data]
            model_ids,                 # [N_total]
            model_names,               # list[str]
            size_buckets,              # [N_total] or None
            family_ids,                # [N_total] or None
            ms_repr_per_model,         # [N_total, 10, D_ms] or None
            ms_interactions_batch,     # [N_total, D_ms] or None
            metric_ids,                # [N_total] or None
        )  # [N_total], [N_total]

        scores = scores / self.tau

        # --------- List-wise ranking loss per dataset ---------
        loss_listwise = scores.new_zeros(())
        valid_groups = 0

        for b in range(B):
            start = ds_ptr[b].item()
            end = ds_ptr[b + 1].item()
            M = end - start
            if M <= 1:
                continue

            s = scores[start:end]                         # [M]
            rev_s = s.flip(0)
            log_cumsum_rev = torch.logcumsumexp(rev_s, dim=0)
            log_den = log_cumsum_rev.flip(0)              # [M]

            per_pos_loss = log_den - s                    # [M]
            # loss_d = per_pos_loss.sum()
            loss_d = per_pos_loss.mean() 

            loss_listwise = loss_listwise + loss_d
            valid_groups += 1

        loss_listwise = loss_listwise / max(valid_groups, 1)

        # --------- Pointwise: Regression z-value ---------
        loss_point = F.mse_loss(z_pred, values, reduction="mean")

        loss_total = (
            loss_listwise +
            self.point_loss_weight * loss_point
        )
        
        model_ref = _unwrap_model(model)
        if self.use_l2_reg and hasattr(model_ref, "l2_reg") and model_ref.l2_reg > 0:
            l2_term = torch.tensor(0.0, device=device)
            for p in model.parameters():
                if p.requires_grad:
                    l2_term = l2_term + p.pow(2).sum()
            loss_total = loss_total + model_ref.l2_reg * l2_term

        # --------- logging stats ---------
        stats = {
            "loss_list":  float(loss_listwise.detach().item()),
            "loss_point": float(loss_point.detach().item()),
            "loss_total": float(loss_total.detach().item()),
            "score_mean": float(scores.mean().item()),
            "score_std":  float(scores.std(unbiased=False).item()),
            "num_datasets_in_batch": int(B),
            "avg_models_per_dataset": float(N_total) / max(B, 1),
        }
        
        return loss_total, stats

class ListwiseRankLoss(nn.Module):
    """
    List-wise ranking loss over per-dataset candidate sets.

    For each dataset d, given a set of model predictions {s_m},
    calculate:
        L(d) = sum_m -log( exp(s_m / tau) / sum_l exp(s_l / tau) )
        point_loss_pos = (z_pos_pred - pos_z) ** 2          # [B]
        point_loss_neg = (z_neg_pred - neg_z) ** 2          # [B]
        point_loss = point_loss_pos + point_loss_neg        # [B]
        if self.use_sample_weight and (weight is not None):
            point_loss = point_loss * weight
        point_loss = point_loss.mean()        # scalar
        return point_loss
             = - sum_m log_softmax(s / tau)_m
    finally average over all datasets in the batch
    """
    def __init__(
        self,
        args,
        temperature: float = 1.0,
        use_l2_reg: bool = False,
        max_grad_norm: float | None = 5.0,
    ):
        super().__init__()
        self.tau = getattr(args, "listwise_temperature", temperature)
        self.use_l2_reg = use_l2_reg
        self.max_grad_norm = max_grad_norm
        self.lambda_mono = getattr(args, "lambda_mono", 0.0)

    @torch.no_grad()
    def clip_grads(self, model: nn.Module):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def forward(self, model: nn.Module, batch):
        task_ids      = batch["task_ids"]       # [B]
        ds_ids        = batch["ds_ids"]           # [B]
        dataset_embs  = batch["dataset_embs"]     # [B, D_data]
        ms_repr_batch = batch.get("ms_repr_batch", None)  # [B, 10, D_ms] or None
        ms_interactions_batch = batch.get("ms_interactions_batch", None)  # [N_total, D_ms] or None
        model_ids     = batch["model_ids"]        # [N_total]
        metric_ids    = batch.get("metric_ids", None)  # [N_total] or None
        size_buckets  = batch.get("size_buckets", None)  # [N_total] or None
        family_ids    = batch.get("family_ids", None)    # [N_total] or None
        values        = batch["values"]                 # [N_total]
        ds_ptr        = batch["ds_ptr"]           # [B+1]
        model_names   = batch["model_names"]      # [N_total]

        device = dataset_embs.device
        task_ids     = task_ids.to(device)
        dataset_embs = dataset_embs.to(device)
        model_ids = model_ids.to(device)
        if metric_ids is not None:
            metric_ids = metric_ids.to(device)
        values       = values.to(device)
        ds_ptr = ds_ptr.to(device)
        if size_buckets is not None:
            size_buckets = size_buckets.to(device)
        if family_ids is not None:
            family_ids = family_ids.to(device)
        if ms_repr_batch is not None:
            ms_repr_batch = ms_repr_batch.to(device)
        if ms_interactions_batch is not None:
            ms_interactions_batch = ms_interactions_batch.to(device)

        B = dataset_embs.size(0)       # num datasets in this batch
        N_total = model_ids.size(0)    # total (d,m) pairs in this batch

        # === construct dataset embedding for each (d,m) pair ===
        ds_index_for_each_model = torch.empty(
            N_total, dtype=torch.long, device=device
        )
        for b in range(B):
            start = ds_ptr[b].item()
            end = ds_ptr[b + 1].item()
            ds_index_for_each_model[start:end] = b

        dataset_emb_per_model = dataset_embs[ds_index_for_each_model]   # [N_total, D_data]
        task_ids_per_model    = task_ids[ds_index_for_each_model]      # [N_total]
        if ms_repr_batch is not None:
            ms_repr_per_model = ms_repr_batch[ds_index_for_each_model]  # [N_total, 10, D_ms]
        else:
            ms_repr_per_model = None

        # === call model to score ===
        scores, _ = _call_model_with_optional_metric(
            model,
            task_ids_per_model,        # [N_total]
            dataset_emb_per_model,     # [N_total, D_data]
            model_ids,                 # [N_total]
            model_names,               # list[str]
            size_buckets,              # [N_total] or None
            family_ids,                # [N_total] or None
            ms_repr_per_model,         # [N_total, 10, D_ms] or None
            ms_interactions_batch,     # [N_total, D_ms] or None
            metric_ids,                # [N_total] or None
        )  # [N_total], [N_total] - 只使用 scores，忽略 z_pred

        scores = scores / self.tau
            
        # === List-wise ranking loss per dataset ===
        loss_listwise = scores.new_zeros(())
        valid_groups = 0
        
        for b in range(B):
            start = ds_ptr[b].item()
            end = ds_ptr[b + 1].item()
            M = end - start
            if M <= 1:
                continue
            
            s = scores[start:end]
            rev_s = s.flip(0)                                   # [M]
            log_cumsum_rev = torch.logcumsumexp(rev_s, dim=0)   
            log_den = log_cumsum_rev.flip(0)             

            # per-position loss: log_den_m - s_m
            per_pos_loss = log_den - s                          
            loss_d = per_pos_loss.sum()

            loss_listwise = loss_listwise + loss_d
            valid_groups += 1

        # average over all datasets in the batch
        loss_listwise = loss_listwise / max(valid_groups, 1)

        # === optional L2 regularization (aligned with PairwiseBPRLoss) ===
        model_ref = _unwrap_model(model)
        if self.use_l2_reg and hasattr(model_ref, "l2_reg") and model_ref.l2_reg > 0:
            l2_term = torch.tensor(0.0, device=device)
            for p in model.parameters():
                if p.requires_grad:
                    l2_term = l2_term + p.pow(2).sum()
            loss_listwise = loss_listwise + model_ref.l2_reg * l2_term

        # === optional size prior monotonicity regularization (aligned with PairwiseBPRLoss) ===
        mono_penalty = 0.0
        if self.lambda_mono > 0.0 and getattr(model_ref, "num_size_buckets", 0) > 0:
            size_bucket_order = list(range(model_ref.num_size_buckets))[:-1]
            if len(size_bucket_order) > 1:
                with torch.no_grad():
                    pass
                t = model_ref._encode_dataset(dataset_embs)   # [B, D_hidden]
                t_proj = model_ref.W_prior(t)                 # [B, P]
                S = model_ref.size_emb.weight                 # [K, P]
                prior_all = (t_proj @ S.T)                # [B, K]

                idx = torch.tensor(size_bucket_order, device=device)
                ordered = prior_all.index_select(dim=1, index=idx)  # [B, K-1 or K]
                diffs = ordered[:, :-1] - ordered[:, 1:]            # b_k - b_{k+1}
                mono_penalty = F.relu(diffs).mean()

                loss_listwise = loss_listwise + self.lambda_mono * mono_penalty

        # === logging stats ===
        stats = {
            "loss_list": loss_listwise.detach().item(),
            "score_mean": scores.mean().item(),
            "score_std": scores.std(unbiased=False).item(),
            "num_datasets_in_batch": int(B),
            "avg_models_per_dataset": float(N_total) / max(B, 1),
        }

        return loss_listwise, stats
