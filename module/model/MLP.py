import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re
import numpy as np
from types import SimpleNamespace
from module.model.modelNameEncoder import ModelNameAvgEncoder
from module.model.registry import register_model

def batch_logq(size_id: torch.LongTensor, n_size_bucket: int, eps: float = 1e-6):
    """
    size_id: [B]  the size-bucket id (0..n_size_bucket-1) for each sample
    Here we estimate q_s by batch; if you want to estimate q_s by each task, you can change it to count within each task.
    """
    B = size_id.numel()
    # count the number of each bucket in the batch
    counts = torch.bincount(size_id, minlength=n_size_bucket).float()  # [S]
    probs  = counts / max(B, 1)
    logq_all = torch.log(probs + eps)                                  # [S]
    return logq_all.gather(0, size_id)                                  # [B]

# ---- module-level name-alias / desc-matrix helpers ---------------------------
# Extracted from internal ranker variants so ModelLens can stand alone.

def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _name_aliases(name: str) -> list[str]:
    n = _normalize_name(name)
    cands = [
        n,
        n.replace("-", ""),
        n.replace("_", ""),
        n.replace(" ", ""),
        re.sub(r"[^a-z0-9]+", "", n),
    ]
    if "/" in n:
        base = n.split("/")[-1]
        cands.extend([
            base,
            base.replace("-", ""),
            base.replace("_", ""),
            re.sub(r"[^a-z0-9]+", "", base),
        ])
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _load_model2id(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_desc_matrix(model2id: dict, emb_path: str, emb_dim: int) -> torch.Tensor:
    num_models = max(model2id.values()) + 1 if model2id else 0
    mat = np.zeros((num_models, emb_dim), dtype=np.float32)
    if num_models == 0:
        return torch.zeros(0, emb_dim, dtype=torch.float32)
    if not os.path.exists(emb_path):
        print(f"[ModelLens] warning: description embedding file not found: {emb_path}; using zeros.")
        return torch.from_numpy(mat)

    payload = np.load(emb_path, allow_pickle=True)
    if "model_names" not in payload or "embeddings" not in payload:
        return torch.from_numpy(mat)
    names = payload["model_names"]
    embs = payload["embeddings"]
    if embs.ndim != 2:
        return torch.from_numpy(mat)
    use_dim = min(emb_dim, int(embs.shape[1]))

    lookup = {}
    for i, raw_name in enumerate(names.tolist()):
        n = str(raw_name)
        for a in _name_aliases(n):
            if a not in lookup:
                lookup[a] = i

    for model_name, mid in model2id.items():
        idx = None
        for a in _name_aliases(model_name):
            if a in lookup:
                idx = lookup[a]
                break
        if idx is None:
            continue
        mat[int(mid), :use_dim] = embs[idx, :use_dim]
    return torch.from_numpy(mat)


# Internal base class — not exposed in the public model registry.
class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.use_id_emb:
            self.model_embedding = nn.Embedding(args.num_models, args.model_dim)
        else:
            self.model_embedding = None

        self.task_embedding = nn.Embedding(args.num_tasks, args.task_dim)
        self.model_info_encoder = ModelNameAvgEncoder(args)
        self.size_embedding = nn.Embedding(args.num_size_buckets, args.size_dim)
        self.num_size_buckets = args.num_size_buckets
        self.use_size_prior = args.use_size_prior

        # Family prior support
        self.use_family_prior = getattr(args, "use_family_prior", False)
        if self.use_family_prior:
            self.num_families = args.num_families
            family_dim = getattr(args, "family_dim", args.size_dim)  # default to same dim as size
            self.family_embedding = nn.Embedding(args.num_families, family_dim)
            self.family_dim = family_dim
        else:
            self.family_dim = 0

        model_info_dim = (
            args.token_dim
            + (args.model_dim if args.use_id_emb else 0)
        )
        self.model_info_dim = model_info_dim
        dataset_info_dim = (
            args.dataset_desp_dim
            + args.task_dim
        )
        self.use_ms_spider_repr = getattr(args, "use_ms_spider_repr", False)
        self.ms_fusion_dim = int(getattr(args, "ms_fusion_dim", 128))
        if self.use_ms_spider_repr:
            self.ms_repr_dim = int(getattr(args, "ms_repr_dim", 1024))
            self.ms_num_models = int(getattr(args, "ms_num_models", 10))
            self.ms_query = nn.Linear(model_info_dim, self.ms_fusion_dim)
            self.ms_key = nn.Linear(self.ms_repr_dim, self.ms_fusion_dim)
            self.ms_val = nn.Linear(self.ms_repr_dim, self.ms_fusion_dim)
            self.ms_out = nn.Sequential(
                nn.Linear(self.ms_fusion_dim, self.ms_fusion_dim),
                nn.ReLU(),
            )
        else:
            self.ms_repr_dim = 0
            self.ms_num_models = 0
            self.ms_fusion_dim = 0
        # Include family_dim in backbone input if family prior is used
        backbone_in_dim = model_info_dim + dataset_info_dim + args.size_dim + self.family_dim + self.ms_fusion_dim
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
        )
        # -------- Pairwise Projector：residual score --------
        self.pairwise_head = nn.Linear(args.hidden_dim, 1)
        # -------- Pointwise Projector：z-score --------
        self.pointwise_head = nn.Linear(args.hidden_dim, 1)

        # Prior head: takes size_emb (+ family_emb if enabled) and outputs prior score
        prior_in_dim = args.size_dim + self.family_dim
        self.prior_head = nn.Sequential(
            nn.Linear(prior_in_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_dim // 2, 1)
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def encode_model(self, model_ids: torch.LongTensor, model_names: list[str]) -> torch.Tensor:
        h_model = self.model_info_encoder(model_ids, model_names)
        return h_model

    def _compute_ms_feature(
        self,
        h_model: torch.Tensor,
        ms_repr: torch.Tensor | None,
        ms_interaction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        h_model: [B, model_info_dim]
        ms_repr: [B, K, ms_repr_dim]
        return:  [B, ms_fusion_dim]
        """
        if not self.use_ms_spider_repr:
            return h_model.new_zeros(h_model.shape[0], self.ms_fusion_dim)

        if ms_interaction is not None:
            # Direct (dataset,model)-specific interaction embedding from Model-Spider.
            return self.ms_out(self.ms_val(ms_interaction))

        if ms_repr is None:
            return h_model.new_zeros(h_model.shape[0], self.ms_fusion_dim)

        q = self.ms_query(h_model)                             # [B, F]
        k = self.ms_key(ms_repr)                               # [B, K, F]
        v = self.ms_val(ms_repr)                               # [B, K, F]
        logits = torch.einsum("bf,bkf->bk", q, k) / (self.ms_fusion_dim ** 0.5)
        attn = torch.softmax(logits, dim=-1)                   # [B, K]
        ctx = torch.einsum("bk,bkf->bf", attn, v)              # [B, F]
        return self.ms_out(ctx)

    def forward(
        self,
        task_ids: torch.LongTensor,            # [B]
        dataset_desp: torch.Tensor,            # [B, dataset_desp_dim]
        model_ids: torch.LongTensor,           # [B]
        model_names: list[str],                # len=B
        size_id: torch.LongTensor,             # [B]  (bucket id)
        family_id: torch.LongTensor = None,    # [B]  (family id, optional)
        ms_repr: torch.Tensor = None,          # [B, K, ms_repr_dim] (optional)
        ms_interaction: torch.Tensor = None,   # [B, ms_repr_dim] (optional)
    ):
        B = task_ids.size(0)
        device = task_ids.device

        # --- encode ---
        h_model = self.encode_model(model_ids, model_names)                    # [B, Mdim]
        h_data  = dataset_desp                                                 # [B, Ddim]
        h_task  = self.task_embedding(task_ids)                                # [B, Tdim]
        h_size  = self.size_embedding(size_id)                                 # [B, Sdim]
        if self.use_ms_spider_repr:
            h_ms = self._compute_ms_feature(h_model, ms_repr, ms_interaction)
        else:
            h_ms = h_model.new_zeros(B, self.ms_fusion_dim)

        # --- family embedding ---
        if self.use_family_prior and family_id is not None:
            h_family = self.family_embedding(family_id)                        # [B, Fdim]
        else:
            h_family = None

        # --- residual logit s(m,t) ---
        if h_family is not None:
            residual_inp = torch.cat([h_model, h_data, h_size, h_family, h_task, h_ms], dim=-1)
        else:
            residual_inp = torch.cat([h_model, h_data, h_size, h_task, h_ms], dim=-1)
        h = self.backbone(residual_inp)
        s_residual = self.pairwise_head(h).squeeze(-1)                         # [B]
        z_pred = self.pointwise_head(h).squeeze(-1)                            # [B]

        # --- size + family aware prior ---
        if self.use_size_prior or self.use_family_prior:
            # Combine size and family embeddings for prior computation
            if h_family is not None:
                prior_inp = torch.cat([h_size, h_family], dim=-1)              # [B, Sdim + Fdim]
            else:
                prior_inp = h_size                                              # [B, Sdim]
            s_prior = self.prior_head(prior_inp).squeeze(-1)                   # [B]
        else:
            s_prior = torch.zeros(B, device=device)

        # --- combine final logit (with temperature) ---
        tilde_s = (s_residual + s_prior) / torch.clamp(self.temperature, min=1e-3)

        return tilde_s, z_pred
    
    @torch.no_grad()
    def build_model_cache(
        self,
        all_model_names: list[str],
        all_model_size_ids: torch.LongTensor,
        all_model_family_ids: torch.LongTensor = None,
        device=None,
    ):
        """
        Pre-encode the model-side fixed features for reuse in score_matrix.
        """
        if device is None:
            device = next(self.parameters()).device

        size_ids = all_model_size_ids.to(device=device, dtype=torch.long)  # [M]
        M = len(all_model_names)
        assert size_ids.shape[0] == M, "all_model_names and all_model_size_ids must have the same length"

        model_ids = torch.arange(M, device=device, dtype=torch.long)

        # encode model + size
        h_model = self.encode_model(model_ids, all_model_names)   # [M, Mdim]
        h_size  = self.size_embedding(size_ids)                   # [M, Sdim]

        cache = {
            "h_model": h_model,   # [M, Mdim]
            "h_size":  h_size,    # [M, Sdim]
            "size_ids": size_ids, # [M]
        }

        # Family embedding cache
        if self.use_family_prior and all_model_family_ids is not None:
            family_ids = all_model_family_ids.to(device=device, dtype=torch.long)  # [M]
            h_family = self.family_embedding(family_ids)           # [M, Fdim]
            cache["h_family"] = h_family
            cache["family_ids"] = family_ids
        else:
            cache["h_family"] = None
            cache["family_ids"] = None

        return cache
    
    @torch.no_grad()
    def score_matrix(
        self,
        task_ids: torch.LongTensor,          # [B]
        dataset_desp_batch: torch.Tensor,    # [B, Ddim]
        model_cache: dict,
        ms_repr_batch: torch.Tensor = None,  # [B, K, ms_repr_dim] (optional)
        chunk_size: int = 8192,
    ):
        device = dataset_desp_batch.device
        B = dataset_desp_batch.size(0)

        h_task = self.task_embedding(task_ids)                 # [B, Tdim]
        h_data = dataset_desp_batch                            # [B, Ddim]

        h_model_all = model_cache["h_model"]                   # [M, Mdim]
        h_size_all  = model_cache["h_size"]                    # [M, Sdim]
        h_family_all = model_cache.get("h_family")             # [M, Fdim] or None

        M = h_model_all.size(0)

        # Compute prior for all models
        if self.use_size_prior or self.use_family_prior:
            if h_family_all is not None:
                prior_inp_all = torch.cat([h_size_all, h_family_all], dim=-1)  # [M, Sdim + Fdim]
            else:
                prior_inp_all = h_size_all                                      # [M, Sdim]
            prior_all = self.prior_head(prior_inp_all).squeeze(-1)             # [M]
        else:
            prior_all = torch.zeros(M, device=device)

        out = torch.empty(B, M, device=device)
        T = torch.clamp(self.temperature, min=1e-3)

        start = 0
        while start < M:
            end = min(start + chunk_size, M)
            h_model = h_model_all[start:end]                   # [m, Mdim]
            h_size  = h_size_all[start:end]                    # [m, Sdim]
            m = end - start

            # expand
            h_model_exp = h_model.unsqueeze(0).expand(B, m, -1)   # [B, m, Mdim]
            h_size_exp  = h_size.unsqueeze(0).expand(B, m, -1)    # [B, m, Sdim]
            h_data_exp  = h_data.unsqueeze(1).expand(B, m, -1)    # [B, m, Ddim]
            h_task_exp  = h_task.unsqueeze(1).expand(B, m, -1)    # [B, m, Tdim]
            if self.use_ms_spider_repr and ms_repr_batch is not None:
                q = self.ms_query(h_model).unsqueeze(0).expand(B, m, -1)            # [B, m, F]
                k = self.ms_key(ms_repr_batch)                                       # [B, K, F]
                v = self.ms_val(ms_repr_batch)                                       # [B, K, F]
                attn = torch.softmax(torch.einsum("bmf,bkf->bmk", q, k) / (self.ms_fusion_dim ** 0.5), dim=-1)
                h_ms = torch.einsum("bmk,bkf->bmf", attn, v)                         # [B, m, F]
                h_ms = self.ms_out(h_ms.reshape(B * m, -1)).reshape(B, m, -1)
            else:
                h_ms = h_model_exp.new_zeros(B, m, self.ms_fusion_dim)

            # Include family in backbone if enabled
            if h_family_all is not None:
                h_family = h_family_all[start:end]                 # [m, Fdim]
                h_family_exp = h_family.unsqueeze(0).expand(B, m, -1)  # [B, m, Fdim]
                residual_inp = torch.cat(
                    [h_model_exp, h_data_exp, h_size_exp, h_family_exp, h_task_exp, h_ms],
                    dim=-1
                )   # [B, m, in_dim]
            else:
                residual_inp = torch.cat(
                    [h_model_exp, h_data_exp, h_size_exp, h_task_exp, h_ms],
                    dim=-1
                )   # [B, m, in_dim]

            h = self.backbone(residual_inp.reshape(B*m, -1))        # [B*m, H]
            s_chunk = self.pairwise_head(h).reshape(B, m)           # [B, m]

            prior_chunk = prior_all[start:end].unsqueeze(0)         # [1, m]

            out[:, start:end] = (s_chunk + prior_chunk) / T

            start = end

        return out

    @torch.no_grad()
    def rank_models_for_dataset(
        self,
        task_name: str,
        dataset_description: torch.Tensor,
        all_model_names: list[str],
        all_model_size_ids: torch.LongTensor,
        task2id: dict,
        model2id: dict,
        all_model_family_ids: torch.LongTensor = None,
    ):
        """
        Rank all models for a single dataset with a given task.

        Args:
            task_name: str, the task name (e.g., "question answering")
            dataset_description: [D_data], dataset description embedding
            all_model_names: list of all model names
            all_model_size_ids: [M], size bucket id for each model
            task2id: dict mapping task name to task id
            model2id: dict mapping model name to model id

        Returns:
            scores: [M], scores for all models
        """
        device = next(self.parameters()).device

        # Get task id
        task_id = task2id.get(task_name, 0)  # default to 0 if not found
        task_ids = torch.tensor([task_id], dtype=torch.long).to(device)

        # Expand dataset description to [1, D_data]
        dataset_desp_batch = dataset_description.unsqueeze(0).to(device)

        # Build model cache
        model_cache = self.build_model_cache(
            all_model_names,
            all_model_size_ids,
            all_model_family_ids=all_model_family_ids,
            device=device,
        )

        # Get score matrix [1, M]
        scores = self.score_matrix(task_ids, dataset_desp_batch, model_cache)

        # Return flattened scores [M]
        return scores.squeeze(0)


# Internal base class — not exposed in the public model registry.
class MLPMetric(MLP):
    """
    MLP variant with metric embedding lookup.
    """
    def __init__(self, args):
        super().__init__(args)
        self.use_metric_embedding = bool(getattr(args, "use_metric_feature", True))
        self.num_metrics = int(getattr(args, "num_metrics", 1))
        self.metric_dim = int(getattr(args, "metric_dim", args.task_dim))
        self.unknown_metric_id = int(getattr(args, "unknown_metric_id", 0))

        if self.use_metric_embedding:
            self.metric_embedding = nn.Embedding(max(self.num_metrics, 1), self.metric_dim)
            in_features = self.backbone[0].in_features + self.metric_dim
            hidden = self.backbone[0].out_features
            dropout = self.backbone[2].p
            self.backbone = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.metric_embedding = None

    def _metric_embed(self, metric_ids: torch.LongTensor | None, batch_size: int, device) -> torch.Tensor | None:
        if not self.use_metric_embedding or self.metric_embedding is None:
            return None
        if metric_ids is None:
            metric_ids = torch.full(
                (batch_size,),
                int(self.unknown_metric_id),
                dtype=torch.long,
                device=device,
            )
        return self.metric_embedding(metric_ids)

    def forward(
        self,
        task_ids: torch.LongTensor,
        dataset_desp: torch.Tensor,
        model_ids: torch.LongTensor,
        model_names: list[str],
        size_id: torch.LongTensor,
        family_id: torch.LongTensor = None,
        ms_repr: torch.Tensor = None,
        ms_interaction: torch.Tensor = None,
        metric_ids: torch.LongTensor = None,
    ):
        B = task_ids.size(0)
        device = task_ids.device

        h_model = self.encode_model(model_ids, model_names)
        h_data = dataset_desp
        h_task = self.task_embedding(task_ids)
        h_size = self.size_embedding(size_id)
        h_metric = self._metric_embed(metric_ids, B, device)
        if self.use_ms_spider_repr:
            h_ms = self._compute_ms_feature(h_model, ms_repr, ms_interaction)
        else:
            h_ms = h_model.new_zeros(B, self.ms_fusion_dim)

        if self.use_family_prior and family_id is not None:
            h_family = self.family_embedding(family_id)
        else:
            h_family = None

        parts = [h_model, h_data, h_size, h_task, h_ms]
        if h_family is not None:
            parts.insert(3, h_family)
        if h_metric is not None:
            parts.append(h_metric)
        residual_inp = torch.cat(parts, dim=-1)
        h = self.backbone(residual_inp)
        s_residual = self.pairwise_head(h).squeeze(-1)
        z_pred = self.pointwise_head(h).squeeze(-1)

        if self.use_size_prior or self.use_family_prior:
            if h_family is not None:
                prior_inp = torch.cat([h_size, h_family], dim=-1)
            else:
                prior_inp = h_size
            s_prior = self.prior_head(prior_inp).squeeze(-1)
        else:
            s_prior = torch.zeros(B, device=device)

        tilde_s = (s_residual + s_prior) / torch.clamp(self.temperature, min=1e-3)
        return tilde_s, z_pred

    @torch.no_grad()
    def score_matrix(
        self,
        task_ids: torch.LongTensor,
        dataset_desp_batch: torch.Tensor,
        model_cache: dict,
        ms_repr_batch: torch.Tensor = None,
        chunk_size: int = 8192,
        metric_ids: torch.LongTensor = None,
    ):
        device = dataset_desp_batch.device
        B = dataset_desp_batch.size(0)

        h_task = self.task_embedding(task_ids)
        h_data = dataset_desp_batch
        h_metric = self._metric_embed(metric_ids, B, device)

        h_model_all = model_cache["h_model"]
        h_size_all = model_cache["h_size"]
        h_family_all = model_cache.get("h_family")
        M = h_model_all.size(0)

        if self.use_size_prior or self.use_family_prior:
            if h_family_all is not None:
                prior_inp_all = torch.cat([h_size_all, h_family_all], dim=-1)
            else:
                prior_inp_all = h_size_all
            prior_all = self.prior_head(prior_inp_all).squeeze(-1)
        else:
            prior_all = torch.zeros(M, device=device)

        out = torch.empty(B, M, device=device)
        T = torch.clamp(self.temperature, min=1e-3)

        start = 0
        while start < M:
            end = min(start + chunk_size, M)
            h_model = h_model_all[start:end]
            h_size = h_size_all[start:end]
            m = end - start

            h_model_exp = h_model.unsqueeze(0).expand(B, m, -1)
            h_size_exp = h_size.unsqueeze(0).expand(B, m, -1)
            h_data_exp = h_data.unsqueeze(1).expand(B, m, -1)
            h_task_exp = h_task.unsqueeze(1).expand(B, m, -1)
            if h_metric is not None:
                h_metric_exp = h_metric.unsqueeze(1).expand(B, m, -1)
            else:
                h_metric_exp = None
            if self.use_ms_spider_repr and ms_repr_batch is not None:
                q = self.ms_query(h_model).unsqueeze(0).expand(B, m, -1)
                k = self.ms_key(ms_repr_batch)
                v = self.ms_val(ms_repr_batch)
                attn = torch.softmax(torch.einsum("bmf,bkf->bmk", q, k) / (self.ms_fusion_dim ** 0.5), dim=-1)
                h_ms = torch.einsum("bmk,bkf->bmf", attn, v)
                h_ms = self.ms_out(h_ms.reshape(B * m, -1)).reshape(B, m, -1)
            else:
                h_ms = h_model_exp.new_zeros(B, m, self.ms_fusion_dim)

            parts = [h_model_exp, h_data_exp, h_size_exp]
            if h_family_all is not None:
                h_family = h_family_all[start:end]
                h_family_exp = h_family.unsqueeze(0).expand(B, m, -1)
                parts.append(h_family_exp)
            parts.extend([h_task_exp, h_ms])
            if h_metric_exp is not None:
                parts.append(h_metric_exp)
            residual_inp = torch.cat(parts, dim=-1)

            h = self.backbone(residual_inp.reshape(B * m, -1))
            s_chunk = self.pairwise_head(h).reshape(B, m)
            prior_chunk = prior_all[start:end].unsqueeze(0)
            out[:, start:end] = (s_chunk + prior_chunk) / T
            start = end

        return out




@register_model("ModelLens", aliases=["modellens", "MLPMetricFull"])
class ModelLens(MLPMetric):
    """
    Full-feature model using ALL available features.

    Model features (residual backbone):
      - model_id:   learned embedding              [model_dim]
      - model_name: hashed-token average            [token_dim]
      - model_desc: pre-computed description emb    [model_desp_emb_dim]  (frozen buffer)

    Dataset features (residual backbone):
      - dataset_id:   learned embedding             [dataset_id_emb_dim]
      - dataset_desc: pre-computed description emb  [dataset_desp_emb_dim] (frozen buffer)

    Fixed features:
      - task_id:      learned embedding  [task_dim]
      - metric_id:    learned embedding  [metric_dim]
      - size_prior:   learned embedding  [size_dim]     (prior head)
      - family_prior: learned embedding  [family_dim]   (prior head)

    Architecture:
      final_score = (s_residual + s_prior) / temperature
      - s_prior:    MLP(size_emb || family_emb)
      - s_residual: backbone(model_feats || dataset_feats || task || size || family || metric)

    Requires ``use_dataset_id_as_desp: True`` in config so that the data
    pipeline passes global dataset IDs (as floats) in the ``dataset_desp`` slot.
    The model intercepts these IDs to look up both a learned dataset-ID
    embedding and a frozen dataset-description embedding.
    """

    def __init__(self, args):
        # ---- dimension bookkeeping ----
        self.dataset_id_emb_dim = int(getattr(args, "dataset_id_emb_dim", 256))
        self.dataset_desp_emb_dim = int(getattr(args, "dataset_desp_emb_dim", 1536))
        self.model_desp_emb_dim = int(getattr(args, "model_desp_emb_dim", 1536))

        # ID dropout: per-step random masking so the model generalizes to unseen IDs
        self.model_id_dropout_rate = float(getattr(args, "model_id_dropout_rate",
                                                   getattr(args, "id_dropout_rate", 0.1)))
        self.dataset_id_dropout_rate = float(getattr(args, "dataset_id_dropout_rate",
                                                     getattr(args, "id_dropout_rate", 0.1)))

        # Size-feature ablation flag. When False, the size embedding is
        # removed from BOTH the residual backbone and the prior head input.
        # Cascades: a size prior cannot exist without the size feature.
        self.use_size_feature = bool(getattr(args, "use_size_feature", True))
        if not self.use_size_feature:
            args.use_size_prior = False  # ensure parent doesn't think size_prior is on

        # Information-source ablation flags. Each gates one feature stream so
        # the same ModelLens architecture can express Semantic-only,
        # Interaction-only, Structural-only, and combination variants.
        self.use_model_id_emb   = bool(getattr(args, "use_model_id_emb",   True))
        self.use_model_name_emb = bool(getattr(args, "use_model_name_emb", True))
        self.use_model_desc_emb = bool(getattr(args, "use_model_desc_emb", True))
        self.use_dataset_id_emb   = bool(getattr(args, "use_dataset_id_emb",   True))
        self.use_dataset_desc_emb = bool(getattr(args, "use_dataset_desc_emb", True))

        # Tell the parent that the dataset side is (dataset_id_emb + dataset_desc_emb)
        # so the parent's backbone placeholder is sized correctly (we rebuild it anyway).
        orig_desp_dim = args.dataset_desp_dim
        args.dataset_desp_dim = self.dataset_id_emb_dim + self.dataset_desp_emb_dim
        super().__init__(args)
        args.dataset_desp_dim = orig_desp_dim  # restore to avoid side-effects

        # ==== Model-side components (clean separation, gated by flags) ====

        if self.use_model_name_emb:
            # Name encoder without model-ID mixed in
            args_name_only = SimpleNamespace(**vars(args))
            args_name_only.use_id_emb = False
            self._name_encoder = ModelNameAvgEncoder(args_name_only)
        else:
            self._name_encoder = None

        if self.use_model_id_emb:
            # Standalone model-ID embedding (+1 slot for [UNK])
            self._id_emb = nn.Embedding(args.num_models + 1, args.model_dim)
            self.unk_model_id = args.num_models
        else:
            self._id_emb = None
            self.unk_model_id = 0

        if self.use_model_desc_emb:
            # Pre-computed model-description embedding (frozen buffer)
            data_name = getattr(args, "data_name", "unified")
            model_emb_path = str(getattr(
                args, "model_desp_emb_path",
                os.path.join("data", data_name, "model2desp_embeddings.npz"),
            ))
            model2id_path = str(getattr(
                args, "model2id_path",
                os.path.join("data", data_name, "model2id.json"),
            ))
            model2id_map = _load_model2id(model2id_path)
            model_desc_matrix = _build_desc_matrix(
                model2id_map, model_emb_path, self.model_desp_emb_dim,
            )
            self.register_buffer("model_desc_matrix", model_desc_matrix)
        else:
            self.register_buffer("model_desc_matrix", torch.zeros(0, self.model_desp_emb_dim))

        # ==== Dataset-side components (gated by flags) ====

        num_datasets = int(getattr(args, "num_datasets", 100000))
        if self.use_dataset_id_emb:
            # Learned dataset-ID embedding (+1 slot for [UNK])
            self.dataset_id_embedding = nn.Embedding(num_datasets + 2, self.dataset_id_emb_dim)
            self.unk_dataset_id = num_datasets + 1
        else:
            self.dataset_id_embedding = None
            self.unk_dataset_id = 0

        if self.use_dataset_desc_emb:
            ds_desc_matrix = self._build_dataset_desc_matrix(args, num_datasets + 1)
            self.register_buffer("dataset_desc_matrix", ds_desc_matrix)
        else:
            self.register_buffer("dataset_desc_matrix", torch.zeros(0, self.dataset_desp_emb_dim))

        # ==== Recompute dimensions and rebuild backbone ====
        model_info_dim = (
            (args.token_dim if self.use_model_name_emb else 0)
            + (args.model_dim if self.use_model_id_emb else 0)
            + (self.model_desp_emb_dim if self.use_model_desc_emb else 0)
        )
        self.model_info_dim = model_info_dim

        dataset_emb_dim = (
            (self.dataset_id_emb_dim if self.use_dataset_id_emb else 0)
            + (self.dataset_desp_emb_dim if self.use_dataset_desc_emb else 0)
        )
        self.dataset_emb_dim = dataset_emb_dim
        dataset_info_dim = dataset_emb_dim + args.task_dim
        metric_dim = self.metric_dim if self.use_metric_embedding else 0
        size_emb_dim_eff = args.size_dim if self.use_size_feature else 0
        backbone_in = (
            model_info_dim
            + dataset_info_dim
            + size_emb_dim_eff
            + self.family_dim
            + self.ms_fusion_dim
            + metric_dim
        )
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
        )

        # Rebuild prior_head whose input dim must match exactly which prior
        # components are active (size, family, both, or neither).
        prior_in_actual = 0
        if self.use_size_prior and self.use_size_feature:
            prior_in_actual += args.size_dim
        if self.use_family_prior:
            prior_in_actual += self.family_dim
        if prior_in_actual > 0:
            self.prior_head = nn.Sequential(
                nn.Linear(prior_in_actual, args.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(args.hidden_dim // 2, 1),
            )

        if self.use_ms_spider_repr:
            self.ms_query = nn.Linear(model_info_dim, self.ms_fusion_dim)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset_desc_matrix(args, num_slots: int) -> torch.Tensor:
        """
        Build a ``[num_slots, desc_dim]`` buffer of pre-computed dataset
        description embeddings, indexed by *global* dataset ID.

        For the train split, global_id == local_id, so ``train_vecs.npz``
        can be indexed directly.
        """
        data_name = getattr(args, "data_name", "unified")
        data_dir = os.path.join("data", data_name)
        desc_dim = int(getattr(args, "dataset_desp_emb_dim", 1536))

        mat = np.zeros((num_slots, desc_dim), dtype=np.float32)

        train_vecs_path = os.path.join(data_dir, "train", "train_vecs.npz")
        train_ds_path = os.path.join(data_dir, "train", "train_datasets.json")

        if not os.path.exists(train_vecs_path) or not os.path.exists(train_ds_path):
            print(
                f"[ModelLens] warning: dataset description files not found "
                f"({train_vecs_path}); using zeros."
            )
            return torch.from_numpy(mat)

        vecs = np.load(train_vecs_path)["vecs"]  # [num_train_datasets, 1536]
        with open(train_ds_path, "r", encoding="utf-8") as f:
            ds2id = json.load(f)

        use_dim = min(desc_dim, int(vecs.shape[1]))
        for _name, local_id in ds2id.items():
            # For train split: local_id == global_id
            if local_id < vecs.shape[0] and local_id < num_slots:
                mat[local_id, :use_dim] = vecs[local_id, :use_dim].astype(np.float32)

        return torch.from_numpy(mat)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _ds_ids_from_desp(self, dataset_desp: torch.Tensor) -> torch.LongTensor:
        """Extract integer dataset IDs from the 1-dim float tensor."""
        ids = dataset_desp[:, 0].long()
        # Use whichever buffer/embedding is active to determine the upper bound;
        # both encode the same id-space (size = num_datasets + 1 or +2).
        if self.dataset_id_embedding is not None:
            upper = self.dataset_id_embedding.num_embeddings - 1
        elif self.dataset_desc_matrix.shape[0] > 0:
            upper = self.dataset_desc_matrix.shape[0] - 1
        else:
            upper = max(int(ids.max().item()), 0)
        return ids.clamp(0, upper)

    def _encode_dataset(self, dataset_desp: torch.Tensor) -> torch.Tensor:
        """
        Convert raw ``dataset_desp`` (carrying global ds_id as a float)
        into a rich dataset representation:  learned_id_emb || frozen_desc_emb.
        Either component may be gated off by config flags. Always returns a
        tensor whose width matches ``self.dataset_emb_dim`` so the backbone's
        input dimension is stable across the configured flags.
        """
        B = dataset_desp.shape[0]
        device = dataset_desp.device
        ds_ids = self._ds_ids_from_desp(dataset_desp) if (
            self.use_dataset_id_emb or self.use_dataset_desc_emb
        ) else None

        parts = []
        if self.use_dataset_id_emb:
            ids_for_emb = ds_ids
            # Per-step random dataset-ID dropout: replace a random subset with [UNK]
            # so the backbone learns to rank on unseen datasets via description alone.
            if self.training and self.dataset_id_dropout_rate > 0:
                mask = torch.rand(ds_ids.size(0), device=ds_ids.device) < self.dataset_id_dropout_rate
                ids_for_emb = ds_ids.clone()
                ids_for_emb[mask] = self.unk_dataset_id
            parts.append(self.dataset_id_embedding(ids_for_emb))             # [B, dataset_id_emb_dim]

        if self.use_dataset_desc_emb:
            if self.dataset_desc_matrix.shape[0] > 0:
                safe_ids = ds_ids.clamp(0, self.dataset_desc_matrix.shape[0] - 1)
                parts.append(self.dataset_desc_matrix[safe_ids])              # [B, dataset_desp_emb_dim]
            else:
                # Frozen-desc file missing: fall back to zeros so backbone dim stays stable.
                parts.append(torch.zeros(B, self.dataset_desp_emb_dim, device=device))

        if not parts:
            return torch.zeros(B, 0, device=device)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def encode_model(
        self, model_ids: torch.LongTensor, model_names: list[str],
    ) -> torch.Tensor:
        B = model_ids.shape[0]
        device = model_ids.device

        parts = []
        if self.use_model_name_emb:
            parts.append(self._name_encoder(model_ids, model_names))         # [B, token_dim]

        if self.use_model_id_emb:
            ids = model_ids
            # Per-step random ID dropout: replace a random subset with [UNK]
            # so the backbone learns to rank unseen models via name/desc/size.
            if self.training and self.model_id_dropout_rate > 0:
                mask = torch.rand(ids.size(0), device=ids.device) < self.model_id_dropout_rate
                ids = ids.clone()
                ids[mask] = self.unk_model_id
            parts.append(self._id_emb(ids))                                  # [B, model_dim]

        if self.use_model_desc_emb:
            if self.model_desc_matrix.shape[0] > 0:
                safe_ids = model_ids.clamp(0, self.model_desc_matrix.shape[0] - 1)
                parts.append(self.model_desc_matrix[safe_ids])               # [B, model_desp_emb_dim]
            else:
                # Frozen-desc file missing: zeros so backbone dim stays stable.
                parts.append(torch.zeros(B, self.model_desp_emb_dim, device=device))

        if not parts:
            return torch.zeros(B, 0, device=device)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        task_ids: torch.LongTensor,
        dataset_desp: torch.Tensor,
        model_ids: torch.LongTensor,
        model_names: list[str],
        size_id: torch.LongTensor,
        family_id: torch.LongTensor = None,
        ms_repr: torch.Tensor = None,
        ms_interaction: torch.Tensor = None,
        metric_ids: torch.LongTensor = None,
    ):
        B = task_ids.size(0)
        device = task_ids.device

        h_model = self.encode_model(model_ids, model_names)                  # [B, model_info_dim]
        h_data = self._encode_dataset(dataset_desp)                          # [B, ds_id_dim + ds_desc_dim]
        h_task = self.task_embedding(task_ids)                               # [B, task_dim]
        h_size = self.size_embedding(size_id) if self.use_size_feature else None
        h_metric = self._metric_embed(metric_ids, B, device)

        if self.use_ms_spider_repr:
            h_ms = self._compute_ms_feature(h_model, ms_repr, ms_interaction)
        else:
            h_ms = h_model.new_zeros(B, self.ms_fusion_dim) if h_model.shape[1] > 0 \
                else torch.zeros(B, self.ms_fusion_dim, device=device)

        if self.use_family_prior and family_id is not None:
            h_family = self.family_embedding(family_id)                      # [B, family_dim]
        else:
            h_family = None

        # ---- residual backbone (skip zero-width parts when a feature is gated off) ----
        parts = []
        if h_model.shape[1] > 0:
            parts.append(h_model)
        if h_data.shape[1] > 0:
            parts.append(h_data)
        if h_size is not None:
            parts.append(h_size)
        if h_family is not None:
            parts.append(h_family)
        parts.append(h_task)
        if h_ms.shape[1] > 0:
            parts.append(h_ms)
        if h_metric is not None:
            parts.append(h_metric)
        residual_inp = torch.cat(parts, dim=-1)
        h = self.backbone(residual_inp)
        s_residual = self.pairwise_head(h).squeeze(-1)                       # [B]
        z_pred = self.pointwise_head(h).squeeze(-1)                          # [B]

        # ---- prior (size and/or family) ----
        prior_parts = []
        if self.use_size_prior and h_size is not None:
            prior_parts.append(h_size)
        if self.use_family_prior and h_family is not None:
            prior_parts.append(h_family)
        if prior_parts:
            prior_inp = torch.cat(prior_parts, dim=-1) if len(prior_parts) > 1 else prior_parts[0]
            s_prior = self.prior_head(prior_inp).squeeze(-1)
        else:
            s_prior = torch.zeros(B, device=device)

        tilde_s = (s_residual + s_prior) / torch.clamp(self.temperature, min=1e-3)
        return tilde_s, z_pred

    # ------------------------------------------------------------------
    # build_model_cache  (evaluation) -- override to gate size_embedding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_model_cache(
        self,
        all_model_names: list[str],
        all_model_size_ids: torch.LongTensor,
        all_model_family_ids: torch.LongTensor = None,
        device=None,
    ):
        if device is None:
            device = next(self.parameters()).device

        size_ids = all_model_size_ids.to(device=device, dtype=torch.long)
        M = len(all_model_names)
        assert size_ids.shape[0] == M, "all_model_names and all_model_size_ids must have the same length"

        model_ids = torch.arange(M, device=device, dtype=torch.long)

        h_model = self.encode_model(model_ids, all_model_names)
        h_size = self.size_embedding(size_ids) if self.use_size_feature else None

        cache = {
            "h_model": h_model,
            "h_size": h_size,
            "size_ids": size_ids,
        }

        if self.use_family_prior and all_model_family_ids is not None:
            family_ids = all_model_family_ids.to(device=device, dtype=torch.long)
            h_family = self.family_embedding(family_ids)
            cache["h_family"] = h_family
            cache["family_ids"] = family_ids
        else:
            cache["h_family"] = None
            cache["family_ids"] = None

        return cache

    # ------------------------------------------------------------------
    # score_matrix  (evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_matrix(
        self,
        task_ids: torch.LongTensor,
        dataset_desp_batch: torch.Tensor,
        model_cache: dict,
        ms_repr_batch: torch.Tensor = None,
        chunk_size: int = 8192,
        metric_ids: torch.LongTensor = None,
    ):
        device = dataset_desp_batch.device
        B = dataset_desp_batch.size(0)

        h_task = self.task_embedding(task_ids)
        h_data = self._encode_dataset(dataset_desp_batch)
        h_metric = self._metric_embed(metric_ids, B, device)

        h_model_all = model_cache["h_model"]
        h_size_all = model_cache["h_size"] if self.use_size_feature else None
        h_family_all = model_cache.get("h_family")
        M = h_model_all.size(0)

        # prior for all models (size and/or family)
        prior_parts_all = []
        if self.use_size_prior and h_size_all is not None:
            prior_parts_all.append(h_size_all)
        if self.use_family_prior and h_family_all is not None:
            prior_parts_all.append(h_family_all)
        if prior_parts_all:
            prior_inp_all = (
                torch.cat(prior_parts_all, dim=-1) if len(prior_parts_all) > 1 else prior_parts_all[0]
            )
            prior_all = self.prior_head(prior_inp_all).squeeze(-1)
        else:
            prior_all = torch.zeros(M, device=device)

        out = torch.empty(B, M, device=device)
        T = torch.clamp(self.temperature, min=1e-3)

        start = 0
        while start < M:
            end = min(start + chunk_size, M)
            h_model = h_model_all[start:end]
            m = end - start

            h_model_exp = h_model.unsqueeze(0).expand(B, m, -1) if h_model.shape[1] > 0 else None
            h_data_exp = h_data.unsqueeze(1).expand(B, m, -1) if h_data.shape[1] > 0 else None
            h_task_exp = h_task.unsqueeze(1).expand(B, m, -1)
            if h_size_all is not None:
                h_size_exp = h_size_all[start:end].unsqueeze(0).expand(B, m, -1)
            else:
                h_size_exp = None
            if h_metric is not None:
                h_metric_exp = h_metric.unsqueeze(1).expand(B, m, -1)
            else:
                h_metric_exp = None

            if self.use_ms_spider_repr and ms_repr_batch is not None:
                q = self.ms_query(h_model).unsqueeze(0).expand(B, m, -1)
                k = self.ms_key(ms_repr_batch)
                v = self.ms_val(ms_repr_batch)
                attn = torch.softmax(
                    torch.einsum("bmf,bkf->bmk", q, k) / (self.ms_fusion_dim ** 0.5),
                    dim=-1,
                )
                h_ms = torch.einsum("bmk,bkf->bmf", attn, v)
                h_ms = self.ms_out(h_ms.reshape(B * m, -1)).reshape(B, m, -1)
            else:
                h_ms = None

            parts = []
            if h_model_exp is not None:
                parts.append(h_model_exp)
            if h_data_exp is not None:
                parts.append(h_data_exp)
            if h_size_exp is not None:
                parts.append(h_size_exp)
            if h_family_all is not None:
                h_family = h_family_all[start:end]
                h_family_exp = h_family.unsqueeze(0).expand(B, m, -1)
                parts.append(h_family_exp)
            parts.append(h_task_exp)
            if h_ms is not None and self.ms_fusion_dim > 0:
                parts.append(h_ms)
            if h_metric_exp is not None:
                parts.append(h_metric_exp)
            residual_inp = torch.cat(parts, dim=-1)

            h = self.backbone(residual_inp.reshape(B * m, -1))
            s_chunk = self.pairwise_head(h).reshape(B, m)
            prior_chunk = prior_all[start:end].unsqueeze(0)
            out[:, start:end] = (s_chunk + prior_chunk) / T
            start = end

        return out

