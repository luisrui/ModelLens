import re, math, hashlib, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ModelNameAvgEncoder(nn.Module):
    """
    Split model name by '-' and hash the tokens, then average the embeddings.
    Optionally, concatenate with ID embedding and project to get the final model embedding.

    When ``id_dropout_rate > 0`` and the module is in training mode, each
    sample's model-ID embedding is replaced with a learned ``[UNK]`` vector
    with the given probability.  This forces the backbone to rely on the
    name-token / size / family signal when the ID is unavailable, enabling
    zero-shot ranking for unseen models at inference time.
    """
    def __init__(self,
                 args,
                 hash_buckets: int = 10000):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.tok_emb = nn.Embedding(self.hash_buckets, args.token_dim)
        self.use_id_emb = args.use_id_emb
        self.id_dropout_rate = float(getattr(args, "id_dropout_rate", 0.1))
        if args.use_id_emb:
            # +1 slot: the last row is the learnable [UNK] embedding
            self.id_emb = nn.Embedding(args.num_models + 1, args.model_dim)
            self.unk_model_id = args.num_models  # index of [UNK]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tok_emb.weight)
        if self.use_id_emb:
            nn.init.xavier_uniform_(self.id_emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _split(name: str):
        # Keep both full repo-style name and finer tokens (org/model, '-', '_', space).
        n = (name or "").strip().lower()
        if not n:
            return []

        toks = [n]
        if "/" in n:
            toks.append(n.split("/")[-1])

        toks.extend([t for t in re.split(r"[\/_\-\s]+", n) if t])

        # de-dup while preserving order
        out = []
        seen = set()
        for t in toks:
            if t in seen:
                continue
            out.append(t)
            seen.add(t)
        return out

    def _hash(self, tok: str):
        return int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.hash_buckets

    def _parse_meta(self, toks):
        s = "-".join(toks)
        # steps
        m = re.search(r"(\d+)\s*steps", s)
        steps = float(m.group(1)) if m else 0.0
        log_steps = math.log(steps+1.0)

        # quant bits
        qb = 0.0
        m2 = re.search(r"(\d+)\s*bit", s)
        if m2:
            qb = float(m2.group(1))
        elif any(t in {"4bit","8bit"} for t in toks):
            qb = 4.0 if "4bit" in toks else (8.0 if "8bit" in toks else 0.0)

        flags = {
            "is_sft": 1.0 if "sft" in toks else 0.0,
            "is_dpo": 1.0 if "dpo" in toks else 0.0,
            "is_ppo": 1.0 if "ppo" in toks else 0.0,
            "is_qlora": 1.0 if "qlora" in toks else 0.0,
            "is_bnb": 1.0 if "bnb" in toks else 0.0,
            "is_mtbc": 1.0 if "mtbc" in toks else 0.0,
        }
        meta = [log_steps, qb, flags["is_sft"], flags["is_dpo"], flags["is_ppo"],
                flags["is_qlora"], flags["is_bnb"], flags["is_mtbc"]]
        return meta

    def forward(self, model_ids: torch.LongTensor, model_names: list[str]):
        device = model_ids.device
        B = len(model_names)

        # 1) name -> tokens -> hashed ids
        tok_lists = [self._split(n) for n in model_names]
        # 2) token embedding & average
        vecs = []
        for toks in tok_lists:
            if not toks:
                vecs.append(torch.zeros(self.tok_emb.embedding_dim, device=device))
                continue
            idxs = torch.tensor([self._hash(t) for t in toks], device=device, dtype=torch.long)
            emb = self.tok_emb(idxs)                            # [L, D]
            h_name = emb.mean(dim=0)
            vecs.append(h_name)
        h_name = torch.stack(vecs, dim=0)                       # [B, D]

        feats = [h_name]
        if self.use_id_emb:
            ids = model_ids
            # During training, randomly replace model IDs with [UNK] so the
            # backbone learns to rank unseen models using name/size/family.
            if self.training and self.id_dropout_rate > 0:
                mask = torch.rand(B, device=device) < self.id_dropout_rate
                ids = ids.clone()
                ids[mask] = self.unk_model_id
            feats.append(self.id_emb(ids))                       # [B, model_dim]

        h = torch.cat(feats, dim=-1)
        return h


class ModelNameDescAvgEncoder(nn.Module):
    """
    Encode model name + model description by average hashed-token embeddings.
    Optionally concatenate model-id embedding (disabled by default in ablation runs).
    """
    def __init__(self, args, hash_buckets: int = 10000):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.tok_emb = nn.Embedding(self.hash_buckets, args.token_dim)
        desc_dim = int(getattr(args, "desc_token_dim", args.token_dim))
        self.desc_tok_emb = nn.Embedding(self.hash_buckets, desc_dim)
        self.use_id_emb = bool(getattr(args, "use_id_emb", False))
        self.id_dropout_rate = float(getattr(args, "id_dropout_rate", 0.1))
        if self.use_id_emb:
            self.id_emb = nn.Embedding(args.num_models + 1, args.model_dim)
            self.unk_model_id = args.num_models
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tok_emb.weight)
        nn.init.xavier_uniform_(self.desc_tok_emb.weight)
        if self.use_id_emb:
            nn.init.xavier_uniform_(self.id_emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _split(text: str):
        n = (text or "").strip().lower()
        if not n:
            return []
        toks = [t for t in re.split(r"[\/_\-\s\.,:;()\[\]{}]+", n) if t]
        out = []
        seen = set()
        for t in toks:
            if t in seen:
                continue
            out.append(t)
            seen.add(t)
        return out

    def _hash(self, tok: str):
        return int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.hash_buckets

    def _avg_tok_emb(self, toks, emb_table: nn.Embedding, device: torch.device):
        if not toks:
            return torch.zeros(emb_table.embedding_dim, device=device)
        idxs = torch.tensor([self._hash(t) for t in toks], device=device, dtype=torch.long)
        emb = emb_table(idxs)
        return emb.mean(dim=0)

    def forward(
        self,
        model_ids: torch.LongTensor,
        model_names: list[str],
        model_descs: Optional[list[str]] = None,
    ):
        device = model_ids.device
        B = len(model_names)
        if model_descs is None:
            model_descs = [""] * B
        if len(model_descs) != B:
            raise ValueError("model_descs length must match model_names length")

        name_vecs = []
        desc_vecs = []
        for n, d in zip(model_names, model_descs):
            n_toks = self._split(n)
            d_toks = self._split(d)
            name_vecs.append(self._avg_tok_emb(n_toks, self.tok_emb, device))
            desc_vecs.append(self._avg_tok_emb(d_toks, self.desc_tok_emb, device))

        h_name = torch.stack(name_vecs, dim=0)
        h_desc = torch.stack(desc_vecs, dim=0)

        feats = [h_name, h_desc]
        if self.use_id_emb:
            ids = model_ids
            if self.training and self.id_dropout_rate > 0:
                mask = torch.rand(B, device=device) < self.id_dropout_rate
                ids = ids.clone()
                ids[mask] = self.unk_model_id
            feats.append(self.id_emb(ids))
        return torch.cat(feats, dim=-1)
