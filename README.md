# ModelLens: Finding the Best Model for Your Task from Myriads of Models

> *A unified ranking framework that learns directly from public leaderboard
> interactions to recommend the best pretrained model for an unseen
> dataset — without ever running a candidate on the target task.*

This repository contains the official implementation of **ModelLens**, the
metric-aware ranking framework introduced in our paper *"ModelLens: Finding
the Best for Your Task from Myriads of Models"*. 

---

## TL;DR

The open-source model ecosystem now contains hundreds of thousands of
pretrained models, and new models and datasets emerge continuously. Existing
selection paradigms — AutoML, transferability estimation, and model routing —
either (i) require a forward pass per candidate on the target dataset, or
(ii) presuppose a small curated pool. Neither scales.

ModelLens reframes model selection as a **ranking problem over (model,
dataset, task, metric) tuples**, learned from the large-scale but noisy
trace of public benchmark records. Once trained, it ranks **unseen models on
unseen datasets** zero-shot, using only metadata (names, descriptions, model
size, architecture family) — no forward pass on the target dataset is
required.

On a benchmark of **1.62M evaluation records spanning 47K models and 9.6K
datasets**, ModelLens surpasses both metadata-only and forward-pass
transferability baselines, and its recommended Top-K pools improve five
representative routers by **21%–81%** across QA benchmarks.

---

## What's in this repo

```
ModelLens/
├── config/
│   ├── FinalModel_unified_augmented.yaml      # main model config (Table 1)
│   ├── method_ablation/                       # loss-objective ablations
│   ├── ablation_information/                  # structural/semantic/interaction ablations
│   ├── ablation_size/                         # size-prior / size-feature ablations
│   └── ablation_family/                       # family-prior / family-holdout ablations
├── module/
│   ├── data/        # leaderboard corpus loader, name tokenizer
│   ├── model/       # MLP backbone, MLPMetric, MLPMetricFull (the paper model)
│   ├── procedure/   # listwise / pairwise / pointwise / ensemble training loops
│   └── utils/       # metrics (Kendall-w τ, NDCG@K, Hit@K, Rec@K), family extractor
├── src/main.py      # entry point: parse YAML, build model, train, evaluate
└── scripts/         # one-shot training and ablation drivers
```

---

## Method overview

ModelLens combines **structured inductive bias** with **flexible interaction
modeling**. Three components instantiate this principle.

### 1. Multi-view representations

Each model `m` is encoded as a concatenation of three complementary parts:

```
h_m = [ e_m^id  ‖  e_m^name  ‖  e_m^desc ]
```

* `e_m^id` — learned ID embedding (memorisation of training-time behaviour)
* `e_m^name` — token-averaged hashed name embedding (compositional)
* `e_m^desc` — frozen pretrained-text-encoder embedding of the model card

Each dataset `d` is encoded analogously as `h_d = [ e_d^id ‖ e_d^desc ]`. Two
**structural** model attributes are encoded separately: a **size** bucket
embedding `e_m^size` (capturing neural-scaling effects) and an **architecture
family** embedding `e_m^fam` (capturing shared inductive biases).

### 2. Residual + prior decomposition

The compatibility score is decomposed into a *structural prior* — depending
only on size and family — and a *residual interaction* term:

```
s_prior(m)        = MLP_prior( e_m^size ‖ e_m^fam )
s_residual(m,d,t,μ) = w_pair^T · MLP_backbone([h_m ‖ h_d ‖ e_m^size ‖ e_m^fam ‖ e_t ‖ e_μ])
ŝ(m,d,t,μ)        = (s_residual + s_prior) / max(τ, ε)
```

where `t` and `μ` are task-type and metric embeddings. A **pointwise head**
`ẑ = w_point^T · h` predicts the within-group standardised score `z(m,d)`,
which provides absolute-magnitude grounding for the shared backbone.

### 3. ID dropout for cold-start generalisation

Learned ID embeddings are powerful for memorisation but useless for unseen
entities. During training each ID embedding is independently replaced with a
shared `[UNK]` vector with probability `p_m`, `p_d`. This trains a single
parameter set that operates in both regimes — when IDs are visible it
memorises; when they are masked it relies entirely on names, descriptions,
size, and family. At inference, unseen models or datasets simply map to
`[UNK]` and are scored without any architectural change.

### 4. Multi-objective training

ModelLens is supervised with a weighted combination of three complementary
losses (Section 3.5 of the paper):

```
L = λ_list · L_list  +  λ_pair · L_pair  +  λ_point · L_point
```

* `L_list` — Plackett–Luce listwise likelihood (global ranking structure)
* `L_pair` — BPR pairwise loss (local preferences)
* `L_point` — MSE on the within-group standardised score (absolute calibration)

The full ensemble (`loss_type: ensemble` in YAML) reproduces Table 1 of the
paper. Single-loss variants are provided in `config/method_ablation/`.

---

## Installation

The recommended setup is conda — it pins both Python and CUDA-capable PyTorch:

```bash
conda env create -f environment.yml
conda activate modellens
```

If you prefer pip / venv:

```bash
# Python 3.10+ recommended; install PyTorch separately to match your CUDA.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU training requires a CUDA-capable PyTorch build. Distributed (DDP)
training is supported out of the box; see `scripts/train.sh`.

---

## Data

The training corpus and pretrained checkpoints are **not** included in this
repository — they are large and partially derived from third-party sources
(HuggingFace, Open LLM Leaderboard, Papers-with-Code) whose licences must be
respected when redistributing. Instructions for obtaining the cleaned corpus
are below.

The expected layout under `data/<data_name>/` is:

```
data/unified_augmented/
├── data.csv                     # (model_id, dataset_id, task_id, metric_id, score) records
├── model2id.json                # model name -> integer id
├── task2id.json                 # task type -> integer id
├── metric2id.json               # metric name -> integer id
├── model2family.json            # model name -> architecture family
├── model_profile.json           # canonical model metadata (size in B params, family, ...)
├── model2desp_embeddings.npz    # frozen text embeddings of model cards
├── dataset2desp.json            # dataset description text (per dataset id)
├── train/  val/  test/          # split-specific (model, dataset, metric, score) files
└── new_dataset_evaluation/      # held-out unseen-dataset / unseen-model splits
```

Once available, place the data under `./data/unified_augmented/` (or set
`data_name` in the YAML to point at a different subdirectory).

> **Where to get the data.** We are preparing a public release of the
> 1.62M-record corpus and pretrained ModelLens checkpoints on
> HuggingFace Datasets. A download script will be added to this repo when
> the release is finalised. In the meantime, please contact the authors.

---

## Quick start

Once data is in place:

```bash
# Train the full ModelLens model (ensemble loss, all features)
bash scripts/train.sh

# or, equivalently, single-GPU
python src/main.py --config config/FinalModel_unified_augmented.yaml

# Multi-GPU (DDP). nproc_per_node should match your number of devices.
USE_DDP=1 NPROC=4 bash scripts/train.sh
```

Reproduce the loss-objective and information-source ablations:

```bash
bash scripts/run_method_ablations.sh
bash scripts/run_feature_ablations.sh
```

Outputs:

* Checkpoints — `checkpoint/mlp/<data_name>/<trail_name>/`
* Logs — `log/mlp/<data_name>/<trail_name>/train.log`
* Optional W&B run — controlled by `use_wandb` in the YAML

---

## Configuration

All hyperparameters live in YAML. Key knobs (see
`config/FinalModel_unified_augmented.yaml` for defaults):

| Field | Meaning |
|---|---|
| `model_name` | One of `MLP`, `MLPMetric`, `MLPMetricFull` (the paper model). |
| `loss_type` | `ensemble`, `listwise`, `pairwise`, `pairwise_pointwise`, `listwise_pointwise`, `listwise_pairwise`. |
| `id_dropout_rate` | Probability of masking a learned model/dataset ID with `[UNK]`. |
| `use_size_prior`, `use_family_prior` | Toggle the structural-prior head terms. |
| `use_size_feature` | If `False`, drops the size embedding from both backbone and prior. |
| `use_dataset_id_as_desp` | When `True`, the dataloader passes a global dataset id in the dataset-description slot, which the model intercepts to look up *both* a learned dataset embedding and a frozen description embedding. Required by `MLPMetricFull`. |
| `lambda_list`, `lambda_pair`, `point_loss_weight` | Loss weights `λ_list`, `λ_pair`, `λ_point`. |
| `tau` | Initial value of the learnable temperature `τ`. |
| `topk` | List of `K` values for Hit@K / NDCG@K / Rec@K. |

---

## Evaluation protocol

ModelLens supports the two settings from Section 4.2.1 of the paper:

1. **Performance completion** — randomly mask entries from a partially
   observed `(model × dataset)` matrix and predict their values.
2. **Cold-start generalisation** — hold out entire *datasets* or entire
   *models* (`new_dataset_evaluation` / `new_model_evaluation` split modes)
   and score them zero-shot.

Ranking quality is reported with **Kendall-weighted τ_w** (the primary
metric, emphasising top-rank correctness) and **NDCG@K, Hit@K, Rec@K**, all
implemented in [`module/utils/metric.py`](module/utils/metric.py).

---

## Citation

If you find ModelLens useful in your research, please cite:

```bibtex
@article{cai2026modellens,
  title   = {{ModelLens}: Finding the Best for Your Task from Myriads of Models},
  author  = {Cai, Rui and Mo, Weijie Jacky and Wen, Xiaofei and Ma, Qiyao and
             Zhu, Wenhui and Chen, Xiwen and Chen, Muhao and Zhao, Zhe},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
