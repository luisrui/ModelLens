# ModelLens: Finding the Best Model for Your Task from Myriads of Models

<p align="center">
  <a href="https://arxiv.org/pdf/2605.07075" target="_blank">📄 Paper</a>
  &nbsp;|&nbsp;
  <a href="https://luisrui.github.io/ModelLens/" target="_blank">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/luisrui/ModelLens" target="_blank">🤗 Demo</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/luisrui/modellens-atlas" target="_blank">🗺️ Atlas Graph</a>
</p>

> *A unified ranking framework that learns directly from public leaderboard
> interactions to recommend the best pretrained model for an unseen
> dataset — without ever running a candidate on the target task.*

This repository contains the official implementation of **ModelLens**, the
metric-aware ranking framework introduced in our paper *"ModelLens: Finding
the Best for Your Task from Myriads of Models"*.

<p align="center">
  <img src="figures/teaser_figure/teaser_figure_v3.png" alt="ModelLens teaser" width="100%" />
</p>

<p align="center"><sub>
<b>Left:</b> the <b>learned model–dataset atlas</b> — a single embedding
space, trained on 1.62M public benchmark records, that co-locates
<i>every</i> model and <i>every</i> dataset. Models from the same family
(BERT / LLaMA / T5 / ViT / Whisper / …) cluster together, and datasets from
the same domain (NLP / Vision / Speech / Retrieval / Multimodal / Math &
Code) form their own neighborhoods. The geometry reflects <i>what works on
what</i>, not just text similarity.
&nbsp;&nbsp;<b>Right:</b> given an unseen target dataset (here:
<b>MMMU</b>), ModelLens returns top-K candidates that are
<i>task-appropriate</i> — multimodal LMs such as Gemini-2.5-Pro,
Step-3-VL-108B and Qwen3-VL-235B — in stark contrast to the nearest
text-embedding neighbors (DeBERTa-MNLI, mDeBERTa-Vietnamese, MiniLM-IMDb)
which match the <i>description</i> but solve the wrong problem.
</sub></p>

---

## Why ModelLens

The open-source model ecosystem is exploding. HuggingFace alone now hosts
**hundreds of thousands** of pretrained models across thousands of
architectures, and a practitioner facing a new task has to answer one
deceptively simple question:

> *Which of these myriad models will do best on my dataset?*

Existing answers are unsatisfying for very different reasons:

* **AutoML / fine-tune-and-rank.** Train every candidate on the target
  task and pick the winner. Optimal in the limit, infeasible at the scale
  of hundreds of thousands of models.
* **Transferability estimation** (LEEP, NCE, LogME, …). Cheaper than full
  fine-tuning, but still requires *a forward pass per candidate on the
  target dataset*. The cost grows linearly with the candidate pool, and
  most estimators assume a single, well-defined task setup.
* **Model routing** (RouterBench, RouteLLM, …). Fast at inference, but
  presupposes a tiny, hand-curated pool of ~5–30 models. Asks "which of
  these few?", not "which of these many?".
* **Metadata-only retrieval.** Embed the model card and the dataset
  description with a frozen text encoder, return nearest neighbors. Cheap
  and scalable, but as the right panel of the teaser shows, *text
  similarity is not task similarity*: a Vietnamese DeBERTa is among the
  nearest text-neighbors of MMMU but a hopeless choice for solving it.

ModelLens reframes model selection as a **ranking problem over
`(model, dataset, task, metric)` tuples**, learned directly from the
large-scale but noisy trace of public benchmark records. Once trained, it
ranks **unseen models on unseen datasets** zero-shot, using only metadata
(names, descriptions, model size, architecture family) — no forward pass
on the target dataset, no curated pool.

On a benchmark of **1.62M evaluation records spanning ~47K models and
~9.6K datasets**, ModelLens surpasses both metadata-only and forward-pass
transferability baselines, and its recommended Top-K pools improve five
representative routers by **21%–81%** across QA benchmarks.

---

## What ModelLens learns: the model–dataset atlas

A useful side-effect of training a single ranker over all
`(model, dataset)` interactions is that we can inspect the resulting
latent space directly. Each star below is a model, colored by
**architecture family**; the surrounding scatter / mesh shows the
datasets it has been evaluated on, colored by **task domain**.

<table>
  <tr>
    <td align="center" width="50%"><sub><b>Semantic-only baseline</b> — atlas built from frozen text-embedding similarity between model cards and dataset descriptions (i.e. what a metadata-only retriever sees).</sub></td>
    <td align="center" width="50%"><sub><b>ModelLens (full data)</b> — the same projection, but using the learned latents that absorb 1.62M co-evaluation records.</sub></td>
  </tr>
  <tr>
    <td><img src="figures/teaser_figure/atlas_A_semantic_only.png" alt="Atlas — semantic only" /></td>
    <td><img src="figures/teaser_figure/atlas_B_full_data.png" alt="Atlas — full data (ModelLens)" /></td>
  </tr>
</table>

The two atlases tell the same story from opposite ends:

* The **semantic-only** atlas (left) shows that text similarity alone
  produces a tangled mass: families overlap heavily in the centre, and
  many task-relevant distinctions (e.g. encoder-only LMs vs decoder-only
  LMs, multimodal vs vision-only) collapse together because their
  *descriptions* read similarly.
* The **full-data** atlas (right), driven by actual evaluation
  interactions, untangles this geometry: speech models (orange) detach
  cleanly from the text continent, retrieval embedders (green) form
  their own arc, and vision / multimodal models bridge the vision–text
  boundary. Family structure is **recovered from co-evaluation patterns**,
  not supplied as a label.

The practical consequence is the right panel of the teaser: in the learned
space, *nearest-neighbor* in fact means *task-appropriate*, while in the
semantic-only space it means *text-similar*. ModelLens's recommendation
quality is, in large part, a downstream effect of having the right
geometry to begin with.

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
├── figures/         # teaser & atlas figures used in this README
└── scripts/         # one-shot training and ablation drivers
```

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

The training corpus and pretrained checkpoints are **not** included in
this repository — they are large and partially derived from third-party
sources (HuggingFace, Open LLM Leaderboard, Papers-with-Code) whose
licences must be respected when redistributing.

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
