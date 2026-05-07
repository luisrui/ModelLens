from tqdm import tqdm
import torch
import torch.distributed as dist
import os
import json
import pandas as pd
from module.utils.metric import *
from module.utils.general_util import _to_device, format_eval_summary
import wandb


def _model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process(args) -> bool:
    return int(getattr(args, "rank", 0)) == 0


def _set_distributed_epoch(args, train_loader, epoch: int):
    if not bool(getattr(args, "is_distributed", False)):
        return
    if hasattr(train_loader, "list_train_loader"):
        for dl in (train_loader.list_train_loader, train_loader.pair_train_loader):
            sampler = getattr(dl, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        return
    sampler = getattr(train_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def train_mlp(
    args,
    model,
    train_loader,
    valid_data_loader,
    criterion,
    optimizer,
    logger,
    procedure_name: str = "mlp",
):
    num_epochs = args.num_epochs
    is_main = _is_main_process(args)
    best_val = float('-inf')
    best_val_ic8x10 = float('-inf')
    stop_count = 0
    run = None
    
    ckpt_dir = f"checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/{args.model_name}.pt"
    ckpt_ic8x10_path = f"{ckpt_dir}/{args.model_name}_best_ic8x10.pt"
    
    if isinstance(args.topk, int):
        topk_list = [args.topk]
    else:
        topk_list = list(args.topk)

    track_ic8x10_ckpt = bool(getattr(args, "save_best_ic8x10_checkpoint", False))
    ic8x10_target_models_file = getattr(
        args, "ic8x10_target_models_file", "model_candidates/image_classification.txt"
    )
    ic8x10_target_datasets = getattr(
        args,
        "ic8x10_target_datasets",
        ["Aircraft", "Cars", "DTD", "Pets", "Flowers102", "Food101", "Country211", "EuroSAT"],
    )
    ic8x10_task_name = getattr(args, "ic8x10_target_task", "Image Classification")
        
    if is_main and getattr(args, 'use_wandb', False):
        cfg = dict(vars(args))
        if 'device' in cfg:
            cfg['device'] = str(cfg['device'])
        run_name = f"{getattr(args, 'trail_name', 'trail')}-{getattr(args, 'model_name', procedure_name)}"
        wandb_id = getattr(args, "wandb_id", None)
        try:
            init_kwargs = dict(
                project=getattr(args, 'wandb_project', 'ModelProfile'),
                entity=getattr(args, 'wandb_entity', None),
                config=cfg,
            )
            if wandb_id is not None:
                init_kwargs.update(dict(id=wandb_id, resume="allow", name=run_name))
            else:
                init_kwargs.update(dict(name=run_name))

            run = wandb.init(**init_kwargs)
            logger.info(f"[wandb] initialized run: {run_name} (id={run.id})")
        except Exception as e:
            logger.warning(f"[wandb] init failed: {e}")
            run = None
            
    # save args
    if is_main:
        args_path = os.path.join(ckpt_dir, "args.json")
        with open(args_path, "w") as f:
            args_to_save = dict(vars(args))
            if 'device' in args_to_save:
                try:
                    args_to_save['device'] = str(args_to_save['device'])
                except Exception:
                    args_to_save['device'] = str(args.device)
            json.dump(args_to_save, f, default=str)
        logger.info(f"[Args] saved to {args_path}")
    
    for epoch in range(args.start_epoch, num_epochs + 1):
        _set_distributed_epoch(args, train_loader, epoch)
        model.train()
        total_loss = 0.0
        total_list_loss = 0.0
        total_pair_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"[Ensemble weights: List : Pair = {args.lambda_list} : {args.lambda_pair}] Epoch {epoch}",
            ncols=100,
            disable=not is_main,
        )
        for batch in pbar:
                
            batch = _to_device(batch, args.device)

            loss, stats = criterion(model, batch)

            optimizer.zero_grad()
            loss.backward()
            criterion.clip_grads(model)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if stats is not None:
                l_list = stats.get("loss_list", None)
                l_pair = stats.get("loss_pair", None)   
                if l_list is not None:
                    total_list_loss += l_list
                if l_pair is not None:
                    total_pair_loss += l_pair
                    
            avg_loss = total_loss / max(1, num_batches)
            desc = f"[Ensemble] Epoch {epoch} | loss={avg_loss:.4f}"
            if is_main:
                pbar.set_description(desc)

        avg_loss = total_loss / max(1, num_batches)
        avg_list_loss = total_list_loss / max(1, num_batches)
        avg_pair_loss = total_pair_loss / max(1, num_batches)
            
        if is_main:
            logger.info(
                f"[Train] epoch={epoch}  "
                f"avg_total_loss={avg_loss:.4f}  "
                f"avg_list_loss={avg_list_loss:.4f}  "
                f"avg_pair_loss={avg_pair_loss:.4f}  "
            )

        if run is not None:
            try:
                wandb.log(
                    {
                        "train/avg_total_loss": float(avg_loss),
                        "train/avg_list_loss": float(avg_list_loss),
                        "train/avg_pair_loss": float(avg_pair_loss),
                        "epoch": int(epoch),
                    },
                    step=epoch,
                )
            except Exception:
                pass
            
        should_stop = False
        if _is_distributed():
            dist.barrier()

        if is_main and (epoch % getattr(args, "eval_every", 1) == 0) and (valid_data_loader is not None):
            summary_fixed = None
            if getattr(args, "val_eval_target_models_all_datasets", False):
                summary, per_dataset_stats = evaluate_ranking_mlp_target_models_all_datasets(
                    model=model,
                    data_loader=valid_data_loader,
                    device=args.device,
                    topk_list=topk_list,
                    target_models_file=getattr(
                        args, "val_target_models_file", "model_candidates/image_classification.txt"
                    ),
                    task_name=getattr(args, "val_target_task", "Image Classification"),
                    strict_target_models=getattr(args, "val_target_models_strict", False),
                    min_models_per_dataset=int(getattr(args, "val_target_models_min_per_dataset", 2)),
                )
            elif getattr(args, "val_eval_fixed_backbones", False):
                summary, per_dataset_stats = evaluate_ranking_mlp_fixed_backbones(
                    model=model,
                    data_loader=valid_data_loader,
                    device=args.device,
                    topk_list=topk_list,
                    target_models_file=getattr(
                        args, "val_target_models_file", "model_candidates/image_classification.txt"
                    ),
                    target_datasets=getattr(args, "val_target_datasets", None),
                    task_name=getattr(args, "val_target_task", "Image Classification"),
                    strict_target_datasets=getattr(args, "val_target_datasets_strict", False),
                )
                summary_fixed = summary
            else:
                summary, per_dataset_stats = evaluate_ranking_mlp(
                    model, valid_data_loader, args.device, topk_list
                )
            logger.info("\n" + format_eval_summary(summary, topk_list, tag="Valid", epoch=epoch))
            if "target_datasets_requested" in summary:
                req = summary.get("target_datasets_requested", [])
                cov = summary.get("target_datasets_evaluated", [])
                miss = summary.get("target_datasets_missing", [])
                logger.info(
                    f"[Valid-Coverage] requested={len(req)} evaluated={len(cov)} missing={len(miss)} "
                    f"covered={cov} missing_list={miss}"
                )

            judge_score = 1.0 * summary["mean_weighted_tau"]

            if run is not None:
                try:
                    log_payload = {
                        "epoch": int(epoch),
                        "valid/judge_score": float(judge_score),
                    }
                    for k, v in summary.items():
                        log_payload[f"valid/{k}"] = v
                    wandb.log(log_payload, step=epoch)
                except Exception:
                    pass

            if track_ic8x10_ckpt:
                if summary_fixed is None:
                    summary_fixed, _ = evaluate_ranking_mlp_fixed_backbones(
                        model=model,
                        data_loader=valid_data_loader,
                        device=args.device,
                        topk_list=topk_list,
                        target_models_file=ic8x10_target_models_file,
                        target_datasets=ic8x10_target_datasets,
                        task_name=ic8x10_task_name,
                        strict_target_datasets=getattr(args, "ic8x10_target_datasets_strict", False),
                    )
                ic8x10_score = float(summary_fixed["mean_weighted_tau"])
                logger.info(
                    f"[Valid-IC8x10] epoch={epoch} datasets={summary_fixed.get('datasets_evaluated', 0)} "
                    f"mean_weighted_tau={ic8x10_score:.4f}"
                )
                if run is not None:
                    try:
                        wandb.log(
                            {
                                "epoch": int(epoch),
                                "valid_ic8x10/judge_score": ic8x10_score,
                            },
                            step=epoch,
                        )
                    except Exception:
                        pass
                if ic8x10_score > best_val_ic8x10:
                    best_val_ic8x10 = ic8x10_score
                    torch.save(_model_state_dict(model), ckpt_ic8x10_path)
                    logger.info(f"  -> saved IC8x10-best checkpoint at epoch {epoch}: {ckpt_ic8x10_path}")
                    if run is not None:
                        try:
                            wandb.save(ckpt_ic8x10_path)
                            wandb.summary["best_ic8x10_judge_score"] = float(best_val_ic8x10)
                            wandb.summary["best_ic8x10_epoch"] = int(epoch)
                        except Exception:
                            pass

            if judge_score > best_val:
                best_val = judge_score
                torch.save(_model_state_dict(model), ckpt_path)
                logger.info(f"  -> saved checkpoint at epoch {epoch}")
                if run is not None:
                    try:
                        wandb.save(ckpt_path)
                        wandb.summary["best_judge_score"] = float(best_val)
                        wandb.summary["best_epoch"] = int(epoch)
                    except Exception:
                        pass
                stop_count = 0
            else:
                stop_count += 1

            if stop_count >= args.early_stop:
                logger.info(
                    f"[Joint List+Pair] Early stopping at epoch {epoch} "
                    f"with best valid judge_score = {best_val:.4f}"
                )
                should_stop = True

        if _is_distributed():
            stop_flag = torch.tensor(int(should_stop) if is_main else 0, device=args.device)
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(stop_flag.item())
            dist.barrier()
        if should_stop:
            break

    if is_main:
        logger.info(f"Training done. Best valid [Ensemble] judge_score = {best_val:.4f}")
        if track_ic8x10_ckpt:
            logger.info(
                f"Training done. Best valid [IC8x10] judge_score = {best_val_ic8x10:.4f}; "
                f"ckpt={ckpt_ic8x10_path}"
            )
    if is_main and run is not None:
        try:
            wandb.finish()
        except Exception:
            pass
        
    ### Test the model on the test set based on the best model checkpoint
    return ckpt_path 

def test_mlp(args, model, data_loader, logger, mode: str = "in-domain", procedure_name: str = "mlp"):
    model.eval()

    if isinstance(args.topk, int):
        topk_list = [args.topk]
    else:
        topk_list = list(args.topk)

    with torch.no_grad():
        if getattr(args, "test_eval_target_models_all_datasets", False):
            summary, per_dataset_stats = evaluate_ranking_mlp_target_models_all_datasets(
                model=model,
                data_loader=data_loader,
                device=args.device,
                topk_list=topk_list,
                target_models_file=getattr(
                    args,
                    "test_target_models_file",
                    getattr(args, "val_target_models_file", "model_candidates/image_classification.txt"),
                ),
                task_name=getattr(
                    args,
                    "test_target_task",
                    getattr(args, "val_target_task", "Image Classification"),
                ),
                strict_target_models=getattr(args, "test_target_models_strict", False),
                min_models_per_dataset=int(getattr(args, "test_target_models_min_per_dataset", 2)),
            )
        else:
            summary, per_dataset_stats = evaluate_ranking_mlp(
                model,
                data_loader,
                args.device,
                topk_list=topk_list
            )
    logger.info("\n" + format_eval_summary(summary, topk_list, tag="Test"))
    
    # Save the aligned dataframe of each dataset to a csv file
    aligned_dir = f"checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}"
    os.makedirs(aligned_dir, exist_ok=True)
    for (task_name, ds_name), stats in per_dataset_stats.items():
        num_models = stats.get("num_models", 0)
        if num_models >= 1000:
            stats["aligned"].to_csv(
                f"{aligned_dir}/{args.model_name}_aligned_{task_name}_{ds_name}.csv",
                index=False,
            )
    logger.info(
        f"[Test] saved aligned csv files to checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}/{args.model_name}_aligned_*.csv"
    )
    
    parts = []
    for k in topk_list:
        ndcg = summary.get(f"mean_ndcg@{k}", "nan")
        hit  = summary.get(f"mean_hit@{k}", "nan")
        n_ds = summary.get(f"num_datasets_for_k@{k}", "nan")
        recall = summary.get(f"mean_recall@{k}", "nan")

        parts.append(f"{ndcg:.4f}, {hit:.4f}, {recall:.4f}")

    csv_line = f"{summary['mean_weighted_tau']:.4f}, {', '.join(parts)}"

    logger.info(f"[Test] copy-paste metrics: {csv_line}")
        
    # Generate a single CSV file for all metrics
    rows = []
    for (task_name, ds_name), stats in per_dataset_stats.items():
        row = {
            "task": task_name,
            "dataset": ds_name,
            "num_models": stats.get("num_models", None),
            "weightedtau": stats.get("weightedtau", None),
        }
        # put all topk ndcg/hit metrics into the csv
        for k in topk_list:
            row[f"ndcg@{k}"] = stats.get(f"ndcg@{k}", None)
            row[f"hit@{k}"] = stats.get(f"hit@{k}", None)
            row[f"recall@{k}"] = stats.get(f"recall@{k}", None)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_path = (
        f"checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}/"
        f"{args.model_name}_per_dataset_metrics_{mode}.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"[Test] saved per-dataset metrics to {metrics_path}")
    
    # log to wandb if available and active
    if getattr(args, 'use_wandb', False) and (getattr(wandb, "run", None) is not None):
        try:
            log_payload = {}
            for k, v in summary.items():
                log_payload[f'test/{k}'] = v

            # per-dataset table
            cols = ['dataset', 'num_models', 'weightedtau']

            # table = wandb.Table(columns=cols)
            # for (task_name, ds_name), stats in per_dataset_stats.items():
            #     row = [
            #         task_name,
            #         ds_name,
            #         stats.get('num_models', None),
            #         stats.get('weightedtau', None),
            #     ]
            #     table.add_data(*row)

            # wandb.log({'test/per_dataset_table': table})
            
            # upload csvs
            for (task_name, ds_name), stats in per_dataset_stats.items():    
                csv_path = f"checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}/{args.model_name}_aligned_{task_name}_{ds_name}_{mode}.csv"
                wandb.save(csv_path)

            wandb.log(log_payload)
        except Exception as e:
            logger.warning(f"[wandb] test logging failed: {e}")
    
    return summary, per_dataset_stats
