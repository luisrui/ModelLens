from tqdm import tqdm
import torch
import os
import json
import pandas as pd
from module.utils.metric import *
from module.utils.general_util import _to_device, format_eval_summary
import wandb


def _model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def train_mlp_pairPointwise(args, model, train_data_loader, valid_data_loader, criterion, optimizer, logger, procedure_name: str = "mlp"):
    num_epochs = args.num_epochs
    best_val = float('-inf')
    stop_count = 0
    run = None
    
    ckpt_dir = f"checkpoint/mlp/{args.data_name}/{args.trail_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if isinstance(args.topk, int):
        topk_list = [args.topk]
    else:
        topk_list = list(args.topk)

    # wandb init (optional)
    if getattr(args, 'use_wandb', False):
        cfg = dict(vars(args))
        if 'device' in cfg:
            cfg['device'] = str(cfg['device'])
        run_name = f"{getattr(args, 'trail_name', 'trail')}-{getattr(args, 'model_name', 'MLP')}"
        try:
            run = wandb.init(
                project=getattr(args, 'wandb_project', 'ModelProfile'),
                entity=getattr(args, 'wandb_entity', None),
                name=run_name,
                config=cfg,
            )
            logger.info(f"[wandb] initialized run: {run_name}")
        except Exception as e:
            logger.warning(f"[wandb] init failed: {e}")
            run = None
            
    # save args
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
        
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_pair_loss = 0.0
        total_point_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_data_loader, desc=f"[Pairwise] Epoch {epoch}", ncols=100)
        for batch in pbar:
            batch = tuple(_to_device(x, args.device) for x in batch)
            
            loss, stats = criterion(model, batch)
            optimizer.zero_grad()
            loss.backward()
            criterion.clip_grads(model)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if stats is not None:
                pair_l = stats.get("loss_pair", None)
                point_l = stats.get("loss_point", None)
                if pair_l is not None:
                    total_pair_loss += pair_l
                if point_l is not None:
                    total_point_loss += point_l
                
            avg_loss = total_loss / max(1, num_batches)
            desc = f"[Pair+Point] Epoch {epoch} | loss={avg_loss:.4f}"
            pbar.set_description(desc)
                    
        avg_loss = total_loss / max(1, num_batches)
        avg_pair = total_pair_loss / max(1, num_batches)
        avg_point = total_point_loss / max(1, num_batches)
        
        logger.info(
            f"[Train] epoch={epoch}  "
            f"avg_total_loss={avg_loss:.4f}  "
            f"avg_pair_loss={avg_pair:.4f}  "
            f"avg_point_loss={avg_point:.4f}"
        )
        if run is not None:
            try:
                wandb.log(
                    {
                        "train/avg_total_loss": float(avg_loss),
                        "train/avg_pair_loss": float(avg_pair),
                        "train/avg_point_loss": float(avg_point),
                        "epoch": int(epoch),
                    },
                    step=epoch,
                )
            except Exception:
                pass
        
        # Evaluate the model on the validation set using weighted Kendall's tau
        if (epoch % getattr(args, 'eval_every', 1) == 0) and (valid_data_loader is not None):
            summary, per_dataset_stats = evaluate_ranking_mlp(model, valid_data_loader, args.device, topk_list)
            logger.info("\n" + format_eval_summary(summary, topk_list, tag="Valid", epoch=epoch))
            judge_score = 1 * summary["mean_weighted_tau"]
            
            # log to wandb if available and active
            if run is not None:
                try:
                    log_payload = {'epoch': int(epoch), 'valid/judge_score': float(judge_score)}
                    for k, v in summary.items():
                        log_payload[f'valid/{k}'] = v
                    # per-dataset table
                    cols = ['dataset', 'num_models', 'weightedtau']
                    table = wandb.Table(columns=cols)
                    # for (task_name, ds_name), stats in per_dataset_stats.items():
                    #     table.add_data(task_name, ds_name, stats.get('num_models', None), stats.get('weightedtau', None))
                    # log_payload['valid/per_dataset_table'] = table
                    wandb.log(log_payload, step=epoch)
                except Exception:
                    pass
            
            # Save the best model checkpoint
            if judge_score > best_val:
                best_val = judge_score
                ckpt_path = f"checkpoint/mlp/{args.data_name}/{args.trail_name}/{args.model_name}.pt"
                torch.save(_model_state_dict(model), ckpt_path)
                logger.info(f"  -> [Pair+Point] saved checkpoint at epoch {epoch}")
                if run is not None:
                    try:
                        wandb.save(ckpt_path)
                        wandb.summary['best_judge_score'] = float(best_val)
                        wandb.summary['best_epoch'] = int(epoch)
                    except Exception:
                        pass
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= args.early_stop:
                logger.info(f"Early stopping at epoch {epoch} with best valid judge_score = {best_val:.4f}")
                break
    
    logger.info(f"Training done. Best valid [Pair+Point] judge_score = {best_val:.4f}")
    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    
    ### Test the model on the test set based on the best model checkpoint
    return ckpt_path 

def train_mlp_pairwise(args, model, train_data_loader, valid_data_loader, criterion, optimizer, logger, procedure_name: str = "mlp"):
    num_epochs = args.num_epochs
    best_val = float('-inf')
    stop_count = 0
    run = None
    
    ckpt_dir = f"checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if isinstance(args.topk, int):
        topk_list = [args.topk]
    else:
        topk_list = list(args.topk)

    # wandb init (optional)
    if getattr(args, 'use_wandb', False):
        cfg = dict(vars(args))
        if 'device' in cfg:
            cfg['device'] = str(cfg['device'])
        run_name = f"{getattr(args, 'trail_name', 'trail')}-{getattr(args, 'model_name', procedure_name)}"
        try:
            run = wandb.init(
                project=getattr(args, 'wandb_project', 'ModelProfile'),
                entity=getattr(args, 'wandb_entity', None),
                name=run_name,
                config=cfg,
            )
            logger.info(f"[wandb] initialized run: {run_name}")
        except Exception as e:
            logger.warning(f"[wandb] init failed: {e}")
            run = None
            
    # save args
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
        model.train()
        total_loss = 0.0
        total_pair_loss = 0.0
        total_point_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_data_loader, desc=f"[Pairwise] Epoch {epoch}", ncols=100)
        for batch in pbar:
            batch = tuple(_to_device(x, args.device) for x in batch)
            
            loss, stats = criterion(model, batch)
            optimizer.zero_grad()
            loss.backward()
            criterion.clip_grads(model)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if stats is not None:
                pair_l = stats.get("loss_pair", None)
                if pair_l is not None:
                    total_pair_loss += pair_l
                
            avg_loss = total_loss / max(1, num_batches)
            desc = f"[Pair] Epoch {epoch} | loss={avg_loss:.4f}"
            pbar.set_description(desc)
                    
        avg_loss = total_loss / max(1, num_batches)
        avg_pair = total_pair_loss / max(1, num_batches)
        
        logger.info(
            f"[Train] epoch={epoch}  "
            f"avg_total_loss={avg_loss:.4f}  "
            f"avg_pair_loss={avg_pair:.4f}  "
        )
        if run is not None:
            try:
                wandb.log(
                    {
                        "train/avg_total_loss": float(avg_loss),
                        "train/avg_pair_loss": float(avg_pair),
                        "epoch": int(epoch),
                    },
                    step=epoch,
                )
            except Exception:
                pass
        
        # Evaluate the model on the validation set using weighted Kendall's tau
        if (epoch % getattr(args, 'eval_every', 1) == 0) and (valid_data_loader is not None):
            summary, per_dataset_stats = evaluate_ranking_mlp(model, valid_data_loader, args.device, topk_list)
            logger.info("\n" + format_eval_summary(summary, topk_list, tag="Valid", epoch=epoch))
            judge_score = 1 * summary["mean_weighted_tau"]
            
            # log to wandb if available and active
            if run is not None:
                try:
                    log_payload = {'epoch': int(epoch), 'valid/judge_score': float(judge_score)}
                    for k, v in summary.items():
                        log_payload[f'valid/{k}'] = v
                    # per-dataset table
                    cols = ['dataset', 'num_models', 'weightedtau']
                    table = wandb.Table(columns=cols)
                    # for ds_name, stats in per_dataset_stats.items():
                    #     table.add_data(ds_name, stats.get('num_models', None), stats.get('weightedtau', None))
                    # log_payload['valid/per_dataset_table'] = table
                    wandb.log(log_payload, step=epoch)
                except Exception:
                    pass
            
            # Save the best model checkpoint
            if judge_score > best_val:
                best_val = judge_score
                ckpt_path = f"checkpoint/{procedure_name}/{args.data_name}/{args.trail_name}/{args.model_name}.pt"
                torch.save(_model_state_dict(model), ckpt_path)
                logger.info(f"  -> saved checkpoint at epoch {epoch}")
                if run is not None:
                    try:
                        wandb.save(ckpt_path)
                        wandb.summary['best_judge_score'] = float(best_val)
                        wandb.summary['best_epoch'] = int(epoch)
                    except Exception:
                        pass
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= args.early_stop:
                logger.info(f"Early stopping at epoch {epoch} with best valid judge_score = {best_val:.4f}")
                break
    
    logger.info(f"Training done. Best valid judge_score = {best_val:.4f}")
    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    
    ### Test the model on the test set based on the best model checkpoint
    return ckpt_path 

