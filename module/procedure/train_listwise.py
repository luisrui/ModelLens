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


def train_mlp_listPointwise(args, model, train_data_loader, valid_data_loader, criterion, optimizer, logger, procedure_name: str = "mlp"):
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
        
    if getattr(args, 'use_wandb', False):
        cfg = dict(vars(args))
        if 'device' in cfg:
            cfg['device'] = str(cfg['device'])
        run_name = f"{getattr(args, 'trail_name', 'trail')}-{getattr(args, 'model_name', 'MLP')}"
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
        total_list_loss = 0.0
        total_point_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_data_loader, desc=f"[Listwise] Epoch {epoch}", ncols=100)
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
                list_l = stats.get("loss_list", None)
                point_l = stats.get("loss_point", None)
                if list_l is not None:
                    total_list_loss += list_l
                if point_l is not None:
                    total_point_loss += point_l
                
            avg_loss = total_loss / max(1, num_batches)
            desc = f"[List+Point] Epoch {epoch} | loss={avg_loss:.4f}"
            pbar.set_description(desc)

        avg_loss = total_loss / max(1, num_batches)
        avg_list = total_list_loss / max(1, num_batches)
        avg_point = total_point_loss / max(1, num_batches)
        
        logger.info(
            f"[Train] epoch={epoch}  "
            f"avg_total_loss={avg_loss:.4f}  "
            f"avg_list_loss={avg_list:.4f}  "
            f"avg_point_loss={avg_point:.4f}"
        )

        if run is not None:
            try:
                wandb.log(
                    {
                        "train/avg_total_loss": float(avg_loss),
                        "train/avg_list_loss": float(avg_list),
                        "train/avg_point_loss": float(avg_point),
                        "epoch": int(epoch),
                    },
                    step=epoch,
                )
            except Exception:
                pass
            
        if (epoch % getattr(args, 'eval_every', 1) == 0) and (valid_data_loader is not None):
            summary, per_dataset_stats = evaluate_ranking_mlp(
                model, valid_data_loader, args.device, topk_list
            )
            logger.info("\n" + format_eval_summary(summary, topk_list, tag="Valid", epoch=epoch))

            judge_score = 1.0 * summary["mean_weighted_tau"]

            # log 到 wandb
            if run is not None:
                try:
                    log_payload = {
                        'epoch': int(epoch),
                        'valid/judge_score': float(judge_score),
                    }
                    for k, v in summary.items():
                        log_payload[f'valid/{k}'] = v

                    # cols = ['dataset', 'num_models', 'weightedtau']
                    # table = wandb.Table(columns=cols)
                    # for ds_name, stats in per_dataset_stats.items():
                    #     table.add_data(
                    #         ds_name,
                    #         stats.get('num_models', None),
                    #         stats.get('weightedtau', None),
                    #     )
                    # log_payload['valid/per_dataset_table'] = table
                    wandb.log(log_payload, step=epoch)
                except Exception:
                    pass

            # ====== early stopping & best ckpt ======
            if judge_score > best_val:
                best_val = judge_score
                ckpt_path = f"checkpoint/mlp/{args.data_name}/{args.trail_name}/{args.model_name}.pt"
                torch.save(_model_state_dict(model), ckpt_path)
                logger.info(f"  -> [List+Point] saved best checkpoint at epoch {epoch}, judge_score={best_val:.4f}")
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
                logger.info(
                    f"[List+Point] Early stopping at epoch {epoch} with best valid judge_score = {best_val:.4f}"
                )
                break

    logger.info(f"Training done. Best valid [List+Point] judge_score = {best_val:.4f}")
    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass
        
    ### Test the model on the test set based on the best model checkpoint
    return ckpt_path 

def train_mlp_listwise(args, model, train_data_loader, valid_data_loader, criterion, optimizer, logger, procedure_name=None):
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
    
    for epoch in range(args.start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_data_loader, desc=f"[Listwise] Epoch {epoch}", ncols=100)
        for batch in pbar:
            batch = _to_device(batch, args.device)

            loss, stats = criterion(model, batch)

            optimizer.zero_grad()
            loss.backward()
            criterion.clip_grads(model)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_data_loader))
        logger.info(f"[Train-Listwise] epoch={epoch}  avg_loss={avg_loss:.4f}")
        if run is not None:
            try:
                wandb.log({'train_listwise/avg_loss': float(avg_loss), 'epoch': int(epoch)}, step=epoch)
            except Exception:
                pass

        if (epoch % getattr(args, 'eval_every', 1) == 0) and (valid_data_loader is not None):
            summary, per_dataset_stats = evaluate_ranking_mlp(
                model, valid_data_loader, args.device, topk_list
            )
            logger.info("\n" + format_eval_summary(summary, topk_list, tag="Valid-Listwise", epoch=epoch))

            # 评分指标：还是用 weighted Kendall tau
            judge_score = 1.0 * summary["mean_weighted_tau"]

            # log 到 wandb
            if run is not None:
                try:
                    log_payload = {
                        'epoch': int(epoch),
                        'valid_listwise/judge_score': float(judge_score),
                    }
                    for k, v in summary.items():
                        log_payload[f'valid_listwise/{k}'] = v

                    # per-dataset table (照抄你原来的写法)
                    cols = ['dataset', 'num_models', 'weightedtau']
                    table = wandb.Table(columns=cols)
                    for ds_name, stats in per_dataset_stats.items():
                        table.add_data(
                            ds_name,
                            stats.get('num_models', None),
                            stats.get('weightedtau', None),
                        )
                    log_payload['valid_listwise/per_dataset_table'] = table
                    wandb.log(log_payload, step=epoch)
                except Exception:
                    pass

            # ====== early stopping & best ckpt ======
            if judge_score > best_val:
                best_val = judge_score
                ckpt_path = f"checkpoint/mlp/{args.data_name}/{args.trail_name}/{args.model_name}.pt"
                torch.save(_model_state_dict(model), ckpt_path)
                logger.info(f"  -> [Listwise] saved best checkpoint at epoch {epoch}, judge_score={best_val:.4f}")
                if run is not None:
                    try:
                        wandb.save(ckpt_path)
                        wandb.summary['best_listwise_judge_score'] = float(best_val)
                        wandb.summary['best_listwise_epoch'] = int(epoch)
                    except Exception:
                        pass
                stop_count = 0
            else:
                stop_count += 1

            if stop_count >= getattr(args, 'early_stop', 10):
                logger.info(
                    f"[Listwise] Early stopping at epoch {epoch} with best valid judge_score = {best_val:.4f}"
                )
                break

    logger.info(f"Training done. Best valid judge_score = {best_val:.4f}")
    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass
        
    ### Test the model on the test set based on the best model checkpoint
    return ckpt_path 
