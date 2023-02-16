import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import time
import argparse
import logging
import math
from pathlib import Path
from typeguard import typechecked

from zsvision.zs_utils import load_json_config
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from mergedeep import Strategy, merge

from parse_config import ConfigParser
import models.metric as module_metric
from utils import set_seeds, write_json
from trainer.ddp_trainer import DdpTrainer, ctxt_mgr, verbose, tensor_dict_apply
from utils.util import init_obj, count_params, shared_inner_product
from logger.logger import setup_logging


def dist_init(config):
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    config["rank"] = rank
    config["world_size"] = world_size
    config["local_rank"] = local_rank
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="file://" + config._args.sync_file,
        world_size=world_size,
        rank=rank)


@typechecked
def load_test_data(config: ConfigParser) -> torch.utils.data.DataLoader:

    ckpt_path = config._args.resume
    train_config = Path(ckpt_path).parent / "config.json"
    train_config = load_json_config(train_config)
    config["data_loader"]["test_collate_fn"] = train_config[
        "data_loader"]["test_collate_fn"]

    dataset = init_obj(config["data_loader"]["dataset_settings"]["test"])
    
    if "test_collate_fn" in config["data_loader"]:
        collate_config = config["data_loader"]["test_collate_fn"]
    else:
        collate_config = config["data_loader"]["collate_fn"]
    collate_fn = init_obj(collate_config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=config["data_loader"]["num_workers"],
        collate_fn=collate_fn)

    return dataloader


@typechecked
def load_model(config: ConfigParser,
               logger: logging.Logger,
               device: str) -> torch.nn.Module:

    ckpt_path = config._args.resume
    model = init_obj(config["model"])
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


def evaluate(config, logger=None, trainer=None):
    if logger is None:
        logger = config.get_logger('test')

    if getattr(config._args, "eval_from_training_config", False):
        eval_conf = copy.deepcopy(config)
        merge(eval_conf._config, config["eval_settings"], strategy=Strategy.REPLACE)
        config = eval_conf

    # seed = config["seed"]
    # logger.info(f"Setting experiment random seed to {seed}")

    # set_seeds(seed)

    disable_gpu = config.get("disable_gpu",
                             config["eval_settings"].get("disable_gpu", True))
    if torch.cuda.is_available() and not disable_gpu:
        device = "cuda"
    else:
        device = "cpu"

    logger.info(f"Running evaluation on {device}")

    exp_config_path = Path(config.resume).with_name("config.json")
    exp_config = load_json_config(exp_config_path)
    
    config._config["model"] = exp_config["model"]

    model = load_model(config, logger, device)
    dataloader = load_test_data(config)
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    model = model.to(device)
    model.eval()

    safe_size = config._args.batch_size
    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            disable_nan_checks = config._config["disable_nan_checks"]
            batch_size = samples["waveform"].size(0)
            if batch_size > safe_size:
                partitions = math.ceil(batch_size / safe_size)
                audio_emb_chunks = []
                text_emb_chunks = []
                for chunk_idx in trange(partitions, ascii=True):
                    chunk_start = chunk_idx * safe_size
                    chunk_stop = (chunk_idx + 1) * safe_size
                    sample_chunk = tensor_dict_apply(samples, lambda x: x[
                        chunk_start: chunk_stop])
                    if len(sample_chunk["waveform"]) == 0:
                        continue
                    with ctxt_mgr(sample_chunk, device, disable_nan_checks) as xx:
                        output = model.evaluate_retrieval(xx)
                    audio_emb_chunks.append(output["audio_emb"])
                    text_emb_chunks.append(output["text_emb"])

                audio_emb = torch.cat(audio_emb_chunks, dim=0)
                text_emb = torch.cat(text_emb_chunks, dim=0)
            else:
                with ctxt_mgr(samples, device, disable_nan_checks) as xx:
                    output = model.evaluate_retrieval(xx)
                audio_emb = output["audio_emb"]
                text_emb = output["text_emb"]

            sims = shared_inner_product(audio_emb, text_emb, "none").cpu().numpy()

            dataset = config["data_loader"]["dataset"]
            nested_metrics = {}
            for metric in metrics:
                metric_name = metric.__name__
                res = metric(sims)
                verbose(epoch=0, metrics=res, name=dataset, mode=metric_name)
                if trainer is not None:
                    metric_name_ = f"test_{metric_name}"
                    trainer.log_metrics(res, metric_name=metric_name_, mode ="val")
                nested_metrics[metric_name] = res

    log = {}
    for subkey, subval in nested_metrics.items():
        for subsubkey, subsubval in subval.items():
            log[f"test_{subkey}_{subsubkey}"] = subsubval
    for key, value in log.items():
        logger.info(" {:15s}: {}".format(str(key),value))


def train(config):
    logger = config.get_logger('train')
    rank = config["rank"]

    seed = config._args.seed
    tic = time.time()
    if rank == 0:
        logger.info(f"Setting experiment random seed to {seed}")
    set_seeds(seed)
    config["seed"] = seed

    if rank == 0:
        print("Initializing model...")

    audio_encoder = init_obj(config["model"]["audio_encoder"])
    text_encoder = init_obj(config["model"]["text_encoder"])

    model = init_obj(config["model"],
                       audio_encoder=audio_encoder,
                       text_encoder=text_encoder)

    if rank == 0:
        logger.info(config._args)
        for line in str(model).split("\n"):
            logger.info(line)
        count_params(model, ["audio_encoder", "text_encoder",
            "audio_proj", "text_proj"], logger.info)
        print("Initializing dataloaders")
            
    num_workers = config["data_loader"]["num_workers"]
    collate_fn = init_obj(config["data_loader"]["collate_fn"])

    if "test_collate_fn" in config["data_loader"]:
        collate_config = config["data_loader"]["test_collate_fn"]
        test_collate_fn = init_obj(collate_config)
    else:
        test_collate_fn = collate_fn

    dataset_settings = config["data_loader"]["dataset_settings"]
    train_dataset = init_obj(dataset_settings["train"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["data_loader"]["batch_size"] // config["world_size"],
        sampler=DistributedSampler(train_dataset, shuffle=True),
        num_workers=num_workers,
        collate_fn=collate_fn)
    val_dataset = init_obj(dataset_settings["val"])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        sampler=DistributedSampler(val_dataset, shuffle=False),
        num_workers=num_workers,
        collate_fn=test_collate_fn)
 
    data_loaders = {
        "train": train_dataloader,
        "val": val_dataloader,
    }

    if config.get("manual_linear_init", False):
        if rank == 0:
            logger.info("manually setting init for linear layers")

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        model.apply(init_weights)

    loss = init_obj(config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = init_obj(config["optimizer"], trainable_params)
    lr_scheduler = init_obj(config["lr_scheduler"], optimizer)
        
    
    if rank == 0:
        print("Start to train...")

    trainer = DdpTrainer(
        model,
        loss,
        metrics,
        optimizer,
        config=config,
        data_loaders=data_loaders,
        lr_scheduler=lr_scheduler,
        train_iterations=config["trainer"]["train_iterations"],
        val_interval=config["trainer"]["val_interval"],
        mini_train=config._args.mini_train,
        disable_nan_checks=config["disable_nan_checks"],
        val_freq=config["trainer"].get("val_freq", 1),
        distil_loss=config.get("distil_loss", False),
        distil_params=config.get("distil_params", None),
        force_cpu_val=config.get("force_cpu_val", False),
        skip_tboard=config.get("skip_tboard", False),
        skip_first_n_saves=config["trainer"].get("skip_first_n_saves", 0),
        num_keep_ckpts=config["trainer"].get("num_keep_ckpts", 3),
        include_optim_in_ckpts=config["trainer"].get("include_optim_in_ckpts", 0)
    )
    trainer.train()
    best_ckpt_path = config.save_dir / "trained_model.pth"
    duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
    if rank == 0:
        logger.info(f"Training took {duration}")

    if rank == 0:
        print(f"Log file stored at {config.log_path}")
        print(f"The best performing ckpt can be found at {str(best_ckpt_path)}")


def main():
    parser = argparse.ArgumentParser(description="Main entry point")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", help="config file path", required=True)
    train_parser.add_argument("--sync_file", help="distributed data parallel "\
                              "syncronization file path", required=True)
    train_parser.add_argument("--resume", help="path to latest checkpoint (default: None)")
    train_parser.add_argument("--finetune", action="store_true", default=False,
                              help="set to true if finetuning (default: None)")
    train_parser.add_argument("--device", help="indices of GPUs to enable")
    train_parser.add_argument("--mini_train", action="store_true")
    train_parser.add_argument("--group_id", help="if supplied, group these experiments")
    train_parser.add_argument("--disable_workers", action="store_true")
    train_parser.add_argument("--refresh_lru_cache", action="store_true")
    train_parser.add_argument("--train_single_epoch", action="store_true")
    train_parser.add_argument("--purge_exp_dir", action="store_true",
                      help="remove all previous experiments with the given config")
    train_parser.add_argument("--dbg", default="ipdb.set_trace")
    train_parser.add_argument("--custom_args", help="qualified key,val pairs")
    train_parser.add_argument("--seed", default=0, help="random seed")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", default=None, type=str, help="config file path")
    eval_parser.add_argument("--resume", type=Path,
                             help="path to checkpoint for evaluation",
                             required=True)
    eval_parser.add_argument("--device", help="indices of GPUs to enable")
    eval_parser.add_argument("--eval_from_training_config", action="store_true",
                             help="if true, evaluate directly from a training config file.")
    eval_parser.add_argument("--custom_args", help="qualified key,val pairs")
    eval_parser.add_argument("--batch_size", help="inference batch size", type=int, default=256)
    eval_parser.add_argument("--output", help="output filename", type=str,
                             default=None)
    eval_parser.add_argument("--per_class", action="store_true",
                             help="if true, evaluate retrieval task only on specific class")

    
    args = ConfigParser(parser)

    if args._args.mode == "train":
        dist_init(args)

        if args["rank"] == 0:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            args.log_dir.mkdir(parents=True, exist_ok=True)
            # save updated config file to the checkpoint dir
            write_json(args.config, args.save_dir / "config.json")
            # configure logging module
            args.log_path = setup_logging(args.log_dir)
        args["data_loader"]["args"]["refresh_lru_cache"] = args._args.refresh_lru_cache
        if args["rank"] == 0:
            print("Launching experiment with config:")
            print(args)
        train(args)
    elif args._args.mode == "evaluate":
        args._log_dir = Path(args.resume).parent
        args.log_path = setup_logging(args.log_dir,
                                      args._args.output,
                                      "logger/eval_logger_config.json")
        cfg_msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
        assert args._args.resume, cfg_msg
        evaluate(args)


if __name__ == '__main__':
    main()
