import math
import re
import sys
from pathlib import Path
import pickle
import time
from contextlib import contextmanager

import copy
import torch
import numpy as np
import diffdist

from trainer.trainer import IterationTrainer
from utils import memory_summary, shared_inner_product
from logger.visualization import TensorboardWriter


def verbose(epoch, metrics, mode, name='TEST'):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f". R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"

def verbose_iteration(iteration, metrics, mode, name='TEST'):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} iteration {iteration}, R@1: {r1:.1f}"
    msg += f". R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"

def tensor_dict_apply(tensor_dict, fn):
    data = {}
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            data[k] = fn(v)
        else:
            data[k] = v
    return data

@contextmanager
def ctxt_mgr(samples, device, disable_nan_checks):
    """Provide a context for managing temporary, cloned copies of retrieval
    sample tensors.

    The rationale here is that to use nan-checking in the model (to validate the 
    positions of missing experts), we need to modify the underlying tensors. This 
    function lets the evaluation code run (and modify) temporary copies, without
    modifying the originals.
    """
    if disable_nan_checks:
        print("running without nan checks")
        yield samples
    else:
        samples_ = tensor_dict_apply(samples, lambda x: x.clone().to(device))
        try:
           yield samples_
        finally:
            del samples_


class DdpTrainer(IterationTrainer):
    """
    Trainer which validates and saves checkpoints every certain iterations
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loaders,
                 lr_scheduler, train_iterations, val_interval, disable_nan_checks,
                 skip_first_n_saves, include_optim_in_ckpts, force_cpu_val,
                 distil_loss, distil_params, num_keep_ckpts=3,
                 mini_train=False, val_freq=1, skip_tboard=False):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.start_epoch = 1
        self.iteration = 0
        self.model = model
        self.include_optim_in_ckpts = include_optim_in_ckpts
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if config.resume is not None:
            if "exclude_pretrained_keys" in config:
                exclude_keys = config["exclude_pretrained_keys"]
            else:
                exclude_keys = None
            self._resume_checkpoint(config.resume, exclude_keys)

        local_rank = self.config["local_rank"]
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.model = self.model.to(self.device)
        if "sync_bn" in config["model"] and config["model"]["sync_bn"]:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)

        if config.resume is not None and self.include_optim_in_ckpts:
            if self.optimizer.__class__.__name__ == "Adam":
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

        self.loss = loss
        self.metrics = metrics
        self.num_keep_ckpts = num_keep_ckpts
        self.skip_tboard = skip_tboard or mini_train

        # This property can be overriden in the subclass
        self.skip_first_n_saves = 0

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_only_best = cfg_trainer.get("save_only_best", True)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = cfg_trainer.get('early_stop', np.inf)

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        if self.config["rank"] == 0:
            if not self.skip_tboard:
                summary_dir = config.log_dir / f"seed-{config['seed']}"
                self.writer = TensorboardWriter(summary_dir, self.logger,
                                                cfg_trainer['tensorboard'])

        self.data_loaders = data_loaders
        self.mini_train = mini_train
        self.disable_nan_checks = disable_nan_checks
        # self.len_epoch = len(self.data_loaders["train"])
        self.log_step = math.ceil(math.sqrt(data_loaders["train"].batch_size * self.config["world_size"]))
        self.force_cpu_val = force_cpu_val
        self.val_freq = val_freq
        self.skip_first_n_saves = skip_first_n_saves
        self.seen = { "train": 0, "val": 0 }
        self.distil_loss = distil_loss
        self.distil_params = distil_params
        self.tt_loss = torch.nn.SmoothL1Loss(reduction="elementwise_mean")
        self.train_dataiter = iter(self.data_loaders["train"])
        self.train_iterations = train_iterations
        self.val_interval = val_interval
        self.epochs = math.ceil(train_iterations / val_interval)

    def train(self):
        """Full training logic.  Responsible for iterating over epochs, early stopping,
        checkpointing and logging metrics.
        """
        rank = self.config["rank"]
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            logging_info = ""
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'nested_val_metrics':
                    # NOTE: currently only supports two layers of nesting
                    for subkey, subval in value.items():
                        for subsubkey, subsubval in subval.items():
                            log[f"val_{subkey}_{subsubkey}"] = subsubval
                            if subsubkey in ["R1", "R5", "R10"]:
                                log_subkey = subkey.replace("_metrics", "")
                                logging_info += f"{log_subkey}_{subsubkey}: {subsubval:.2f}   "
                elif key == 'nested_test_metrics':
                    # NOTE: currently only supports two layers of nesting
                    for subkey, subval in value.items():
                        for subsubkey, subsubval in subval.items():
                            log[f"test_{subkey}_{subsubkey}"] = subsubval
                else:
                    log[key] = value

            log["val_t2a_a2t"] = (
                log["val_t2a_metrics_geometric_mean_R1-R5-R10"] + \
                log["val_a2t_metrics_geometric_mean_R1-R5-R10"]) / 2
            logging_info += f"a2t_t2a: {log['val_t2a_a2t']:.2f}"

            # print logged informations to the screen
            if rank == 0:
                # for key, value in log.items():
                    # self.logger.info('    {:15s}: {}'.format(str(key), value))
                self.logger.info("epoch: {}, loss: {:.3f}, lr: {:.3g}".format(
                    log["epoch"], log["loss"], self.lr_scheduler.get_last_lr()[0]))
                self.logger.info(logging_info)

            # eval model according to configured metric, save best # ckpt as trained_model
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether specified metric improved or not, according to
                    # specified metric(mnt_metric)
                    lower = log[self.mnt_metric] <= self.mnt_best
                    higher = log[self.mnt_metric] >= self.mnt_best
                    improved = (self.mnt_mode == 'min' and lower) or \
                               (self.mnt_mode == 'max' and higher)
                except KeyError:
                    msg = "Warning: Metric '{}' not found, perf monitoring is disabled."
                    self.logger.warning(msg.format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0
                    raise ValueError("Pick a metric that will save checkpoints!!!!!!!!")

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    cpu_model = copy.deepcopy(self.model).cpu()
                    self.best_checkpoint = {"epoch": epoch, "model": cpu_model}
                    if rank == 0:
                        self._save_checkpoint(epoch, save_best=True)
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                # if rank == 0:
                    # self.logger.info("not improved count: {}, early stop: {}".
                        # format(not_improved_count, self.early_stop))

                if not_improved_count == self.early_stop:
                    if rank == 0:
                        self.logger.info("Val performance didn\'t improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                    break

            if self.save_only_best:
                if epoch == self.epochs:
                    best_ckpt = self.best_checkpoint
                    self.model = best_ckpt["model"]
                    if rank == 0:
                        print(f"saving the best ckpt to disk (epoch {epoch})")
                        self._save_checkpoint(best_ckpt["epoch"], save_best=True)
                continue

            # If checkpointing is done intermittently, still save models that outperform
            # the best metric
            save_best = True

            # Due to the fast runtime/slow HDD combination, checkpointing can dominate
            # the total training time, so we optionally skip checkpoints for some of
            # the first epochs
            if epoch < self.skip_first_n_saves and not self.save_only_best:
                msg = f"Skipping ckpt save at epoch {epoch} <= {self.skip_first_n_saves}"
                if rank == 0:
                    self.logger.info(msg)
                continue

            if epoch % self.save_period == 0 and save_best:
                if rank == 0:
                    self._save_checkpoint(epoch, save_best=best)
                    print("This epoch, the save best :{}".format(best))

            if rank == 0:
                if epoch > self.num_keep_ckpts:
                    self.purge_stale_checkpoints()

        if rank == 0:
            self.logger.info(f"the best ckpt is in epoch {self.best_checkpoint['epoch']}")

    def _train_epoch(self, epoch):
        total_loss = 0
        self.model.train()
        rank = self.config["rank"]
        world_size = self.config["world_size"]
        if rank == 0:
            memory_summary()

        for batch_idx in range(self.val_interval):
            if self.iteration >= self.train_iterations:
                break

            try:
                minibatch = next(self.train_dataiter)
            except StopIteration:
                self.data_loaders["train"].sampler.set_epoch(self.iteration // self.val_interval)
                self.train_dataiter = iter(self.data_loaders["train"])
                minibatch = next(self.train_dataiter)

            batch_size = len(minibatch["waveform"])
            if batch_size == 1:
                continue

            self.optimizer.zero_grad()
            # if self.specaug:
                # forward_batch["specaug"] = True
            forward_batch = tensor_dict_apply(minibatch, lambda x: x.to(self.device))
            output = self.model(forward_batch)

            audio_emb = [torch.empty_like(output["audio_emb"]) for _ in range(
                world_size)]
            audio_emb = diffdist.functional.all_gather(audio_emb, output["audio_emb"])
            audio_emb = torch.cat(audio_emb)

            text_emb = [torch.empty_like(output["text_emb"]) for _ in range(
                world_size)]
            text_emb = diffdist.functional.all_gather(text_emb, output["text_emb"])
            text_emb = torch.cat(text_emb)
            
            gathered_output = {
                "audio_emb": audio_emb,
                "text_emb": text_emb
            }
            if "logit_scale" in output:
                gathered_output["logit_scale"] = output["logit_scale"]


            loss = self.loss(**gathered_output)
 
            loss.backward()
            self.optimizer.step()

            self.seen["train"] += batch_size * world_size

            if rank == 0:
                if not self.skip_tboard:
                    self.writer.set_step(self.seen["train"], mode="train")
                    self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()

            if rank == 0:
                if batch_idx % self.log_step == 0:
                    prog = self._progress(batch_idx)
                    # self.logger.info(f"Train Epoch: {epoch} {prog} Loss: {loss.item():.6f}")
                    print(f"Train Epoch: {epoch} {prog} Loss: {loss.item():.6f}")
                    sys.stdout.flush()
                    
                
            if self.mini_train and batch_idx > 3:
                break

            if self.config["lr_scheduler"]["type"] in ["Noam", "CosineWithWarmup"]:
                self.lr_scheduler.step()

            self.iteration += 1

        log = {"loss": total_loss / self.val_interval}
        nested_log = self._valid(self.iteration)
        log.update(nested_log)
        

        if self.lr_scheduler is not None:
            if self.config["lr_scheduler"]["type"] not in ["Noam", "CosineWithWarmup"]:
                self.lr_scheduler.step()

        # self.logger.info(f"LR {self.lr_scheduler.get_last_lr()}")
        return log

    def _valid(self, iteration):
        """
        Validate mode after an epoch of training and store results to disk.

        Args:
            epoch (int): the current epoch

        Returns:
            A log that contains information about validation

        NOTE: The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        rank = self.config["rank"]
        world_size = self.config["world_size"]
        if rank == 0:
            if not self.skip_tboard:
                self.writer.mode = "val"

        with torch.no_grad():
            for batch_idx, samples in enumerate(self.data_loaders["val"]):
                batch_size = samples["waveform"].size(0)
                self.seen["val"] += batch_size * world_size
                safe_size = self.config["eval_settings"]["batch_size"]
                if batch_size > safe_size:
                    partitions = math.ceil(batch_size / safe_size)
                    audio_emb_chunks = []
                    text_emb_chunks = []
                    for chunk_idx in range(partitions):
                        chunk_start = chunk_idx * safe_size
                        chunk_stop = (chunk_idx + 1) * safe_size
                        sample_chunk = tensor_dict_apply(samples, lambda x: x[chunk_start: chunk_stop])
                        if len(sample_chunk["waveform"]) == 0:
                            continue
                        with ctxt_mgr(sample_chunk, self.device, self.disable_nan_checks) as xx:
                            output = self.model.module.evaluate_retrieval(xx)
                        audio_emb_chunks.append(output["audio_emb"])
                        text_emb_chunks.append(output["text_emb"])

                    audio_emb = torch.cat(audio_emb_chunks, dim=0)
                    text_emb = torch.cat(text_emb_chunks, dim=0)
                else:
                    with ctxt_mgr(samples, self.device, self.disable_nan_checks) as xx:
                        output = self.model.module.evaluate_retrieval(xx)
                    audio_emb = output["audio_emb"]
                    text_emb = output["text_emb"]
                logit_scale = output["logit_scale"]

                gathered_audio = [torch.empty_like(audio_emb) for _ in range(
                    world_size)]
                torch.distributed.all_gather(gathered_audio, audio_emb)
                audio_emb = torch.cat(gathered_audio)

                gathered_text = [torch.empty_like(text_emb) for _ in range(
                    world_size)]
                torch.distributed.all_gather(gathered_text, text_emb)
                text_emb = torch.cat(gathered_text)

                if text_emb.ndim == 3:
                    sims = shared_inner_product(audio_emb, text_emb, "none")
                    # sample the loss (using only the first query for each audio)
                    queries_per_audio = samples["num_captions"]
                    sims_ = sims.view(-1, queries_per_audio, sims.size(-1))
                    loss = self.loss(sims=sims_[:, 0, :].contiguous(), logit_scale=logit_scale)
                else:
                    sims = text_emb @ audio_emb.T
                    loss = self.loss(sims=sims, logit_scale=logit_scale)
                if rank == 0:
                    if not self.skip_tboard:
                        self.writer.add_scalar("first_query_loss", loss.item())
                dataset = self.config["data_loader"]["dataset"]
                sims = sims.cpu().numpy()
                nested_metrics = {}
                for metric in self.metrics:
                    metric_name = metric.__name__
                    res = metric(sims)
                    if rank == 0:
                        if metric_name == "mean_average_precision":
                            print(f"Iteration: {iteration}, mean AP: {res['mAP']}")
                        else:
                            verbose_iteration(iteration=iteration, metrics=res, name=dataset, mode=metric_name)
                    
                        self.log_metrics(res, metric_name=metric_name, mode="val")
                    nested_metrics[metric_name] = res
                
                # num_test_caps = self.data_loaders.num_test_captions
                # if num_test_caps == 1 and meta["raw_captions"] is not None:
                    # if self.visualizer is not None:
                        # self.visualizer.visualize_ranking(
                            # sims=sims,
                            # meta=meta,
                            # epoch=epoch,
                            # nested_metrics=nested_metrics,
                        # )
                return {"nested_val_metrics": nested_metrics}

    def _save_checkpoint(self, epoch, save_best=False):
        """Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'trained_model.pth'
        """

        state = {
            'epoch': epoch,
            'iteration': self.iteration,
            'state_dict': self.model.module.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config.config
        }
        if self.include_optim_in_ckpts:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        filename = str(self.checkpoint_dir /
                       'checkpoint-epoch{}.pth'.format(epoch))
        tic = time.time()
        print("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        print(f"Done in {time.time() - tic:.3f}s")
        if save_best:
            print("Updating 'best' checkpoint: {} ...".format(filename))
            best_path = str(self.checkpoint_dir / 'trained_model.pth')
            torch.save(state, best_path)
            print(f"Done in {time.time() - tic:.3f}s")

    def _resume_checkpoint(self, resume_path, exclude_keys=None):
        """ Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        rank = self.config["rank"]
        if rank == 0:
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        if self.config.finetune:
            self.start_epoch = 1
            self.mnt_best = 0
            self.iteration = 0
        else:
            self.iteration = checkpoint['iteration']
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            msg = ("Warning: Architecture configuration given in config file is "
                   "different from that of checkpoint. This may yield an exception"
                   " while state_dict is being loaded.")
            if rank == 0:
                self.logger.warning(msg)
        if exclude_keys is not None:
            for key in exclude_keys:
                del checkpoint["state_dict"][key]
        self.model.load_state_dict(checkpoint['state_dict'])

        if self.include_optim_in_ckpts:
            # load optimizer state from ckpt only when optimizer type is not changed.
            optim_args = checkpoint['config']['optimizer']
            if optim_args['type'] != self.config['optimizer']['type']:
                msg = ("Warning: Optimizer type given in config file differs from that"
                       " of checkpoint. Optimizer parameters not being resumed.")
                if rank == 0:
                    self.logger.warning(msg)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        if rank == 0:
            self.logger.info(f"Ckpt loaded. Resume training from epoch {self.start_epoch}")

    def purge_stale_checkpoints(self):
        """Remove checkpoints that are no longer neededself.

        NOTE: This function assumes that the `best` checkpoint has already been renamed
        to have a format that differs from `checkpoint-epoch<num>.pth`
        """
        all_ckpts = list(self.checkpoint_dir.glob("*.pth"))
        found_epoch_ckpts = list(self.checkpoint_dir.glob("checkpoint-epoch*.pth"))
        if len(all_ckpts) <= self.num_keep_ckpts:
            return

        msg = "Expected at the best checkpoint to have been renamed to a different format"
        if not len(all_ckpts) > len(found_epoch_ckpts):
            print("Warning, purging ckpt, but the best epoch was not saved!")
        # assert len(all_ckpts) > len(found_epoch_ckpts), msg

        # purge the oldest checkpoints
        regex = r".*checkpoint-epoch(\d+)[.]pth$"
        epochs = [int(re.search(regex, str(x)).groups()[0]) for x in found_epoch_ckpts]
        sorted_ckpts = sorted(list(zip(epochs, found_epoch_ckpts)), key=lambda x: -x[0])

        for epoch, stale_ckpt in sorted_ckpts[self.num_keep_ckpts:]:
            tic = time.time()
            stale_ckpt.unlink()
            msg = f"removing stale ckpt [epoch {epoch}] [took {time.time() - tic:.2f}s]"
            print(msg)
