import argparse
import os
import time
from collections import defaultdict
from shutil import copy2

import numpy as np
import torch
import torch.distributed
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from fires.configs.default import get_cfg_defaults
from fires.data.omnidataloader import get_dataloader
from fires.model.spnet import SpDRDFNet, compute_loss


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description="3DFIRES: Few Image 3D REconstruction for Scenes with Hidden Surfaces"
    )
    parser.add_argument("config", type=str, help="The configuration file.")

    # distributed training
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed nodes."
    )
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:9991",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use multi-processing distributed training to "
        "launch N processes per node, which has N GPUs. "
        "This is the fastest way to use PyTorch for "
        "either single node or multi node data parallel "
        "training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="GPU id to use. None means using all " "available GPUs.",
    )

    # Resume:
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument(
        "--pretrained", default=None, type=str, help="Pretrained checkpoint"
    )

    # Test run:
    parser.add_argument("--test_run", default=False, action="store_true")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    # Currently save dir and log_dir are the same
    cfg.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
    cfg.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    cfg.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    os.makedirs(cfg.log_dir + "/config")
    copy2(args.config, cfg.log_dir + "/config")
    return args, cfg


def build_optimizer(cfg, model):
    """
    Build an optimizer from config.
    """
    if cfg.TRAIN.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.TRAIN.SOLVER.BASE_LR),
            momentum=cfg.TRAIN.SOLVER.MOMENTUM,
            weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY,
        )
    else:
        assert 0
    if cfg.TRAIN.SOLVER.LR_SCHEDULER_NAME == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.TRAIN.SOLVER.STEPS, gamma=cfg.TRAIN.SOLVER.GAMMA
        )
    else:
        assert 0
    return optimizer, scheduler


class BaseTrainer:
    def __init__(self, cfg, args):
        pass

    def update(self, data, *args, **kwargs):
        raise NotImplementedError("Trainer [update] not implemented.")

    def epoch_end(self, epoch, writer=None, **kwargs):
        # Signal now that the epoch ends....
        pass

    def log_train(
        self,
        train_info,
        train_data,
        writer=None,
        step=None,
        epoch=None,
        visualize=False,
        **kwargs
    ):
        raise NotImplementedError("Trainer [log_train] not implemented.")

    def validate(self, test_loader, epoch, *args, **kwargs):
        raise NotImplementedError("Trainer [validate] not implemented.")

    def log_val(self, val_info, writer=None, step=None, epoch=None, **kwargs):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    writer.add_scalar(k + "_step", v, step)
                else:
                    writer.add_scalar(k + "_epoch", v, epoch)

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        raise NotImplementedError("Trainer [save] not implemented.")

    def resume(self, path, strict=True, **kwargs):
        raise NotImplementedError("Trainer [resume] not implemented.")


class Trainer(BaseTrainer):

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        self.model = SpDRDFNet(cfg).cuda()
        self.optimizer, self.scheduler = build_optimizer(cfg, self.model)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler is not None:
            self.scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar("train/opt_dec_lr", self.scheduler.get_lr()[0], epoch)

    def update(self, data, *args, **kwargs):
        if "no_update" in kwargs:
            no_update = kwargs["no_update"]
        else:
            no_update = False
        if not no_update:
            self.model.train()
            self.optimizer.zero_grad()
        loss_dict = self.model(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
        else:
            losses = sum(loss_dict.values())
        if not no_update:
            losses.backward()
            self.optimizer.step()
        return {"loss": losses.detach().cpu().item()}

    def log_train(
        self,
        train_info,
        train_data,
        writer=None,
        step=None,
        epoch=None,
        visualize=False,
        **kwargs
    ):
        if writer is None:
            return

        # Log training information to tensorboard
        train_info = {
            k: (v.cpu() if not isinstance(v, float) else v)
            for k, v in train_info.items()
        }
        for k, v in train_info.items():
            if not ("loss" in k):
                continue
            if step is not None:
                writer.add_scalar("train/" + k, v, step)
            else:
                assert epoch is not None
                writer.add_scalar("train/" + k, v, epoch)

        if visualize:
            with torch.no_grad():
                return

    def validate(self, test_loader, epoch, stage="test/", *args, **kwargs):
        self.model.eval()
        loss_list = defaultdict(list)
        with torch.no_grad():
            for bidx, data in enumerate(tqdm(test_loader)):

                processed = self.model(data)
                pred_drdf = processed[0]["pred_drdf"].squeeze(-1)
                gt_drdf = torch.clamp(data[0]["gt_drdf"].to(pred_drdf.device), -1, 1)
                first_hit_masks = (
                    data[0]["first_hit_masks"].to(pred_drdf.device).view(-1)
                )
                loss_weights = data[0]["loss_weights"].to(pred_drdf.device)
                losses = compute_loss(pred_drdf, gt_drdf, loss_weights, first_hit_masks)

                for k, v in losses.items():
                    loss_list[k].append(v.cpu().numpy())
        for k, v in loss_list.items():
            loss_list[k] = np.average(loss_list[k])
        return loss_list

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {"model": self.model.state_dict(), "epoch": epoch, "step": step}
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        os.makedirs(os.path.join(self.cfg.save_dir, "checkpoints"), exist_ok=True)
        torch.save(d, path)

    def resume(self, path, strict=True, multi_gpu=False, **kwargs):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        else:
            start_epoch = ckpt["iteration"] // 3781
        return start_epoch

    def log_val(self, val_info, writer=None, step=None, epoch=None, **kwargs):
        if writer is not None:
            for k, v in val_info.items():
                if "pr_curve" in k:
                    plot_pr_curve(v[0], v[1], writer, step=step, epoch=epoch, name=k)
                else:
                    if step is not None:
                        writer.add_scalar(k + "_step", v, step)
                    else:
                        writer.add_scalar(k + "_epoch", v, epoch)


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True
    writer = SummaryWriter(logdir=cfg.log_name)

    train_loader, test_loader = get_dataloader(cfg)
    trainer = Trainer(cfg, args)

    start_epoch = 0
    start_time = time.time()

    if args.resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained) + 1

    # If test run, go through the validation loop first
    if args.test_run:
        trainer.save(epoch=-1, step=-1)

        val_info = trainer.validate(test_loader, epoch=-1)
        trainer.log_val(val_info, writer=writer, epoch=-1)
        trainer.log_val(val_info, writer=writer, step=-1)

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.TRAIN.EPOCHS))
    step = 0
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):

        # train for one epoch
        for bidx, data in enumerate(train_loader):
            step = bidx + len(train_loader) * epoch + 1
            logs_info = trainer.update(data)
            if step % int(cfg.VIS_PERIOD) == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    "Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f"
                    % (epoch, bidx, len(train_loader), duration, logs_info["loss"])
                )
                trainer.log_train(
                    logs_info, data, writer=writer, epoch=epoch, step=step
                )
            if step % int(cfg.EVAL_PERIOD) == 0:
                val_info = trainer.validate(test_loader, epoch=epoch)
                trainer.log_val(val_info, writer=writer, step=step)

        # Save first so that even if the visualization bugged,
        # we still have something
        if (epoch + 1) % int(cfg.SAVE_PERIOD) == 0 and int(cfg.SAVE_PERIOD) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.SAVE_PERIOD) == 0 and int(cfg.EVAL_PERIOD) > 0:
            val_info = trainer.validate(test_loader, epoch=epoch)
            trainer.log_val(val_info, writer=writer, epoch=epoch)

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)
    writer.close()


if __name__ == "__main__":
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
