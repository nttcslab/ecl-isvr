#! -*- coding: utf-8
import logging
import os
import os.path as path
import typing
from collections import OrderedDict
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import torch
import torchvision
from datasets import CIFAR10, FashionMNIST
from model import (LogisticsModel, densenet121, densenet161,
                   densenet169, vgg11_bn, vgg13_bn, vgg16_bn)
from sklearn.metrics import accuracy_score

try:
    if os.name == "POSIX":
        import matplotlib
        matplotlib.use("Agg")
finally:
    from matplotlib import pyplot as plt


TORCH_SUMMARY_STRING = True
try:
    from torchsummary import summary_string
except ImportError:
    TORCH_SUMMARY_STRING = False


__all__ = ["ATrainer"]


class ATrainer(object):
    def __init__(self, outdir: str, seed: int,
                 datadir: str = "./datas",
                 dataset_name: str = "cifar10",
                 model_name: str = "resnet50",
                 group_channels: int = 32,
                 drop_rate: float = 0.1, last_drop_rate: float = 0.5,
                 data_init_seed: int = 11,
                 model_init_seed: int = 13,
                 train_data_length: int = 12800,
                 cuda: bool = True,
                 cuda_device_no: int = 0,
                 **kwargs):
        self.__outdir__ = outdir
        self.__seed__ = seed
        self.__data_init_seed__ = data_init_seed
        self.__datadir__ = datadir
        self.__dataset_name__ = dataset_name
        self.__datasets__ = None
        self.__optim__ = None
        self.__train_data_length__ = train_data_length
        self.__sleep_factor__ = 0.0

        if cuda:
            if not torch.cuda.is_available():
                raise ValueError("CUDA device is'nt available!!!")
            self.__device__ = "cuda:%d" % (cuda_device_no)
        else:
            self.__device__ = "cpu"
        self.__cuda__ = cuda

        os.makedirs(self.outdir, exist_ok=True)

        self.__model_name__ = model_name
        self.__model__ = self.build_model(model_init_seed=model_init_seed,
                                          group_channels=group_channels,
                                          drop_rate=drop_rate,
                                          last_drop_rate=last_drop_rate)

        self.model.to(self.device)
        logging.info("device: %s %d/%d, %s" % (self.device,
                                               torch.cuda.current_device(),
                                               torch.cuda.device_count(),
                                               torch.cuda.get_device_name(torch.cuda.current_device())))

    @property
    def outdir(self): return self.__outdir__
    @property
    def seed(self): return self.__seed__
    @property
    def data_init_seed(self): return self.__data_init_seed__
    @property
    def datadir(self): return self.__datadir__
    @property
    def cuda(self): return self.__cuda__
    @property
    def device(self): return self.__device__
    @property
    def dataset_name(self): return self.__dataset_name__

    @property
    def datasets(self):
        if self.__datasets__ is None:
            self.__datasets__ = self.load_datas(
                train_data_length=self.train_data_length)
        return self.__datasets__

    @property
    def train_data_length(self): return self.__train_data_length__

    @property
    def train_log(self): return path.join(self.outdir, "train.log")
    @property
    def eval_log(self): return path.join(self.outdir, "eval.log")
    @property
    def state_dict_file(self): return path.join(self.outdir, "model.pth")

    @property
    def score_filenames(self): return {k: "scores_%s.csv" % k
                                       for k in self.datasets}

    @property
    def fig_prefix(self): return ""
    @property
    def fig_suffix(self): return ""

    @property
    def sleep_factor(self): return self.__sleep_factor__
    @sleep_factor.setter
    def sleep_factor(self, val): self.__sleep_factor__ = float(val)

    @property
    def model(self) -> torch.nn.Module: return self.__model__
    @property
    def model_name(self) -> str: return self.__model_name__

    def build_model(self, model_init_seed: int = 32,
                    group_channels: int = 64,
                    drop_rate: float = 0.1,
                    last_drop_rate: float = 0.5,
                    **kwargs) -> torch.nn.Module:
        torch.manual_seed(model_init_seed)

        if self.dataset_name == "cifar10":
            image_shape = (3, 32, 32)
            num_classes = 10
        elif self.dataset_name == "fashion":
            image_shape = (1, 28, 28)
            num_classes = 10
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        if self.model_name == "vgg11":
            model = vgg11_bn(in_channels=image_shape[0],
                             drop_rate=drop_rate,
                             last_drop_rate=last_drop_rate,
                             num_classes=num_classes,
                             group_channels=group_channels)
        elif self.model_name == "vgg13":
            model = vgg13_bn(in_channels=image_shape[0],
                             drop_rate=drop_rate,
                             last_drop_rate=last_drop_rate,
                             num_classes=num_classes,
                             group_channels=group_channels)
        elif self.model_name == "vgg16":
            model = vgg16_bn(in_channels=image_shape[0],
                             drop_rate=drop_rate,
                             last_drop_rate=last_drop_rate,
                             num_classes=num_classes,
                             group_channels=group_channels)
        elif self.model_name == "resnet32":
            from model import resnet32
            model = resnet32(in_channel=image_shape[0],
                             drop_rate=drop_rate,
                             last_drop_rate=last_drop_rate,
                             num_classes=num_classes,
                             norm_layer=torch.nn.GroupNorm,
                             group_channels=group_channels)
        elif self.model_name == "resnet50":
            raise NotImplementedError()
            # model = resnet50(in_channel=image_shape[0],
            #                  drop_rate=drop_rate,
            #                  last_drop_rate=last_drop_rate,
            #                  num_classes=num_classes,
            #                  norm_layer=torch.nn.GroupNorm,
            #                  group_channels=group_channels)
        elif self.model_name == "densenet121":
            model = densenet121(in_channels=image_shape[0],
                                drop_rate=drop_rate,
                                last_drop_rate=last_drop_rate,
                                num_classes=num_classes,
                                group_channels=group_channels)
        elif self.model_name == "densenet161":
            model = densenet161(in_channels=image_shape[0],
                                drop_rate=drop_rate,
                                last_drop_rate=last_drop_rate,
                                num_classes=num_classes,
                                group_channels=group_channels)
        elif self.model_name == "densenet169":
            model = densenet169(in_channels=image_shape[0],
                                drop_rate=drop_rate,
                                last_drop_rate=last_drop_rate,
                                num_classes=num_classes,
                                group_channels=group_channels)
        elif self.model_name == "logistics":
            model = LogisticsModel(in_features=np.prod(image_shape),
                                   num_classes=num_classes)
        else:
            raise ValueError("Unknonw model name: %s" % self.model_name)

        if TORCH_SUMMARY_STRING:
            # print to log
            summary, _ = summary_string(model, image_shape,
                                        device="cpu",
                                        dtypes=[torch.float]*len(image_shape))
            logging.info("===== Model Summary =====\n%s" % summary)

        for n, l in model.named_modules():
            if isinstance(l, torch.nn.GroupNorm):
                logging.info("%s: %s, groups: %d, channels: %d, eps: %f, affine: %s" % (
                    n, str(type(l)), l.num_groups, l.num_channels, l.eps, l.affine))
            else:
                logging.info("%s: %s" % (n, str(type(l))))

        model_size = 0
        b = 32/8
        for n, p in model.named_parameters():
            param_size = np.prod(p.size())*b
            logging.info("parameter size: %s %dB" % (n, param_size))
            model_size += param_size
        logging.info("Model: %s size: %.3fMiB" %
                     (self.model_name, model_size/(1024*1024)))

        return model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optim__

    def load_datas(self, indices: typing.List[int] = None, train_data_length: int = None):
        # uint8(0-255)を0-1に変換
        transforms = [torchvision.transforms.ToTensor()]
        if self.dataset_name == "cifar10":
            transforms.append(torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transforms = torchvision.transforms.Compose(transforms)

        if self.dataset_name == "cifar10":
            trains = CIFAR10(self.datadir, train=True,
                             download=True, transform=transforms,
                             indices=indices, data_length=train_data_length,
                             shuffle=True)
            vals = CIFAR10(self.datadir, train=True,
                           download=True, transform=transforms)
            tests = CIFAR10(self.datadir, train=False,
                            download=True, transform=transforms)
        elif self.dataset_name == "fashion":
            trains = FashionMNIST(self.datadir, train=True,
                                  download=True, transform=transforms,
                                  indices=indices, data_length=train_data_length,
                                  shuffle=True)
            vals = FashionMNIST(self.datadir, train=True,
                                download=True, transform=transforms)
            tests = FashionMNIST(self.datadir, train=False,
                                 download=True, transform=transforms)
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        return OrderedDict([("train", trains), ("val", vals), ("test", tests)])

    def build_criterion(self, reduction: str = "mean", *args, **kwargs):
        return torch.nn.CrossEntropyLoss(reduction=reduction)

    def build_optimizer(self, optimizer: str = "sgd", args: typing.Dict = {}):
        if optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), **args)
        elif optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), **args)
        else:
            raise ValueError("Unknonw optimizer: %s" % (optimizer))

    def epoch(self, dataloader, criterion, optimizer,
              l2_lambda: float = 0.01, **kwargs):
        losses = OrderedDict([("loss", 0.0)])
        ndata, nbatch = 0, 0
        l2_loss = 0.0
        total_exchange_time = 0.0
        for i, (x, y) in enumerate(dataloader):
            x = x.float().to(self.device)
            y = y.to(self.device)
            o = self.model(x)

            loss = criterion(o, y).sum(dim=0)
            l2 = torch.tensor(0.0).to(self.device)
            for param in self.model.parameters():
                l2 += ((0.5*torch.sum(param**2)))
            if loss.requires_grad and optimizer is not None:
                optimizer.zero_grad()
                update_loss = loss.sum() + (l2 * l2_lambda)
                update_loss.backward()
                start = datetime.now()
                optimizer.step()
                total_exchange_time += (datetime.now() - start).total_seconds()
            loss = loss.sum().detach().cpu().item()
            losses["loss"] += loss
            l2_loss = l2.detach().cpu().item()

            ndata += len(x)
            nbatch += 1
            logging.debug("[%04d] loss: %.3f, l2 loss: %.3f" %
                          (i, loss, l2_loss))
            del x, y, o, loss
        if criterion.reduction == "mean":
            losses = OrderedDict([(k, v/nbatch) for k, v in losses.items()])
        else:  # reduction is sum or none
            losses = OrderedDict([(k, v/ndata) for k, v in losses.items()])
        losses["l2"] = l2_loss
        losses["excange_proc"] = total_exchange_time
        return losses

    def metric(self, dataloader, batch_size: int = 64, output: str = None,
               **kwargs):
        scores, labels = [], []
        for x, y in dataloader:
            x = x.float().to(self.device)
            o = self.model(x)
            labels.append(y.detach().cpu().numpy())
            scores.append(o.detach().cpu().numpy())
            del x, y, o
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)

        probs = 1/(1+np.exp(-scores))  # convert score to probavility.
        preds = np.argmax(probs, axis=1)
        metrics = OrderedDict([("acc", accuracy_score(labels, preds))])

        if output is not None:
            # columns: true label, predict label, sequence of prob per each label...
            outputs = OrderedDict([("label", labels)] +
                                  [("predict", preds)] +
                                  [(f"prob_{i}", probs[:, i]) for i in range(probs.shape[1])])
            outputs = pd.DataFrame(outputs)
            outputs.to_csv(output, index=False)
        return metrics

    def dispose(self):
        import gc
        if hasattr(self.__optim__, "notice_train_ending"):
            self.__optim__.notice_train_ending()
        del self.__optim__
        gc.collect()
        logging.info("GC: check garbage %s" % (str(gc.garbage)))
        gc.collect()

    def train(self, epochs: int, batch_size: int,
              optimizer: str = "sgd",
              optim_args: typing.Dict = {},
              scheduler_name: str = None,
              scheduler_args: list = [],
              scheduler_kwargs: dict = {},
              adameps: float = 1e-8,
              **kwargs):
        loaders = OrderedDict([(k, torch.utils.data.DataLoader(v,
                                                               batch_size=batch_size,
                                                               shuffle=(k == "train")))
                               for k, v in self.datasets.items()])

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        criterion = self.build_criterion(**kwargs)
        self.__optim__ = self.build_optimizer(optimizer=optimizer,
                                              args=optim_args)
        optimizer = self.__optim__
        if scheduler_name is not None and hasattr(torch.optim.lr_scheduler, scheduler_name):
            last_epoch = scheduler_kwargs["last_epoch"] if "last_epoch" in scheduler_kwargs else -1
            if not last_epoch == -1 and not scheduler_name == "ReduceLROnPlateau":
                # initial_lrがoptimizerのパラメータとして必要なので、追加で設定する。
                # ReduceLROnPlateauだけは例外
                for group in optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
            scheduler_ctor = getattr(torch.optim.lr_scheduler, scheduler_name)
            scheduler = scheduler_ctor(optimizer,
                                       *scheduler_args, **scheduler_kwargs)
            logging.info("Setup scheduler: %s" % (str(scheduler)))
        else:
            scheduler = None

        if hasattr(optimizer, "edges"):
            num_edges = len(
                [edge for edge in optimizer.edges() if edge.is_connected])
        else:
            num_edges = 0
        logging.debug("[%4d/%4d] connecting edge count: %d" % (
            0, epochs, num_edges))

        random_state = np.random.RandomState(self.seed)
        with open(self.train_log, "wt") as f:
            for epoch in range(1, epochs+1):
                logs = OrderedDict([("epoch", epoch)])
                start = datetime.now()

                epoch_losses, epoch_metrics = OrderedDict(), OrderedDict()
                # train
                self.model.train()
                epoch_losses["train"] = self.epoch(loaders["train"],
                                                   criterion,
                                                   optimizer,
                                                   **kwargs)
                train_proc = (datetime.now() - start).total_seconds()
                if scheduler is not None and hasattr(scheduler, "step"):
                    scheduler.step()

                if hasattr(optimizer, "edges"):
                    curr_num_edges = len(
                        [edge for edge in optimizer.edges() if edge.is_connected])
                    logging.info("[%4d/%4d] connecting edge count: %d" % (
                        epoch, epochs, curr_num_edges))
                else:
                    curr_num_edges = 0

                # unset data augmentator
                if isinstance(loaders["train"].dataset, torch.utils.data.ConcatDataset):
                    for d in loaders["train"].dataset.datasets:
                        d.data_aug = None
                else:
                    loaders["train"].dataset.data_aug = None

                # validation and test
                start = datetime.now()
                self.model.eval()
                with torch.no_grad():
                    for k, loader in loaders.items():
                        if not k == "train":
                            epoch_losses[k] = self.epoch(loader, criterion, None,
                                                         **kwargs)
                        epoch_metrics[k] = self.metric(loader,
                                                       batch_size=batch_size,
                                                       output=None,
                                                       **kwargs)
                eval_proc = (datetime.now() - start).total_seconds()

                # merge train, test loss and metrics
                for phase, losses in epoch_losses.items():
                    for k, v in losses.items():
                        if "loss" in k or "proc" in k:
                            logs["%s_%s" % (phase, k)] = v
                        else:
                            logs["%s_%s_loss" % (phase, k)] = v
                for phase, metrics in epoch_metrics.items():
                    for k, v in metrics.items():
                        logs["%s_%s" % (phase, k)] = v
                if hasattr(optimizer, "diff"):
                    diff = optimizer.diff()
                    if isinstance(diff, torch.Tensor):
                        diff = diff.detach().cpu().item()
                else:
                    diff = 0.0
                logs["diff"] = diff
                logs["train_proc"] = train_proc
                logs["eval_proc"] = eval_proc
                if hasattr(optimizer, "get_communication_time"):
                    logs["comm_proc"] = optimizer.get_communication_time()
                else:
                    logs["comm_proc"] = 0.0
                logs["timestamp"] = datetime.now()

                # write log file.
                logging.info("[%4d/%4d] loss: (train=%s, val=%s, test=%s, l2=%s), acc: (train=%s, val=%s, test=%s), diff=%.8f, proc(train=%.3fsec, eval=%.3fsec)" % (
                    epoch, epochs,
                    *["%.3f" % logs[k] for k in ["train_loss", "val_loss", "test_loss", "train_l2_loss",
                                                 "train_acc", "val_acc", "test_acc"]],
                    diff, train_proc, eval_proc))

                sleep_proc = (train_proc * (0.5 + 0.5 *
                                            self.sleep_factor * random_state.rand()))
                logs["sleep_proc"] = sleep_proc if self.sleep_factor > 0.0 else 0.0
                if self.sleep_factor > 0.0 and sleep_proc > 0.0:
                    logging.info("[%4d/%4d] Sleep %fsec" %
                                 (epoch, epochs, sleep_proc))
                    sleep(sleep_proc)

                if num_edges > curr_num_edges:
                    logging.info("[%4d/%4d] found edge disconnection: %d -> %d. finished train." % (
                        epoch, epochs, num_edges, curr_num_edges))
                    break
                num_edges = np.max([num_edges, curr_num_edges])

                # write train log.
                if epoch == 1:
                    f.write(",".join(logs.keys()))
                    f.write("\n")
                f.write(",".join(map(str, logs.values())))
                f.write("\n")

    def evaluate(self, batch_size: int,
                 optimizer: str = "sgd", lr: float = 0.01,
                 **kwargs):
        # 学習用の乱数初期化前にデータセットを初期化する
        dataloaders = OrderedDict([(k, torch.utils.data.DataLoader(v,
                                                                   batch_size=batch_size,
                                                                   shuffle=False))
                                   for k, v in self.datasets.items()])

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        criterion = self.build_criterion(**kwargs)
        self.model.eval()

        outdir = self.outdir
        os.makedirs(outdir, exist_ok=True)
        with open(self.eval_log, "wt") as f, torch.no_grad():
            logs = OrderedDict([("epoch", 0)])
            start = datetime.now()
            eval_losses, eval_metrics = OrderedDict(), OrderedDict()
            for k, loader in dataloaders.items():
                eval_losses[k] = self.epoch(loader, criterion, None,
                                            **kwargs)

                eval_metrics[k] = self.metric(loader,
                                              batch_size=batch_size,
                                              output=path.join(outdir,
                                                               self.score_filenames[k]),
                                              **kwargs)
            proc = (datetime.now() - start).total_seconds()

            # merge train, test loss and metrics
            for phase, losses in eval_losses.items():
                for k, v in losses.items():
                    if k == "loss":
                        logs["%s_loss" % (phase)] = v
                    else:
                        logs["%s_%s_loss" % (phase, k)] = v
            for phase, metrics in eval_metrics.items():
                for k, v in metrics.items():
                    logs["%s_%s" % (phase, k)] = v
            logs["proc"] = proc

            # write header.
            f.write(",".join(logs.keys()))
            f.write("\n")
            # write eval log.
            f.write(",".join(map(str, logs.values())))
            f.write("\n")
            logging.info("[EVAL] loss: (train=%s, val=%s, test=%s, l2=%s), acc: (train=%s, val=%s, test=%s), proc=%.3fsec" % (
                *["%.3f" % logs[k] for k in ["train_loss", "val_loss", "test_loss", "train_l2_loss",
                                             "train_acc", "val_acc", "test_acc"]],
                proc))

    def save_state_dict(self, **kwrags):
        torch.save(self.model.state_dict(), self.state_dict_file)

    def plot_figs(self, **kwargs):
        outdir = path.join(self.outdir, "imgs")
        os.makedirs(outdir, exist_ok=True)

        # read train log
        train_log = pd.read_csv(self.train_log)
        # read predict files.
        pred_vals = OrderedDict([(k, pd.read_csv(
            path.join(self.outdir, self.score_filenames[k]))) for k in self.datasets.keys()])

        # plot loss, acc
        for metric in ["loss", "acc", "l2_loss"]:
            figname = path.join(outdir, "%s%s%s.png" % (
                self.fig_prefix, metric, self.fig_suffix))
            for k in self.datasets.keys():
                plt.plot(train_log["%s_%s" % (k, metric)].values, label=k)
            plt.title(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()
