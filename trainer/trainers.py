#! -*- coding: utf-8
import logging
import os.path as path
import typing
from collections import OrderedDict

import numpy as np
import torchvision

from .abstracts import ATrainer

__all__ = ["Trainer", "PdmmTrainer"]


class Trainer(ATrainer):
    def __init__(self, outdir: str, seed: int,
                 datadir: str = "./datas",
                 dataset_name: str = "cifar10",
                 model_name: str = "resnet32",
                 group_channels: int = 32,
                 drop_rate: float = 0.1,
                 last_drop_rate: float = 0.5,
                 data_init_seed: int = 11,
                 model_init_seed: int = 13,
                 train_data_length: int = 12800,
                 cuda: bool = True,
                 cuda_device_no: int = 0, **kwargs):
        super(Trainer, self).__init__(outdir, seed,
                                      datadir=datadir,
                                      dataset_name=dataset_name,
                                      model_name=model_name,
                                      group_channels=group_channels,
                                      drop_rate=drop_rate,
                                      last_drop_rate=last_drop_rate,
                                      data_init_seed=data_init_seed,
                                      model_init_seed=model_init_seed,
                                      train_data_length=train_data_length,
                                      cuda=cuda,
                                      cuda_device_no=cuda_device_no,
                                      **kwargs)


class PdmmTrainer(ATrainer):
    def __init__(self, outdir: str,
                 nodename: str, edges: typing.Dict, hosts: typing.Dict,
                 seed: int,
                 datadir: str = "./datas",
                 dataset_name: str = "cifar10",
                 model_name: str = "resnet32",
                 group_channels: int = 32,
                 drop_rate: float = 0.1,
                 last_drop_rate: float = 0.5,
                 data_init_seed: int = 11,
                 model_init_seed: int = 13,
                 train_data_length: int = 12800,
                 cuda: bool = True,
                 cuda_device_no: int = 0,
                 **kwargs):
        super(PdmmTrainer, self).__init__(outdir, seed,
                                          datadir=datadir,
                                          dataset_name=dataset_name,
                                          model_name=model_name,
                                          group_channels=group_channels,
                                          drop_rate=drop_rate, last_drop_rate=last_drop_rate,
                                          data_init_seed=data_init_seed,
                                          model_init_seed=model_init_seed,
                                          train_data_length=train_data_length,
                                          cuda=cuda,
                                          cuda_device_no=cuda_device_no,
                                          **kwargs)
        self.__nodename__ = nodename
        self.__nodeindex__ = sorted([n["name"] for n in hosts]).index(nodename)
        self.__hosts__ = hosts
        self.__edges__ = edges
        self.__common_classes__ = 0
        self.__node_per_classes__ = 8
        self.__class_inbalanced_lambda__ = 1.0
        self.__data_inbalanced_lambda__ = 0.2
        self.__nshift_of_nodes__ = 4
        self.__data_split_mode__ = "split"

    @property
    def train_log(self): return path.join(self.outdir,
                                          "%s.train.log" % (self.nodename))

    @property
    def eval_log(self): return path.join(self.outdir,
                                         "%s.eval.log" % (self.nodename))

    @property
    def state_dict_file(self): return path.join(self.outdir,
                                                "%s.model.pth" % (self.nodename))

    @property
    def model_file(self): return path.join(self.outdir,
                                           "%s.model.pt" % (self.nodename))

    @property
    def score_filenames(self): return {k: "%s.scores_%s.csv" % (self.nodename, k)
                                       for k in self.datasets}

    @property
    def fig_prefix(self): return "%s." % (self.nodename)
    @property
    def fig_suffix(self): return ""

    @property
    def nodename(self): return self.__nodename__
    @property
    def nodeindex(self): return self.__nodeindex__
    @property
    def hosts(self): return self.__hosts__
    @property
    def edges(self): return self.__edges__
    @property
    def common_classes(self): return self.__common_classes__
    @property
    def node_per_classes(self): return self.__node_per_classes__
    @property
    def nshift_of_nodes(self): return self.__nshift_of_nodes__
    @property
    def data_split_mode(self): return self.__data_split_mode__
    @property
    def class_inbalanced_lambda(self): return self.__class_inbalanced_lambda__
    @property
    def data_inbalanced_lambda(self): return self.__data_inbalanced_lambda__

    def build_optimizer(self,
                        optimizer: str = "PdmmISVR",
                        args: typing.Dict = {}):
        round_cnt = args["round"]
        del args["round"]
        if optimizer in ["PdmmISVR", "AdmmISVR"]:
            from edgecons import Ecl
            return Ecl(self.nodename,
                       round_cnt,
                       self.edges,
                       self.hosts,
                       self.model,
                       device=self.device,
                       **args)
        elif optimizer == "DSgd":
            from edgecons import DSgd
            return DSgd(self.nodename,
                        round_cnt,
                        self.edges,
                        self.hosts,
                        self.model,
                        device=self.device,
                        **args)
        else:
            raise ValueError("Unknonw optimizer: %s" % (optimizer))

    def load_datas(self,
                   indices: typing.List[int] = None,
                   train_data_length: int = None):
        if self.dataset_name == "cifar10":
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif self.dataset_name == "fashion":
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        if self.data_split_mode == "split":
            return self.split_load_datas(indices=indices, train_data_length=train_data_length)
        elif self.data_split_mode == "same":
            return super(PdmmTrainer, self).load_datas(indices=indices, train_data_length=train_data_length)
        elif self.data_split_mode == "split_class_data_same":
            return self.split_class_data_same_load_datas(indices=indices, train_data_length=train_data_length)
        elif self.data_split_mode == "split_class_data_even":
            return self.split_class_data_even_load_datas(indices=indices, train_data_length=train_data_length)
        else:
            raise ValueError("Unsupported data split mode: %s" %
                             (self.data_split_mode))

    def split_load_datas(self,
                         indices: typing.List[int] = None,
                         train_data_length: int = None):
        if self.dataset_name == "cifar10":
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif self.dataset_name == "fashion":
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        assert num_classes - self.common_classes - self.node_per_classes > 0
        class_indices = list(range(num_classes))
        common_class_indices = class_indices[:self.common_classes]
        split_class_indices = class_indices[self.common_classes:]

        node_to_class_idx = [[] for i in range(len(self.hosts))]
        class_to_node_idx = [[] for i in range(num_classes)]
        class_data_weights = []
        # 共通クラスはノード間のデータ数も同じにする
        for class_idx in common_class_indices:
            for node_idx, node_classes in enumerate(node_to_class_idx):
                node_classes.append(class_idx)
                class_to_node_idx[class_idx].append(node_idx)
            class_data_weights.append(
                np.array([1.0/len(self.hosts)] * len(self.hosts)))

        if self.node_per_classes > 0:
            assert self.node_per_classes < len(split_class_indices)
            assert self.nshift_of_nodes > 0
            np.random.shuffle(split_class_indices)
            for node_idx, node_indices in enumerate(node_to_class_idx):
                sidx = (node_idx * self.nshift_of_nodes) % (len(split_class_indices))
                eidx = sidx + self.node_per_classes
                class_indices = split_class_indices[sidx:eidx]
                if eidx > len(split_class_indices):
                    class_indices += split_class_indices[:(
                        eidx-len(split_class_indices))]
                node_indices.extend(class_indices)
                for class_idx in class_indices:
                    class_to_node_idx[class_idx].append(node_idx)

        for node_idxs in class_to_node_idx[len(class_data_weights):]:
            w = (np.random.rand(len(node_idxs)) * self.data_inbalanced_lambda +
                 (1.0 - self.data_inbalanced_lambda))
            w /= np.sum(w)
            class_data_weights.append(w)

        # logging
        for node_idx, class_idxs in enumerate(node_to_class_idx):
            logging.info("node%d: [%s]" %
                         (node_idx, ",".join(map(str, class_idxs))))
        for class_idx, node_idxs in enumerate(class_to_node_idx):
            logging.info("class %d, [%s]" % (class_idx,
                                             ",".join(["node%d:%.3f" % (n, class_data_weights[class_idx][i])
                                                       for i, n in enumerate(node_idxs)])))

        class_data_idxs = [np.where(targets == i)[0]
                           for i in range(num_classes)]
        node_data_indices = [[] for _ in range(len(self.hosts))]
        for class_idx, (node_idxs, node_weights, class_datas) in enumerate(zip(class_to_node_idx,
                                                                               class_data_weights,
                                                                               class_data_idxs)):
            # 各ノードごとのデータ数を計算
            ndata_of_nodes = (node_weights * len(class_datas)).astype(np.int)
            offsets = []
            offset = 0
            for n in ndata_of_nodes:
                offset += n
                offsets.append(offset)
            offsets[-1] = len(class_datas)  # 端数は末尾に押し込む
            offset = 0
            for node_idx, end_offset in zip(node_idxs, offsets):
                node_data_indices[node_idx].append(
                    class_datas[offset:end_offset])
                offset = end_offset

        node_data_indices = [np.concatenate(d, axis=0)
                             for d in node_data_indices]

        # write train data summary...
        for node_idx, data_indices in enumerate(node_data_indices):
            node_targets = targets[data_indices]
            classes, counts = np.unique(node_targets, return_counts=True)
            logging.info("train data node%d: %s" % (node_idx,
                                                    ",".join(["%d:%d" % (i, n) for i, n in zip(classes, counts)])))
        node_indices = node_data_indices[self.nodeindex]
        datasets = super(PdmmTrainer, self).load_datas(indices=node_indices,
                                                       train_data_length=train_data_length)
        return datasets

    def split_class_data_same_load_datas(self,
                                         indices: typing.List[int] = None,
                                         train_data_length: int = None):
        if self.dataset_name == "cifar10":
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif self.dataset_name == "fashion":
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        assert num_classes - self.common_classes - self.node_per_classes > 0
        class_indices = list(range(num_classes))
        common_class_indices = class_indices[:self.common_classes]
        split_class_indices = class_indices[self.common_classes:]

        node_to_class_idx = [[] for i in range(len(self.hosts))]
        class_to_node_idx = [[] for i in range(num_classes)]
        # class_data_weights = []
        # 共通クラスはノード間のデータ数も同じにする
        for class_idx in common_class_indices:
            for node_idx, node_classes in enumerate(node_to_class_idx):
                node_classes.append(class_idx)
                class_to_node_idx[class_idx].append(node_idx)
            # class_data_weights.append(
            #     np.array([1.0/len(self.hosts)] * len(self.hosts)))

        if self.node_per_classes > 0:
            assert self.node_per_classes < len(split_class_indices)
            assert self.nshift_of_nodes > 0
            np.random.shuffle(split_class_indices)
            for node_idx, node_indices in enumerate(node_to_class_idx):
                sidx = (node_idx * self.nshift_of_nodes) % (len(split_class_indices))
                eidx = sidx + self.node_per_classes
                class_indices = split_class_indices[sidx:eidx]
                if eidx > len(split_class_indices):
                    class_indices += split_class_indices[:(
                        eidx-len(split_class_indices))]
                node_indices.extend(class_indices)
                for class_idx in class_indices:
                    class_to_node_idx[class_idx].append(node_idx)

        # logging
        for node_idx, class_idxs in enumerate(node_to_class_idx):
            logging.info("node%d: [%s]" %
                         (node_idx, ",".join(map(str, class_idxs))))
        for class_idx, node_idxs in enumerate(class_to_node_idx):
            logging.info("class %d, [%s]" % (class_idx,
                                             ",".join(["node%d:%.3f" % (n, 1.0)
                                                       for i, n in enumerate(node_idxs)])))

        class_data_idxs = [np.where(targets == i)[0]
                           for i in range(num_classes)]
        node_data_indices = [[] for _ in range(len(self.hosts))]
        for node_idxs, class_datas in zip(class_to_node_idx,
                                          class_data_idxs):
            for node_idx in node_idxs:
                node_data_indices[node_idx].append(class_datas)

        node_data_indices = [np.concatenate(d, axis=0)
                             for d in node_data_indices]

        # write train data summary...
        for node_idx, data_indices in enumerate(node_data_indices):
            node_targets = targets[data_indices]
            classes, counts = np.unique(node_targets, return_counts=True)
            logging.info("train data node%d: %s" % (node_idx,
                                                    ",".join(["%d:%d" % (i, n) for i, n in zip(classes, counts)])))
        node_indices = node_data_indices[self.nodeindex]
        datasets = super(PdmmTrainer, self).load_datas(indices=node_indices,
                                                       train_data_length=train_data_length)
        return datasets

    def split_class_data_even_load_datas(self,
                                         indices: typing.List[int] = None,
                                         train_data_length: int = None):
        dataset_type = self.dataset_type(self.dataset_name)
        if dataset_type == 0:
            trains = torchvision.datasets.CIFAR10(self.datadir, train=True,
                                                  download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif dataset_type == 1:
            trains = torchvision.datasets.EMNIST(self.datadir, "letters", train=True,
                                                 download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        elif dataset_type == 2:
            trains = torchvision.datasets.FashionMNIST(self.datadir, train=True,
                                                       download=True)
            num_classes = len(trains.classes)
            class_to_idx = trains.class_to_idx
            targets = np.asarray(trains.targets)
        else:
            raise ValueError("Unknown dataset name: %s" % (self.dataset_name))

        np.random.seed(self.data_init_seed)
        assert num_classes - self.common_classes - self.node_per_classes > 0
        class_indices = list(range(num_classes))
        common_class_indices = class_indices[:self.common_classes]
        split_class_indices = class_indices[self.common_classes:]

        node_to_class_idx = [[] for i in range(len(self.hosts))]
        class_to_node_idx = [[] for i in range(num_classes)]
        # class_data_weights = []
        # 共通クラスはノード間のデータ数も同じにする
        for class_idx in common_class_indices:
            for node_idx, node_classes in enumerate(node_to_class_idx):
                node_classes.append(class_idx)
                class_to_node_idx[class_idx].append(node_idx)

        if self.node_per_classes > 0:
            assert self.node_per_classes < len(split_class_indices)
            assert self.nshift_of_nodes > 0
            np.random.shuffle(split_class_indices)
            for node_idx, node_indices in enumerate(node_to_class_idx):
                sidx = (node_idx * self.nshift_of_nodes) % (len(split_class_indices))
                eidx = sidx + self.node_per_classes
                class_indices = split_class_indices[sidx:eidx]
                if eidx > len(split_class_indices):
                    class_indices += split_class_indices[:(
                        eidx-len(split_class_indices))]
                node_indices.extend(class_indices)
                for class_idx in class_indices:
                    class_to_node_idx[class_idx].append(node_idx)

        # logging
        for node_idx, class_idxs in enumerate(node_to_class_idx):
            logging.info("node%d: [%s]" %
                         (node_idx, ",".join(map(str, class_idxs))))
        for class_idx, node_idxs in enumerate(class_to_node_idx):
            logging.info("class %d, [%s]" % (class_idx,
                                             ",".join(["node%d:%.3f" % (n, 1.0)
                                                       for i, n in enumerate(node_idxs)])))

        class_data_idxs = [np.where(targets == i)[0]
                           for i in range(num_classes)]
        node_data_indices = [[] for _ in range(len(self.hosts))]
        for node_idxs, class_datas in zip(class_to_node_idx,
                                          class_data_idxs):
            # 各ノードごとのデータ数を計算
            splited = np.array_split(class_datas, len(node_idxs))
            for node_idx, n in zip(node_idxs, splited):
                node_data_indices[node_idx].append(n)

        node_data_indices = [np.concatenate(d, axis=0)
                             for d in node_data_indices]

        # write train data summary...
        for node_idx, data_indices in enumerate(node_data_indices):
            node_targets = targets[data_indices]
            classes, counts = np.unique(node_targets, return_counts=True)
            logging.info("train data node%d: %s" % (node_idx,
                                                    ",".join(["%d:%d" % (i, n) for i, n in zip(classes, counts)])))
        node_indices = node_data_indices[self.nodeindex]
        datasets = super(PdmmTrainer, self).load_datas(indices=node_indices,
                                                       train_data_length=train_data_length)
        return datasets
