#! -*- coding: utf-8
import gc
import json
import logging
import os
import os.path as path
import sys
from argparse import ArgumentParser
from enum import IntEnum

import numpy as np

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)


LOGFORMAT = "[%(asctime)s] <%(levelname)s> %(message)s"


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def name_of(cls, name: str):
        name = name.upper()
        for e in cls:
            if e.name == name:
                return e
        raise ValueError(f"Unknown {cls.__name__} name: {name}")

    @classmethod
    def value_of(cls, value: int):
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown {cls.__name__} value: {value}")

    def __str__(self):
        return self.name.upper()


def config_logger(logformat: str = LOGFORMAT,
                  loglevel: LogLevel = LogLevel.INFO,
                  logfile: str = None):

    if logfile is None:
        logging.basicConfig(level=loglevel, format=logformat,
                            stream=sys.stderr)  # ログファイル未指定の場合、標準エラーに出力
    else:
        logdir = path.dirname(logfile)
        os.makedirs(logdir, exist_ok=True)
        logging.basicConfig(level=loglevel, format=logformat,
                            filename=logfile)  # ログファイル未指定の場合、標準エラーに出力


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datadir", type=str, help="データセットのパス")
    parser.add_argument("outdir", type=str, help="出力ディレクトリ")

    parser.add_argument("--epochs", type=int, default=1000, help="学習回数")
    parser.add_argument("--batch_size", type=int, default=64, help="バッチサイズ")
    parser.add_argument("--seed", type=int, default=569, help="学習用の乱数種")
    parser.add_argument("--data_init_seed", type=int,
                        default=11, help="データセット初期化時の乱数種")
    parser.add_argument("--model_init_seed", type=int,
                        default=13, help="モデル初期化時の乱数種")
    parser.add_argument("--l2_lambda", type=float, default=0.001,
                        help="L2正則化項の係数")
    parser.add_argument("--model_name", type=str, default="resnet32",
                        choices=["logistics", "resnet32"])
    parser.add_argument("--dataset_name", type=str,
                        default="cifar10",
                        choices=["cifar10", "fashion"])
    parser.add_argument("--train_data_length", type=int, default=12800,
                        help="train_data_length / batch_size = 1epoch = n roundとする。")
    parser.add_argument("--group_channels", type=int, default=32)
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="dropout層のdropout rate")
    parser.add_argument("--last_drop_rate", type=float, default=0.5,
                        help="最後のdropout層のdropout rate")
    parser.add_argument("--cpu", dest="cuda", default=True, action="store_false",
                        help="CPUを使用して学習する。デフォルトはCUDAを使用する。")
    parser.add_argument("--cuda_device_no", type=int, default=0)
    parser.add_argument("--skip_plots", dest="plot", default=True, action="store_false",
                        help="指定時、各種画像のプロット処理をスキップする")

    parser.add_argument("--scheduler", type=str,
                        default="none", choices=["none", "StepLR"])
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.90)

    parser.add_argument("--sleep_factor", type=float, default=0.0)

    parser.add_argument("--loglevel",
                        type=lambda x: LogLevel.name_of(x),
                        default=LogLevel.INFO,
                        help="ログ出力レベルを指定")
    parser.add_argument("--logfile", type=str, default=None,
                        help="ログ出力先。未指定の場合、標準エラーに出力する")

    optim_parsers = parser.add_subparsers(title="optimizer", dest="optimizer")
    optim = optim_parsers.add_parser("sgd")
    optim.add_argument("--lr", type=float, default=0.002)
    optim.add_argument("--momentum", type=float, default=0.0)
    optim.add_argument("--dampening", type=float, default=0.0)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--nesterov", default=False, action="store_true")
    optim = optim_parsers.add_parser("adam")
    optim.add_argument("--lr", type=float, default=0.002)
    optim.add_argument("--betas", type=float, default=(0.9, 0.999), nargs=2)
    optim.add_argument("--eps", type=float, default=1e-8)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--amsgrad", default=False, action="store_true")
    optim = optim_parsers.add_parser("PdmmISVR")
    optim.add_argument("nodename", type=str)
    optim.add_argument("conf", type=str)
    optim.add_argument("host", type=str)
    optim.add_argument("--lr", type=float, default=0.002)
    optim.add_argument("--use_gcoef", default=False, action="store_true")
    optim.add_argument("--piw", type=float, default=1.0,
                       choices=[0.5, 1.0, 2.0])
    optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
    optim.add_argument("--swap_timeout", type=int, default=100)
    optim = optim_parsers.add_parser("AdmmISVR")
    optim.add_argument("nodename", type=str)
    optim.add_argument("conf", type=str)
    optim.add_argument("host", type=str)
    optim.add_argument("--lr", type=float, default=0.002)
    optim.add_argument("--use_gcoef", default=False, action="store_true")
    optim.add_argument("--piw", type=float, default=1.0,
                       choices=[0.5, 1.0, 2.0])
    optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
    optim.add_argument("--swap_timeout", type=int, default=1000)
    optim = optim_parsers.add_parser("DSgd")
    optim.add_argument("nodename", type=str)
    optim.add_argument("conf", type=str)
    optim.add_argument("host", type=str)
    optim.add_argument("--lr", type=float, default=0.002)
    optim.add_argument("--momentum", type=float, default=0.0)
    optim.add_argument("--dampening", type=float, default=0.0)
    optim.add_argument("--weight_decay", type=float, default=0.0)
    optim.add_argument("--nesterov", default=False, action="store_true")
    optim.add_argument("--weight", type=float, default=1.0)
    optim.add_argument("--async_step", dest="round_step",
                       default=True, action="store_false")
    optim.add_argument("--swap_timeout", type=int, default=1)

    args = parser.parse_args()

    config_logger(loglevel=args.loglevel, logfile=args.logfile)
    logging.info(args)

    optim_args = {}
    if args.optimizer == "sgd":
        optim_args["lr"] = args.lr
        optim_args["momentum"] = args.momentum
        optim_args["weight_decay"] = args.weight_decay
        optim_args["dampening"] = args.dampening
        optim_args["nesterov"] = args.nesterov
    elif args.optimizer == "adam":
        optim_args["lr"] = args.lr
        optim_args["betas"] = args.betas
        optim_args["eps"] = args.eps
        optim_args["weight_decay"] = args.weight_decay
        optim_args["amsgrad"] = args.amsgrad
    elif args.optimizer == "PdmmISVR":
        optim_args["lr"] = args.lr
        optim_args["round_step"] = args.round_step
        optim_args["use_gcoef"] = args.use_gcoef
        optim_args["drs"] = False
        optim_args["piw"] = args.piw
        optim_args["swap_timeout"] = args.swap_timeout
    elif args.optimizer == "AdmmISVR":
        optim_args["lr"] = args.lr
        optim_args["round_step"] = args.round_step
        optim_args["use_gcoef"] = args.use_gcoef
        optim_args["drs"] = True
        optim_args["piw"] = args.piw
        optim_args["swap_timeout"] = args.swap_timeout
    elif args.optimizer == "DSgd":
        optim_args["lr"] = args.lr
        optim_args["momentum"] = args.momentum
        optim_args["weight_decay"] = args.weight_decay
        optim_args["dampening"] = args.dampening
        optim_args["nesterov"] = args.nesterov
        optim_args["weight"] = args.weight
        optim_args["round_step"] = args.round_step
        optim_args["swap_timeout"] = args.swap_timeout

    scheduler_args = []
    scheduler_kwargs = {}
    if args.scheduler == "none":
        pass
    elif args.scheduler == "StepLR":
        scheduler_args.append(args.step_size)
        scheduler_kwargs["gamma"] = args.gamma

    if args.optimizer in ["sgd", "adam"]:
        from trainer import Trainer
        trainer = Trainer(args.outdir, args.seed,
                          datadir=args.datadir,
                          dataset_name=args.dataset_name,
                          model_name=args.model_name,
                          group_channels=args.group_channels,
                          drop_rate=args.drop_rate, last_drop_rate=args.last_drop_rate,
                          data_init_seed=args.data_init_seed, model_init_seed=args.model_init_seed,
                          cuda=args.cuda, cuda_device_no=args.cuda_device_no)
    elif args.optimizer in ["PdmmISVR", "AdmmISVR", "DSgd"]:
        from trainer import PdmmTrainer as Trainer

        with open(args.conf) as f:
            conf = json.load(f)
        logging.info(conf)
        nodes = conf["nodes"][args.nodename]
        edges = nodes["edges"]
        optim_args["round"] = nodes["round"]
        with open(args.host) as f:
            hosts = json.load(f)["hosts"]

        # 分散学習を行う場合、学習用のseedには異なる値を使用する
        nodeidx = sorted([n["name"] for n in hosts]).index(args.nodename)
        np.random.seed(args.seed)
        seed = np.random.randint(0, 25485227, len(hosts))[nodeidx]

        trainer = Trainer(args.outdir,
                          args.nodename, edges, hosts,
                          seed,
                          datadir=args.datadir,
                          dataset_name=args.dataset_name,
                          model_name=args.model_name,
                          group_channels=args.group_channels,
                          drop_rate=args.drop_rate, last_drop_rate=args.last_drop_rate,
                          data_init_seed=args.data_init_seed, model_init_seed=args.model_init_seed,
                          train_data_length=args.train_data_length,
                          cuda=args.cuda,
                          cuda_device_no=args.cuda_device_no)
        trainer.sleep_factor = args.sleep_factor
    else:
        raise ValueError("Unknown optimizer: %s" % (args.optimizer))
    trainer.train(args.epochs, args.batch_size,
                  optimizer=args.optimizer, optim_args=optim_args,
                  scheduler_name=args.scheduler, scheduler_args=scheduler_args, scheduler_kwargs=scheduler_kwargs,
                  l2_lambda=args.l2_lambda)
    trainer.evaluate(args.batch_size)
    trainer.dispose()
    trainer.save_state_dict()
    if args.plot:
        trainer.plot_figs()
    del trainer
    gc.collect()
    logging.info("GC: check garbage %s" % (str(gc.garbage)))
    gc.collect()
