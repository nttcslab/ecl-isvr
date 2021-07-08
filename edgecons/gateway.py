# -*- coding: utf-8 -*-
import logging
import os
import threading
from collections import OrderedDict

import torch.distributed as dist
import zmq
from torch.optim.optimizer import Optimizer

from edgecons.mnw.edge import Edge


class ServerHandler(threading.Thread):
    def __init__(self, name, edges, nodes, model, is_prm, info, device="cpu", swap_timeout=10000):
        super(ServerHandler, self).__init__()
        self._name = name
        self._edge = edges
        self._nodes = nodes
        self._device = device
        self._model = model
        self._is_prm = is_prm
        self._info = info
        self._swap_timeout = swap_timeout

        bind_addr = 'tcp://0.0.0.0:' + str(nodes[name].port)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(bind_addr)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def __del__(self):
        self._socket.close()
        self._context.destroy()

    def run(self):
        while self._info["node"]["training"]:
            if self._poller.poll(self._swap_timeout):
                idx = self._socket.recv_json(0)
                if idx["kind"] == "hello":
                    self.hello(idx["src"])
                elif idx["kind"] == "info":
                    self._nodes[idx["src"]].info = idx["info"]
                    self._socket.send_json(self._info["node"], 0)
                elif idx["kind"] == "function":
                    self._info["func"](idx["src"])
                    self._socket.send_json({"status": 200}, 0)
                else:
                    self.swap(idx["src"], idx["kind"])

    def hello(self, name, weight=1.0):
        if name not in self._edge.keys():
            self._nodes[name].weight = weight
            self._edge[name] = Edge(self._nodes[name], self._model.parameters(), -1, self._device, self._is_prm)

        if not self._edge[name].is_connected:
            self._edge[name].is_connected = True
            logging.info("<SRV> %s : edge setup.", self._nodes[name].name)

        self._socket.send_json({"status": 200}, 0)

    def swap(self, name, kind):
        self._socket.send_json({"status": 200}, 0)
        edge = self._edge[name]
        edge.swap_receive_params(kind)


class MnwGateway(Optimizer):
    def __init__(self, name, edges, nodes, model, is_prm, defaults, device="cpu", weight=1.0, swap_timeout=10000):
        super(MnwGateway, self).__init__(model.parameters(), defaults)
        self._edge = OrderedDict()
        self._edge_num = len(edges)
        self._server_th = None
        self._info = {"node": {"training": True, "round": 0, "user": {}}, "func": None}

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('gloo', rank=nodes[name].index, world_size=len(nodes))
        is_prm["rcv_buf"] = True

        for edge_name, swap_cnt in edges.items():
            if edge_name not in self._edge.keys():
                self._edge[edge_name] = Edge(nodes[edge_name], model.parameters(), swap_cnt, device, is_prm)

        self._server_th = ServerHandler(name, self._edge, nodes, model, is_prm, self._info,
                                        device, swap_timeout=swap_timeout)
        self._server_th.start()

    def __del__(self):
        if self._server_th is not None:
            self._info["node"]["training"] = False
            self._server_th.join()

    def notice_train_ending(self):
        self._info["node"]["training"] = False
