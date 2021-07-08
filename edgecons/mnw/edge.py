# -*- coding: utf-8 -*-
import copy
import io
import torch
import logging
import torch.distributed as dist
import threading
from abc import ABCMeta, abstractmethod


class EdgeBase(metaclass=ABCMeta):
    def __init__(self, node_info, parameters, swap_cnt, device, is_prm):
        self._node = node_info
        self._swap_cnt = swap_cnt
        self._device = device
        self._update_cnt = 1
        self._is_swap = False
        self._is_prm = is_prm

        self._prm_state = None
        self._prm_dual = None
        self._dual_avg = None
        self._c_prm = None
        self._state_prv = None
        self._is_connected = False

        self._prm_state = {"snd": [], "rcv": []}
        self._prm_dual = {"snd": [], "rcv": []}
        if is_prm["avg"]:
            self._dual_avg = []
        if is_prm["c_prm"]:
            self._c_prm = []
        if is_prm["rcv_buf"]:
            self._rcv_buf = []

        for p in parameters:
            self._prm_state["snd"].append(copy.deepcopy(p))
            self._prm_state["rcv"].append(copy.deepcopy(p))
            self._prm_dual["snd"].append(torch.zeros(p.size(), device=self._device))
            self._prm_dual["rcv"].append(torch.zeros(p.size(), device=self._device))
            if self._is_prm["avg"]:
                self._dual_avg.append(copy.deepcopy(p))
            if self._is_prm["c_prm"]:
                self._c_prm.append(copy.deepcopy(p))
            if self._is_prm["rcv_buf"]:
                self._rcv_buf.append(torch.zeros(p.size(), device="cpu"))

    def req_state(self):
        ret = False
        if self.prm_a > 0:
            ret = True
        return ret

    @abstractmethod
    def swap_params(self, kind="stat"):
        pass

    def is_swap_cnt(self, round_cnt):
        rsv_swap = False
        if round_cnt == self._swap_cnt:
            rsv_swap = True
        return rsv_swap

    @abstractmethod
    def hello(self):
        pass

    @property
    def name(self):
        return self._node.name

    @property
    def dst_index(self):
        return self._node.index

    @property
    def is_swap(self):
        return self._is_swap

    @is_swap.setter
    def is_swap(self, val):
        self._is_swap = val

    @property
    def prm_state(self):
        return self._prm_state

    @prm_state.setter
    def prm_state(self, val):
        self._prm_state = val

    @property
    def prm_dual(self):
        return self._prm_dual

    @prm_dual.setter
    def prm_dual(self, val):
        self._prm_dual = val

    @property
    def dual_avg(self):
        return self._dual_avg

    @dual_avg.setter
    def dual_avg(self, val):
        self._dual_avg = val

    @property
    def c_prm(self):
        return self._c_prm

    @c_prm.setter
    def c_prm(self, val):
        self._c_prm = val

    @property
    def rcv_buf(self):
        return self._rcv_buf

    @property
    def prm_a(self):
        return self._node.prm_a

    @property
    def weight(self):
        return self._node.weight

    @weight.setter
    def weight(self, weight):
        self._node.weight = weight

    @property
    def is_connected(self):
        return self._is_connected

    @is_connected.setter
    def is_connected(self, val):
        self._is_connected = val

    @property
    def update_cnt(self):
        return self._update_cnt

    @update_cnt.setter
    def update_cnt(self, val):
        self._update_cnt = val

    def set_state_rcv_params(self, params):
        try:
            self._prm_state["rcv"] = torch.load(io.BytesIO(params), map_location=torch.device(self._device))
            self._update_cnt = 1
            self._is_swap = True
        except RuntimeError:
            logging.warning("received data can't load (srv)")

    def set_dual_rcv_params(self, params):
        try:
            self._prm_dual["rcv"] = torch.load(io.BytesIO(params), map_location=torch.device(self._device))
            self._update_cnt = 1
            self._is_swap = True
        except RuntimeError:
            logging.warning("received data can't load (srv)")


class Edge(EdgeBase):
    def __init__(self, node_info, parameters, swap_cnt, device, is_prm, err_max_cnt=1):
        super().__init__(node_info, parameters, swap_cnt, device, is_prm)
        self._err_max_cnt = err_max_cnt
        self._params_lock = threading.Lock()

    def hello(self):
        self._node.hello()
        self.weight = 1.0
        self._is_connected = True
        return True

    def swap_params(self, kind="state"):
        is_con = self._node.swap_params(kind)
        try:
            if is_con:
                if kind == "dual":
                    self._swap(self._prm_dual)
                else:
                    self._swap(self._prm_state)

        except RuntimeError:
            logging.warning("received data can't load (cli)")

        return is_con

    def _swap(self, params):
        for i, rcv_buf in enumerate(self._rcv_buf):
            dist.send(tensor=params["snd"][i].to("cpu"), dst=self.dst_index, tag=i)

        self._params_lock.acquire()
        for i, rcv_buf in enumerate(self._rcv_buf):
            dist.recv(tensor=rcv_buf, src=self.dst_index, tag=i)
            params["rcv"][i] = self._rcv_buf[i].to(self._device)
        self._params_lock.release()

    def swap_receive_params(self, kind):
        if kind == "dual":
            swap_params = self._prm_dual
        else:
            swap_params = self._prm_state

        self._params_lock.acquire()
        for i, rcv_buf in enumerate(self._rcv_buf):
            dist.recv(tensor=rcv_buf, src=self.dst_index, tag=i)
            swap_params["rcv"][i] = rcv_buf.to(self._device)
        self._params_lock.release()

        for i, rcv_buf in enumerate(self._rcv_buf):
            dist.send(tensor=swap_params["snd"][i].to("cpu"), dst=self.dst_index, tag=i)

        self._update_cnt = 1
        self._is_swap = True

