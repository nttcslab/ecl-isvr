# -*- coding: utf-8 -*-
import time
import logging
import threading
import torch
import torch.nn as nn
from .gateway import MnwGateway
from .mnw.node import Node


class Contract(MnwGateway):
    def __init__(self, name, round_cnt, edges, hosts, model, defaults, device="cpu", round_step=True, weight=1.0,
                 is_state=True, is_dual=True, is_avg=False, is_c_prm=False, start_wait=60, swap_timeout=10):

        is_prm = {"state": is_state, "dual": is_dual, "avg": is_avg, "c_prm": is_c_prm, "rcv_buf": False}
        self._is_round_step = round_step
        self._nodes = {}
        self._com_time = 0.
        prm_a = 1
        swap_timeout_zmq = swap_timeout * 1000
        for i, host in enumerate(hosts):
            if host["name"] == name:
                prm_a = -1
            self._nodes[host["name"]] = Node(name, host["addr"], host["port"], prm_a, i, swap_timeout=swap_timeout_zmq)

        super(Contract, self).__init__(name, edges, self._nodes, model, is_prm, defaults, device, weight, swap_timeout)

        self._round_cnt = 0
        self._round_cnt_max = round_cnt
        self._prm_num = 0
        m_state = model.state_dict()
        for state_name in m_state:
            for dim in m_state[state_name].shape:
                self._prm_num += dim

        start_wait_cnt = 0
        while start_wait_cnt < start_wait:
            con_failed_cnt = 0
            for i, edge_name in enumerate(edges.keys()):
                if not self._edge[edge_name].is_connected:
                    con = self._edge[edge_name].hello()
                    if con:
                        logging.info("<CLI> %s : edge setup.", self._edge[edge_name].name)
                        self._nodes[edge_name].weight = self._edge[edge_name].weight
                        if self._edge[edge_name].req_state():
                            self._edge[edge_name].swap_params("state")
                            for group in self.param_groups:
                                for j, p in enumerate(group['params']):
                                    p.data = self._edge[edge_name].prm_state["rcv"][j]
                    else:
                        con_failed_cnt += 1

            if con_failed_cnt > 0:
                time.sleep(1)
                start_wait_cnt += 1
            else:
                break

    @property
    def round_cnt(self):
        return self._round_cnt

    def edges(self):
        return list(self._edge.values())

    def edge_nums(self):
        return len(list(self._edge.values()))

    def swap_params(self, kind="state"):
        disconnect_edges = []
        edges = self._edge.items()
        for name, edge in edges:
            if edge.is_swap_cnt(self._round_cnt):
                start_time = time.time()
                is_con = edge.swap_params(kind)
                self._com_time += time.time() - start_time
                if not is_con:
                    disconnect_edges.append(name)

        for edge_name in disconnect_edges:
            self._edge.pop(edge_name)
            logging.info(" => %s : edge disconnect.", self._nodes[edge_name].name)

    def round_update(self):
        self._round_cnt += 1
        if self._round_cnt_max <= self._round_cnt:
            self._round_cnt = 0
            if self._is_round_step:
                self._round_step()

    def _round_step(self):
        self._info["node"]["round"] += 1
        th_s = []
        for node in list(self._nodes.values()):
            th = threading.Thread(target=self._round_step_worker, args=(node,))
            th.start()
            th_s.append(th)

        for th in th_s:
            th.join()

    def _round_step_worker(self, node):
        ret = node.swap_info(self._info["node"])
        if self._is_round_step:
            while True:
                if self._info["node"]["round"] == ret["round"]:
                    break
                time.sleep(0.001)
                ret = node.swap_info(self._info["node"])

    def push_node_info(self, user_info: dict):
        self._info["node"]["user"] = user_info
        for node in list(self._nodes.values()):
            node.swap_info(self._info["node"])

    def node_info(self, node_name):
        return self._nodes[node_name].info["user"]

    def regist_callback_func(self, func):
        self._info["func"] = func

    def push_exec_func(self):
        for node in list(self._nodes.values()):
            node.push_exec_func()

    def get_communication_time(self):
        ret_time = self._com_time
        self._com_time = 0.
        return ret_time

    @torch.no_grad()
    def diff(self):
        diff = 0.
        state_sum = 0.
        criterion = nn.MSELoss(reduction="sum")

        for edge in self.edges():
            edge.swap_params("state")

            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    diff += criterion(p.data, edge.prm_state["rcv"][i])
                    state_sum += torch.mean(torch.abs(edge.prm_state["rcv"][i]))

        return diff / self._prm_num
