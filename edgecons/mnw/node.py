# -*- coding: utf-8 -*-
import zmq


class Node:
    def __init__(self, name, addr, port, prm_a, index, weight=1.0, swap_timeout=10000):
        self._name = name
        self._addr = addr
        self._port = port
        self._prm_a = prm_a
        self._weight = weight
        self._index = index
        self._swap_timeout = swap_timeout
        self._info = {"node": {"training": True, "round": 0, "user": {}}, "func": None}

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect("tcp://" + addr + ":" + port)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

    def __del__(self):
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.close()
        self._context.term()

    def hello(self):
        self._socket.send_json({"kind": "hello", "src": self._name})
        self._socket.recv(0)

    def swap_info(self, info):
        self._socket.send_json({"kind": "info", "src": self._name, "info": info})
        self._info["node"] = self._socket.recv_json(0)
        return self._info["node"]

    def push_exec_func(self):
        self._socket.send_json({"kind": "function", "src": self._name})
        self._socket.recv_json(0)

    def swap_params(self, kind):
        is_con = True
        try:
            self._socket.send_json({"kind": kind, "src": self._name})
            if self._poller.poll(self._swap_timeout):
                self._socket.recv(0)
            else:
                is_con = False

        except zmq.ZMQError:
            is_con = False

        return is_con

    @property
    def name(self):
        return self._name

    @property
    def info(self):
        return self._info["node"]

    @info.setter
    def info(self, val):
        self._info["node"] = val

    @property
    def port(self):
        return self._port

    @property
    def prm_a(self):
        return self._prm_a

    @property
    def index(self):
        return self._index

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, val):
        self._weight = val

