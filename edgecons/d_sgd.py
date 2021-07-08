# -*- coding: utf-8 -*-
import logging

import torch

from .contract import Contract

__all__ = ["DSgd"]


class DSgd(Contract):
    def __init__(self, name, round_cnt, edges, hosts, model, device="cpu",
                 lr=0.002, momentum=0, dampening=0, weight_decay=0, nesterov=False, round_step=False, weight=1.0,
                 swap_timeout=10):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, initial_lr=lr)
        super(DSgd, self).__init__(name, round_cnt, edges, hosts, model, defaults, device, round_step, weight,
                                   is_dual=False, swap_timeout=swap_timeout)
        logging.info(f"Optimizer {type(self)} params: {defaults}")

    def __setstate__(self, state):
        super(DSgd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        edges = self.edges()
        edge_num = len(edges) + 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

                state = self.state[p]
                state['p_sum'] = torch.zeros_like(p)
                p_sum = (state['p_sum'])

                p_sum += p.data
                for edge in edges:
                    p_sum += edge.prm_state["rcv"][i]

                p.data = p_sum / edge_num
                for edge in edges:
                    edge.prm_state["snd"][i] = p.data

        self.swap_params("state")
        self.round_update()

        return loss
