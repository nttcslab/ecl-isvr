# -*- coding: utf-8 -*-
import logging

import torch

from .contract import Contract

__all__ = ["Ecl"]


class Ecl(Contract):
    def __init__(self, name, round_cnt, edges, hosts, model, device="cpu", lr=0.002,
                 drs=False, use_gcoef=False, piw: float = 1.0,
                 round_step=False, weight=1.0, swap_timeout=10):
        if drs:
            is_avg = True
        else:
            is_avg = False
        defaults = dict(lr=lr, round=round_cnt, initial_lr=lr,
                        piw=piw, use_gcoef=use_gcoef)
        super(Ecl, self).__init__(name, round_cnt, edges, hosts, model, defaults, device, round_step, weight,
                                  is_avg=is_avg, swap_timeout=swap_timeout)
        logging.info(f"Optimizer {type(self)} params: {defaults}")

    def __setstate__(self, state):
        super(Ecl, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        edges = self.edges()
        edge_nums = len(edges)

        for group in self.param_groups:
            lr = group["lr"]
            av_in_loop = group["round"]
            piw = group["piw"]
            use_gcoef = group["use_gcoef"]

            for i, p in enumerate(group['params']):
                state = self.state[p]
                m_grad = p.grad.data

                state['ctr_var'] = torch.zeros_like(p)
                state['denom'] = torch.zeros_like(p)
                state['v_grad'] = torch.zeros_like(p)

                ctr_var, denom, v_grad = (
                    state['ctr_var'],
                    state['denom'],
                    state['v_grad']
                )

                torch.nn.init.constant_(v_grad, (1 / lr))
                denom = v_grad.clone()

                eta = piw / (lr * (av_in_loop * edge_nums +
                                   av_in_loop - piw * edge_nums))

                for edge in edges:
                    if piw == 2:
                        edge.dual_avg[i] = torch.div(
                            (edge.dual_avg[i] + edge.prm_dual["rcv"][i]), 2)
                        ctr_var += edge.prm_a * edge.dual_avg[i] * eta
                    else:
                        ctr_var += edge.prm_a * edge.prm_dual["rcv"][i] * eta
                    denom += eta

                if not use_gcoef:
                    p.data = (v_grad * p.data - m_grad + ctr_var) / denom
                else:
                    gcoef = (1 - lr * eta * edge_nums * (av_in_loop/piw - 1))
                    p.data = ((v_grad * p.data - gcoef *
                               m_grad + ctr_var) / denom)

                for edge in edges:
                    edge.prm_state["snd"][i] = p.data
                    edge.prm_dual["snd"][i] = edge.prm_dual["rcv"][i] - \
                        2 * edge.prm_a * p.data

        self.swap_params("dual")
        self.round_update()

        if closure is not None:
            loss = closure()
        return loss
